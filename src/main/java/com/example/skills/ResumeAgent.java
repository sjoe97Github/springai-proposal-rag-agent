package com.example.skills;

import match.AggregateGroupScoreType;
import match.AggregateScoringAlgorithm;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Component;

import java.util.*;
import java.util.stream.Collectors;

@Component
public class ResumeAgent {
    Logger logger = LoggerFactory.getLogger(ResumeAgent.class);

    @Value("${app.match.top-k-similarity}")
    private int similaritySearchTopK;

    @Value("${app.match.top-k-aggregate}")
    private int aggregateGroupsTopK;

    @Value("${app.match.softmax-temperature}")
    private double softmaxTemperature;

    @Value("${app.match.top-k-documents}")
    private long rerankedDocumentsTopK;

    @Value("${app.match.chunk-window-size}")
    private int chunkWindowSize;

    private final VectorStore vectorStore;
    private final HypotheticalSearchStrategy hypotheticalSearchStrategy;
    private final AggregateScoringAlgorithm aggregateScoringAlgorithm;
    private final JdbcTemplate jdbcTemplate;

    public ResumeAgent(VectorStore vectorStore,
                       HypotheticalSearchStrategy hypotheticalSearchStrategy,
                       AggregateScoringAlgorithm aggregateScoringAlgorithm,
                       JdbcTemplate jdbcTemplate) {
        this.vectorStore = vectorStore;
        this.hypotheticalSearchStrategy = hypotheticalSearchStrategy;
        this.aggregateScoringAlgorithm = aggregateScoringAlgorithm;
        this.jdbcTemplate = jdbcTemplate;
    }

    public List<Document> relevantResumes(String userPrompt) {
        return relevantResumes(userPrompt, true);
    }

    public List<Document> relevantResumes(String userPrompt, boolean useHypotheticalPrompt) {
        // Generate hypothetically ideal resume search prompt, only if requested (useHypotheticalPrompt = true).
        String resumePrompt = useHypotheticalPrompt ? hypotheticalSearchStrategy.generatePrompt(userPrompt) : userPrompt;
        System.out.println("Hypothetical Resume Prompt:\n" + resumePrompt);

        // Search for similar resumes
        SearchRequest searchRequest = SearchRequest.builder()
                .topK(similaritySearchTopK) // Increase top_k for better recall
                .query(resumePrompt)
                .build();

        // Use the hypothetical resume prompt for the similarity search
        List<Document> results = vectorStore.similaritySearch(searchRequest);
        for (Document doc : results) {
            logger.info("Pre-refined similarity matched vector ID: " + doc.getId());
        }

        //
        // ============================================================================================
        //
        // Note regarding similaritySearch results:
        //
        // The results returned by similaritySearch are only chunks of a resume and therefore not sufficiently
        // representative of an entire resume.
        //
        // Each chunk has metadata including "groupId" (same for all chunks of the same resume)
        // and "chunkIndex" (the position of the chunk within the resume).  This metadata can be used to associate
        // chunks with other chunks of their parent resume, and to order chunks within the resume.
        //
        // ============================================================================================
        //

        Map<String, Double> rerankedGroups = groupAndScore(results);

        // Pick the top groupIds from reranked groups.
        List<String> topGroupIds = topGroupIds(rerankedGroups);

        /*
            Gather all chunks for each top groupId and re-score score the entire group which effectively
            scores resumes instead of a subset of resume chunks.
         */
        Map<String, Double> groupScores = gatherGroupChunksAndScoreGroups(topGroupIds, resumePrompt);

        // Order the top groupIds by their newly computed group (resume) scores
        List<String> orderedGroupIds = topGroupIds.stream()
            .sorted((grpId_a,grpId_b) -> Double.compare(groupScores.get(grpId_b), groupScores.get(grpId_a)))
            .toList();

        /*
            For each top groupId, aggregate (assemble) all chunks in the group, essentially reconstructing the resume
            represented by the groupId.  Ultimately, an LLM can be used to summarize or extract key points from all
            chunks in the group
        */
        return gatherGroupChunksTogether(orderedGroupIds, resumePrompt, groupScores);
    }

    private List<String> topGroupIds(Map<String, Double> rerankedGroups) {
        return rerankedGroups.entrySet().stream()
                .sorted((a, b) -> Double.compare(b.getValue(), a.getValue()))
                .limit(rerankedDocumentsTopK)
                .map(Map.Entry::getKey)
                .toList();
    }

    private Map<String, Double> gatherGroupChunksAndScoreGroups(List<String> topGroupIds, String userPrompt) {
        Map<String, Double> groupScores = new HashMap<>();
        for (String gid : topGroupIds) {
            List<Document> allGroupDocs = vectorStore.similaritySearch(
                    SearchRequest.builder()
                            .query(userPrompt)
                            .topK(aggregateGroupsTopK) // presumably large enough to cover whole doc
                            .filterExpression("groupId == '" + gid + "'")
                            .build()
            );

            groupScores.put(gid, aggregateGroupScore(allGroupDocs));
        }
        return groupScores;
    }

    /**
     * Organize chunks by groupId (by resume) and compute a score for each group. Then return a map of groupId to score
     * which is effectively a map of resume to score.
     *
     * @param docs - list of chunks (documents) returned by similarity search
     * @return map of groupId to score (effectively a map of resume to score)
     */
    private Map<String, Double> groupAndScore(List<Document> docs) {
        Map<String, Double> scoresByGroupId = new HashMap<>();

        Map<String, List<Document>> documentsByGroupId = docs.stream()
                .filter(d -> d.getMetadata().get("groupId") != null)
                .collect(Collectors.groupingBy(d -> (String) d.getMetadata().get("groupId")));

        for (var e : documentsByGroupId.entrySet()) {
            scoresByGroupId.put(e.getKey(), aggregateGroupScore(e.getValue()));
        }

        return scoresByGroupId;
    }

    private double aggregateGroupScore(List<Document> docs) {
        AggregateGroupScoreType type = AggregateGroupScoreType.fromString(aggregateScoringAlgorithm.getScoringAlgorithm());
        if (type == null) {
            return 0.0;
        }
        return switch (type) {
            case SUM -> aggregatedSumScore(docs);
            case AVG -> aggregatedAvgScore(docs);
            case MAX -> aggregatedMaxScore(docs);
            case SOFTMAX -> aggregatedSoftMaxScore(docs);
        };
    }

    private double aggregatedAvgScore(List<Document> docs) {
        return docs.stream().mapToDouble(Document::getScore).average().orElse(0.0);
    }

    private double aggregatedSumScore(List<Document> docs) {
        return docs.stream().mapToDouble(Document::getScore).sum();
    }

    private double aggregatedMaxScore(List<Document> docs) {
        return docs.stream().mapToDouble(Document::getScore).max().orElse(0.0);
    }

    private double aggregatedSoftMaxScore(List<Document> docs) {
        double score = 0.0;

        // Note regarding softmax temperature:
        //
        // 0.05–0.2 works well ...
        // Higher the score result in a flatter or more even distribution.
        // Lower scores result in a distribution of varying peaks, the higher the peak the heavier the weight.
        // (see https://en.wikipedia.org/wiki/Softmax_function#Temperature)

        double[] documentScores = docs.stream().mapToDouble(Document::getScore).toArray();
        double maxDocumentScore = Arrays.stream(documentScores).max().orElse(0);
        double denominator = Arrays.stream(documentScores)
                .map(v -> Math.exp((v - maxDocumentScore)/softmaxTemperature))
                .sum();

        score = Arrays.stream(documentScores)
                .map(v -> (Math.exp((v - maxDocumentScore) / softmaxTemperature) / denominator) * v)
                .sum();

        return score;
    }

    List<Document> fetchAllGroupChunks(String groupId, String userPrompt) {
        // TODO - Why 20? Consider making configurable?
        int maxChunksPerGroup = 20;

        // TODO - Consider making filter and query configurable as different vector stores may require different values.
        SearchRequest sr = SearchRequest.builder()
                // The query can't be empty otherwise an exception is thrown; however,
                // we’re filtering by groupId and not relying on similarity here so the query is not important.
                .query(userPrompt)
                .topK(maxChunksPerGroup)
                .filterExpression("groupId == '" + groupId + "'")
                .build();

        return vectorStore.similaritySearch(sr);
    }

    public List<Document> gatherGroupChunksTogether(List<String> topGroupIds, String userPrompt, Map<String, Double> groupScores) {
        List<Document> results = new ArrayList<>();

        for (String gid : topGroupIds) {
            List<Document> windowedChunks = centeredGroupWindow(fetchAllGroupChunks(gid, userPrompt), chunkWindowSize);

            if (windowedChunks.isEmpty()) {
                logger.warn("No chunks found for groupId: " + gid);
                continue;
            }

            // "stitch" the chunks together to form a single text blob,
            String stitched = windowedChunks.stream()
                    .map(Document::getText)
                    .collect(Collectors.joining("\n---\n"));

            // There must be at least one Document (chunk) for the groupId from which to get the filename.
            Map<String, Object> metadata = windowedChunks.getFirst().getMetadata();
            metadata.put("score", groupScores.getOrDefault(gid, 0.0d));
            results.add(new Document(stitched, metadata));
        }

        return results;
    }

    protected static List<Document> centeredGroupWindow(List<Document> chunks, int windowSize) {
        if (chunks.isEmpty()) return chunks;
        if (chunks.size() <= windowSize) return chunks;

        // Highest scoring chunk
        Document bestChunk = Collections.max(chunks, Comparator.comparingDouble(d -> d.getScore() != null ? d.getScore() : 0.0));
        int indexOfBestChunk = Integer.parseInt(bestChunk.getMetadata().getOrDefault("chunkIndex", 0).toString());
        int halfWindowSize = windowSize / 2;
        int start;
        int end;

        // If highest scoring chunk is close enough to the start, then start index = 0
        if ((indexOfBestChunk - halfWindowSize) <= 0) {
            start = 0;
            end = windowSize - 1;
        } else if ((indexOfBestChunk + halfWindowSize) >= chunks.size()) {
            // If highest scoring chunk is close enough to the end, then start index = end - (windowSize - 1)
            end = chunks.size() - 1;
            start = end - windowSize;
        } else {
            // Otherwise center the window around the highest scoring chunk
            start = indexOfBestChunk - halfWindowSize;
            end = indexOfBestChunk + halfWindowSize;
        }

        return chunks.stream()
                .sorted(Comparator.comparingInt(d -> Integer.parseInt(d.getMetadata().getOrDefault("chunkIndex", 0).toString())))
                .filter(d -> {
                    int idx = Integer.parseInt(d.getMetadata().getOrDefault("chunkIndex", 0).toString());
                    return idx >= start && idx <= end;
                })
                .toList();
    }
}
