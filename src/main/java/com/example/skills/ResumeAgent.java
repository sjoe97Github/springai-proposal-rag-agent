package com.example.skills;

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
    private final JdbcTemplate jdbcTemplate;

    public ResumeAgent(VectorStore vectorStore, HypotheticalSearchStrategy hypotheticalSearchStrategy, JdbcTemplate jdbcTemplate) {
        this.vectorStore = vectorStore;
        this.hypotheticalSearchStrategy = hypotheticalSearchStrategy;
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

        /* Rerank similarity search chunks using softmax-weighted score.
            Organize chunks by groupId and re-score the groups of chunks as an aggregated whole, effectively scoring
            each group of related chunks.  Where related chunks are parts of the same resume.
            Map<String, Double> rerankedGroups = groupAndScore(results);
        */
        Map<String, Double> rerankedGroups = groupAndScore(results);

        // Pick the top groupIds from reranked groups.
        List<String> topGroupIds = topGroupIds(rerankedGroups);

        /*
            Gather all chunks for each top groupId and re-score score the entire group which effectively
            scores resumes instead of a subset of resume chunks.
         */
        Map<String, Double> groupScores = gatherGroupChunksAndScoreGroups(topGroupIds, resumePrompt);

        // Final doc ranking
        List<String> orderedGroupIds = topGroupIds.stream()
            .sorted((grpId_a,grpId_b) -> Double.compare(groupScores.get(grpId_b), groupScores.get(grpId_a)))
            .toList();

        /*
            For each top groupId, aggregate (assemble) all chunks in the group, essentially reconstructing the resume
            represented by the groupId.

            Other possible aggregation strategies:
             - Just return the single best chunk for each groupId
             - Return the top N chunks for each groupId
             - Return chunks until a certain token limit is reached
             - Return a "window" of chunks around the best scoring chunk
             - Return a collection of the highest scoring chunks that cover different sections of the resume
             - Use a clustering algorithm to identify and select representative chunks from the group
             - Use Maximal Marginal Relevance (MMR) to select diverse and relevant chunks
             - Use a graph-based approach to identify and select the most central chunks in the group
             - Use a machine learning model to predict the relevance of each chunk and select the top ones

            Ultimately LLM can be used to summarize or extract key points from all chunks in the group
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

            groupScores.put(gid, aggregatedSoftMaxScore(allGroupDocs));
        }
        return groupScores;
    }

    private Map<String, Double> groupAndScore(List<Document> docs) {
        Map<String, Double> scoresByGroupId = new HashMap<>();

        Map<String, List<Document>> documentsByGroupId = docs.stream()
                .filter(d -> d.getMetadata().get("groupId") != null)
                .collect(Collectors.groupingBy(d -> (String) d.getMetadata().get("groupId")));

        for (var e : documentsByGroupId.entrySet()) {
//            scoresByGroupId.put(e.getKey(), aggregatedSoftMaxScore(e.getValue()));
            scoresByGroupId.put(e.getKey(), e.getValue().stream().max(Comparator.comparingDouble(Document::getScore))
                    .map(Document::getScore)
                    .orElse(0.0));
        }

        return scoresByGroupId;
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
        int maxChunksPerGroup = 20;  // TODO - Why 20? Make configurable?
        SearchRequest sr = SearchRequest.builder()
                // query can be empty-ish; we’re filtering by groupId and not relying on similarity here
                .query(userPrompt)
                .topK(maxChunksPerGroup)
//                .filterExpression("metadata->>'groupId' == '" + groupId + "'")
                .filterExpression("groupId == '" + groupId + "'")
                .build();

        return vectorStore.similaritySearch(sr);
    }

    public List<Document> gatherGroupChunksTogether(List<String> topGroupIds, String userPrompt, Map<String, Double> groupScores) {
        List<Document> results = new ArrayList<>();

        for (String gid : topGroupIds) {
//            List<Document> allChunks = fetchAllGroupChunks(gid, userPrompt);
//            String stitched = allChunks.stream()
//                    .sorted(Comparator.comparingInt(sd -> ((Number)
//                            sd.getMetadata().getOrDefault("chunkIndex", 0)).intValue()))
//                    .map(Document::getText)
//                    .collect(Collectors.joining("\n---\n"));

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
