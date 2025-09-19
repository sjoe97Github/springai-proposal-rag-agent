package com.example.proposals;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class ResumeAgent {
    Logger logger = LoggerFactory.getLogger(ResumeAgent.class);

//    @Value("${spring.ai.ollama.embedding.options.top-k}")
    @Value("${spring.ai.openai.embedding.options.top-k}")
    private int topK;

    private final VectorStore vectorStore;
    private final HypotheticalSearchStrategy hypotheticalSearchStrategy;

    public ResumeAgent(VectorStore vectorStore, HypotheticalSearchStrategy hypotheticalSearchStrategy) {
        this.vectorStore = vectorStore;
        this.hypotheticalSearchStrategy = hypotheticalSearchStrategy;
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
                .topK(topK) // Increase top_k for better recall
                .query(resumePrompt)
                .build();

        // Use the hypothetical resume prompt for the similarity search
//        return vectorStore.similaritySearch(searchRequest);
        List<Document> results = vectorStore.similaritySearch(searchRequest);
        for (Document doc : results) {
            logger.info("Matched vector ID: " + doc.getId());
        }
        return results;
    }
}
