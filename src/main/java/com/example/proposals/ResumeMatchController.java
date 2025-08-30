package com.example.proposals;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@RestController
@RequestMapping("/resume-match")
public class ResumeMatchController {

    @Value("${spring.ai.vectorstore.topk:4}")
    private int topK;

    @Value("classpath:/resume-ranking-template.txt")
    private Resource defaultPromptTemplate;

    private final ChatClient aiClient;
    private final VectorStore vectorStore;

    // Chat history by sessionId
    private final Map<String, List<Message>> chatHistories = new ConcurrentHashMap<>();

    // Custom prompt templates by sessionId
    private final Map<String, String> customPromptTemplates = new ConcurrentHashMap<>();

    private final ObjectMapper objectMapper = new ObjectMapper();

    public ResumeMatchController(ChatClient aiClient, VectorStore vectorStore) {
        this.aiClient = aiClient;
        this.vectorStore = vectorStore;
    }

    @PostMapping("/query")
    public ResumeMatchResponse matchResumes(@RequestBody JobQuery query,
                                            @RequestParam(required = false) String sessionId) throws JsonProcessingException {
        // Create session ID if not provided
        if (sessionId == null || sessionId.isEmpty()) {
            sessionId = UUID.randomUUID().toString();
        }

        // Get or create chat history
        List<Message> chatHistory = chatHistories.computeIfAbsent(sessionId, k -> new ArrayList<>());

        // Add user query to history
        UserMessage userMessage = new UserMessage(query.getQuery());
        chatHistory.add(userMessage);

        // Search for similar resumes
        SearchRequest searchRequest = SearchRequest.builder()
                .topK(topK)
                .query(query.getQuery())
                .build();

        List<Document> similarResumes = vectorStore.similaritySearch(searchRequest);
        // TODO - Null check similarResumes?
        similarResumes = deduplicateResumes(similarResumes);

        // Format resumes for prompt context
        String resumeContext = formatResumesForPrompt(similarResumes);

        // Get prompt template (custom or default)
        String promptText = getPromptTemplate(sessionId);

        // Create prompt with parameters
        PromptTemplate template = new PromptTemplate(promptText);
        Map<String, Object> params = new HashMap<>();
        params.put("input", query.getQuery());
        params.put("ranked_resumes", resumeContext);

        Prompt prompt = template.create(params);

        System.out.println("\nPrompt:\n" + prompt);

        ChatResponse chatResponse = aiClient.prompt(prompt).call().chatResponse();

        // Get AI response
        String response = chatResponse.getResult().getOutput().getText();

        System.out.println("\n\nAI Response: " + response);

        // Add to chat history
        chatHistory.add(new AssistantMessage(response));

        // Create response
        ResumeMatchResponse result = new ResumeMatchResponse();
        result.setSessionId(sessionId);
        result.setQuery(query.getQuery());
        result.setResults(response);
        result.setChatHistory(chatHistory);

        System.out.println("\n\nresult: " + objectMapper.writeValueAsString(result));
        return result;
    }

    @PostMapping("/set-prompt-template")
    public Map<String, String> setCustomPromptTemplate(@RequestBody PromptTemplateRequest request) {
        customPromptTemplates.put(request.getSessionId(), request.getPromptTemplate());

        return Map.of(
                "status", "success",
                "message", "Custom prompt template set successfully",
                "sessionId", request.getSessionId()
        );
    }

    @GetMapping("/chat-history/{sessionId}")
    public Map<String, Object> getChatHistory(@PathVariable String sessionId) {
        List<Message> history = chatHistories.getOrDefault(sessionId, Collections.emptyList());

        return Map.of(
                "sessionId", sessionId,
                "chatHistory", history
        );
    }

    private String getPromptTemplate(String sessionId) {
        if (customPromptTemplates.containsKey(sessionId)) {
            return customPromptTemplates.get(sessionId);
        } else {
            try {
                return new String(defaultPromptTemplate.getInputStream().readAllBytes());
            } catch (IOException e) {
                // Fallback default template
                return "You are a technical recruiter assistant. "
                        + "Given a user job/skills query and a set of candidate resumes, "
                        + "return a JSON array ranking the candidates from best to worst match. "
                        + "Each item must include: candidateId, finalScore (0-100), and shortExplanation."
                        + "\n\nUSER QUERY:\n{input}\n\nRESUMES:\n{resume_contexts}\n\n"
                        + "Return ONLY JSON like:\n"
                        + "[{\"candidateId\":\"...\", \"finalScore\": 0-100, \"shortExplanation\":\"...\"}]";
            }
        }
    }

    private String formatResumesForPrompt(List<Document> resumes) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < resumes.size(); i++) {
            Document doc = resumes.get(i);
            Map<String, Object> metadata = doc.getMetadata();

            sb.append("CandidateID: ").append(metadata.getOrDefault("source", "unknown-" + i)).append("\n");
            sb.append("InitialScore: ").append(String.format("%.4f", doc.getScore())).append("\n");
            sb.append("Path: ").append(metadata.getOrDefault("path", "unknown")).append("\n");
            sb.append("ResumeSnippet:\n").append(truncateText(doc.getText(), 1200)).append("\n---\n");
        }
        return sb.toString();
    }

    private String truncateText(String text, int maxLength) {
        if (text == null || text.length() <= maxLength) {
            return text;
        }
        return text.substring(0, maxLength) + "…";
    }

    private List<Document> deduplicateResumes(List<Document> resumes) {
        Set<String> seenIds = new HashSet<>();
        List<Document> uniqueResumes = new ArrayList<>();

        for (Document doc : resumes) {
            String id = (String) doc.getMetadata().getOrDefault("source", UUID.randomUUID().toString());
            if (seenIds.add(id)) {
                uniqueResumes.add(doc);
            }
        }
        return uniqueResumes;
    }
}

// Required data classes
class JobQuery {
    private String query;

    public String getQuery() { return query; }
    public void setQuery(String query) { this.query = query; }
}

class ResumeMatchResponse {
    private String sessionId;
    private String query;
    private String results;
    private List<Message> chatHistory;

    // Getters and setters
    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }

    public String getQuery() { return query; }
    public void setQuery(String query) { this.query = query; }

    public String getResults() { return results; }
    public void setResults(String results) { this.results = results; }

    public List<Message> getChatHistory() { return chatHistory; }
    public void setChatHistory(List<Message> chatHistory) { this.chatHistory = chatHistory; }
}

class PromptTemplateRequest {
    private String sessionId;
    private String promptTemplate;

    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }

    public String getPromptTemplate() { return promptTemplate; }
    public void setPromptTemplate(String promptTemplate) { this.promptTemplate = promptTemplate; }
}