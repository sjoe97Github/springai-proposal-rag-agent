package com.example.skills;

import com.example.skills.datatypes.*;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import ingest.ChatPromptSystemContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.template.st.StTemplateRenderer;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.ApplicationContext;
import org.springframework.core.io.Resource;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@RestController
@RequestMapping("/resume-match")
public class ResumeMatchController {
    Logger logger = LoggerFactory.getLogger(ResumeMatchController.class);

    @Value("classpath:/resume-ranking-template.txt")
    private Resource defaultPromptTemplate;

    @Value("classpath:/mcp-repos-system-context.txt")
    private Resource mcpReposSystemContext;

    @Autowired
    @Qualifier("githubPromptSystemContext")
    private ChatPromptSystemContext gitHubLookupSystemContext;

    @Autowired
    @Qualifier("linkedInPromptSystemContext")
    private ChatPromptSystemContext linkedInPromptSystemContext;

    @Autowired
    @Qualifier("skillsQueryPrompt")
    private ChatPromptSystemContext skillsQueryPrompt;

    private final ResumeAgent resumeAgent;

    private final ChatClient aiClient;
    private final ChatClient githubMcpServerChatClient;
//    private final ToolCallbackProvider toolCallbackProvider;

    private final ApplicationContext applicationContext;

    // Chat history by sessionId
    private final Map<String, List<Message>> chatHistories = new ConcurrentHashMap<>();

    // Custom prompt templates by sessionId
    private final Map<String, String> customPromptTemplates = new ConcurrentHashMap<>();

    private final ObjectMapper objectMapper = new ObjectMapper();

    public ResumeMatchController(ChatClient.Builder chatClientBuilder,
                                 ChatClient githubMcpServerChatClient,
                                 ToolCallbackProvider tools,
                                 ResumeAgent resumeAgent,
                                 ApplicationContext applicationContext) {
        this.resumeAgent = resumeAgent;
        this.applicationContext = applicationContext;

        // Create ChatClient with MCP tools
        this.aiClient = chatClientBuilder
                .defaultSystem("Answer all questions with complete sentences.")
                .defaultToolCallbacks(tools)
                .build();

        this.githubMcpServerChatClient = githubMcpServerChatClient;
    }

    @PostMapping("/query")
    public ResumeMatchResponse matchResumes(@RequestBody ResumeMatchQuery query,
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

        List<Document> similarResumes = resumeAgent.relevantResumes(query.getQuery(), false);

        // TODO - Null check similarResumes?
        similarResumes = deduplicateResumes(similarResumes);

        String resumeContext = formatResumesForPrompt(similarResumes);

        // Get prompt template
        String promptText = skillsQueryPrompt.getSystemContext();

        // Create prompt with parameters
        Map<String, Object> params = new HashMap<>();
        params.put("input", query.getQuery());
        params.put("ranked_resumes", resumeContext);
        PromptTemplate template = PromptTemplate.builder()
                .renderer(
                        StTemplateRenderer
                        .builder().startDelimiterToken('<')
                        .endDelimiterToken('>')
                        .build())
                .template(promptText)
                .variables(params)
                .build();

        Prompt prompt = template.create(params);

        logger.info("Prompt: {}", prompt);

        ChatResponse chatResponse = aiClient.prompt(prompt).call().chatResponse();

        String response = chatResponse.getResult().getOutput().getText();

        logger.info("AI Response: {}", response);

        // Add to chat history
        chatHistory.add(new AssistantMessage(response));

        // Parse AI response using Jackson to ensure valid JSON
        // TODO - Re-evaluate the Jackson parsing approach given that the result being parsed is returned by the LLM
        //        and therefore has a non-deterministic shape (may not be valid JSON). Consider using a more flexible
        //        method to extract structured data from whatever shape is returned in the chat response.
        ResumeResult[] resumeResults = new ResumeResult[0];
        try {
            // Trim any leading/trailing ```json ``` wrapper, if present.
            response = response.trim();
            if (response.startsWith("```json")) {
                response = response.substring(7).trim();
            }
            if (response.endsWith("```")) {
                response = response.substring(0, response.length() - 3).trim();
            }

            // If the response string does not start and end with square brackets, add them to form a valid JSON array
            if (!response.trim().startsWith("[")) {
                response = "[" + response;
            }
            if (!response.trim().endsWith("]")) {
                response = response + "]";
            }
            resumeResults = objectMapper.readValue(response, ResumeResult[].class);
        } catch (JsonProcessingException e) {
            logger.warn("Failed to parse AI response: {}", e.getMessage());
        }

        // Create response
        ResumeMatchResponse result = new ResumeMatchResponse(
            sessionId,
            query.getQuery(),
            resumeResults != null ? Arrays.asList(resumeResults) : Collections.emptyList()
        );
        logger.info("result: {}", objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result));

        for (ResumeResult rr : result.results()) {
            List<GithubRep> repos = new ArrayList<>();
            if (rr.getGithub() != null) {
                repos = getRepositories(rr);
                rr.setReposList(repos);
            }
            logger.debug("Repos for candidate {}: {}", rr.getCandidateId(), repos);
        }

        String finalResult = objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result);
        logger.debug("Final Result: {}", finalResult);

        return result;
    }

    private List<GithubRep> getRepositories(ResumeResult resumeResult) {
        String githubUrl = resumeResult.getGithub().toString();
        String candidateId = resumeResult.getCandidateId();

        String response = "No response from AI client for candidate: " + candidateId;

//        String reposPromptTemplate = """
//            Use the tool `spring_ai_mcp_client_local_mcp_service_list_repos` to list all repositories for user %s.
//            Return only the tool result, not code.
//        """;
        String reposPromptTemplate = """
            list repositories for the GitHub url: %s
        """;

        logger.info("GitHub MCP Client System context: {}", gitHubLookupSystemContext.getSystemContext());

        String reposPrompt = String.format(reposPromptTemplate, githubUrl);

        ChatResponse reposChatResponse = githubMcpServerChatClient.prompt(PromptTemplate.builder()
                .template(reposPrompt).build().create())
                .system(gitHubLookupSystemContext.getSystemContext())
                .call().chatResponse();

        if (reposChatResponse != null) {
            // TODO - Guard against null result?
            Generation generatedResponse = reposChatResponse.getResult();
            response = generatedResponse.getOutput().getText();

            logger.info("\nGitHub Repos for candidate={}: {}", candidateId, response);
        } else {
            logger.warn(response);
        }

        // Using jackson to parse the JSON response into a list of GithubRep objects
        List<GithubRep> repos = new ArrayList<>();
        try {
            // TODO - Crude Workaround! If the response string does not start and end with square brackets,
            //                          add them to form a valid JSON array
            if (!response.trim().startsWith("[")) {
                response = "[" + response;
            }
            if (!response.trim().endsWith("]")) {
                response = response + "]";
            }
            repos = objectMapper.readValue(response, objectMapper.getTypeFactory().constructCollectionType(List.class, GithubRep.class));
        } catch (JsonProcessingException e) {
            logger.warn("Failed to parse github Repos response: {}", e.getMessage());
        }

        return repos;
    }

    @GetMapping("/chat/history/{sessionId}")
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
                        + "\n\nSKILL_QUERY:\n{input}\n\nRESUMES:\n{resume_contexts}\n\n"
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

            sb.append("CandidateID: ").append(metadata.getOrDefault("file", "unknown-" + i)).append("\n");
            sb.append("InitialScore: ").append(metadata.getOrDefault("score", 0.0d)).append("\n");
            sb.append("Path: ").append(metadata.getOrDefault("file", "unknown")).append("\n");
            sb.append("ResumeSnippet:\n").append(truncateText(doc.getText(), 2500)).append("\n\n");
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
