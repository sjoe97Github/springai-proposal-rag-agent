package com.example.proposals;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.HttpClientSseClientTransport;
import io.modelcontextprotocol.spec.McpSchema;
import jakarta.annotation.PostConstruct;
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
import org.springframework.ai.mcp.SyncMcpToolCallback;
import org.springframework.ai.model.ModelResponse;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.Resource;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;
import java.net.URL;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/resume-match")
public class ResumeMatchController {
    // Add Class level logger
    Logger logger = LoggerFactory.getLogger(ResumeMatchController.class);

//    @Value("${spring.ai.ollama.embedding.options.top-k}")
    @Value("${spring.ai.openai.embedding.options.top-k}")
    private int topK;

    @Value("classpath:/resume-ranking-template.txt")
    private Resource defaultPromptTemplate;

    @Value("classpath:/mcp-repos-system-context.txt")
    private Resource mcpReposSystemContext;

    private final ResumeAgent resumeAgent;

    private final ChatClient aiClient;
    private final ChatClient githubMcpServerChatClient;
    private final VectorStore vectorStore;
//    private final ToolCallbackProvider toolCallbackProvider;

    // Chat history by sessionId
    private final Map<String, List<Message>> chatHistories = new ConcurrentHashMap<>();

    // Custom prompt templates by sessionId
    private final Map<String, String> customPromptTemplates = new ConcurrentHashMap<>();

    private final ObjectMapper objectMapper = new ObjectMapper();

    private final ApplicationContext applicationContext;

//    public ResumeMatchController(ChatClient.Builder chatClientBuilder, VectorStore vectorStore, ResumeAgent resumeAgent, ApplicationContext applicationContext) {
//        this.resumeAgent = resumeAgent;
//        this.vectorStore = vectorStore;
//        this.applicationContext = applicationContext;
//
//        // Create ChatClient with MCP tools
//        this.aiClient = createChatClientWithMcpTools(chatClientBuilder);
//    }
    public ResumeMatchController(ChatClient.Builder chatClientBuilder,
                                 ChatClient githubMcpServerChatClient,
                                 ToolCallbackProvider tools,
                                 VectorStore vectorStore,
                                 ResumeAgent resumeAgent,
                                 ApplicationContext applicationContext) {
        this.resumeAgent = resumeAgent;
        this.vectorStore = vectorStore;
        this.applicationContext = applicationContext;

        // Create ChatClient with MCP tools
        this.aiClient = chatClientBuilder
                .defaultSystem("Answer all questions with complete sentences.")
                .defaultToolCallbacks(tools)
                .build();

        this.githubMcpServerChatClient = githubMcpServerChatClient;
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

        List<Document> similarResumes = resumeAgent.relevantResumes(query.getQuery(), false);

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

        logger.info("Prompt: {}", prompt);

        ChatResponse chatResponse = aiClient.prompt(prompt).call().chatResponse();

//        String experimentalPromptText = """
//            In the curl statement, `curl -s https://api.github.com/users/user_segment/repos`, replace the `user-segment` portion of the url with
//            everything after the last `/` character in https://github.com/sjoe97Github
//            Execute the curl command to get a list of public repositories
//            Keep only the .html_url parts of the response.
//        """;

//        String experimentalPromptText = """
//            Use the tool named `list_repos` to return all of the repositories for https://github.com/sjoe97Github
//            Parameters: per_page=10, visibility=all, sort=updated
//            Show the tool request and response.
//            Do not fabricate the response.
//        """;
//        String experimentalPromptText = """
//                Get repositories for the owner of https://github.com/sjoe97Github.
//                Execute the tool and return the actual response data, not made up response.
//            """;
//        ChatResponse chatResponse = aiClient.prompt(PromptTemplate.builder().template(experimentalPromptText).build()
//                    .create()).toolNames("list_repos").call().chatResponse();
//
        // Get AI response
        String response = chatResponse.getResult().getOutput().getText();

        logger.info("AI Response: {}", response);

        // Add to chat history
        chatHistory.add(new AssistantMessage(response));

        // Parse AI response using Jackson to ensure valid JSON
        // TODO - Re-evaluate the Jackson parsing approach given that the result being parsed is returned by the LLM
        //        and therefore has a non-deterministic shape (may not be valid JSON). Consider using a more flexible
        //        method to extract structured data from whatever shape string is returned in the chat response.
        ResumeResult[] resumeResults = new ResumeResult[0];
        try {
            // TODO - Crude Workaround! If the response string does not start and end with square brackets,
            //                          add them to form a valid JSON array
            if (!response.trim().startsWith("[")) {
                response = "[" + response;
            }
            if (!response.trim().endsWith("]")) {
                response = response + "]";
            }
            resumeResults = objectMapper.readValue(response, ResumeResult[].class);
        } catch (JsonProcessingException e) {
            // TODO - Use a logging framework
            logger.warn("Failed to parse AI response: {}", e.getMessage());
        }

        // Create response
        ResumeMatchResponse result = new ResumeMatchResponse(
            sessionId,
            query.getQuery(),
            resumeResults != null ? Arrays.asList(resumeResults) : Collections.emptyList()
        );
        logger.info("result: {}", objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(result));

        // for each resume result, extract candidateId, linkedin, and github fields if present.
        for (ResumeResult rr : result.results()) {
            logger.info("CandidateID: {}, LinkedIn: {}, GitHub: {}}",
                    rr.candidateId(),
                    rr.linkedin() != null ? rr.linkedin().toString() : "N/A",
                    rr.github() != null ? rr.github().toString() : "N/A");

            String repos = "No Repos";
            if (rr.github() != null) {
                repos = getRepositories(rr);
            } else {
                repos = getRepositories("https://github.com/ai-ml-workshops", rr.candidateId());
            }
            logger.debug("Repos for candidate {}: {}", rr.candidateId(), repos);
        }

        return result;
    }

    private String getRepositories(ResumeResult resumeResult) {
        return getRepositories(resumeResult.github().toString(), resumeResult.candidateId());
    }

    private String getRepositories(String githubUrl, String candidateId) {
        String response = "No response from AI client for candidate: " + candidateId;

//        String reposPromptTemplate = """
//            Use the tool `spring_ai_mcp_client_local_mcp_service_list_repos` to list all repositories for user %s.
//            Return only the tool result, not code.
//        """;
        String reposPromptTemplate = """
            list repositories for the GitHub url: %s
        """;

        String reposPrompt = String.format(reposPromptTemplate, githubUrl);

        String promptText = getMcpReposSystemContext();
        logger.debug("GitHub MCP Client System context: {}", promptText);

//        ChatResponse reposChatResponse = githubMcpServerChatClient.prompt(PromptTemplate.builder()
//                .template(reposPrompt).build().create())
//                .system(promptText).call().chatResponse();
        ChatResponse reposChatResponse = githubMcpServerChatClient.prompt(PromptTemplate.builder()
                .template(reposPrompt).build().create())
                .call().chatResponse();

        if (reposChatResponse != null) {
            // Guard against null result
            Generation generatedResponse = reposChatResponse.getResult();
            response = generatedResponse.getOutput().getText();

            logger.info("\nGitHub Repos for candidate={}: {}", candidateId, response);
        } else {
            logger.warn(response);
        }


        return response;
    }

    private String getMcpReposSystemContext() {
        try {
            return new String(mcpReposSystemContext.getInputStream().readAllBytes());
        } catch (IOException e) {
            // Fallback ...
            return """
                You are a helpful GitHub research assistant.
                You may call the "list_repos" tool to list user repositories.
                Only return repository names, URLs, and languages.
            
                Use tool results to answer clearly and concisely.
                Always respond with valid JSON only. No other text allowed.
                Return the final answer in JSON format similar to this example:
                {
                  "answer": [
                    {
                      "full_name": "",
                      "language": "",
                      "description": null,
                      "created_at": "2025-01-26T03:36:26Z",
                      "updated_at": "2025-02-03T13:02:41Z",
                      "private": false
                    },
                    {
                      "full_name": "",
                      "url": "",
                      "language": "",
                      "description": null,
                      "created_at": "2025-01-26T03:36:26Z",
                      "updated_at": "2025-02-03T13:02:41Z",
                      "private": true
                    }
                  ],
                  "usedTool": "list_repos"
                  "note": "Some repositories are private and do not have public descriptions."
                }
             """;
        }
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
            sb.append("ResumeSnippet:\n").append(truncateText(doc.getText(), 2400)).append("\n---\n");
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

    private ChatClient createChatClientWithMcpTools(ChatClient.Builder chatClientBuilder) {
        try {
            // Get the MCP sync clients
            Object mcpSyncClients = applicationContext.getBean("mcpSyncClients");

            if (mcpSyncClients instanceof List) {
                List<?> clientsList = (List<?>) mcpSyncClients;
                List<ToolCallback> allToolCallbacks = new ArrayList<>();

                for (Object client : clientsList) {
                    if (client instanceof McpSyncClient) {
                        McpSyncClient syncClient = (McpSyncClient) client;

                        // Get all tools from this MCP client
                        var toolsResponse = syncClient.listTools();

                        // Create SyncMcpToolCallback for each tool
                        for (var tool : toolsResponse.tools()) {
                            SyncMcpToolCallback toolCallback = new SyncMcpToolCallback(syncClient, tool);
                            allToolCallbacks.add(toolCallback);
                            System.out.println("Registered MCP tool: " + tool.name());
                        }
                    }
                }

                // Check all MCP-related beans
                String[] allBeans = applicationContext.getBeanDefinitionNames();
                System.out.println("\nAll registered beans containing 'mcp' or 'tool':");
                Arrays.stream(allBeans)
                        .filter(name -> name.toLowerCase().contains("mcp") ||
                                name.toLowerCase().contains("tool"))
                        .forEach(beanName -> {
                            Object bean = applicationContext.getBean(beanName);
                            System.out.println(" - " + beanName + ": " + bean.getClass().getName());
                        });
                if (!allToolCallbacks.isEmpty()) {
                    System.out.println("Creating ChatClient with " + allToolCallbacks.size() + " MCP tools");
                    return chatClientBuilder
                            .defaultToolCallbacks(allToolCallbacks.toArray(new ToolCallback[0]))
                            .build();
                }
            }
        } catch (Exception e) {
            System.err.println("Error setting up MCP tools: " + e.getMessage());
            e.printStackTrace();
        }

        // Fallback to regular ChatClient without tools
        System.out.println("Creating ChatClient without MCP tools");
        return chatClientBuilder.build();
    }

    //@PostConstruct
    public void checkTools() {
        // Check if MCP client beans exist
        System.out.println("=== MCP Debug Information ===");

        Object mcpSyncClients = applicationContext.getBean("mcpSyncClients");
        System.out.println("Found mcpSyncClients bean: " + mcpSyncClients.getClass().getName());

        if (mcpSyncClients instanceof List) {
            List<?> clientsList = (List<?>) mcpSyncClients;
            System.out.println("MCP clients found: " + clientsList.size());

            for (int i = 0; i < clientsList.size(); i++) {
                Object client = clientsList.get(i);
                System.out.println("MCP Client[" + i + "]: " + client.getClass().getName());

                // Check if it's a SyncMcpClient or has access to tools
                if (client instanceof McpSyncClient) {
                    McpSyncClient syncClient = (McpSyncClient) client;
                    syncClient.listTools().tools().forEach(tool -> {
                        System.out.printf("  - Tool: %s, Description: %s%n", tool.name(), tool.description());
                        try {
                            McpSchema.JsonSchema schema = tool.inputSchema();
                            System.out.println("    Input Schema: " + objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(schema));
                        } catch (JsonProcessingException e) {
                            System.err.println("    Failed to parse input schema for tool " + tool.name() + ": " + e.getMessage());
                        }
                    });
                }
            }
        }

        try {
            ToolCallbackProvider toolProvider = applicationContext.getBean(ToolCallbackProvider.class);
            System.out.println("Found ToolCallbackProvider: " + toolProvider.getClass().getName());

            ToolCallback[] callbacks = toolProvider.getToolCallbacks();
            System.out.println("Available tools: " + callbacks.length);

            for (ToolCallback callback : callbacks) {
                ToolDefinition def = callback.getToolDefinition();
                System.out.println("  - Tool: " + def.name() + " | " + def.description());
            }

        } catch (Exception e) {
            System.out.println("No ToolCallbackProvider found: " + e.getMessage());
        }

        // Check all beans
        String[] allBeans = applicationContext.getBeanNamesForType(Object.class);
        System.out.println("Total beans: " + allBeans.length);

        // Filter for MCP-related beans
        List<String> mcpBeans = Arrays.stream(allBeans)
                .filter(name -> name.toLowerCase().contains("mcp") ||
                        name.toLowerCase().contains("tool") ||
                        name.toLowerCase().contains("stdio") ||
                        name.toLowerCase().contains("client"))
                .collect(Collectors.toList());

        System.out.println("MCP/Tool-related beans found: " + mcpBeans.size());
        mcpBeans.forEach(System.out::println);

        // Check ChatClient details
        System.out.println("ChatClient class: " + aiClient.getClass().getName());

        // Check if MCP client beans are actually created
        try {
            Object mcpClient = applicationContext.getBean("spring.ai.mcp.client.stdio.connections.local-mcp-service");
            System.out.println("MCP client bean found: " + mcpClient.getClass().getName());
        } catch (Exception e) {
            System.err.println("Error checking MCP setup: " + e.getMessage());
        }
    }
}

// Required data classes
class JobQuery {
    private String query;

    public String getQuery() { return query; }
    public void setQuery(String query) { this.query = query; }
}

class PromptTemplateRequest {
    private String sessionId;
    private String promptTemplate;

    public String getSessionId() { return sessionId; }
    public void setSessionId(String sessionId) { this.sessionId = sessionId; }

    public String getPromptTemplate() { return promptTemplate; }
    public void setPromptTemplate(String promptTemplate) { this.promptTemplate = promptTemplate; }
}

record ResumeMatchResponse(
    String sessionId,
    String query,
    List<ResumeResult> results
) {}

record ResumeResult(
        String candidateId,
        int finalScore,
        String shortExplanation,
        URL linkedin,
        URL github
) {}
