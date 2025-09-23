package com.example.skills;

import com.example.skills.vector.VectorInitializer;
import com.example.skills.vector.VectorStoreMaintenanceService;
import com.sun.istack.logging.Logger;
import ingest.ChatPromptSystemContext;
import ingest.IngestResources;
import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.HttpClientSseClientTransport;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider;
import org.springframework.ai.tool.ToolCallbackProvider;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

import com.example.skills.config.IngestProperties;
import org.springframework.context.annotation.ComponentScan;

@SpringBootApplication
@EnableConfigurationProperties(IngestProperties.class)
@ComponentScan(basePackages = {"com.example.skills", "match"})
public class SkillsMatcherApplication {
    private static final Logger logger = Logger.getLogger(SkillsMatcherApplication.class);

    public static void main(String[] args) {
        SpringApplication.run(SkillsMatcherApplication.class, args);
    }

    @Autowired
    private IngestProperties ingestProperties;

    @Autowired
    private VectorStoreMaintenanceService vectorStoreMaintenanceService;

    @Autowired
    @Qualifier("pgVectorInitializer")
    VectorInitializer vectorInitializer;

    @Autowired
    @Qualifier("githubPromptSystemContext")
    private ChatPromptSystemContext gitHubLookupSystemContext;

    @Bean
    ApplicationRunner startSkillsMatcher(VectorStore vectorStore,
                                         @Qualifier("fileSystemResumeIngest") IngestResources resourceIngest) {
        return args -> {
            vectorInitializer.initialize(vectorStore, resourceIngest);
        };
    }

    @Bean
    ApplicationRunner toolDebugger(ChatClient.Builder chatClientBuilder, ToolCallbackProvider tools) {
        // Create ChatClient with MCP tools
        ChatClient chatClient = chatClientBuilder
                .defaultSystem("Answer all questions with complete sentences.")
                .defaultToolCallbacks(tools)
                .build();

        return args -> {
            String debugPrompt = "What tools are available to you?";
            ChatResponse response = chatClient.prompt(debugPrompt).call().chatResponse();
            logger.info("Available tools: " + response.getResult().getOutput().getText());
        };
    }

    @Bean
    McpSyncClient githubMcpSyncClient(@Value("${github-mcp-server-url}") String githubMcpServerUrl) {
        var mcp = McpClient
                .sync(HttpClientSseClientTransport
                        .builder(githubMcpServerUrl)
                        .build())
                .build();
        mcp.initialize();
        return mcp;
    }

    @Bean
    ChatClient githubMcpServerChatClient(
            ChatClient.Builder builder,
            McpSyncClient githubMcpSyncClient
            ) {
        return builder
                .defaultToolCallbacks(new SyncMcpToolCallbackProvider(githubMcpSyncClient))
                .defaultSystem(gitHubLookupSystemContext.getSystemContext())
                .build();
    }
}
