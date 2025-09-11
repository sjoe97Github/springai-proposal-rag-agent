package com.example.proposals;

import java.util.*;

import ingest.IngestResources;
import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.HttpClientSseClientTransport;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.PromptChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.document.Document;
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.Resource;
import org.springframework.jdbc.core.simple.JdbcClient;

@SpringBootApplication
public class ResumeIngestRunner {

    public static void main(String[] args) {
        SpringApplication.run(ResumeIngestRunner.class, args);
    }

    @Bean
    ChatClient chatClient(ChatClient.Builder chatClientBuilder) {
        return chatClientBuilder.build();
    }

    // Batch size for pushing to the vector store
    @Value("${app.ingest.batchSize}")
    private int batchSize;

    @Value("${app.ingest.chunkSize}")
    private int chunkSize;

    @Autowired
    @Qualifier("fileSystemResumeIngest")
    private IngestResources fileSystemIngest;

    @Bean
    ApplicationRunner applicationRunner(VectorStore vectorStore) {
        return args -> {
            TextSplitter splitter = TokenTextSplitter.builder().withChunkSize(chunkSize).build();

            List<Resource> fileResources = fileSystemIngest.getResources();

            // Read → split → index in batches
            List<Document> buffer = new ArrayList<>(batchSize);
            for (Resource res : fileResources) {
                List<Document> docs = new TikaDocumentReader(res).get();
                List<Document> splitDocs = splitter.apply(docs);
                for (Document d : splitDocs) {
                    buffer.add(d);
                    if (buffer.size() >= batchSize) {
                        vectorStore.accept(buffer);
                        buffer.clear();
                    }
                }
            }
            if (!buffer.isEmpty()) {
                vectorStore.accept(buffer);
            }
        };
    }
    @Bean
    ApplicationRunner toolDebugger(ChatClient chatClient) {
        return args -> {
            String debugPrompt = "What tools are available to you?";
            ChatResponse response = chatClient.prompt(debugPrompt).call().chatResponse();
            System.out.println("Available tools: " + response.getResult().getOutput().getText());
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

//        var system = """
//                You are an AI powered assistant to help people adopt a dog from the adoption\s
//                agency named Pooch Palace with locations in Antwerp, Seoul, Tokyo, Singapore, Paris,\s
//                Mumbai, New Delhi, Barcelona, San Francisco, and London. Information about the dogs available\s
//                will be presented below. If there is no information, then return a polite response suggesting we\s
//                don't have any dogs available.
//                """;
        var system = """
                You are a helpful GitHub research assistant.
                You may call the "list_repos" tool to list user repositories.
                Only return repository names, URLs, and languages.
                Include repositories that were forked from another repository.
            
                Use tool results to answer clearly and concisely.
                Always respond with valid JSON only. No other text allowed.
                The tool returns a list of repositories in a JSON format similar to this example:
                [
                    {
                        "url":"https://github.com/bswanson58/NoiseMusicSystem",
                        "visibility":"PUBLIC",
                        "language":"C#",
                        "createdAt":1372698378.000000000,
                        "updatedAt":1676375106.000000000,
                        "pushedAt":1697816613.000000000
                    }
                ]
                """;
        return builder
                .defaultToolCallbacks(new SyncMcpToolCallbackProvider(githubMcpSyncClient))
                .defaultSystem(system)
                .build();
    }
}
