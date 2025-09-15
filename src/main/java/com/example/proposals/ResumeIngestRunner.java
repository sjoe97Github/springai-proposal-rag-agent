package com.example.proposals;

import java.util.*;

import ingest.ChatPromptSystemContext;
import ingest.IngestResources;
import ingest.ResourceChunker;
import io.modelcontextprotocol.client.McpClient;
import io.modelcontextprotocol.client.McpSyncClient;
import io.modelcontextprotocol.client.transport.HttpClientSseClientTransport;
import org.springframework.ai.chat.client.ChatClient;
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

@SpringBootApplication
public class ResumeIngestRunner {

    public static void main(String[] args) {
        SpringApplication.run(ResumeIngestRunner.class, args);
    }

    @Bean
    ChatClient chatClient(ChatClient.Builder chatClientBuilder) {
        return chatClientBuilder.build();
    }

    @Value("${app.ingest.skipResumeIngest:false}")
    private boolean skipResumeIngest;

    @Value("${app.ingest.batchSize}")
    private int batchSize;

    @Value("${app.ingest.chunkSize}")
    private int chunkSize;

    @Value("${app.ingest.overlapSize}")
    private int overlapSize;

    @Autowired
    @Qualifier("githubPromptSystemContext")
    private ChatPromptSystemContext gitHubLookupSystemContext;

    @Bean
    ApplicationRunner initialize(VectorStore vectorStore,
                                 @Qualifier("fileSystemResumeIngest") IngestResources resourceIngest) {
        return args -> {
            if (!skipResumeIngest) {
                TextSplitter splitter = TokenTextSplitter.builder().withChunkSize(chunkSize).build();

                List<Resource> fileResources = resourceIngest.getResources();

                // Read → split → index in batches
                List<Document> buffer = new ArrayList<>(batchSize);
                for (Resource res : fileResources) {
                    // TikaDocumentReader(res).get() reads the file resource res and returns a List<Document>,
                    // where each Document represents the content extracted from the file which is often a single document
                    // per file, but there could be more than one document depending on the file type.
                    List<Document> docs = new TikaDocumentReader(res).get();

                    // splitter.apply(docs) takes the list of Document objects and splits their text content into
                    // smaller chunks, according to the chunkSize specified when building the TokenTextSplitter.
                    // It returns a new List<Document>, where each Document contains a chunk of the original text
                    List<Document> splitDocs = splitter.apply(docs);

                    // TODO - Consider eliminating the splitDocs references and just put the splitter.apply(docs)
                    //        directly into the ResourceChunker call.
                    List<Document> overlappingSplits = ResourceChunker.overlappingChunk(splitDocs, chunkSize, overlapSize);

                    for (Document d : overlappingSplits) {
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
        return builder
                .defaultToolCallbacks(new SyncMcpToolCallbackProvider(githubMcpSyncClient))
                .defaultSystem(gitHubLookupSystemContext.getSystemContext())
                .build();
    }
}
