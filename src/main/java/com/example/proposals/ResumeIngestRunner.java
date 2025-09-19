package com.example.proposals;

import java.io.IOException;
import java.util.*;

import com.sun.istack.logging.Logger;
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
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.Resource;

import com.example.proposals.config.IngestProperties;

@SpringBootApplication
@EnableConfigurationProperties(IngestProperties.class)
public class ResumeIngestRunner {
    private static final Logger logger = Logger.getLogger(ResumeIngestRunner.class);

    public static void main(String[] args) {
        SpringApplication.run(ResumeIngestRunner.class, args);
    }

    @Bean
    ChatClient chatClient(ChatClient.Builder chatClientBuilder) {
        return chatClientBuilder.build();
    }

    @Autowired
    private IngestProperties ingestProperties;

    @Autowired
    @Qualifier("githubPromptSystemContext")
    private ChatPromptSystemContext gitHubLookupSystemContext;

    @Autowired
    private VectorStoreMaintenanceService vectorStoreMaintenanceService;

    private Map<String, Object> documentMetadataDecorator(Document document, String name, long size, long lastModified) {
        Map<String, Object> metadata = document.getMetadata();

        metadata.put("filename", name);
        metadata.put("size", size);
        metadata.put("lastModified", lastModified);

        return metadata;
    }

    @Bean
    ApplicationRunner initialize(VectorStore vectorStore,
                                 @Qualifier("fileSystemResumeIngest") IngestResources resourceIngest) {
        return args -> {
            if (!ingestProperties.isSkipResumeIngest()) {
                // drop existing vectors, if there are any ...
                // TODO - Improve the configuration and startup control because during testing/development there might be
                //        cases where the clearPgVectorTable() method was successfully called, but a subsequent ingest
                //        failure resulted in an empty vector store.  Therefore, the next startup would result in an
                //        error related to trying to truncate an already empty table.
                //
                if (vectorStoreMaintenanceService.countVectors() > 0) {
                    vectorStoreMaintenanceService.clearPgVectorTable();
                }

                TextSplitter splitter = TokenTextSplitter.builder().withChunkSize(ingestProperties.getChunkSize()).build();

                List<Resource> fileResources = resourceIngest.getResources();

                // Read → split → index in batches
                List<Document> buffer = new ArrayList<>(ingestProperties.getBatchSize());
                for (Resource res : fileResources) {
                    // TikaDocumentReader(res).get() reads the file resource res and returns a List<Document>,
                    // where each Document represents the content extracted from the file which is often a single document
                    // per file, but there could be more than one document depending on the file type.
                    List<Document> docs = new TikaDocumentReader(res).get();

                    //
                    // TODO - Is this overkill?  The point it to use the filename, size, and lastModified to check
                    //        whether or not the file is already in the vector store.  However, there is currently
                    //        no mechanism to lookup the vector by filename or metadata.   It would be possible to
                    //        simply maintain a simple lookup table representing the "memory" of what has been ingested;
                    //        and this table might use a sequence for the id and support a composite
                    //        key of filename+size+lastModified.   This downside of this approach is there is no
                    //        known way to define a foreign key relationship between this table and the underlying
                    //        vector store table; therefore there is no way to enforce referential integrity.
                    //
//                    docs.forEach(d -> {
//                        try {
//                            documentMetadataDecorator(d, res.getFilename(), res.contentLength(), res.lastModified());
//                        } catch (IOException e) {
//                            logger.info("Unable to get file resource metadata while ingesting document: " + d.getMetadata().get("source"));
//                        }
//                    });

                    // splitter.apply(docs) takes the list of Document objects and splits their text content into
                    // smaller chunks, according to the chunkSize specified when building the TokenTextSplitter.
                    // It returns a new List<Document>, where each Document contains a chunk of the original text
                    List<Document> splitDocs = splitter.apply(docs);

                    // TODO - Consider eliminating the splitDocs references and just put the splitter.apply(docs)
                    //        directly into the ResourceChunker call.
                    List<Document> overlappingSplits = ResourceChunker.overlappingChunk(splitDocs, ingestProperties.getChunkSize(), ingestProperties.getOverlapSize());

                    for (Document d : overlappingSplits) {
                        buffer.add(d);
                        if (buffer.size() >= ingestProperties.getBatchSize()) {
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
