package com.example.proposals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.sql.SQLOutput;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import ingest.FileSystemIngest;
import ingest.IngestResources;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
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
public class GenerateProposalsRagExample {

    public static void main(String[] args) {
        SpringApplication.run(GenerateProposalsRagExample.class, args);
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
//    @Qualifier("fileSystemProposalIngest")
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
}
