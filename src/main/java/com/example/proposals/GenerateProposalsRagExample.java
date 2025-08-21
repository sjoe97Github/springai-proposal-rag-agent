package com.example.proposals;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.ApplicationRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.core.io.Resource;
import org.springframework.core.io.FileSystemResource;

@SpringBootApplication
public class GenerateProposalsRagExample {

    public static void main(String[] args) {
        SpringApplication.run(GenerateProposalsRagExample.class, args);
    }

    @Bean
    ChatClient chatClient(ChatClient.Builder chatClientBuilder) {
        return chatClientBuilder.build();
    }

    @Value("${app.resource}")
    private Resource documentResource;

    // Toggle recursion and basic filename filtering if desired
    @Value("${app.scan.recursive:true}")
    private boolean recursive;

    // Comma-separated list of lowercase extensions to include (empty = all)
    @Value("${app.scan.extensions:txt,pdf,doc,docx,md,html}")
    private String includeExtensions;

    // Batch size for pushing to the vector store
    @Value("${app.index.batchSize:100}")
    private int batchSize;

    @Bean
    ApplicationRunner applicationRunner(VectorStore vectorStore) {
        return args -> {
            TextSplitter splitter = new TokenTextSplitter();

            List<Resource> fileResources = resolveFileResources(documentResource, recursive, parseExtensions(includeExtensions));

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

    private static List<Resource> resolveFileResources(Resource root, boolean recursive, Set<String> includeExts) throws IOException {
        // Prefer filesystem paths for directory scanning
        if (root.isFile() && root.getFile().isDirectory()) {
            Path dir = root.getFile().toPath();
            Stream<Path> walker = recursive ? Files.walk(dir) : Files.list(dir);
            try (walker) {
                return walker
                        .filter(Files::isRegularFile)
                        .filter(p -> includeExts.isEmpty() || includeExts.contains(getExtLower(p)))
                        .sorted()
                        .map(FileSystemResource::new)
                        .collect(Collectors.toList());
            }
        }

        // Single file (or a non-filesystem resource). Just return it as-is.
        return List.of(root);
    }

    private static Set<String> parseExtensions(String extensions) {
        if (extensions == null || extensions.isBlank()) return Collections.emptySet();
        return Arrays.stream(extensions.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .map(s -> s.startsWith(".") ? s.substring(1) : s)
                .map(String::toLowerCase)
                .collect(Collectors.toCollection(LinkedHashSet::new));
    }

    private static String getExtLower(Path p) {
        String name = p.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot >= 0 ? name.substring(dot + 1).toLowerCase() : "";
    }
}
