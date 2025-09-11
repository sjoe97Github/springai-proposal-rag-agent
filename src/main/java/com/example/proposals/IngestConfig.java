package com.example.proposals;

import ingest.FileSystemIngest;
import ingest.IngestResources;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;

@Configuration
public class IngestConfig {

    @Bean("fileSystemResumeIngest")
    public IngestResources fileSystemResumeIngest(
            @Value("${app.ingest.resume-resources}") Resource documentResource,
            @Value("${app.scan.recursive:true}") boolean recursive,
            @Value("${app.scan.extensions:txt,pdf,doc,docx,md,html}") String includeExtensions) {
        return new FileSystemIngest(documentResource, recursive, includeExtensions);
    }
}
