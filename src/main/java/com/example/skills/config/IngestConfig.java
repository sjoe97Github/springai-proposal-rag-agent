package com.example.skills.config;

import ingest.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.core.io.Resource;

@Configuration
public class IngestConfig {

    @Autowired
    private IngestProperties ingestProperties;

    @Bean("fileSystemResumeIngest")
    public IngestResources fileSystemResumeIngest(
            @Value("${app.scan.recursive:true}") boolean recursive,
            @Value("${app.scan.extensions:txt,pdf,doc,docx,md,html}") String includeExtensions) {
        Resource documentResource = new DefaultResourceLoader().getResource(ingestProperties.getResumeResources());
        return new FileSystemIngest(documentResource, recursive, includeExtensions);
    }

    @Bean("githubPromptSystemContext")
    public ChatPromptSystemContext githubPromptSystemContext() {
        return new GitHubLookupSystemContext();
    }

    @Bean("linkedInPromptSystemContext")
    public ChatPromptSystemContext linkedInPromptSystemContext() {
        return new LinkedInLookupSystemContext();
    }
}
