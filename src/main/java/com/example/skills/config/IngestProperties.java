package com.example.skills.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app.ingest", ignoreUnknownFields = false)
public class IngestProperties {
    private boolean skipResumeIngest = false;
    private int batchSize;
    private int chunkSize;
    private int overlapSize;
    private String resumeResources;

    public boolean isSkipResumeIngest() {
        return skipResumeIngest;
    }
    public void setSkipResumeIngest(boolean skipResumeIngest) {
        this.skipResumeIngest = skipResumeIngest;
    }
    public int getBatchSize() {
        return batchSize;
    }
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
    public int getChunkSize() {
        return chunkSize;
    }
    public void setChunkSize(int chunkSize) {
        this.chunkSize = chunkSize;
    }
    public int getOverlapSize() {
        return overlapSize;
    }
    public void setOverlapSize(int overlapSize) {
        this.overlapSize = overlapSize;
    }
    public String getResumeResources() {
        return resumeResources;
    }
    public void setResumeResources(String resumeResources) {
        this.resumeResources = resumeResources;
    }
}

