package com.example.skills.vector;

import ingest.IngestResources;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Qualifier;

public interface VectorInitializer {
    void initialize(VectorStore vectorStore, IngestResources resourceIngest) throws Exception;
}
