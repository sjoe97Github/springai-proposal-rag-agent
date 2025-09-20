package com.example.skills.vector;

import com.example.skills.config.IngestProperties;
import ingest.IngestResources;
import ingest.ResourceChunker;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;

@Component
@Qualifier("pgVectorInitializer")
public class PgVectorInitializer implements VectorInitializer {
    @Autowired
    private IngestProperties ingestProperties;

    @Autowired
    private VectorStoreMaintenanceService vectorStoreMaintenanceService;

    @Override
    public void initialize(VectorStore vectorStore, IngestResources resourceIngest) throws Exception {
        if (!ingestProperties.isSkipResumeIngest()) {
            if (vectorStoreMaintenanceService.countVectors() > 0) {
                vectorStoreMaintenanceService.clearPgVectorTable();
            }

            TextSplitter splitter = TokenTextSplitter.builder().withChunkSize(ingestProperties.getChunkSize()).build();

            List<Resource> fileResources = resourceIngest.getResources();

            // Read → split → index in batches
            List<Document> buffer = new ArrayList<>(ingestProperties.getBatchSize());
            for (Resource res : fileResources) {
                // Generate a unique groupId for all documents containing chunks of this current resource
                String groupId = UUID.randomUUID().toString();
                String fullyQualifiedFileName = res.getFile().getAbsolutePath();

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
                List<Document> overlappingSplits = ResourceChunker.overlappingChunk(splitDocs, ingestProperties.getChunkSize(), ingestProperties.getOverlapSize());

                for (int i = 0; i < overlappingSplits.size(); i++) {
                    Document d = overlappingSplits.get(i);
                    // Add index, groupId, and fully qualified filename to document metadata
                    documentMetadataDecorator(d, i, groupId, fullyQualifiedFileName);

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
    }

    private Map<String, Object> documentMetadataDecorator(Document document, int chunkIndex, String groupId, String qualifiedFileName) {
        Map<String, Object> metadata = document.getMetadata();

        metadata.put("file", qualifiedFileName);
        metadata.put("groupId", groupId);
        metadata.put("chunkIndex", chunkIndex);

        return metadata;
    }
}
