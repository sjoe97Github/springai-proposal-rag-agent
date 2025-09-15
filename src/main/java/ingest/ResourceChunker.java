package ingest;

import org.springframework.ai.document.Document;

import java.util.ArrayList;
import java.util.List;

public class ResourceChunker {
    public static List<Document> overlappingChunk(List<Document> documents, int chunkSize, int overlapSize) {
        List<Document> chunkedDocs = new ArrayList<>();
        for (Document originalDocument : documents) {
            String originalDocumentText = originalDocument.getText();
            // TODO - Could getText() actually return null?
            if (originalDocumentText == null) continue;

            int originalDocumentLength = originalDocumentText.length();
            int chunkStartPos = 0;
            while (chunkStartPos < originalDocumentLength) {
                int end = Math.min(chunkStartPos + chunkSize, originalDocumentLength);
                String chunkText = originalDocumentText.substring(chunkStartPos, end);
                Document chunkDoc = new Document(chunkText, originalDocument.getMetadata());
                chunkedDocs.add(chunkDoc);
                if (end == originalDocumentLength) break;
                chunkStartPos += (chunkSize - overlapSize);
            }
        }
        return chunkedDocs;
    }
}
