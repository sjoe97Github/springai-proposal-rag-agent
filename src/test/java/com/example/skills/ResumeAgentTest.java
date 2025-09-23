package com.example.skills;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class ResumeAgentTest {
    List<Document> chunksSkewedTowardStart;
    List<Document> chunksSkewedTowardEnd;
    List<Document> chunksSkewedTowardMiddle;
    @BeforeEach
    void setUp() {
        chunksSkewedTowardStart = List.of(
                Document.builder().text("Chunk 1.").metadata(Map.of("chunkIndex", "0")).score(1.0).build(),
                Document.builder().text("Chunk 2.").metadata(Map.of("chunkIndex", "1")).score(10.0).build(),
                Document.builder().text("Chunk 3.").metadata(Map.of("chunkIndex", "2")).score(3.0).build(),
                Document.builder().text("Chunk 4.").metadata(Map.of("chunkIndex", "3")).score(4.0).build(),
                Document.builder().text("Chunk 5.").metadata(Map.of("chunkIndex", "4")).score(5.0).build()
        );
        chunksSkewedTowardEnd = List.of(
                Document.builder().text("Chunk 1.").metadata(Map.of("chunkIndex", "0")).score(1.0).build(),
                Document.builder().text("Chunk 2.").metadata(Map.of("chunkIndex", "1")).score(2.0).build(),
                Document.builder().text("Chunk 3.").metadata(Map.of("chunkIndex", "2")).score(3.0).build(),
                Document.builder().text("Chunk 4.").metadata(Map.of("chunkIndex", "3")).score(10.0).build(),
                Document.builder().text("Chunk 5.").metadata(Map.of("chunkIndex", "4")).score(4.0).build()
        );
        chunksSkewedTowardMiddle = List.of(
                Document.builder().text("Chunk 1.").metadata(Map.of("chunkIndex", "0")).score(1.0).build(),
                Document.builder().text("Chunk 2.").metadata(Map.of("chunkIndex", "1")).score(2.0).build(),
                Document.builder().text("Chunk 3.").metadata(Map.of("chunkIndex", "2")).score(10.0).build(),
                Document.builder().text("Chunk 4.").metadata(Map.of("chunkIndex", "3")).score(3.0).build(),
                Document.builder().text("Chunk 5.").metadata(Map.of("chunkIndex", "4")).score(4.0).build()
        );
    }

    @Test
    void testGroupSizeLessWindowSize() {
        int windowSize = 8;

        List<Document> chunkWindow = ResumeAgent.centeredGroupWindow(chunksSkewedTowardEnd, windowSize);

        assertEquals(chunkWindow, chunksSkewedTowardEnd);
    }

    @Test
    void testGroupSizeEqualWindowSize() {
        int windowSize = 7;

        List<Document> chunkWindow = ResumeAgent.centeredGroupWindow(chunksSkewedTowardEnd, windowSize);

        assertEquals(chunkWindow, chunksSkewedTowardEnd);
    }

    @Test
    void testGroupSizeLessWindowSizeSkewedToStart() {
        int windowSize = 3;
        List<Integer> expectedChunks = List.of(0, 1, 2);

        List<Document> chunkWindow = ResumeAgent.centeredGroupWindow(chunksSkewedTowardStart, windowSize);

        List<Integer> actualChunks = chunkWindow.stream()
                .map(doc -> Integer.parseInt(doc.getMetadata().get("chunkIndex").toString()))
                .collect(java.util.stream.Collectors.toList());

        assertEquals(expectedChunks, actualChunks);
    }

    @Test
    void testGroupSizeLessWindowSizeSkewedToEnd() {
        int windowSize = 3;
        List<Integer> expectedChunks = List.of(2, 3, 4);

        List<Document> chunkWindow = ResumeAgent.centeredGroupWindow(chunksSkewedTowardEnd, windowSize);

        List<Integer> actualChunks = chunkWindow.stream()
                .map(doc -> Integer.parseInt(doc.getMetadata().get("chunkIndex").toString()))
                .collect(java.util.stream.Collectors.toList());

        assertEquals(expectedChunks, actualChunks);
    }

    @Test
    void testGroupSizeLessWindowSizeSkewedToMiddle() {
        int windowSize = 3;
        List<Integer> expectedChunks = List.of(1, 2, 3);

        List<Document> chunkWindow = ResumeAgent.centeredGroupWindow(chunksSkewedTowardMiddle, windowSize);

        List<Integer> actualChunks = chunkWindow.stream()
                .map(doc -> Integer.parseInt(doc.getMetadata().get("chunkIndex").toString()))
                .collect(java.util.stream.Collectors.toList());

        assertEquals(expectedChunks, actualChunks);
    }
}