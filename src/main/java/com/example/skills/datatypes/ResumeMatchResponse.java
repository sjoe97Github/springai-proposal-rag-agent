package com.example.skills.datatypes;

import java.util.List;

public record ResumeMatchResponse(
        String sessionId,
        String query,
        List<ResumeResult> results
) {
}
