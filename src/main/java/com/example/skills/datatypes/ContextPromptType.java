package com.example.skills.datatypes;

public enum ContextPromptType {
    GITHUB,
    LINKEDIN,
    SKILLSQUERY;

    public static ContextPromptType fromString(String type) {
        return switch (type.toLowerCase()) {
            case "github" -> GITHUB;
            case "linkedin" -> LINKEDIN;
            case "skillsquery" -> SKILLSQUERY;
            default -> null;
        };
    }
}

