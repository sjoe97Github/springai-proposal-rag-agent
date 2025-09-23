package com.example.skills.datatypes;

public enum PromptType {
    GITHUB,
    LINKEDIN,
    SKILLSQUERY;

    public static PromptType fromString(String type) {
        return switch (type.toLowerCase()) {
            case "github" -> GITHUB;
            case "linkedin" -> LINKEDIN;
            case "skillsquery" -> SKILLSQUERY;
            default -> null;
        };
    }
}

