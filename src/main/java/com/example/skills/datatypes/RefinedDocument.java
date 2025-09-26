package com.example.skills.datatypes;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.annotation.Nulls;

public class RefinedDocument {
    private String candidateId;
    private String initialScore;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String resumeSnippet;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String shortExplanation;
    private String resumeSummary;
    private String relevanceScore;
    private int finalScore;

    @JsonInclude(JsonInclude.Include.ALWAYS)
    @JsonProperty("github")
    @JsonSetter(nulls = Nulls.SET)
    private String github;

    @JsonInclude(JsonInclude.Include.ALWAYS)
    @JsonProperty("linkedIn")
    @JsonSetter(nulls = Nulls.SET)
    private String linkedIn;

    public RefinedDocument() {}
    public RefinedDocument(String candidateId, String initialScore, String resumeSnippet) {
        this.candidateId = candidateId;
        this.initialScore = initialScore;
        this.resumeSnippet = resumeSnippet;
    }
    public String getCandidateId() {
        return candidateId;
    }
    public void setCandidateId(String candidateId) {
        this.candidateId = candidateId;
    }
    public String getInitialScore() {
        return initialScore;
    }
    public void setInitialScore(String initialScore) {
        this.initialScore = initialScore;
    }
    public String getRelevanceScore() {
        return relevanceScore;
    }
    public void setRelevanceScore(String relevanceScore) {
        this.relevanceScore = relevanceScore;
    }

    public int getFinalScore() {
        return finalScore;
    }

    public void setFinalScore(int finalScore) {
        this.finalScore = finalScore;
    }

    public String getResumeSnippet() {
        return resumeSnippet;
    }
    public void setResumeSnippet(String resumeSnippet) {
        this.resumeSnippet = resumeSnippet;
    }
    public String getShortExplanation() {
        return shortExplanation;
    }
    public void setShortExplanation(String shortExplanation) {
        this.shortExplanation = shortExplanation;
    }
    public String getResumeSummary() {
        return resumeSummary;
    }
    public void setResumeSummary(String resumeSummary) {
        this.resumeSummary = resumeSummary;
    }
    public String getGithub() {
        return github;
    }
    public void setGithub(String github) {
        this.github = github;
    }
    @JsonProperty("linkedIn")
    public String getLinkedIn() {
        return linkedIn;
    }
    @JsonProperty("linkedIn")
    public void setLinkedIn(String linkedIn) {
        this.linkedIn = linkedIn;
    }
}
