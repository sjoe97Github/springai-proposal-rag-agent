package com.example.skills.datatypes;

import com.fasterxml.jackson.annotation.*;

import java.net.URL;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class ResumeResult {
    private String candidateId;
    private String initialScore;
    private String relevanceScore;
    private int finalScore;
    private String shortExplanation;
    @JsonInclude(JsonInclude.Include.ALWAYS)
    @JsonProperty("linkedIn")
    @JsonSetter(nulls = Nulls.SET)
    private URL linkedin;
    @JsonInclude(JsonInclude.Include.ALWAYS)
    @JsonProperty("github")
    @JsonSetter(nulls = Nulls.SET)
    private URL github;

    //@JsonIgnore
    private List<GithubRep> reposList;

    public ResumeResult() {}

    // TODO - Replace with Builder pattern
    public ResumeResult(String candidateId, int finalScore, String shortExplanation, URL linkedin, URL github, List<GithubRep> reposList) {
        this.candidateId = candidateId;
        this.finalScore = finalScore;
        this.shortExplanation = shortExplanation;
        this.linkedin = linkedin;
        this.github = github;
        this.reposList = reposList;
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

    public String getShortExplanation() {
        return shortExplanation;
    }

    public void setShortExplanation(String shortExplanation) {
        this.shortExplanation = shortExplanation;
    }

    @JsonProperty("linkedIn")
    public URL getLinkedin() {
        return linkedin;
    }

    @JsonProperty("linkedIn")
    public void setLinkedin(URL linkedin) {
        this.linkedin = linkedin;
    }

    public URL getGithub() {
        return github;
    }

    public void setGithub(URL github) {
        this.github = github;
    }

    public List<GithubRep> getReposList() {
        return reposList;
    }

    public void setReposList(List<GithubRep> reposList) {
        this.reposList = reposList;
    }
}

