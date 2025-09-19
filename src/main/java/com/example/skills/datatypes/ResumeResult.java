package com.example.skills.datatypes;

import java.net.URL;
import java.util.List;

public class ResumeResult {
    private String candidateId;
    private int finalScore;
    private String shortExplanation;
    private URL linkedin;
    private URL github;

//    @JsonIgnore
    private List<GithubRep> reposList;

    public ResumeResult() {}

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

    public URL getLinkedin() {
        return linkedin;
    }

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

