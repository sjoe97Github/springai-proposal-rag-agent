"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.DataStore = void 0;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const constants_1 = require("./constants");
class DataStore {
    constructor() {
        this.resumeMatchData = this.loadResumeMatchData();
        this.allCandidates = this.extractAllCandidates();
        this.contextStore = new Map();
        this.chatHistoryStore = new Map();
    }
    loadResumeMatchData() {
        try {
            const dataPath = path_1.default.join(__dirname, '../data/resume-matches.json');
            const rawData = fs_1.default.readFileSync(dataPath, 'utf-8');
            return JSON.parse(rawData);
        }
        catch (error) {
            console.error('Error loading resume data:', error);
            return this.getDefaultResumeMatchData();
        }
    }
    extractAllCandidates() {
        const candidateMap = new Map();
        // Extract all unique candidates from all queries
        this.resumeMatchData.forEach(matchResponse => {
            matchResponse.results.forEach(candidate => {
                if (!candidateMap.has(candidate.candidateId)) {
                    candidateMap.set(candidate.candidateId, candidate);
                }
            });
        });
        return Array.from(candidateMap.values());
    }
    getDefaultResumeMatchData() {
        return [
            {
                sessionId: "default_session",
                query: "React TypeScript developer",
                results: [
                    {
                        candidateId: "candidate_001",
                        finalScore: 95.5,
                        shortExplanation: "Excellent match with 5+ years React experience and TypeScript expertise",
                        linkedin: "https://linkedin.com/in/john-doe-dev",
                        github: "https://github.com/johndoe",
                        reposList: [
                            { url: "https://github.com/johndoe/react-portfolio", visibility: "PUBLIC", language: "TypeScript" },
                            { url: "https://github.com/johndoe/typescript-utils", visibility: "PUBLIC", language: "TypeScript" }
                        ]
                    },
                    {
                        candidateId: "candidate_002",
                        finalScore: 87.2,
                        shortExplanation: "Strong frontend skills with React and modern JavaScript frameworks",
                        linkedin: "https://linkedin.com/in/jane-smith-frontend",
                        github: "https://github.com/janesmith",
                        reposList: [
                            { url: "https://github.com/janesmith/vue-components", visibility: "PUBLIC", language: "JavaScript" },
                            { url: "https://github.com/janesmith/react-hooks-lib", visibility: "PUBLIC", language: "TypeScript" }
                        ]
                    }
                ]
            }
        ];
    }
    getResumeMatches_Broken_Keyword_Match(query) {
        const queryLower = query.toLowerCase();
        const queryWords = queryLower.split(/\s+/).filter(word => word.length > 2);
        // Score candidates based on query relevance
        const scoredCandidates = this.allCandidates.map(candidate => {
            let relevanceScore = 0;
            const searchText = `${candidate.shortExplanation} ${candidate.candidateId}`.toLowerCase();
            // Check for direct query word matches
            queryWords.forEach(word => {
                if (searchText.includes(word)) {
                    relevanceScore += 10;
                }
            });
            // Check repository languages if available
            if (candidate.reposList && Array.isArray(candidate.reposList)) {
                candidate.reposList.forEach(repo => {
                    const repoText = `${repo.language} ${repo.url}`.toLowerCase();
                    queryWords.forEach(word => {
                        if (repoText.includes(word)) {
                            relevanceScore += 5;
                        }
                    });
                });
            }
            return {
                ...candidate,
                relevanceScore
            };
        });
        // Filter candidates with some relevance and sort by combined score
        return scoredCandidates
            .filter(candidate => candidate.relevanceScore > 0)
            .sort((a, b) => {
            // Primary sort by relevance, secondary by original final score
            if (a.relevanceScore !== b.relevanceScore) {
                return b.relevanceScore - a.relevanceScore;
            }
            return b.finalScore - a.finalScore;
        })
            .map(({ relevanceScore, ...candidate }) => candidate); // Remove relevanceScore from result
    }
    // Randomly pick one resume match response and return all its results    
    getResumeMatches(query) {
        // If no match data available, return empty array
        if (this.resumeMatchData.length === 0) {
            console.log(`Query: "${query}" - No resume match data available`);
            return [];
        }
        // Pick a random match response from resumeMatchData
        const randomIndex = Math.floor(Math.random() * this.resumeMatchData.length);
        const randomMatchResponse = this.resumeMatchData[randomIndex];
        console.log(`Query: "${query}" - Returning ${randomMatchResponse.results.length} candidates from match: "${randomMatchResponse.query}"`);
        return randomMatchResponse.results;
    }
    getPromptContext(sessionId, contextType) {
        let sessionContext = this.contextStore.get(sessionId);
        if (!sessionContext) {
            this.setPromptContext(sessionId, constants_1.DEFAULT_CONTEXT_TYPE, constants_1.GITHUB_DEFAULT_CONTEXT);
            // TODO - Refactor!  This just 'feels' like a hack.
            sessionContext = this.contextStore.get(sessionId);
            if (!sessionContext) {
                return null;
            }
        }
        return sessionContext.get(contextType) || null;
    }
    setPromptContext(sessionId, contextType, context) {
        if (!this.contextStore.has(sessionId)) {
            this.contextStore.set(sessionId, new Map());
        }
        const sessionContext = this.contextStore.get(sessionId);
        sessionContext.set(contextType, context);
    }
    getChatHistory(sessionId) {
        return this.chatHistoryStore.get(sessionId) || this.getDefaultChatHistory(sessionId);
    }
    getDefaultChatHistory(sessionId) {
        // Return some sample chat history for demonstration
        const defaultHistory = [
            {
                type: "user",
                content: "experience in Java, C#, Python, and VIM"
            },
            {
                type: "user",
                content: "experience in Java C# TypeScript JavaScript ANTLR"
            },
            {
                type: "user",
                content: "experience in Automation Security DevOps"
            },
            {
                type: "assistant",
                content: "I found several candidates matching your criteria. Here are the top matches based on their skills and experience with these technologies."
            },
            {
                type: "user",
                content: "experience in Automation Security "
            },
            {
                type: "assistant",
                content: "Yes, I've included GitHub links and repository information for candidates who have them. You can review their code contributions and project experience."
            }
        ];
        // Store it for this session
        this.chatHistoryStore.set(sessionId, defaultHistory);
        return defaultHistory;
    }
}
exports.DataStore = DataStore;
//# sourceMappingURL=dataStore.js.map