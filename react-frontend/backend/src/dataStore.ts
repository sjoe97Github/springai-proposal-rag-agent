import fs from 'fs';
import path from 'path';
import { ResumeResult, ResumeMatchResponse, Message, ContextType } from './types';
import { DEFAULT_CONTEXT_TYPE, GITHUB_DEFAULT_CONTEXT } from './constants';

export class DataStore {
    private resumeMatchData: ResumeMatchResponse[];
    private allCandidates: ResumeResult[];
    private contextStore: Map<string, Map<ContextType, string>>;
    private chatHistoryStore: Map<string, Message[]>;

    constructor() {
        this.resumeMatchData = this.loadResumeMatchData();
        this.allCandidates = this.extractAllCandidates();
        this.contextStore = new Map();
        this.chatHistoryStore = new Map();
    }

    private loadResumeMatchData(): ResumeMatchResponse[] {
        try {
            const dataPath = path.join(__dirname, '../data/resume-matches.json');
            const rawData = fs.readFileSync(dataPath, 'utf-8');
            return JSON.parse(rawData);
        } catch (error) {
            console.error('Error loading resume data:', error);
            return this.getDefaultResumeMatchData();
        }
    }

    private extractAllCandidates(): ResumeResult[] {
        const candidateMap = new Map<string, ResumeResult>();

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

    private getDefaultResumeMatchData(): ResumeMatchResponse[] {
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

    public getResumeMatches_Broken_Keyword_Match(query: string): ResumeResult[] {
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
    public getResumeMatches(query: string): ResumeResult[] {
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

    public getPromptContext(sessionId: string, contextType: ContextType): string | null {
        let sessionContext = this.contextStore.get(sessionId);
        if (!sessionContext) {
            this.setPromptContext(sessionId, DEFAULT_CONTEXT_TYPE, GITHUB_DEFAULT_CONTEXT);
            // TODO - Refactor!  This just 'feels' like a hack.
            sessionContext = this.contextStore.get(sessionId);
            if (!sessionContext) {
                return null;
            }
        }
        return sessionContext.get(contextType) || null;
    }

    public setPromptContext(sessionId: string, contextType: ContextType, context: string): void {
        if (!this.contextStore.has(sessionId)) {
            this.contextStore.set(sessionId, new Map());
        }
        const sessionContext = this.contextStore.get(sessionId)!;
        sessionContext.set(contextType, context);
    }

    public getChatHistory(sessionId: string): Message[] {
        return this.chatHistoryStore.get(sessionId) || this.getDefaultChatHistory(sessionId);
    }

    private getDefaultChatHistory(sessionId: string): Message[] {
        // Return some sample chat history for demonstration
        const defaultHistory: Message[] = [
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