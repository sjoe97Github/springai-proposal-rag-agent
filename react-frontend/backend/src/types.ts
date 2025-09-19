// Request/Response interfaces for resume matching service

export interface JobQuery {
    query: string;
}

export interface GithubRep {
    url: string;
    visibility: string;
    language: string;
}

export interface ResumeResult {
    candidateId: string;
    finalScore: number;
    shortExplanation: string;
    linkedin?: string; // URL as string
    github?: string;   // URL as string
    reposList?: GithubRep[]; // Ignored in JSON, but included for completeness
}

export interface ResumeMatchResponse {
    sessionId: string;
    query: string;
    results: ResumeResult[];
}

export interface PromptContext {
    sessionId: string;
    context: string;
}

export type PromptContextResponse = string;

export interface Message {
    type: string;
    content: string;
}

export interface ChatHistoryResponse {
    sessionId: string;
    chatHistory: Message[];
}

// Context types supported by the system
export type ContextType = 'github' | 'linkedin';