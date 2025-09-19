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
    linkedin?: string;
    github?: string;
    reposList?: GithubRep[];
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
export type ContextType = 'github' | 'linkedin';
//# sourceMappingURL=types.d.ts.map