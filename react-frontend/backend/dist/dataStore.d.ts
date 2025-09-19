import { ResumeResult, Message, ContextType } from './types';
export declare class DataStore {
    private resumeMatchData;
    private allCandidates;
    private contextStore;
    private chatHistoryStore;
    constructor();
    private loadResumeMatchData;
    private extractAllCandidates;
    private getDefaultResumeMatchData;
    getResumeMatches_Broken_Keyword_Match(query: string): ResumeResult[];
    getResumeMatches(query: string): ResumeResult[];
    getPromptContext(sessionId: string, contextType: ContextType): string | null;
    setPromptContext(sessionId: string, contextType: ContextType, context: string): void;
    getChatHistory(sessionId: string): Message[];
    private getDefaultChatHistory;
}
//# sourceMappingURL=dataStore.d.ts.map