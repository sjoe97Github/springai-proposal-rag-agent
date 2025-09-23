import express from 'express';
import cors from 'cors';
import {
    JobQuery,
    ResumeMatchResponse,
    ChatHistoryResponse,
    ContextType
} from './types';
import { DataStore } from './dataStore';

const app = express();
const PORT = 3001;

// Initialize data store
const dataStore = new DataStore();

// Middleware
app.use(cors());
app.use(express.json());

// POST /resume-match/query - Match resumes based on query
app.post('/resume-match/query', (req, res) => {
    try {
        const { query }: JobQuery = req.body;
        const sessionId = req.query.sessionId as string;

        if (!query) {
            return res.status(400).json({ error: 'Query is required' });
        }

        const results = dataStore.getResumeMatches(query);
        const effectiveSessionId = sessionId || `session_${Date.now()}`;

        const response: ResumeMatchResponse = {
            sessionId: effectiveSessionId,
            query,
            results
        };

        res.json(response);
    } catch (error) {
        console.error('Error in /resume-match/query:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// GET /resume-match/context/get/:sessionId - Get prompt context
app.get('/resume-match/context/get/:sessionId', (req, res) => {
    try {
        const { sessionId } = req.params;
        const contextType = req.query.type as ContextType;

        if (!sessionId) {
            return res.status(400).json({ error: 'Session ID is required' });
        }

        if (!contextType || !['github', 'linkedin', 'skillsquery'].includes(contextType)) {
            return res.status(400).json({ error: 'Valid context type (github|linkedin|skillsquery) is required' });
        }

        const context = dataStore.getPromptContext(sessionId, contextType);

        if (context === null) {
            return res.status(404).json({ error: 'Context not found for this session' });
        }

        res.send(context);
    } catch (error) {
        console.error('Error in /resume-match/context/get:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// POST /resume-match/context/set/:sessionId - Set prompt context
app.post('/resume-match/context/set/:sessionId', (req, res) => {
    try {
        const { sessionId } = req.params;
        const contextType = req.query.type as ContextType;
        const { context }: { context: string } = req.body;

        if (!sessionId) {
            return res.status(400).json({ error: 'Session ID is required' });
        }

        if (!contextType || !['github', 'linkedin', 'skillsquery'].includes(contextType)) {
            return res.status(400).json({ error: 'Valid context type (github|linkedin|skillsquery) is required' });
        }

        if (!context) {
            return res.status(400).json({ error: 'Context is required' });
        }

        dataStore.setPromptContext(sessionId, contextType, context);
        res.send('Context set successfully');
    } catch (error) {
        console.error('Error in /resume-match/context/set:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// GET /resume-match/chat/history/:sessionId - Get chat history
app.get('/resume-match/chat/history/:sessionId', (req, res) => {
    try {
        const { sessionId } = req.params;

        if (!sessionId) {
            return res.status(400).json({ error: 'Session ID is required' });
        }

        const chatHistory = dataStore.getChatHistory(sessionId);

        const response: ChatHistoryResponse = {
            sessionId,
            chatHistory
        };

        res.json(response);
    } catch (error) {
        console.error('Error in /resume-match/chat/history:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
    console.log(`Resume Match Backend server running on http://localhost:${PORT}`);
    console.log(`Health check: http://localhost:${PORT}/health`);
});