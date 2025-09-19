"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const dataStore_1 = require("./dataStore");
const app = (0, express_1.default)();
const PORT = 3001;
// Initialize data store
const dataStore = new dataStore_1.DataStore();
// Middleware
app.use((0, cors_1.default)());
app.use(express_1.default.json());
// POST /resume-match/query - Match resumes based on query
app.post('/resume-match/query', (req, res) => {
    try {
        const { query } = req.body;
        const sessionId = req.query.sessionId;
        if (!query) {
            return res.status(400).json({ error: 'Query is required' });
        }
        const results = dataStore.getResumeMatches(query);
        const effectiveSessionId = sessionId || `session_${Date.now()}`;
        const response = {
            sessionId: effectiveSessionId,
            query,
            results
        };
        res.json(response);
    }
    catch (error) {
        console.error('Error in /resume-match/query:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});
// GET /resume-match/context/get/:sessionId - Get prompt context
app.get('/resume-match/context/get/:sessionId', (req, res) => {
    try {
        const { sessionId } = req.params;
        const contextType = req.query.type;
        if (!sessionId) {
            return res.status(400).json({ error: 'Session ID is required' });
        }
        if (!contextType || !['github', 'linkedin'].includes(contextType)) {
            return res.status(400).json({ error: 'Valid context type (github|linkedin) is required' });
        }
        const context = dataStore.getPromptContext(sessionId, contextType);
        if (context === null) {
            return res.status(404).json({ error: 'Context not found for this session' });
        }
        res.send(context);
    }
    catch (error) {
        console.error('Error in /resume-match/context/get:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});
// POST /resume-match/context/set/:sessionId - Set prompt context
app.post('/resume-match/context/set/:sessionId', (req, res) => {
    try {
        const { sessionId } = req.params;
        const contextType = req.query.type;
        const { context } = req.body;
        if (!sessionId) {
            return res.status(400).json({ error: 'Session ID is required' });
        }
        if (!contextType || !['github', 'linkedin'].includes(contextType)) {
            return res.status(400).json({ error: 'Valid context type (github|linkedin) is required' });
        }
        if (!context) {
            return res.status(400).json({ error: 'Context is required' });
        }
        dataStore.setPromptContext(sessionId, contextType, context);
        res.send('Context set successfully');
    }
    catch (error) {
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
        const response = {
            sessionId,
            chatHistory
        };
        res.json(response);
    }
    catch (error) {
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
//# sourceMappingURL=server.js.map