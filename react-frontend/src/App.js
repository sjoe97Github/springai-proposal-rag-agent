import React, { useState, useEffect } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:3001/resume-match';

// Generate a random sessionId
const generateSessionId = () => {
    return 'session_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
};

function App() {
    const [sessionId, setSessionId] = useState('');

    // Generate sessionId on component mount
    useEffect(() => {
        setSessionId(generateSessionId());
    }, []);
    const [loading, setLoading] = useState({});

    // State for Match Resumes
    const [query, setQuery] = useState('');
    const [matchResults, setMatchResults] = useState('');

    // State for Get Prompt Context
    const [getContextType, setGetContextType] = useState('github');
    const [getContextResults, setGetContextResults] = useState('');

    // State for Set Prompt Context
    const [setContextType, setSetContextType] = useState('github');
    const [contextToSet, setContextToSet] = useState('');
    const [setContextResults, setSetContextResults] = useState('');

    // State for Chat History
    const [chatHistoryResults, setChatHistoryResults] = useState('');

    const setLoadingState = (endpoint, isLoading) => {
        setLoading(prev => ({ ...prev, [endpoint]: isLoading }));
    };

    const handleMatchResumes = async () => {
        if (!query.trim()) {
            alert('Please enter a query');
            return;
        }

        setLoadingState('matchResumes', true);
        try {
            const url = sessionId
                ? `${API_BASE_URL}/query?sessionId=${encodeURIComponent(sessionId)}`
                : `${API_BASE_URL}/query`;

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify( { query: query }),
            });

            const data = await response.json();
            setMatchResults(JSON.stringify(data, null, 2));
        } catch (error) {
            setMatchResults(`Error: ${error.message}`);
        }
        setLoadingState('matchResumes', false);
    };

    const handleGetPromptContext = async () => {
        setLoadingState('getContext', true);
        try {
            const response = await fetch(
                `${API_BASE_URL}/context/get/${encodeURIComponent(sessionId)}?type=${getContextType}`
            );

            if (response.ok) {
                const data = await response.text();
                setGetContextResults(data);
            } else {
                const errorData = await response.text();
                setGetContextResults(`Error ${response.status}: ${errorData}`);
            }
        } catch (error) {
            setGetContextResults(`Error: ${error.message}`);
        }
        setLoadingState('getContext', false);
    };

    const handleSetPromptContext = async () => {
        if (!contextToSet.trim()) {
            alert('Please enter context to set');
            return;
        }

        setLoadingState('setContext', true);
        try {
            const response = await fetch(
                `${API_BASE_URL}/context/set/${encodeURIComponent(sessionId)}?type=${setContextType}`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        sessionId: sessionId,
                        context: contextToSet
                    }),
                }
            );

            if (response.ok) {
                setSetContextResults('Context set successfully');
            } else {
                const errorData = await response.text();
                setSetContextResults(`Error ${response.status}: ${errorData}`);
            }
        } catch (error) {
            setSetContextResults(`Error: ${error.message}`);
        }
        setLoadingState('setContext', false);
    };

    const handleGetChatHistory = async () => {
        setLoadingState('chatHistory', true);
        try {
            const response = await fetch(
                `${API_BASE_URL}/chat/history/${encodeURIComponent(sessionId)}`
            );

            const data = await response.json();
            setChatHistoryResults(JSON.stringify(data, null, 2));
        } catch (error) {
            setChatHistoryResults(`Error: ${error.message}`);
        }
        setLoadingState('chatHistory', false);
    };

    return (
        <div className="app">
            <h1>Resume Match Controller</h1>
            <p className="session-info">Session ID: {sessionId}</p>

            {/* Match Resumes Section */}
            <div className="section">
                <h2>1. Match Resumes</h2>
                <div className="form-group">
                    <label htmlFor="query">Query:</label>
                    <textarea
                        id="query"
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        placeholder="Enter your job/skills query"
                        className="input-field"
                        rows="3"
                    />
                </div>
                <button onClick={handleMatchResumes} className="submit-button">
                    Submit Query
                </button>

                {loading.matchResumes && <div className="spinner">Loading...</div>}
                <div className="results-container">
          <textarea
              value={matchResults}
              readOnly
              className="results-textbox"
              placeholder="Results will appear here..."
          />
                </div>
            </div>

            {/* Get Prompt Context Section */}
            <div className="section">
                <h2>2. Get Prompt Context</h2>
                <div className="form-group">
                    <label htmlFor="getContextType">Context Type:</label>
                    <select
                        id="getContextType"
                        value={getContextType}
                        onChange={(e) => setGetContextType(e.target.value)}
                        className="select-field"
                    >
                        <option value="github">GitHub</option>
                        <option value="linkedin">LinkedIn</option>
                    </select>
                </div>
                <button onClick={handleGetPromptContext} className="submit-button">
                    Get Context
                </button>

                {loading.getContext && <div className="spinner">Loading...</div>}
                <div className="results-container">
          <textarea
              value={getContextResults}
              readOnly
              className="results-textbox"
              placeholder="Context will appear here..."
          />
                </div>
            </div>

            {/* Set Prompt Context Section */}
            <div className="section">
                <h2>3. Set Prompt Context</h2>
                <div className="form-group">
                    <label htmlFor="setContextType">Context Type:</label>
                    <select
                        id="setContextType"
                        value={setContextType}
                        onChange={(e) => setSetContextType(e.target.value)}
                        className="select-field"
                    >
                        <option value="github">GitHub</option>
                        <option value="linkedin">LinkedIn</option>
                    </select>
                </div>
                <div className="form-group">
                    <label htmlFor="contextToSet">Context to Set:</label>
                    <textarea
                        id="contextToSet"
                        value={contextToSet}
                        onChange={(e) => setContextToSet(e.target.value)}
                        placeholder="Paste the new context string here"
                        className="input-field"
                        rows="5"
                    />
                </div>
                <button onClick={handleSetPromptContext} className="submit-button">
                    Set Context
                </button>

                {loading.setContext && <div className="spinner">Loading...</div>}
                <div className="results-container">
          <textarea
              value={setContextResults}
              readOnly
              className="results-textbox"
              placeholder="Set context results will appear here..."
          />
                </div>
            </div>

            {/* Get Chat History Section */}
            <div className="section">
                <h2>4. Get Chat History</h2>
                <button onClick={handleGetChatHistory} className="submit-button">
                    Get Chat History
                </button>

                {loading.chatHistory && <div className="spinner">Loading...</div>}
                <div className="results-container">
          <textarea
              value={chatHistoryResults}
              readOnly
              className="results-textbox"
              placeholder="Chat history will appear here..."
          />
                </div>
            </div>
        </div>
    );
}

export default App;
// This file is intentionally left blank.