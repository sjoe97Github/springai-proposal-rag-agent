# Resume Match Backend Service

A Node.js TypeScript service that provides REST API endpoints for resume matching functionality.

## Installation

```bash
cd backend
npm install
```

## Development

```bash
# Start in development mode with auto-reload
npm run dev

# Build the project
npm run build

# Start production server
npm start
node dist/server.js 
```

The server will run on `http://localhost:3001`

## API Endpoints

### POST /resume-match/query
Match resumes based on job query
- **Query param**: `sessionId` (optional)
- **Body**: `{ "query": "string" }`
- **Response**: `ResumeMatchResponse`

### GET /resume-match/context/get/:sessionId
Get stored prompt context
- **Query param**: `type` (github|linkedin)
- **Response**: Context string

### POST /resume-match/context/set/:sessionId  
Set prompt context
- **Query param**: `type` (github|linkedin)
- **Body**: `{ "context": "string" }`
- **Response**: Success message

### GET /resume-match/chat/history/:sessionId
Get chat history for session
- **Response**: `ChatHistoryResponse`

### GET /health
Health check endpoint

## Data Storage

The service uses MOCK JSON files in the `data/` directory for mock data:
- `resume-matches.json` - Sample resume data with candidates
- In-memory storage for session contexts and chat history

## Architecture

- **Express.js** with TypeScript
- **CORS** enabled for frontend integration  
- **Type-safe** API contracts
- **JSON file-based** data store
- **Error handling** with proper HTTP status codes