# Multi-source Chatbot (FastAPI + LangChain + FAISS)

This repository contains a FastAPI-based multi-source chatbot that ingests PDFs, web pages, and JSON records, indexes them in a FAISS vector store, and answers queries with a LangChain RAG pipeline.

It is packaged to run inside Docker, persists data between restarts, supports environment-based configuration, and logs queries/responses.

## Features
- Ingest PDF/TXT/MD uploads, web URLs, and JSON records
- LangChain + OpenAI embeddings + FAISS retrieval
- FastAPI endpoints: `/ingest/document`, `/ingest/url`, `/ingest/json`, `/query`, `/health`
- Data persisted under a mounted `data_store` directory
- Logs queries and responses to `logs/queries.log`
- Secured endpoints using an API Key.

## Prerequisites
- Docker and Docker Compose installed
- An OpenAI API key

## Setup
1.  Clone or download the code into a new directory.
2.  Create a `.env` file in the root directory. You can copy `.env.example` to start.
3.  Fill in the required values in the `.env` file.

## Configuration (`.env` file)

Create a `.env` file and add the following variables.

```env
# Required: Your OpenAI API Key for embeddings and generation
OPENAI_API_KEY="sk-..."

# Optional: A secret key to protect your API endpoints. If not set, API is open.
API_KEY="your-secret-api-key"

# Optional: The port the API will run on
PORT=8000

# Optional: The language model to use
LLM_MODEL="gpt-4o"

# Optional: The number of relevant documents to retrieve for a query
RAG_TOP_K=5

# Optional: Directory to store the FAISS index
DATA_DIR="data_store"

# Optional: Path to the log file
LOG_FILE="logs/queries.log"
```

## Build & Run (Docker)

```bash
# Build and start containers in detached mode
docker-compose up --build -d

# View logs
docker-compose logs -f api
```

Data and logs are persisted in the `./data_store` and `./logs` folders via Docker volumes.

## API Usage Examples

All endpoints (except `/health`) are protected. Remember to include your `API_KEY` in the `X-API-Key` header.

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Upload a Document (PDF, TXT, MD)
```bash
curl -X POST "http://localhost:8000/ingest/document" \
  -H "X-API-Key: your-secret-api-key" \
  -F "file=@/path/to/your/file.pdf" \
  -F "source_name=My Document"
```

### Ingest from a URL
```bash
curl -X POST "http://localhost:8000/ingest/url" \
  -H "X-API-Key: your-secret-api-key" \
  -F "url=https://en.wikipedia.org/wiki/Artificial_intelligence" \
  -F "source_name=Wikipedia AI Article"
```

### Ingest JSON Records
```bash
curl -X POST "http://localhost:8000/ingest/json" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "records": [
      {"id": "record-1", "content": "The first rule of Fight Club is: You do not talk about Fight Club."},
      {"id": "record-2", "content": "The second rule of Fight Club is: You do not talk about Fight Club."}
    ],
    "source_name": "Fight Club Rules"
  }'
```

### Ask a Question
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-api-key" \
  -d '{
    "question": "What are the first two rules of Fight Club?"
  }'
```