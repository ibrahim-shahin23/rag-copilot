# Docker Orchestration & Deployment Guide (`DOCKER.md`)

This guide details how to launch the complete RAG Copilot system using Docker Compose, including automatic database initialization and corpus document seeding.

---

## Quick Start (Single Command)

To bring up the entire system (FastAPI web server, relational SQLite database, vector indices, security middleware, and automatic corpus ingestion):

```bash
docker compose up --build
```

### What Happens When You Run `docker compose up`:
1. **App Container (`rag-copilot-app`)**:
   - Builds the Python 3.12 application container.
   - Starts the FastAPI HTTP server on port `8000`.
   - Exposes OpenAPI documentation at `http://localhost:8000/docs`.
   - Mounts a local persistent volume (`rag-data`) for database and vector store persistence.
2. **Seed Container (`rag-copilot-seed`)**:
   - Waits for the `app` container to become healthy (`/health` returns HTTP 200).
   - Automatically executes `python scripts/seed.py`.
   - Ingests and chunks all sample documents from `eval/corpus/` into `data/copilot.db`, `vectors.pkl`, and `bm25.pkl`.

---

## Dedicated Commands

### 1. Bring Up System in Background
```bash
docker compose up -d --build
```

### 2. Run Database Seeding / Ingestion Manually
To re-run the seed ingestion on demand at any time:
```bash
docker compose run --rm seed
```

### 3. Ingest a Custom File into Running Container
```bash
docker compose exec app python interface/cli.py ingest /path/to/your/document.pdf
```

### 4. Interactive Grounded Q&A (CLI inside Container)
```bash
docker compose exec app python interface/cli.py ask "What does FR-2 require for retrieval citations?"
```

### 5. Check System Health & Container Logs
```bash
# Check service status
docker compose ps

# View live API logs
docker compose logs -f app
```

### 6. Shut Down System
```bash
docker compose down
```

To remove all persistent data volumes as well:
```bash
docker compose down -v
```

---

## Environment Variables Configuration

Environment variables can be set in a `.env` file at the root of the repository:

```env
# Optional: Google Gemini API Key for hosted embeddings and completion
GEMINI_API_KEY=your_gemini_api_key_here

# Security & CORS settings
RATE_LIMIT_PER_MINUTE=120
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

*Note: If `GEMINI_API_KEY` is not provided, the system automatically uses call-time fallback providers (`TfidfEmbeddingProvider` and `ExtractiveFallbackProvider`) so all features remain functional out of the box without requiring API keys.*
