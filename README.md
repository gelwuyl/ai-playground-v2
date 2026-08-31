# AI Playground — Little Miss Chatterbox + Little Miss Magic + Mr. Kaypoh Research Agent

A single, portfolio-ready web project that runs **three apps** from one codebase:

1. **Little Miss Chatterbox** — non-stop talking: ask a question, get an answer, and keep a history.
2. **Little Miss Magic** — boundless imagination: generate a gentle bedtime story for kids.
3. **Mr. Kaypoh — Research Agent** — an agentic AI that searches the web, reads pages, and writes a sourced research brief using a ReAct loop.

All three apps share one PostgreSQL database and one LLM backend, and can run in two modes:

- **Local mode** — FastAPI + **Ollama** (local LLM, e.g. `gemma4:e4b-mlx`) for offline development.
- **Cloud mode** — **Vercel** serverless functions + **OpenRouter API** (OpenAI-compatible) for deployment.

An **LLM adapter** (`services/llm_adapter.py`) picks the backend at runtime based on environment variables, so the business logic is shared and not duplicated.

Mr. Kaypoh is a **ReAct research agent** (Reason + Act, after Yao et al. 2022). It runs a **client-driven loop**: the browser polls `POST /api/research_step` repeatedly, and each call executes exactly one tool action — **SEARCH** (SerpApi), **READ** (httpx + BeautifulSoup, capped at 5000 chars), or **FINISH** (write a sourced brief) — and persists it to Postgres. This keeps each serverless invocation short and gives the user a live trace. Safeguards are enforced server-side, not by the model: a **3-page gate** blocks FINISH until at least three different pages are read, duplicate URLs are refused, and a step limit (10) is hard-enforced. Every finding must end with a source URL in brackets or `[no source]`, and the brief prints two separate lists: **Pages read** and **Also found** (not opened). Evaluation runs 6 checks (search used, >1 source, within step limit, has recommendation, ≥3 sources, no `[no source]`). Set `USE_FIXTURES=1` to use saved SerpApi results instead of live queries.

## Architecture

```mermaid
flowchart LR
    B[Browser] --> L[Landing page]
    L --> Q[Little Miss Chatterbox]
    L --> S[Little Miss Magic]
    L --> R[Mr. Kaypoh]
    Q --> A[LLM adapter]
    S --> A
    R --> OR[OpenRouter JSON]
    A -->|OPENROUTER_API_KEY set| OR
    A -->|OLLAMA_BASE_URL set| O[Ollama - local]
    Q --> P[(PostgreSQL)]
    S --> P
    R --> P
    R --> SE[SerpApi search]
    R --> WB[Web page reader]
```

## Project layout

```
api/                       # Vercel serverless route handlers
  ask.py                   # POST /api/ask
  history.py               # GET  /api/history
  story.py                 # POST /api/story
  stories.py               # GET  /api/stories
  healthz.py               # GET  /api/healthz
  research_start.py        # POST /api/research_start  (Mr. Kaypoh)
  research_step.py         # POST /api/research_step   (one ReAct step)
  research_status.py       # GET  /api/research_status (session + steps)
  research_eval.py         # POST /api/research_eval   (6 checks + score)
public/                    # Static pages (plain HTML/CSS/JS)
  index.html               # Landing page with three app cards
  question-log.html        # Little Miss Chatterbox UI
  bedtime-story.html       # Little Miss Magic UI
  research.html            # Mr. Kaypoh Research Agent UI (live trace)
  style.css
services/                  # Shared logic
  llm_adapter.py           # Chooses OpenRouter or Ollama at runtime
  openrouter_service.py    # OpenRouter API call (call_openrouter + JSON mode)
  gemini_service.py        # Legacy Gemini SDK call (unused, kept for reference)
  database.py              # Postgres connection pool (psycopg-pool)
  interaction_service.py   # Question Log DB ops
  story_service.py         # Bedtime Story DB ops
  research_service.py      # Mr. Kaypoh tools (search_web, read_webpage, eval)
  research_engine.py       # Pure ReAct engine (run_one_step, no HTTP imports)
  fixtures.py              # Saved results for USE_FIXTURES=1 fallback
  vercel_handler.py        # Base handler for Vercel serverless functions
app/                       # Local FastAPI app (Ollama) — NOT deployed
  main.py                  # Local entrypoint (uvicorn app.main:app)
  services/ollama_service.py
local/
  run_local.sh             # Helper to start the local app
sql/
  001_create_tables.sql    # Combined schema (interactions + stories)
  002_create_stories.sql   # Stories table (standalone)
  003_research.sql         # Mr. Kaypoh tables (research_sessions + research_steps)
scripts/
  verify_setup.sh          # Local environment checks
vercel.json                # URL rewrites for Vercel
requirements.txt
.env.example
.vercelignore              # Excludes local-only files from Vercel
```

## Prerequisites

- **Python 3.12+** (native arm64 on Apple Silicon)
- **PostgreSQL** running locally (for local mode)
- **Ollama** running locally with a model pulled (for local mode)
- **OpenRouter API key** (for cloud mode)
- **SerpApi key** (for Mr. Kaypoh's SEARCH tool)

## Local mode (Ollama)

```bash
# 1. Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
#   - DATABASE_URL -> your local Postgres
#   - OLLAMA_MODEL  -> e.g. gemma4:12b-mlx

# 3. Create the database and tables
createdb llm_question_log
psql -d llm_question_log -f sql/001_create_tables.sql

# 4. Verify the environment
./scripts/verify_setup.sh

# 5. Run the local app
./local/run_local.sh
# or: uvicorn app.main:app --reload
```

Open <http://localhost:8000>.

## Cloud mode (Vercel + OpenRouter)

1. Push this repo to GitHub.
2. In Vercel, import the repo and set the Python runtime.
3. Provision **Vercel Postgres** (Neon) and run the SQL migrations against it:
   - `sql/001_create_tables.sql` (interactions + stories)
   - `sql/003_research.sql` (research_sessions + research_steps)
   - `sql/004_interview_prep.sql` (Mr. Brave interview tables)
4. Add environment variables in Vercel (Production + Preview):
   - `OPENROUTER_API_KEY` — your OpenRouter API key
   - `OPENROUTER_MODEL` — model slug (e.g. `google/gemini-2.5-flash`)
   - `DATABASE_URL` — Vercel Postgres connection string
   - `SERPAPI_KEY` — SerpApi key for Mr. Kaypoh's SEARCH tool
   - `USE_FIXTURES` — set `1` to use saved results instead of live SerpApi (optional)
5. Deploy. Vercel uses `vercel.json` rewrites and the `api/` handlers.

The `.vercelignore` excludes `app/`, `local/`, and `venv/` so only the serverless code is deployed.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Landing page (three app cards) |
| GET | `/question-log` | Little Miss Chatterbox UI |
| GET | `/bedtime-story` | Little Miss Magic UI |
| GET | `/research` | Mr. Kaypoh Research Agent UI (live trace) |
| POST | `/api/ask` | Ask a question, get an answer |
| GET | `/api/history` | List recent interactions |
| POST | `/api/story` | Generate a bedtime story |
| GET | `/api/stories` | List recent stories |
| POST | `/api/research_start` | Create a research session (returns session_id) |
| POST | `/api/research_step` | Execute one ReAct step (SEARCH / READ / FINISH) |
| GET | `/api/research_status` | Get session + all steps (refresh recovery) |
| POST | `/api/research_eval` | Run 6 evaluation checks on a completed session |
| GET | `/api/healthz` | Health check (OpenRouter + Postgres) |

## License

MIT — see [LICENSE](LICENSE).
