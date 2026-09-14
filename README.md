# AI Playground — Six AI Tools

A portfolio-ready web project that brings together six AI tools in one codebase:

1. **Little Miss Chatterbox** — Ask a question, get an answer, and browse the conversation history.
2. **Little Miss Magic** — Turn a simple idea into a gentle, magical bedtime story.
3. **Mr Kaypoh — Research Agent** — A ReAct research agent that searches the web, reads sources, and writes a cited brief.
4. **Mr Brave — Interview (CrewAI)** — A 3-stage interview-preparation crew that prospects target roles, predicts likely interview questions, and drafts STAR responses with coaching notes.
5. **Mr Bounce — Trip Orchestrator** — a 4-agent crew that turns Google Maps pins into a checked, optimized day-by-day itinerary with swap suggestions.
6. **This or That — Weighted Decision Maker** — Weigh two options across the criteria that matter to you, with AI-backed scores and a clear verdict. (Hosted in AI Studio: https://this-or-that-gel.ai.studio)

The tools share one PostgreSQL database and an OpenRouter-backed cloud LLM integration.

- **Local mode** — FastAPI + **Ollama** for Little Miss Chatterbox and Little Miss Magic.
- **Cloud mode** — Vercel Python serverless functions + the **OpenRouter API** for all six web tools.

The shared OpenRouter model setting is `OPENROUTER_MODEL`, whose code default is `openrouter/free`. Mr Brave may optionally override this with `INTERVIEW_PROSPECTOR_MODEL`, `INTERVIEW_RESEARCHER_MODEL`, and `INTERVIEW_WRITER_MODEL`.

- **Mr Kaypoh — Research Agent** — a ReAct agent that runs one tool action per poll (**SEARCH / READ / FINISH**): short serverless calls, a live trace, and code-enforced safeguards (≥3 pages read before FINISH, duplicate-read refusal, hard step limit). Every finding carries a source URL; the brief separates **Pages read** from **Also found**.

- **Mr Brave — Interview (CrewAI)** — a sequential three-stage interview-prep pipeline (**Prospector → Interview Strategist → Professional Communications Expert**) that prospects roles, predicts likely questions, and drafts STAR answers with coaching notes. It calls OpenRouter directly rather than the CrewAI package, whose bundle exceeded Vercel Hobby's 500 MB serverless-function limit.

- **Mr Bounce — Trip Orchestrator** — a 4-agent crew (**Scout, Reasoner, Alternatives, Compiler**) that turns Google Maps pins into a checked, optimized itinerary. Scout researches pins (hours verified or explicitly flagged); Reasoner builds the travel matrix and the deterministic schedule (hours-feasibility, day-clustering, least-travel, meals), then remediates graded — reorder, compress dwell, consult Alternatives, drop only last — surfacing each call as an advisory note; Alternatives is consult-only. Compiler emits the daily itinerary with per-day Maps route links. Travel and geocoding run on free keyless APIs (OSRM, Photon, Nominatim, Overpass) with SerpApi fallback; the full trace is persisted and surfaced live.

**This or That — Weighted Decision Maker** — weighs two options across the criteria you choose, scoring them with AI and giving a clear verdict. It runs as a separate app in **AI Studio** (https://this-or-that-gel.ai.studio) and is linked out from the landing page, sharing none of this codebase's backend.

## Architecture

```mermaid
flowchart LR
    B[Browser] --> L[Landing page]

    L --> Q[Little Miss Chatterbox]
    L --> S[Little Miss Magic]
    L --> R[Mr Kaypoh<br/>Research Agent]
    L --> I[Mr Brave<br/>Interview (CrewAI)]
    L --> T[Mr Bounce<br/>Trip Orchestrator]

    Q --> A[LLM adapter]
    S --> A
    A -->|Local mode| O[Ollama]
    A -->|Cloud mode| OR[OpenRouter API<br/>openrouter/free default]

    R --> OR
    R --> SE[SerpApi search]
    R --> WB[Web page reader]

    I --> IH["/api/interview<br/>consolidated Vercel handler"]
    IH --> P1["1. Prospector"]
    P1 -->|persist roles; next request| P2["2. Interview Strategist"]
    P2 -->|persist questions; next request| P3["3. Professional Communications Expert"]
    P1 --> OR
    P2 --> OR
    P3 --> OR
    P1 -. saves roles .-> DB
    P2 -. saves questions .-> DB
    P3 -. saves final guide; completed .-> DB

    T --> TH["/api/planner<br/>consolidated Vercel handler"]
    TH --> N1["1. Ingest (tool)"]
    N1 -->|persist pins; next request| N2["2. Scout (agent)"]
    N2 -->|persist research; next request| N3["3. Logistics (tool)"]
    N3 -->|persist legs; next request| N4["4. Scheduler (tool)"]
    N4 -->|persist schedule; next request| N5["5. Critic (agent)"]
    N5 -->|retry if issues; max 3| N4
    N5 -->|persist review; next request| N6["6. Alternatives (agent)"]
    N6 -->|persist swaps; next request| N7["7. Compiler (agent)"]
    N7 -. saves itinerary; completed .-> DB
    N2 --> OR
    N5 --> OR
    N6 --> OR
    N7 --> OR
    N3 --> SE
    N3 --> DB

    Q --> DB[(PostgreSQL)]
    S --> DB
    R --> DB
```

## Project layout

```
api/                       # Vercel serverless route handlers
  ask.py                   # POST /api/ask
  history.py               # GET  /api/history
  story.py                 # POST /api/story
  stories.py               # GET  /api/stories
  healthz.py               # GET  /api/healthz
  research_start.py        # POST /api/research_start  (Mr Kaypoh)
  research_step.py         # POST /api/research_step   (one ReAct step)
  research_status.py       # GET  /api/research_status (session + steps)
  research_eval.py         # POST /api/research_eval   (6 checks + score)
  interview.py             # Mr Brave consolidated handler (start/step/status/delete)
  planner.py               # Mr Bounce consolidated handler (start/step/status/delete)
public/                    # Static pages (plain HTML/CSS/JS)
  index.html               # Landing page with six app cards
  question-log.html        # Little Miss Chatterbox UI
  bedtime-story.html       # Little Miss Magic UI
  research.html            # Mr Kaypoh Research Agent UI (live trace)
  interview-prep.html      # Mr Brave Interview (CrewAI) UI
  trip-planner.html       # Mr Bounce Trip Orchestrator UI (animated node graph)
  style.css
services/                  # Shared logic
  llm_adapter.py           # Chooses OpenRouter or Ollama at runtime
  openrouter_service.py    # OpenRouter API call (call_openrouter + JSON mode)
  gemini_service.py        # Legacy Gemini SDK call (unused, kept for reference)
  database.py              # Postgres connection pool (psycopg-pool)
  interaction_service.py   # Question Log DB ops
  story_service.py         # Bedtime Story DB ops
  research_service.py      # Mr Kaypoh tools (search_web, read_webpage, eval)
  research_engine.py       # Pure ReAct engine (run_one_step, no HTTP imports)
  planner_db.py            # Mr Bounce Postgres tables + _ensure_tables
  planner_types.py         # Mr Bounce canonical data contracts (Pin/PlaceResearch/Leg/Schedule/Trace)
  planner_graph.py         # Mr Bounce graph engine + YAML loader
  planner_ingest.py        # Mr Bounce Ingest tool (short-link + text pins)
  planner_logistics.py     # Mr Bounce Logistics tool (leg cache, SerpApi directions)
  planner_scheduler.py     # Mr Bounce Scheduler tool (clustering + 2-opt + slotting)
  planner_scout.py         # Mr Bounce Scout agent (SerpApi place research)
  planner_critic.py        # Mr Bounce Critic agent (schedule review)
  planner_alternatives.py  # Mr Bounce Alternatives agent (swap proposals)
  planner_compiler.py      # Mr Bounce Compiler agent (final itinerary assembly)
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
  003_research.sql         # Mr Kaypoh tables (research_sessions + research_steps)
  004_interview_prep.sql   # Mr Brave interview tables
sql/
  005_trip_planner.sql     # Mr Bounce tables (planner_sessions/pins/leg_cache/trace/itinerary)
scripts/
  verify_setup.sh          # Local environment checks
planner_graph.yaml         # Mr Bounce 7-node graph definition (ships to Vercel)
tests/                     # Unit tests (scheduler, graph runner, etc.)
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
- **SerpApi key** (for Mr Kaypoh's SEARCH tool)

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
   - `sql/004_interview_prep.sql` (Mr Brave interview tables)
   - `sql/005_trip_planner.sql` (Mr Bounce planner tables)
4. Add environment variables in Vercel (Production + Preview):
   - `OPENROUTER_API_KEY` — your OpenRouter API key
   - `OPENROUTER_MODEL` — model slug shared by all four cloud tools (default: `openrouter/free`)
   - `INTERVIEW_PROSPECTOR_MODEL`, `INTERVIEW_RESEARCHER_MODEL`, `INTERVIEW_WRITER_MODEL` — optional Mr Brave per-stage overrides; when unset, they inherit `OPENROUTER_MODEL`
   - `PLANNER_SCOUT_MODEL`, `PLANNER_CRITIC_MODEL`, `PLANNER_ALTERNATIVES_MODEL`, `PLANNER_COMPILER_MODEL` — optional Mr Bounce per-agent overrides; when unset, they inherit `OPENROUTER_MODEL`
   - `DATABASE_URL` — Vercel Postgres connection string
   - `SERPAPI_KEY` — SerpApi key for Mr Kaypoh's SEARCH tool
   - `USE_FIXTURES` — set `1` to use saved results instead of live SerpApi (optional)
5. Deploy. Vercel uses `vercel.json` rewrites and the `api/` handlers. `planner_graph.yaml` ships to Vercel alongside the Python handlers.

The `.vercelignore` excludes `app/`, `local/`, and `venv/` so only the serverless code is deployed.

## Endpoints

| Method | Path | Description |
|---|---|---|
|| GET | `/` | Landing page (six app cards) |
| GET | `/question-log` | Little Miss Chatterbox UI |
| GET | `/bedtime-story` | Little Miss Magic UI |
| GET | `/research` | Mr Kaypoh Research Agent UI (live trace) |
| GET | `/interview-prep` | Mr Brave Interview (CrewAI) UI |
| GET | `/trip-planner` | Mr Bounce Trip Orchestrator UI (animated node graph) |
| POST | `/api/ask` | Ask a question, get an answer |
| GET | `/api/history` | List recent interactions |
| POST | `/api/story` | Generate a bedtime story |
| GET | `/api/stories` | List recent stories |
| POST | `/api/research_start` | Create a research session (returns session_id) |
| POST | `/api/research_step` | Execute one ReAct step (SEARCH / READ / FINISH) |
| GET | `/api/research_status` | Get session + all steps (refresh recovery) |
| POST | `/api/research_eval` | Run 6 evaluation checks on a completed session |
| POST | `/api/planner_start` | Create a trip-planning session (returns session_id) |
| POST | `/api/planner_step` | Execute one crew node (Ingest through Compiler) |
| GET | `/api/planner_status` | Get session + full node trace + itinerary (refresh recovery) |
| POST | `/api/planner_delete` | Delete a trip-planning session |
| GET | `/api/healthz` | Health check (OpenRouter + Postgres) |

## License

MIT — see [LICENSE](LICENSE).
