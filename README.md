# Multi-Agent Tourism Planning System

An AI-powered travel planning system for **Da Nang, Vietnam** built on a multi-agent architecture. Users describe their trip in natural language (Vietnamese or English) and receive a complete day-by-day itinerary with real distances, hotel recommendations, and an interactive map.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Agent Pipeline](#agent-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Data](#data)
- [Setup & Running](#setup--running)
- [Key Features](#key-features)
- [API Reference](#api-reference)
- [Diagrams](#diagrams)
- [Evaluation](#evaluation)

---

## Overview

The system accepts a free-text travel query such as:

> *"Family of 4 — 3 days in Da Nang. Kids love the beach, parents want seafood and temples. Budget-friendly, near My Khe beach."*

And returns:

- A structured **day-by-day itinerary** (morning / noon / afternoon / evening slots)
- **Real driving distances and ETAs** between each stop via TrackAsia Directions API
- A **budget-aware hotel recommendation** scored by proximity to attractions
- An **interactive map** with markers, day routes, and place detail panels

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  Frontend  (Next.js :3000)                       │
│  ChatShell · ItineraryFlowPanel · DayMap         │
└────────────────────┬─────────────────────────────┘
                     │  REST / SSE
┌────────────────────▼─────────────────────────────┐
│  Backend  (FastAPI :8000)                        │
│                                                  │
│  ┌─────────────────────────────────────────────┐ │
│  │  LangGraph Workflow                         │ │
│  │  intake → retrieval → planning →            │ │
│  │  validator → response                       │ │
│  └─────────────────────────────────────────────┘ │
│                                                  │
│  Services: Builder · RAG Engine · Formatter      │
│            VRP Optimizer · Scoring · Validator   │
└──────┬───────────────┬──────────────────────────-┘
       │               │
┌──────▼──────┐  ┌─────▼──────────┐
│  SQLite     │  │  Chroma        │
│  travel.db  │  │  2162 vectors  │
│  883 places │  │  (local)       │
└─────────────┘  └────────────────┘
       │
┌──────▼──────────────────────────────┐
│  External APIs                      │
│  OpenRouter (gpt-4o-mini) · OpenAI  │
│  TrackAsia (geocode + routing)      │
└─────────────────────────────────────┘
```

---

## Agent Pipeline

### [A1] Intake Agent
Parses the user's free-text message into structured fields using an LLM call (`temperature=0.0`):
- `destination`, `days`, `interests`, `budget`, `companion`
- Infers `budget_tier`: `"low"` | `""` | `"high"` for hotel scoring
- If required fields are missing → returns clarifying questions instead of planning

### [A2] Retrieval Agent
Builds the place candidate pool:
1. Embeds the query via OpenAI `text-embedding-3-small` (dim=1536)
2. Searches Chroma vector store with cosine similarity + BM25-like lexical re-ranking
3. Returns top-20 places as retrieval context

### [A3] Planning Agent
Builds the full itinerary:
1. Loads the complete Da Nang DB pool (883 places, all geocoded)
2. Fills missing coordinates for RAG-retrieved places via `_enrich_coords()`
3. Runs **VRP optimization** to cluster attractions geographically by day
4. Falls back to **LLM planning** (`temperature=0.3`) if VRP is unavailable
5. Selects hotel using a **scoring formula**: proximity to attractions + star rating + type + budget tier
6. Calls TrackAsia Directions API for each leg: `distance_km`, `eta_min`, `mode_label`

**Hotel scoring formula:**
```
score = 32.0
  + dominant_area_bonus  (+10 if same district as attractions)
  + star_bonus           (+1.5 per star)
  + type_bonus           (resort/villa +2.5 | hostel +1.0)
  + budget_bonus         (low budget: hostel +5, penalize >2★
                          high budget: luxury brand +6, +4/star above 3★)
  - 2.2 × avg_km
  - 0.8 × max_km
```

### [A4] Validator Agent
Quality gate before returning to the user:
- **LLM structural checks** (`temperature=0.0`): `daily_count_mismatch`, `missing_daily_structure`, `unrealistic_schedule`, `empty_place_pool`
- **Distance checks**: `too_many_long_legs` (>18 km), `extreme_leg_distance` (>25 km)
- On hard failure → triggers a **single retry** with `strict_mode=True` and a tighter prompt
- Soft issues (`too_many_self_service_meals`) are reported but do not block

### [A5] Response Agent
Assembles the final answer using `format_planning_answer()` (formatter.py):
- Combines research summary + day plan + hotel + tips into a coherent Vietnamese text
- Persists conversation, messages, and plan to SQLite

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, TypeScript, CSS |
| Backend | FastAPI, Python 3.11+ |
| Agent orchestration | LangGraph |
| LLM | gpt-4o-mini via OpenRouter |
| Embeddings | text-embedding-3-small (OpenAI) |
| Vector store | Chroma (persistent local) |
| Relational DB | SQLite |
| Maps & Routing | TrackAsia (geocode + directions + tiles) |
| Route optimization | Custom VRP solver |
| Containerization | Docker + Docker Compose |

---

## Project Structure

```
├── app/
│   ├── core/          # Settings, DB init, session
│   ├── graph/         # LangGraph nodes, state, edges, LLM calls
│   │   ├── nodes.py   # 5 agent node functions
│   │   ├── state.py   # TravelGraphState TypedDict
│   │   ├── intake.py  # Intent extraction prompts
│   │   └── llm.py     # generate_answer(), render output
│   ├── itinerary/     # Planning + validation + formatting
│   │   ├── builder.py      # build_trip_plan_payload(), hotel scorer, VRP
│   │   ├── validation.py   # validate_itinerary_plan(), distance checks
│   │   ├── formatter.py    # format_planning_answer()
│   │   ├── routing.py      # resolve_location_for_map(), leg routing
│   │   └── vrp.py          # VRP itinerary optimizer
│   ├── places/        # RAG engine, scoring, metadata, repository
│   │   ├── vector_rag.py   # retrieve_place_candidates(), re-ranking
│   │   ├── rag.py          # RetrievalArtifacts, build_context_payload()
│   │   ├── scoring.py      # INTEREST_KEYWORDS (bilingual VI+EN)
│   │   ├── metadata.py     # enrich_place_record(), infer_intent_tags()
│   │   ├── repository.py   # upsert_places(), list_places()
│   │   └── chroma.py       # Chroma client wrapper
│   └── tools/         # TrackAsia, Google Places adapters
├── frontend/
│   ├── components/
│   │   ├── chat-shell.tsx  # Main UI: chat + itinerary panel
│   │   └── day-map.tsx     # TrackAsia map widget
│   ├── services/
│   │   └── api.ts          # REST client (chat, sessions, conversations)
│   ├── app/
│   │   └── globals.css     # All styling
│   └── Dockerfile          # Two-stage Next.js production image
├── scripts/
│   ├── preprocess.py            # Raw data → enrich → upsert DB
│   ├── geocode_places.py        # Geocode via TrackAsia (adaptive rate)
│   ├── geocode_places_trackasia.py
│   ├── geocode_google.py
│   ├── ingest_to_chroma.py      # Build/rebuild Chroma vector index
│   ├── build_rag.py             # RAG pipeline builder
│   ├── eval_rag.py              # Giskard RAGET evaluation runner
│   └── test_log.py              # 5-query regression test → logs/
├── data/
│   ├── crawl/processed/         # Source JSON (destinations, restaurants, hotels)
│   ├── processed/               # unified_places.json (883 Da Nang places)
│   ├── chroma/                  # Chroma persistent storage (Docker volume)
│   └── travel.db                # SQLite database (Docker volume)
├── docs/
│   ├── EVALUATION_REPORT.md     # Giskard RAGET evaluation report (Vietnamese)
│   └── diagrams/                # PlantUML source files
│       ├── system_architecture.puml
│       ├── erd.puml
│       ├── sequence_overview.puml
│       ├── sequence_planning.puml
│       └── sequence_validation.puml
├── Dockerfile                   # Backend production image
├── docker-compose.yml           # Compose: backend + frontend
├── .dockerignore
└── logs/                        # Test run logs (test_vi_*.log)
```

---

## Data

### Database (SQLite `travel.db`)

| Table | Rows | Description |
|---|---|---|
| `places` | 883 | All Da Nang places (99.9% geocoded) |
| `place_chunks` | 2162 | Text chunks for vector search |
| `conversations` | — | Chat sessions |
| `messages` | — | User + assistant messages with metadata |
| `plans` | — | Saved structured itineraries |
| `principals` | — | Anonymous / user identities |
| `anonymous_sessions` | — | Cookie-backed sessions |

### Place Categories

| Category | Count | Geocoded |
|---|---|---|
| destination | 30 | 100% |
| restaurant | 84 | 100% |
| accommodation | 759 | 99.9% |
| entertainment | 6 | 100% |
| transport | 4 | 100% |

### Intent Tags (bilingual)
`food` · `beach` · `museum` · `heritage` · `spiritual` · `shopping` · `cafe` · `nature` · `nightlife` · `family`

---

## Setup & Running

### Prerequisites
- API keys: `OPENROUTER_API_KEY`, `TRACKASIA_API_KEY`, `OPENAI_API_KEY` (for embeddings)

---

### Option A — Docker (recommended)

```bash
# 1. Copy and fill in API keys
cp .env.example .env
# Edit .env: add OPENROUTER_API_KEY, TRACKASIA_API_KEY, OPENAI_API_KEY

# 2. Build images
docker compose build

# 3. Start services
docker compose up -d

# Backend:  http://localhost:8000
# Frontend: http://localhost:3000
```

Data (`data/chroma/` and `data/travel.db`) is mounted as a host volume so it persists across restarts.

To stop:
```bash
docker compose down
```

---

### Option B — Local development

**Prerequisites:** Python 3.11+, Node.js 18+

**Backend:**

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env: add OPENROUTER_API_KEY, TRACKASIA_API_KEY, OPENAI_API_KEY

# 3. (First run) Preprocess and build vector index
python scripts/preprocess.py
python scripts/geocode_places.py
python scripts/ingest_to_chroma.py --recreate

# 4. Start backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**

```bash
cd frontend
npm install

# Configure API URL
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api" > .env.local

npm run dev
# Runs on http://localhost:3000
```

**Run regression tests:**

```bash
python scripts/test_log.py
# Results saved to logs/test_vi_<timestamp>.log
```

---

## Key Features

| Feature | Detail |
|---|---|
| **Bilingual input** | Accepts Vietnamese (with/without diacritics) and English queries |
| **Budget-aware hotel selection** | `low` → hostels/mini hotels; `high` → luxury brands/resorts |
| **Real route distances** | TrackAsia Directions API — distance + ETA + transport mode per leg |
| **VRP optimization** | Geographically clusters attractions to minimize total daily travel |
| **Validation + retry** | Structural and distance checks; auto-retries once on failure |
| **Interactive map** | Day-by-day markers, route lines, place detail panel |
| **Full Vietnamese output** | All plan text, labels, and system messages in Vietnamese |
| **Conversation history** | Cookie-backed sessions; persisted conversations and plans |
| **Adaptive geocoding** | 0.15 s/request with exponential backoff on 429 rate limits |
| **Docker deployment** | Single `docker compose up` starts backend + frontend production builds |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Submit query, return full ChatResponse |
| `POST` | `/api/chat/stream` | SSE streaming response |
| `POST` | `/api/session/init` | Create/resume session (sets cookie) |
| `GET` | `/api/session/me` | Get current principal info |
| `GET` | `/api/conversations` | List conversation history |
| `GET` | `/api/conversations/{id}` | Get conversation with messages |
| `DELETE` | `/api/conversations/{id}` | Delete single conversation |
| `DELETE` | `/api/conversations` | Clear all history |
| `POST` | `/api/plans/save` | Save structured plan |
| `GET` | `/api/plans/{id}` | Retrieve saved plan |
| `POST` | `/api/route/geometry` | Get route geometry for map |

### ChatResponse schema (key fields)

```json
{
  "answer": "KE HOACH DU LICH - DA NANG ...",
  "plan": "LICH TRINH 3 NGAY TAI DA NANG ...",
  "route_plan": [{ "day": 1, "from": "...", "to": "...", "distance_km": 3.5, "eta_min": 8 }],
  "recommended_hotel": { "name": "...", "address": "...", "lat": 16.07, "lon": 108.22 },
  "plan_validation": { "passed": true, "issues": [], "retried": false, "metrics": { "max_leg_km": 9.4 } },
  "collected_info": { "destination": "Da Nang", "days": "3", "interests": "beach, food", "budget": "tiet kiem" },
  "verified_places": [...],
  "timings": { "intake_ms": 1200, "retrieval_ms": 2100, "planning_ms": 9800, "validator_ms": 3500 }
}
```

---

## Diagrams

PlantUML source files in [`docs/diagrams/`](docs/diagrams/):

| File | Description |
|---|---|
| `system_architecture.puml` | Full component + data flow diagram |
| `erd.puml` | Database schema (7 tables) |
| `sequence_overview.puml` | High-level agent interaction flow |
| `sequence_planning.puml` | planning_node [A3] internal detail |
| `sequence_validation.puml` | validator_node [A4] + retry logic |

Render with the [PlantUML VS Code extension](https://marketplace.visualstudio.com/items?itemName=jebbs.plantuml) (`Alt+D`) or at [plantuml.com](https://plantuml.com/plantuml/uml/).

---

## Evaluation

A full evaluation using the **Giskard RAGET** framework is documented in [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md).

The report covers:
- Retrieval metrics: Precision@K, Recall@K, MRR, NDCG
- Itinerary quality metrics: constraint satisfaction, diversity, distance efficiency
- Giskard RAGET component scores: RETRIEVER, GENERATOR, REWRITER, ROUTING, KNOWLEDGE_BASE
- Per-question analysis of 20 test cases with root cause breakdowns

To re-run the evaluation:

```bash
python scripts/eval_rag.py
```

---

## Performance

| Metric | Value |
|---|---|
| Average response time | ~17 s |
| Retry rate | ~80% of queries |
| DB coverage (geocoded) | 99.9% (882 / 883 places) |
| Chroma vectors | 2162 chunks |
| Validation pass rate | 100% on regression suite (5 queries) |
