# Next.js Frontend

This directory contains the Next.js 14 (App Router) frontend for the Multi-Agent Tourism Planning System backend.

## Run with Docker (recommended)

From the **project root**:

```bash
docker compose up -d
```

The frontend is available at `http://localhost:3000`.

`NEXT_PUBLIC_API_BASE_URL` is baked into the image at build time via a Docker build arg (defaults to `http://localhost:8000/api`). To point at a different backend, rebuild with:

```bash
docker compose build --build-arg NEXT_PUBLIC_API_BASE_URL=http://your-backend/api
```

---

## Run locally

1. Install packages:

   ```bash
   npm install
   ```

2. Create local env:

   ```bash
   echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api" > .env.local
   ```

3. Start Next.js dev server:

   ```bash
   npm run dev
   # http://localhost:3000
   ```

> `NEXT_PUBLIC_API_BASE_URL` is a **build-time** variable — changes to `.env.local` take effect on restart, not hot-reload.

---

## Build for production (local)

```bash
npm run build
npm start
```

---

## Main files

| Path | Description |
|---|---|
| `app/` | Next.js App Router entrypoints |
| `components/chat-shell.tsx` | Primary chat UI, itinerary flow panel, timeline |
| `components/day-map.tsx` | TrackAsia interactive map widget |
| `services/api.ts` | Browser client for FastAPI session / chat / conversation APIs |
| `app/globals.css` | All application styles |
| `Dockerfile` | Two-stage production image (builder + runner) |
| `.dockerignore` | Excludes `node_modules`, `.next`, `.env*` from build context |

---

## UI panels

**Chat panel** — Handles message input, streaming responses, session management, and conversation history.

**Itinerary Flow Panel** — Rendered when the backend returns a structured plan:
- Day selector tabs
- Timeline of stops (place name + map link) with emoji stripped from LLM-generated text
- Route summary (legs with distance and ETA)
- Hotel recommendation card
- Embedded TrackAsia map (`DayMap`)

---

## Environment variables

| Variable | When set | Description |
|---|---|---|
| `NEXT_PUBLIC_API_BASE_URL` | Build time | Base URL for the FastAPI backend (e.g. `http://localhost:8000/api`) |
