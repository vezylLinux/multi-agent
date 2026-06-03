# Next.js Frontend

This directory contains the App Router frontend for the FastAPI backend.

## Run locally

1. Install packages:

   ```bash
   npm install
   ```

2. Create local env:

   ```bash
   cp .env.local.example .env.local
   ```

3. Start Next.js:

   ```bash
   npm run dev
   ```

The frontend expects the FastAPI backend at `http://localhost:8000/api` by default.

## Main files

- `app/`: Next.js App Router entrypoints
- `components/chat-shell.tsx`: primary chat UI
- `services/api.ts`: browser client for FastAPI session/chat/conversation APIs
