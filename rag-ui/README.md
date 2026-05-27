# Lecture RAG UI

A React + TypeScript + Vite + Tailwind UI for asking questions against the lecture RAG backend.

## Setup

1. Install dependencies in this folder:

```bash
npm install
```

2. Copy the example environment file and adjust the API base URL if needed:

```bash
copy .env.example .env
```

3. Start the Vite dev server:

```bash
npm run dev
```

The UI expects the backend to expose:

- `GET /health`
- `POST /ask`

By default it points at `http://127.0.0.1:8000`.
