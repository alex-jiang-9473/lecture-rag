# Lectures RAG (Qdrant + FastEmbed + Qwen)

A lightweight Retrieval-Augmented Generation (RAG) project for lecture PDFs.

## What This Project Includes

- PDF parsing and chunking to JSONL
- FastEmbed embeddings
- Qdrant vector search
- Qwen chat completion via OpenAI-compatible API (Ollama endpoint style)
- Interactive terminal Q&A for lecture content

## Examples

### Same Question, Different Behavior (LLM vs RAG)

Question:

`what is cosine similarity?`

LLM-only answer (no retrieval):

- Gives a general textbook definition.
- Usually cannot cite your specific lecture files.
- May omit course-specific terms/examples from your notes.

RAG answer (with retrieval from Qdrant):

- Includes the same core definition.
- Grounds the response in your indexed lecture chunks.
- Returns source files from your `lectures/` folder.

Example RAG output pattern:

```text
ANSWER:
Cosine similarity measures the angle-based similarity between vectors...

SOURCES: .../lectures/lsh-1.pdf, .../lectures/collaborative_filtering-1.pdf
RETRIEVED 3 CHUNKS
```

Why this proves augmentation:

- `SOURCES` points to your local lecture PDFs.
- `RETRIEVED N CHUNKS` confirms vector search was used.
- Answer wording should match lecture-specific content, not only generic definitions.

## Project Files

- `chunkpdf.py`: Parse PDFs from `lectures/` and generate `lectures_chunks.jsonl`
- `embeddings.py`: FastEmbed wrapper for batch text embeddings
- `lectures_rag.py`: Interactive RAG chat (retrieve from Qdrant + ask LLM)
- `qwenllm.py`: Simple direct LLM call test
- `fastemb_qdrant.py`: Small standalone Qdrant + FastEmbed demo
- `quickstart.py`: Basic Qdrant vector operations example

## Prerequisites

- Python 3.10+
- Running Qdrant instance (default: `http://localhost:6333`)
- Running OpenAI-compatible LLM endpoint (default: `http://127.0.0.1:11434/v1`)

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install qdrant-client fastembed pymupdf openai transformers
```

## Environment Variables

Optional overrides:

```bash
export EMBED_MODEL=BAAI/bge-small-en-v1.5
export CHUNK_MAX_TOKENS=1000
export CHUNK_OVERLAP_TOKENS=200
export QDRANT_URL=http://localhost:6333
export QDRANT_COLLECTION=lectures_collection
export LLM_BASE_URL=http://127.0.0.1:11434/v1
export LLM_MODEL=qwen3:14b
export RAG_TOP_K=3
```

## End-to-End Workflow

### 1) Put lecture PDFs in `lectures/`

Expected location:

- `./lectures/*.pdf`

### 2) Chunk PDFs into JSONL

```bash
python chunkpdf.py
```

This generates:

- `lectures_chunks.jsonl`

### 3) Index chunks into Qdrant

Use your indexing script if you have one in your local branch.

If not, you can still test Qdrant embedding flow with:

```bash
python fastemb_qdrant.py
```

Note: `lectures_rag.py` expects a collection (default `lectures_collection`) containing payload fields like `text`, `source`, and `page`.

### 4) Run interactive RAG

```bash
python lectures_rag.py
```

Then type questions in terminal. Type `exit` to quit.

## Quick LLM Connectivity Test

```bash
python qwenllm.py
```

## Troubleshooting

- `AttributeError: QdrantClient has no attribute search`
  - Use `query_points()` in your retrieval code (already done in current `lectures_rag.py`).

- No answers or empty retrieval
  - Verify vectors were inserted into the same collection used by `QDRANT_COLLECTION`.
  - Verify payload contains `text` field for retrieved points.

- Model/tokenizer mismatch
  - Keep `EMBED_MODEL` consistent across chunking, indexing, and querying.

## Notes

- Batch embedding with fixed size (e.g. 64) is normal and efficient.
- Token-aware chunking in `chunkpdf.py` uses sentence packing and model tokenizer when available.
