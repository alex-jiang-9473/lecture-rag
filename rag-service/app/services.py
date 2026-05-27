from __future__ import annotations

import asyncio
from typing import Any

from fastapi import HTTPException
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from .config import settings
from .langchain_support import FastEmbedEmbeddingsAdapter, QdrantRetriever, RetrievalQA


class RAGService:
    _MAX_CONTEXT_CHARS = 4500
    _MAX_CHUNK_CHARS = 1200

    def __init__(self) -> None:
        self.embeddings = FastEmbedEmbeddingsAdapter()
        self.qdrant = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
        self.client = Groq(api_key=settings.groq_api_key)
        # Explicit system instruction: only use the provided lecture context.
        # If the answer cannot be found in the context, reply exactly: "I don't know." Do not hallucinate
        # or attempt to answer from outside the provided context.
        self.system_message = (
            "You must answer questions using ONLY the provided lecture context. "
            "Do not use any external knowledge. If the context does not contain the answer, reply exactly: \"I don't know.\""
        )
        # build a LangChain-style retriever + QA wrapper
        retriever = QdrantRetriever(self.qdrant, self.embeddings, settings.collection_name)
        self.qa = RetrievalQA(
            retriever=retriever,
            groq_client=self.client,
            system_message=self.system_message,
            max_context_chars=self._MAX_CONTEXT_CHARS,
            max_chunk_chars=self._MAX_CHUNK_CHARS,
            minimum_relevance_score=getattr(settings, "minimum_relevance_score", 0.0),
            model_name=settings.llm_model,
        )

    async def ensure_ready(self) -> None:
        return None

    def _search_chunks(self, question: str, limit: int) -> list[Any]:
        if not self.qdrant.collection_exists(settings.collection_name):
            return []

        query_vector = self.embeddings.embed_query(question)
        response = self.qdrant.query_points(
            collection_name=settings.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return list(getattr(response, "points", response) or [])

    async def ingest_pdf_directory(self, directory: str, clear_existing: bool = True) -> dict[str, Any]:
        raise HTTPException(status_code=404, detail="PDF ingestion has been removed from this service")

    async def ingest_jsonl(self, path: str, clear_existing: bool = True) -> dict[str, Any]:
        raise HTTPException(status_code=404, detail="JSONL ingestion has been removed from this service")

    async def answer(self, question: str, top_k: int | None = None) -> dict[str, Any]:
        limit = top_k or settings.default_top_k
        result = await asyncio.to_thread(self.qa.run, question, limit)
        return result

    async def _store_chunks(self, chunk_records: list[dict[str, Any]], clear_existing: bool) -> int:
        raise HTTPException(status_code=404, detail="Chunk storage removed; this service only answers questions")

    def _format_context(self, context: list[dict[str, Any]]) -> str:
        formatted_parts: list[str] = []
        total_chars = 0

        for item in context:
            source = item.get("source") or "unknown"
            page = item.get("page") or "?"
            chunk_index = item.get("chunk_index") or "?"
            text = str(item.get("text") or "").strip()
            if not text:
                continue

            if len(text) > self._MAX_CHUNK_CHARS:
                text = text[: self._MAX_CHUNK_CHARS].rstrip() + "..."

            part = f"[Source: {source}, Page {page}, Chunk {chunk_index}]\n{text}"
            if formatted_parts and total_chars + len(part) + 2 > self._MAX_CONTEXT_CHARS:
                break

            formatted_parts.append(part)
            total_chars += len(part)
            

        return "\n\n".join(formatted_parts)

    def _generate_answer(self, context: str, question: str) -> str:
        completion = self.client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": self.system_message},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:",
                },
            ],
            temperature=1,
            max_completion_tokens=8192,
            top_p=1,
            # reasoning_effort="default",
            stream=True,
            stop=None,
        )

        answer_parts: list[str] = []
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            if content:
                answer_parts.append(content)

        final = "".join(answer_parts).strip()
        if not final:
            return "I don't know."
        return final
