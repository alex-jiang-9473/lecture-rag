from __future__ import annotations

import hashlib

from langchain_core.embeddings import Embeddings

from .embeddings import embed_texts


class FastEmbedEmbeddingsAdapter(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_texts([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


def make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    payload = f"{source}|{page}|{chunk_index}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


class QdrantRetriever:
    """Small retriever that queries Qdrant and returns documents in a LangChain-like shape."""

    def __init__(self, qdrant_client, embeddings: FastEmbedEmbeddingsAdapter, collection_name: str):
        self.qdrant = qdrant_client
        self.embeddings = embeddings
        self.collection_name = collection_name

    def get_relevant_documents(self, query: str, k: int = 3):
        vec = self.embeddings.embed_query(query)
        resp = self.qdrant.query_points(collection_name=self.collection_name, query=vec, limit=k, with_payload=True)
        points = list(getattr(resp, "points", resp) or [])
        results = []
        for p in points:
            payload = getattr(p, "payload", {}) or {}
            results.append(
                {
                    "source": payload.get("source"),
                    "page": payload.get("page"),
                    "chunk_index": payload.get("chunk_index"),
                    "text": payload.get("text", ""),
                    "score": float(getattr(p, "score", 0.0) or 0.0),
                }
            )
        return results


class RetrievalQA:
    """Simple RetrievalQA wrapper using a retriever and a Groq client.

    This provides a LangChain-like `run(question)` interface.
    """

    def __init__(self, retriever: QdrantRetriever, groq_client, system_message: str, max_context_chars: int = 4500, max_chunk_chars: int = 1200, minimum_relevance_score: float = 0.0, model_name: str | None = None):
        self.retriever = retriever
        self.client = groq_client
        self.system_message = system_message
        self.model_name = model_name
        self._MAX_CONTEXT_CHARS = max_context_chars
        self._MAX_CHUNK_CHARS = max_chunk_chars
        self.minimum_relevance_score = minimum_relevance_score

    def _format_context(self, docs: list[dict]) -> str:
        parts: list[str] = []
        total = 0
        for d in docs:
            text = str(d.get("text") or "").strip()
            if not text:
                continue
            if len(text) > self._MAX_CHUNK_CHARS:
                text = text[: self._MAX_CHUNK_CHARS].rstrip() + "..."
            part = f"[Source: {d.get('source')}, Page {d.get('page')}, Chunk {d.get('chunk_index')}]\n{text}"
            if parts and total + len(part) + 2 > self._MAX_CONTEXT_CHARS:
                break
            parts.append(part)
            total += len(part)
        return "\n\n".join(parts)

    def run(self, question: str, top_k: int | None = None) -> dict:
        k = top_k or 3
        docs = self.retriever.get_relevant_documents(question, k=k)
        if not docs:
            return {"answer": "I don't know.", "context": [], "sources": []}

        best_score = float(docs[0].get("score") or 0.0)
        if best_score < self.minimum_relevance_score:
            return {"answer": "I don't know.", "context": [], "sources": []}

        prompt_context = self._format_context(docs)

        # determine model to use for the request
        model_to_use = self.model_name or (getattr(self.client, "model", None))
        if not model_to_use:
            raise ValueError("No LLM model configured for Groq client; set `model_name` when creating RetrievalQA.")

        completion = self.client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f"Context:\n{prompt_context}\n\nQuestion:\n{question}\n\nAnswer:"},
            ],
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            stream=True,
            stop=None,
        )

        parts: list[str] = []
        for chunk in completion:
            content = chunk.choices[0].delta.content or ""
            if content:
                parts.append(content)

        answer = "".join(parts).strip()
        if not answer:
            answer = "I don't know."

        sources = list(dict.fromkeys(d.get("source") for d in docs if d.get("source")))
        return {"answer": answer, "context": docs, "sources": sources}
