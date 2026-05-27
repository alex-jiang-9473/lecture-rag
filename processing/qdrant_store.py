from __future__ import annotations

import uuid
from typing import Any, Sequence

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from .config import settings


_MODEL: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    global _MODEL
    if _MODEL is None:
        _MODEL = TextEmbedding(model=settings.embed_model)
    return _MODEL


def _make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    # Use deterministic UUIDv5 based on source|page|chunk_index so IDs are valid UUID strings
    name = f"{source}|{page}|{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _create_collection(client: QdrantClient) -> None:
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=qdrant_models.VectorParams(
            size=settings.embedding_dim,
            distance=qdrant_models.Distance.COSINE,
        ),
    )


def save_vector_records(records: Sequence[dict[str, Any]]) -> int:
    if not records:
        return 0
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    try:
        _create_collection(client)

        model = _get_model()
        eligible = []
        for r in records:
            t = str(r.get("text") or "")
            if "\x00" in t:
                t = t.replace("\x00", "")
            if t.strip():
                # store the cleaned text back into a copy to use for payloads
                r = dict(r)
                r["text"] = t
                eligible.append(r)
        if not eligible:
            return 0

        texts = [str(record["text"]) for record in eligible]
        vectors = []
        for vector in model.embed(texts):
            try:
                vectors.append(vector.tolist())
            except Exception:
                vectors.append(list(vector))

        points = []
        for record, vector in zip(eligible, vectors):
            source = str(record["source"])
            page = int(record["page"])
            chunk_index = int(record["chunk_index"])
            points.append(
                qdrant_models.PointStruct(
                    id=_make_chunk_id(source, page, chunk_index),
                    vector=vector,
                    payload={
                        "source": source,
                        "page": page,
                        "chunk_index": chunk_index,
                        "text": str(record["text"]),
                    },
                )
            )

        # attempt upsert and handle/log errors
        try:
            client.upsert(collection_name=settings.collection_name, points=points)
        except Exception as e:
            print(f"Qdrant upsert error: {e}")
            raise
        return len(points)
    finally:
        try:
            client.close()
        except Exception:
            # best-effort close; ignore errors but avoid noisy warning
            pass