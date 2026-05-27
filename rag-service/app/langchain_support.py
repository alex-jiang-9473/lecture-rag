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
