from typing import List
import os
from fastembed import TextEmbedding

_MODEL_NAME = os.getenv("EMBED_MODEL", None)
try:
    if _MODEL_NAME:
        _MODEL = TextEmbedding(model=_MODEL_NAME)
    else:
        _MODEL = TextEmbedding()
    print("FastEmbed model initialized:", _MODEL_NAME or "default")
except TypeError:
    # Some FastEmbed versions may not accept a model kwarg; fall back
    _MODEL = TextEmbedding()
    print("FastEmbed model initialized (fallback to default)")


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        gen = _MODEL.embed(batch)
        for vec in gen:
            try:
                embeddings.append(vec.tolist())
            except Exception:
                embeddings.append(list(vec))
    return embeddings
