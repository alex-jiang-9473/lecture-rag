from typing import List

from fastembed import TextEmbedding

from .config import settings

_MODEL_NAME = settings.embed_model
_MODEL = TextEmbedding(model=_MODEL_NAME)
print("FastEmbed model initialized:", _MODEL_NAME)


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        generated = _MODEL.embed(batch)
        for vector in generated:
            try:
                embeddings.append(vector.tolist())
            except Exception:
                embeddings.append(list(vector))
    return embeddings
