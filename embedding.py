from fastembed import TextEmbedding 
from typing import List
import numpy as np

print('fastembed ok')

documents: List[str] = [
    "FastEmbed is lighter than Transformers & Sentence-Transformers.",
    "FastEmbed is supported by and maintained by Qdrant.",
]

embedding_model = TextEmbedding()
print("The model BAAI/bge-small-en-v1.5 is ready to use.")

embeddings_generator = embedding_model.embed(documents)
embeddings_list = list(embeddings_generator)
len(embeddings_list[0])  
print("Embeddings:\n", embeddings_list)

