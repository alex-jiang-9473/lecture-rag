import os
from typing import List, Tuple
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from embeddings import embed_texts


class LecturesRAG:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "lectures_collection",
        llm_base_url: str = "http://127.0.0.1:11434/v1",
        llm_model: str = "qwen3:8b",
        llm_timeout: float = 120,
        top_k: int = 3,
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.top_k = top_k
        
        self.llm_client = OpenAI(
            base_url=llm_base_url,
            api_key="dummy",
            timeout=llm_timeout,
        )
        self.llm_model = llm_model

    def retrieve_context(self, query: str) -> List[Tuple[str, dict]]:
        """
        Query Qdrant to retrieve relevant chunks for a given query.
        Returns a list of (text, metadata) tuples.
        """
        # Embed the query using the same model as the chunks
        query_embedding = embed_texts([query])[0]

        # Search Qdrant using query_points
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=self.top_k,
            with_payload=True,
        )

        # Extract text and metadata
        context = []
        for point in search_result.points:
            text = point.payload.get("text", "")
            metadata = {
                "source": point.payload.get("source"),
                "page": point.payload.get("page"),
                "score": point.score,
            }
            context.append((text, metadata))
        return context

    def build_prompt(self, query: str, context: List[Tuple[str, dict]]) -> str:
        """Build a prompt with context for the LLM."""
        context_text = "\n\n".join(
            [f"[Source: {meta['source']}, Page {meta['page']}]\n{text}" for text, meta in context]
        )
        prompt = f"""Based on the following lecture materials, answer the user's question.

CONTEXT:
{context_text}

USER QUESTION:
{query}

ANSWER:"""
        return prompt

    def query(self, query: str, show_context: bool = True) -> dict:
        """
        Main RAG pipeline: retrieve context and query the LLM.
        Returns a dict with 'answer', 'context', and 'sources'.
        """
        # Retrieve relevant chunks from Qdrant
        context = self.retrieve_context(query)
        
        if not context:
            return {
                "answer": "No relevant lecture materials found for this query.",
                "context": [],
                "sources": [],
            }

        # Show retrieved context if requested
        if show_context:
            print(f"\n[RAG DEBUG] Retrieved {len(context)} chunks:")
            for i, (text, meta) in enumerate(context, 1):
                print(f"\n  Chunk {i}:")
                print(f"    Source: {meta['source']}")
                print(f"    Page: {meta['page']}")
                print(f"    Score: {meta['score']:.4f}")
                print(f"    Text: {text[:150]}...")
        
        # Build prompt with context
        prompt = self.build_prompt(query, context)
        
        if show_context:
            print(f"\n[RAG DEBUG] Augmented prompt being sent to LLM:")
            print("-" * 60)
            print(prompt)
            print("-" * 60)

        # Query the LLM
        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content

        # Extract sources
        sources = [meta["source"] for _, meta in context]
        unique_sources = list(set(sources))

        return {
            "answer": answer,
            "context": context,
            "sources": unique_sources,
        }


def main():
    # Initialize RAG system
    rag = LecturesRAG(
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection_name=os.getenv("QDRANT_COLLECTION", "lectures_collection"),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:11434/v1"),
        llm_model=os.getenv("LLM_MODEL", "qwen3:14b"),
        top_k=int(os.getenv("RAG_TOP_K", "3")),
    )

    print("\n" + "="*60)
    print("LECTURES RAG SYSTEM")
    print("="*60)
    print("Enter your questions about the lectures (type 'exit' to quit)\n")

    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ["exit", "quit", "q"]:
            print("\nGoodbye!")
            break
        
        if not query:
            print("Please enter a question.\n")
            continue
        
        print(f"\n{'='*60}")
        print("Processing your question...")
        print('='*60)
        
        result = rag.query(query)
        
        print(f"\nANSWER:\n{result['answer']}")
        print(f"\nSOURCES: {', '.join(result['sources'])}")
        print(f"\nRETRIEVED {len(result['context'])} CHUNKS")
        print()


if __name__ == "__main__":
    main()
