import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List

from transformers import AutoTokenizer

from .config import settings
from .postgres_store import save_chunk_records as save_postgres_chunks
from .qdrant_store import save_vector_records

# Use FastEmbed default model for consistency with embedding
EMBED_MODEL = settings.embed_model
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL)
        print(f"Loaded tokenizer for {EMBED_MODEL}")
    return _TOKENIZER


def split_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.replace('\n', ' '))
    return [s.strip() for s in sentences if s.strip()]


def pack_sentences_by_tokens(sentences: List[str], max_tokens: int, overlap_tokens: int, encoder) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    if encoder:
        token_counts = [len(encoder.encode(s)) for s in sentences]
    else:
        # Fallback: approximate tokens as ~4 chars per token
        token_counts = [max(1, len(s) // 4) for s in sentences]

    i = 0
    while i < len(sentences):
        tcount = token_counts[i]
        if current_tokens + tcount <= max_tokens or not current:
            current.append(sentences[i])
            current_tokens += tcount
            i += 1
        else:
            chunks.append(' '.join(current).strip())
            # prepare overlap: keep trailing sentences that sum to >= overlap_tokens
            if overlap_tokens > 0:
                acc = 0
                overlap_list: List[str] = []
                for s_idx in range(len(current) - 1, -1, -1):
                    if encoder:
                        acc += len(encoder.encode(current[s_idx]))
                    else:
                        acc += max(1, len(current[s_idx]) // 4)
                    overlap_list.insert(0, current[s_idx])
                    if acc >= overlap_tokens:
                        break
                current = overlap_list
                if encoder:
                    current_tokens = sum(len(encoder.encode(s)) for s in current)
                else:
                    current_tokens = sum(max(1, len(s) // 4) for s in current)
            else:
                current = []
                current_tokens = 0

    if current:
        chunks.append(' '.join(current).strip())
    return chunks


def chunk_text_token_aware(text: str, max_tokens: int = 1000, overlap_tokens: int = 200) -> List[str]:
    sentences = split_sentences(text)
    encoder = _get_tokenizer()
    return pack_sentences_by_tokens(sentences, max_tokens, overlap_tokens, encoder)


def gather_pdf_text(pdf_path: str, max_tokens: int, overlap_tokens: int) -> List[tuple]:
    doc = fitz.open(pdf_path)
    chunks_with_meta = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        page_chunks = chunk_text_token_aware(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for i, ch in enumerate(page_chunks):
            meta = {"source": pdf_path, "page": page_num + 1, "chunk_index": i}
            chunks_with_meta.append((ch, meta))
    return chunks_with_meta


def main():
    project_root = Path(__file__).resolve().parents[1]
    lectures_dir = project_root / "lectures"
    if not lectures_dir.is_dir():
        print("No 'lectures' directory found at", lectures_dir)
        return

    pdf_files = [str(path) for path in lectures_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"]
    if not pdf_files:
        print("No PDF files found in lectures/")
        return

    max_tokens = settings.chunk_max_tokens
    overlap_tokens = settings.chunk_overlap_tokens
    database_url = settings.database_url.strip()
    next_id = 1
    chunk_records = []
    for pdf in pdf_files:
        print("Parsing:", pdf)
        for text, meta in gather_pdf_text(pdf, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
            if not text or not str(text).strip():
                print(f"Skipping empty chunk: {pdf} page={meta.get('page')} idx={meta.get('chunk_index')}")
                continue
            # normalize source: prefer meta.source, fall back to pdf path or filename
            src = meta.get("source") if isinstance(meta, dict) else None
            source = str(src or pdf or Path(pdf).name)
            if not source.strip():
                print(f"Warning: empty source detected for {pdf}; using filename fallback")
                source = str(Path(pdf).name)
            record = {
                "id": next_id,
                "text": text,
                "source": source,
                "page": meta.get("page", 0) if isinstance(meta, dict) else 0,
                "chunk_index": meta.get("chunk_index", 0) if isinstance(meta, dict) else 0,
            }
            chunk_records.append(record)
            next_id += 1

    saved_rows = save_postgres_chunks(database_url, chunk_records)
    print(f"Saved {saved_rows} chunks to PostgreSQL")

    saved_vectors = save_vector_records(chunk_records)
    print(f"Saved {saved_vectors} chunks to Qdrant")


if __name__ == "__main__":
    main()
