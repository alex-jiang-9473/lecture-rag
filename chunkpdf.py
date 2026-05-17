import os
import fitz  # PyMuPDF
import json
import re
from typing import List, Optional

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

# Use FastEmbed default model for consistency with embedding
EMBED_MODEL = os.getenv('EMBED_MODEL', 'BAAI/bge-small-en-v1.5')
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None and AutoTokenizer:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(EMBED_MODEL)
            print(f"Loaded tokenizer for {EMBED_MODEL}")
        except Exception as e:
            print(f"Could not load tokenizer: {e}. Falling back to char estimates.")
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
    lectures_dir = os.path.join(os.getcwd(), "lectures")
    if not os.path.isdir(lectures_dir):
        print("No 'lectures' directory found at", lectures_dir)
        return

    pdf_files = [os.path.join(lectures_dir, f) for f in os.listdir(lectures_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("No PDF files found in lectures/")
        return

    max_tokens = int(os.getenv('CHUNK_MAX_TOKENS', os.getenv('CHUNK_MAX_CHARS', '1000')))
    overlap_tokens = int(os.getenv('CHUNK_OVERLAP_TOKENS', os.getenv('CHUNK_OVERLAP', '200')))
    chunks_file = os.path.join(os.getcwd(), "lectures_chunks.jsonl")
    next_id = 1
    with open(chunks_file, "w", encoding="utf-8") as out:
        for pdf in pdf_files:
            print("Parsing:", pdf)
            for text, meta in gather_pdf_text(pdf, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
                record = {
                    "id": next_id,
                    "text": text,
                    "source": meta["source"],
                    "page": meta["page"],
                    "chunk_index": meta["chunk_index"],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                next_id += 1

    print(f"Wrote {next_id-1} chunks to {chunks_file}")


if __name__ == "__main__":
    main()