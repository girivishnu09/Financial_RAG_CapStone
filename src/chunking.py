"""Section B (part 1) — Token-aware chunking with overlap and metadata.

Splits cleaned documents into retrieval-friendly chunks. Each chunk inherits
its parent document's metadata and gains a stable `chunk_id`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.ingest import Document


_ENC = tiktoken.get_encoding(settings.TOKENIZER_NAME)


# Splitters in priority order — try paragraph break, then sentence, then word.
_SPLIT_LEVELS = ["\n\n", "\n", ". ", " "]


def _count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


def _split_by_separator(text: str, separator: str) -> list[str]:
    if separator == " ":
        # Word-level fallback — keep whitespace
        return text.split(" ")
    parts = text.split(separator)
    # Re-attach separator (except sep=="\n\n") to keep readability
    if separator in ("\n\n", "\n"):
        return [p + separator for p in parts[:-1]] + parts[-1:] if parts else []
    if separator == ". ":
        return [p + ". " for p in parts[:-1]] + parts[-1:] if parts else []
    return parts


def _recursive_split(text: str, max_tokens: int, level: int = 0) -> list[str]:
    """Recursively split a string until each piece is under `max_tokens`."""
    if _count_tokens(text) <= max_tokens:
        return [text]
    if level >= len(_SPLIT_LEVELS):
        # Hard truncate by tokens as a last resort
        ids = _ENC.encode(text, disallowed_special=())
        return [_ENC.decode(ids[i : i + max_tokens]) for i in range(0, len(ids), max_tokens)]
    sep = _SPLIT_LEVELS[level]
    pieces = _split_by_separator(text, sep)
    if len(pieces) <= 1:
        return _recursive_split(text, max_tokens, level + 1)
    out: list[str] = []
    for piece in pieces:
        if _count_tokens(piece) <= max_tokens:
            out.append(piece)
        else:
            out.extend(_recursive_split(piece, max_tokens, level + 1))
    return out


def _merge_with_overlap(pieces: list[str], target_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack split pieces into ~target-token chunks with token overlap."""
    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for piece in pieces:
        piece_tokens = _count_tokens(piece)
        if buffer_tokens + piece_tokens <= target_tokens:
            buffer.append(piece)
            buffer_tokens += piece_tokens
            continue
        # Flush current buffer
        if buffer:
            chunks.append("".join(buffer).strip())
        # Start new buffer with token-overlap tail of previous
        if chunks and overlap_tokens > 0:
            prev_ids = _ENC.encode(chunks[-1], disallowed_special=())
            tail = _ENC.decode(prev_ids[-overlap_tokens:])
            buffer = [tail, piece]
            buffer_tokens = _count_tokens(tail) + piece_tokens
        else:
            buffer = [piece]
            buffer_tokens = piece_tokens

    if buffer:
        chunks.append("".join(buffer).strip())
    return [c for c in chunks if c.strip()]


def chunk_document(doc: Document) -> list[Document]:
    pieces = _recursive_split(doc.text, settings.CHUNK_SIZE)
    chunks = _merge_with_overlap(pieces, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    out: list[Document] = []
    base = doc.metadata.get("source", "doc")
    for i, c in enumerate(chunks):
        cid = hashlib.md5(f"{base}:{doc.metadata.get('page', 0)}:{i}:{c[:64]}".encode()).hexdigest()[:16]
        meta = dict(doc.metadata)
        meta["chunk_id"] = cid
        meta["chunk_index"] = i
        meta["chunk_token_count"] = _count_tokens(c)
        out.append(Document(text=c, metadata=meta))
    return out


def chunk_documents(docs: Iterable[Document]) -> list[Document]:
    out: list[Document] = []
    for d in docs:
        out.extend(chunk_document(d))
    return out


def write_chunks_jsonl(chunks: list[Document], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"text": c.text, "metadata": c.metadata}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    from src.ingest import load_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(settings.DATA_RAW))
    parser.add_argument("--output", default=str(settings.DATA_PROCESSED / "chunks.jsonl"))
    args = parser.parse_args()

    docs = load_directory(args.input)
    chunks = chunk_documents(docs)
    write_chunks_jsonl(chunks, args.output)
    print(f"Wrote {len(chunks)} chunks from {len(docs)} document units → {args.output}")
