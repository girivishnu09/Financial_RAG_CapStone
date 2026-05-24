"""One-shot pipeline: ingest -> chunk -> embed -> persist Chroma index.

Usage:
    python scripts/build_index.py                # incremental (skips if already populated)
    python scripts/build_index.py --rebuild      # wipe and rebuild
    python scripts/build_index.py --input path/  # custom input dir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.ingest import load_directory
from src.chunking import chunk_documents, write_chunks_jsonl
from src.indexing import index_chunks, collection_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(settings.DATA_RAW))
    parser.add_argument("--rebuild", action="store_true", help="Wipe and rebuild the collection")
    parser.add_argument("--write-chunks", action="store_true", help="Also write chunks.jsonl for inspection")
    args = parser.parse_args()

    stats_before = collection_stats()
    if stats_before["count"] > 0 and not args.rebuild:
        print(f"Collection '{stats_before['name']}' already has {stats_before['count']} chunks. Use --rebuild to overwrite.")
        return

    print(f"Ingesting from {args.input} ...")
    docs = load_directory(args.input)
    print(f"  -> {len(docs)} document units")

    if not docs:
        print("No documents found. Drop SEC filings (PDF/HTML/TXT) into data/raw/ and rerun.")
        return

    print("Chunking ...")
    chunks = chunk_documents(docs)
    print(f"  -> {len(chunks)} chunks")

    if args.write_chunks:
        out_path = settings.DATA_PROCESSED / "chunks.jsonl"
        write_chunks_jsonl(chunks, out_path)
        print(f"  -> wrote {out_path}")

    print("Embedding + indexing ...")
    final = index_chunks(chunks, rebuild=args.rebuild)
    print(f"Done. Collection now contains {final} chunks at {settings.CHROMA_DIR}")


if __name__ == "__main__":
    main()
