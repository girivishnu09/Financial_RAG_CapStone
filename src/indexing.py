"""Section B (part 2) - Embedding + vector index (Chroma).

Uses Azure OpenAI embeddings (text-embedding-3-small by default). Persists to a
local Chroma collection with metadata for retrieval-time filtering.
"""
from __future__ import annotations

import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import AzureOpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import settings
from src.ingest import Document


_az_client: AzureOpenAI | None = None


def get_azure_client() -> AzureOpenAI:
    global _az_client
    if _az_client is None:
        _az_client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
    return _az_client


def get_chroma_client() -> chromadb.api.ClientAPI:
    settings.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.CHROMA_DIR),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection(client: chromadb.api.ClientAPI | None = None):
    client = client or get_chroma_client()
    return client.get_or_create_collection(
        name=settings.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _flatten_metadata(meta: dict) -> dict:
    """Chroma allows only str/int/float/bool/None - coerce other types."""
    out = {}
    for k, v in meta.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts via Azure OpenAI. Returns list of float vectors."""
    if not texts:
        return []
    client = get_azure_client()
    out: list[list[float]] = []
    # Azure embedding endpoint accepts batched input; we still batch by EMBED_BATCH_SIZE
    # to keep per-request payloads manageable.
    bs = settings.EMBED_BATCH_SIZE
    for start in range(0, len(texts), bs):
        batch = texts[start : start + bs]
        # Azure has an 8192-token-per-input limit for text-embedding-3-small.
        # Chunks are sized to ~800 tokens so we are well below it.
        resp = client.embeddings.create(
            model=settings.AZURE_OPENAI_EMBED_DEPLOYMENT,
            input=batch,
        )
        out.extend([d.embedding for d in resp.data])
    return out


def index_chunks(chunks: list[Document], rebuild: bool = False) -> int:
    """Embed and upsert chunks into Chroma. Returns count indexed."""
    client = get_chroma_client()
    if rebuild:
        try:
            client.delete_collection(settings.COLLECTION_NAME)
        except Exception:
            pass
    collection = get_collection(client)

    if not chunks:
        return collection.count()

    ids = [c.metadata["chunk_id"] for c in chunks]
    docs_text = [c.text for c in chunks]
    metas = [_flatten_metadata(c.metadata) for c in chunks]

    batch = settings.EMBED_BATCH_SIZE
    for start in tqdm(range(0, len(chunks), batch), desc="Indexing"):
        end = start + batch
        embs = embed_texts(docs_text[start:end])
        collection.upsert(
            ids=ids[start:end],
            embeddings=embs,
            documents=docs_text[start:end],
            metadatas=metas[start:end],
        )
    return collection.count()


def collection_stats() -> dict:
    coll = get_collection()
    return {"count": coll.count(), "name": coll.name, "path": str(settings.CHROMA_DIR)}


if __name__ == "__main__":
    print(collection_stats())
