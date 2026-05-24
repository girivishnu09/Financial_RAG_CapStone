"""Centralized configuration for the RAG financial-QA system."""
from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).parent.resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Azure OpenAI ---
    # Required env vars (set in .env):
    #   AZURE_OPENAI_API_KEY
    #   AZURE_OPENAI_ENDPOINT       e.g. https://your-resource.openai.azure.com/
    #   AZURE_OPENAI_API_VERSION    e.g. 2024-10-21
    #   AZURE_OPENAI_CHAT_DEPLOYMENT  the deployment name you created in Azure (used for generation)
    #   AZURE_OPENAI_JUDGE_DEPLOYMENT the deployment used for the RAG-triad judge (often same)
    AZURE_OPENAI_API_KEY: str = Field(default="")
    AZURE_OPENAI_ENDPOINT: str = Field(default="")
    AZURE_OPENAI_API_VERSION: str = Field(default="2024-10-21")
    AZURE_OPENAI_CHAT_DEPLOYMENT: str = Field(default="gpt-4o-mini")
    AZURE_OPENAI_JUDGE_DEPLOYMENT: str = Field(default="gpt-4o-mini")
    # Embedding deployment in Azure (e.g. text-embedding-3-small, text-embedding-ada-002)
    AZURE_OPENAI_EMBED_DEPLOYMENT: str = Field(default="text-embedding-3-small")

    # --- Paths ---
    DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
    CHROMA_DIR: Path = PROJECT_ROOT / "chroma_db"
    EVAL_DIR: Path = PROJECT_ROOT / "eval"

    # --- Chunking ---
    CHUNK_SIZE: int = 800           # target tokens per chunk
    CHUNK_OVERLAP: int = 120        # tokens of overlap
    TOKENIZER_NAME: str = "cl100k_base"  # tiktoken encoding

    # --- Embeddings / index ---
    # Azure OpenAI embeddings (corporate firewall blocks HuggingFace)
    EMBED_MODEL: str = "azure/text-embedding-3-small"  # informational only; deployment set above
    COLLECTION_NAME: str = "sec_filings"
    EMBED_BATCH_SIZE: int = 32  # Azure embedding endpoint accepts up to 2048 inputs but smaller batches are safer

    # --- Retrieval ---
    TOP_K_RETRIEVE: int = 10
    TOP_K_RERANK: int = 4
    MMR_LAMBDA: float = 0.7
    USE_RERANKER: bool = False  # cross-encoder requires HuggingFace download; off by default behind firewalls
    USE_QUERY_REWRITER: bool = True

    # --- Reranker ---
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- LLM hyperparameters ---
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 800

    # --- Guardrails ---
    INSUFFICIENT_CONTEXT_THRESHOLD: float = 0.35  # avg cosine similarity floor


settings = Settings()
