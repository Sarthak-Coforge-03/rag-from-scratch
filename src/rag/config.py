import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv  # uv add python-dotenv

load_dotenv()  # reads .env once, at first import; real env vars take precedence


@dataclass(frozen=True)
class Settings:
    policy_docs_dir: Path = Path(os.getenv("POLICY_DOCS_DIR", "data/policies"))
    max_doc_bytes: int = int(os.getenv("MAX_DOC_BYTES", "5000000"))  # 5 MB
    # Index folders (SQL + vectors) live under index_root; index_name picks one.
    index_root: Path = Path(os.getenv("INDEX_ROOT", "indexes"))
    index_name: str = os.getenv("INDEX_NAME", "default")
    vector_store: str = os.getenv("VECTOR_STORE", "chroma")
    vector_collection: str = os.getenv("VECTOR_COLLECTION", "policy_chunks")
    embed_provider: str = os.getenv("EMBED_PROVIDER", "sentence-transformers")
    embed_model: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "token")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    retriever_strategy: str = os.getenv("RETRIEVER", "vector")
    top_k: int = int(os.getenv("TOP_K", "5"))
    rerank_provider: str = os.getenv("RERANK_PROVIDER", "cross-encoder")
    rerank_model: str = os.getenv(
        "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "20"))
    # The LLM that writes answers. On Foundry, LLM_MODEL is the deployment name.
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic-foundry")
    llm_model: str = os.getenv("LLM_MODEL", "claude-opus-5-5-fast")
    llm_endpoint: str = os.getenv("LLM_ENDPOINT", "")
    # Thinking tokens count toward this cap too, so keep it well above the answer length.
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "16000"))
    # How hard the model thinks: low | medium | high | xhigh | max
    llm_effort: str = os.getenv("LLM_EFFORT", "low")
    # repr=False: printing settings never shows the key.
    llm_api_key: str = field(
        default=os.getenv("ANTHROPIC_FOUNDRY_API_KEY", ""), repr=False
    )


settings = Settings()
