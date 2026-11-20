import os
from dataclasses import dataclass
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


settings = Settings()
