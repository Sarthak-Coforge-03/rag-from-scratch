import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv  # uv add python-dotenv

load_dotenv()  # reads .env once, at first import; real env vars take precedence


@dataclass(frozen=True)
class Settings:
    policy_docs_dir: Path = Path(os.getenv("POLICY_DOCS_DIR", "data/policies"))
    max_doc_bytes: int = int(os.getenv("MAX_DOC_BYTES", "5000000"))  # 5 MB
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///data/docs.db")
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", ".chroma"))
    collection: str = os.getenv("CHROMA_COLLECTION", "policies")
    embed_provider: str = os.getenv("EMBED_PROVIDER", "sentence-transformers")
    embed_model: str = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    chunk_strategy: str = os.getenv("CHUNK_STRATEGY", "token")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "5"))


settings = Settings()
