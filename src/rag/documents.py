"""Read policy markdown files from disk into Document objects."""

import re
from datetime import date
from pathlib import Path

from rag.models import Document

# Matches header lines such as "- **Policy ID:** PTO-001" and captures the key and value.
METADATA_LINE = re.compile(r"^- \*\*(?P<key>[^:*]+):\*\* (?P<value>.+)$", re.MULTILINE)

REQUIRED_KEYS = ["Policy ID", "Version", "Effective date"]


def load_documents(folder: Path) -> list[Document]:
    """Load every .md file in the folder, sorted by file name so the order is stable."""
    markdown_files = sorted(folder.glob("*.md"))
    return [load_document(path) for path in markdown_files]


def load_document(path: Path) -> Document:
    """Load one policy file, reading its title and header metadata.

    Raises ValueError naming the file if the title or any required metadata is missing,
    so bad source data fails loudly at load time instead of silently later.
    """
    text = path.read_text(encoding="utf-8")

    first_line = text.splitlines()[0] if text else ""
    if not first_line.startswith("# "):
        raise ValueError(f"{path.name}: first line must be a '# Title' heading")
    title = first_line.removeprefix("# ").strip()

    metadata = {}
    for match in METADATA_LINE.finditer(text):
        metadata[match.group("key").strip()] = match.group("value").strip()

    missing_keys = [key for key in REQUIRED_KEYS if key not in metadata]
    if missing_keys:
        raise ValueError(f"{path.name}: missing metadata {missing_keys}")

    return Document(
        doc_id=path.stem,
        title=title,
        policy_id=metadata["Policy ID"],
        version=metadata["Version"],
        effective_date=date.fromisoformat(metadata["Effective date"]),
        source_path=path,
        text=text,
    )
