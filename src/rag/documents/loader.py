"""Read markdown files from disk into Document objects.

Files can be free-form markdown. Metadata is optional and comes from a YAML
front-matter block at the very top of the file:

    ---
    title: Paid Time Off (PTO) Policy
    policy_id: PTO-001
    version: "4.0"        # quote versions, or YAML reads 3.10 as the number 3.1
    effective_date: 2026-01-01
    ---

Any key may be missing; unknown keys (e.g. owner) are ignored.

Documents are streamed: iter_documents() yields one Document at a time, so the
whole corpus is never in memory together. Files larger than max_bytes are skipped
before being read, so one oversized file can't exhaust memory either.
"""

import logging
import re
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import yaml

from rag.models import Document

logger = logging.getLogger(__name__)

# A front-matter block: "---" on the first line, YAML, then a closing "---" line.
FRONT_MATTER = re.compile(r"\A---[ \t]*\n(?P<yaml>.*?)\n---[ \t]*(\n|\Z)", re.DOTALL)

# The first top-level markdown heading, used as a fallback title.
TOP_HEADING = re.compile(r"^# +(?P<title>.+?)\s*$", re.MULTILINE)


def iter_documents(folder: Path, max_bytes: int) -> Iterator[Document]:
    """Yield the folder's .md files one Document at a time, sorted by file name.

    Files larger than max_bytes are skipped with a warning instead of being read,
    and the remaining files are still yielded.
    """
    markdown_files = sorted(folder.glob("*.md"))
    for path in markdown_files:
        size_in_bytes = path.stat().st_size
        if size_in_bytes > max_bytes:
            logger.warning(
                "Skipping %s: %d bytes is over the %d-byte limit",
                path.name,
                size_in_bytes,
                max_bytes,
            )
            continue
        yield load_document(path)


def load_document(path: Path) -> Document:
    """Load one markdown file, reading optional front matter and choosing a title.

    Title order: front-matter `title`, else the first "# " heading, else the file name.
    Raises ValueError naming the file if the front matter is not valid YAML mapping,
    so broken metadata fails loudly at load time instead of silently later.
    """
    raw_text = path.read_text(encoding="utf-8")
    front_matter = read_front_matter(raw_text, path)
    body = remove_front_matter(raw_text)

    return Document(
        doc_id=path.stem,
        title=choose_title(front_matter, body, path),
        policy_id=str(front_matter.get("policy_id", "")),
        version=str(front_matter.get("version", "")),
        effective_date=read_date(front_matter.get("effective_date"), path),
        source_path=path,
        text=body,
    )


def read_front_matter(raw_text: str, path: Path) -> dict:
    """Return the front-matter keys as a dict, or {} if the file has no front matter."""
    match = FRONT_MATTER.match(raw_text)
    if match is None:
        return {}
    try:
        parsed = yaml.safe_load(match.group("yaml"))
    except yaml.YAMLError as error:
        raise ValueError(f"{path.name}: front matter is not valid YAML ({error})")
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError(f"{path.name}: front matter must be 'key: value' lines")
    return parsed


def remove_front_matter(raw_text: str) -> str:
    """Return the file text with the front-matter block (if any) cut off the top."""
    match = FRONT_MATTER.match(raw_text)
    if match is None:
        return raw_text
    return raw_text[match.end() :]


def choose_title(front_matter: dict, body: str, path: Path) -> str:
    """Pick the title: front matter, then the first '# ' heading, then the file name."""
    if front_matter.get("title"):
        return str(front_matter["title"])
    heading = TOP_HEADING.search(body)
    if heading is not None:
        return heading.group("title")
    return path.stem


def read_date(value, path: Path) -> date | None:
    """Turn a front-matter date into a date; YAML may already have parsed it, or it may be text."""
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"{path.name}: effective_date {value!r} is not YYYY-MM-DD")
