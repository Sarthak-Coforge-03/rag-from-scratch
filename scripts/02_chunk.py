"""Chunk step: sync the policy folder into one index's SQL database (no embedding).

Two ways to run it:

  Create a brand-new index from the CHUNK_* env settings. It is named
  <strategy>-<size> (e.g. structural-800), or structural-800_1, _2, ... if taken:
      uv run python scripts/02_chunk.py

  Update an existing index with the settings it was created with (CHUNK_* env is
  ignored). Only new, edited or deleted files are processed:
      uv run python scripts/02_chunk.py --sync structural-800

An index's chunker settings never change after creation, so its chunks always come
from one config. Vector folders are not touched; the embed step syncs them later.
"""

import argparse
import logging
import sys

from rag.chunking import create_chunker
from rag.config import settings
from rag.documents import DocumentRepository
from rag.indexes import IndexManager, chunker_fingerprint
from rag.ingest import Ingestor
from rag.models import ChunkerConfig, IngestReport


def main() -> None:
    """Create or update one index, chunk the policy folder into it and print a summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--sync",
        metavar="INDEX_NAME",
        help="existing index to bring up to date; leave out to create a new one from env",
    )
    arguments = parser.parse_args()
    manager = IndexManager(settings.index_root)

    if arguments.sync is None:
        config = ChunkerConfig(
            strategy=settings.chunk_strategy,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        index_name = manager.next_free_name(f"{config.strategy}-{config.chunk_size}")
        print(
            f"Creating new index {index_name!r} from CHUNK_* env settings", flush=True
        )
    else:
        index_name = arguments.sync
        try:
            config = manager.read_chunker_config(index_name)
        except ValueError as error:
            sys.exit(f"Error: {error}; run without --sync to create one")
        print(f"Updating index {index_name!r} with its saved settings", flush=True)

    chunker = create_chunker(
        strategy=config.strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    fingerprint = chunker_fingerprint(chunker)
    if arguments.sync is None:
        layout = manager.create_index(index_name, fingerprint, config)
    else:
        layout = manager.layout(index_name)

    repository = DocumentRepository(layout.database_url)
    ingestor = Ingestor(repository=repository, chunker=chunker)
    report = ingestor.ingest_folder(settings.policy_docs_dir, settings.max_doc_bytes)
    repository.close()

    print_summary(layout.root, fingerprint, report)


def print_summary(index_folder, fingerprint: str, report: IngestReport) -> None:
    """Print where the data went, one line of counts, then the doc_ids in each group.

    Empty groups show only in the counts line, so the output stays short when
    nothing changed.
    """
    groups = {
        "Added": report.added,
        "Updated": report.updated,
        "Unchanged": report.unchanged,
        "Removed": report.removed,
    }
    counts = [f"{len(doc_ids)} {name.lower()}" for name, doc_ids in groups.items()]

    print()
    print(f"Index folder : {index_folder}")
    print(f"Chunker      : {fingerprint}")
    print(f"Data folder  : {settings.policy_docs_dir}")
    print(f"Documents    : {', '.join(counts)}")
    for group_name, doc_ids in groups.items():
        if not doc_ids:
            continue
        print()
        print(f"{group_name} ({len(doc_ids)}):")
        for doc_id in doc_ids:
            print(f"  - {doc_id}")


if __name__ == "__main__":
    main()
