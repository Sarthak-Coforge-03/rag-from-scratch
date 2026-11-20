"""Embed step: turn an index's SQL chunks into vectors, one folder per embedding model.

Two ways to run it:

  Create the vector folder for EMBED_MODEL in an index. If that model already has a
  folder there, it is deleted and every chunk is embedded again from scratch:
      uv run python scripts/03_embed.py --create structural-800

  Sync every vector folder of an index after the chunk step changed its SQL data.
  Each folder uses the model it was created with (EMBED_MODEL is ignored); only new
  chunks are embedded and chunks no longer in SQL are deleted:
      uv run python scripts/03_embed.py --sync structural-800
"""

import argparse
import logging
import sys

from rag.config import settings
from rag.documents import DocumentRepository
from rag.embed import sync_vectors
from rag.embedding import create_embedder
from rag.indexes import IndexManager, embedder_fingerprint
from rag.models import EmbedderConfig, VectorSyncReport
from rag.vectorstore import create_vector_store


def main() -> None:
    """Parse --create / --sync and run it; any ValueError becomes a one-line error, exit 1."""
    # INFO for our own code only; libraries (HuggingFace downloads, progress bars)
    # stay at WARNING so the output is readable.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("rag").setLevel(logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--create",
        metavar="INDEX_NAME",
        help="(re)build the EMBED_MODEL vector folder of this index from scratch",
    )
    mode.add_argument(
        "--sync",
        metavar="INDEX_NAME",
        help="bring every vector folder of this index up to date with its SQL data",
    )
    arguments = parser.parse_args()
    try:
        run(arguments)
    except ValueError as error:
        sys.exit(f"Error: {error}")


def run(arguments: argparse.Namespace) -> None:
    """Pick the vector folders to work on, then embed into each one in turn.

    --create: one folder, for EMBED_MODEL, wiped and recreated before embedding.
    --sync:   every existing folder, each rebuilt from its saved embedder_config.json.
    """
    manager = IndexManager(settings.index_root)
    if arguments.create is not None:
        index_name = arguments.create
        env_config = EmbedderConfig(
            provider=settings.embed_provider, model_name=settings.embed_model
        )
        configs_to_embed = [env_config]
        print(
            f"Creating vectors for {env_config.model_name} in {index_name!r}",
            flush=True,
        )
    else:
        index_name = arguments.sync
        configs_to_embed = manager.read_embedder_configs(index_name)
        if not configs_to_embed:
            raise ValueError(
                f"index {index_name!r} has no vector folders; run with --create first"
            )
        print(
            f"Syncing {len(configs_to_embed)} vector folder(s) in {index_name!r}",
            flush=True,
        )

    layout = manager.open_chunked(index_name)
    repository = DocumentRepository(layout.database_url)
    for config in configs_to_embed:
        # Loads the model once per folder (seconds), then reuses it for every batch.
        embedder = create_embedder(
            provider=config.provider,
            model_name=config.model_name,
            batch_size=settings.embed_batch_size,
        )
        fingerprint = embedder_fingerprint(embedder)
        if arguments.create is not None:
            vector_dir = manager.create_vector_dir(index_name, fingerprint, config)
        else:
            vector_dir = layout.vector_dir(fingerprint)
        vector_store = create_vector_store(
            store=settings.vector_store,
            location=str(vector_dir),
            collection_name=settings.vector_collection,
        )
        report = sync_vectors(repository, embedder, vector_store)
        print_summary(layout.root, fingerprint, vector_dir, report)
    repository.close()


def print_summary(
    index_folder, fingerprint: str, vector_dir, report: VectorSyncReport
) -> None:
    """Print which folder was synced and how many chunks were embedded, deleted or kept."""
    chunks_in_sql = report.embedded + report.unchanged
    print()
    print(f"Index folder  : {index_folder}")
    print(f"Embedder      : {fingerprint}")
    print(f"Vector folder : {vector_dir}")
    print(
        f"Chunks        : {report.embedded} embedded, {report.deleted} deleted, "
        f"{report.unchanged} unchanged ({chunks_in_sql} in SQL)"
    )


if __name__ == "__main__":
    main()
