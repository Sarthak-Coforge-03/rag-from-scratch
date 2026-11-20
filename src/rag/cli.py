"""rag-from-scratch: ask questions about the policy documents from the terminal.

    uv run rag-from-scratch

Menus pick the index, retriever, reranking and LLM (the .env values preselected),
then it keeps asking questions until you enter an empty one. Indexes are prepared
by scripts/02_chunk.py and scripts/03_embed.py; the first menu entry instead builds
a fresh one from the .env settings in a temporary folder, deleted on quit.
"""

import logging
import sys
import tempfile
from pathlib import Path

import questionary

from rag.answering import answer_question, policy_label
from rag.config import settings
from rag.generation import GENERATOR_CLASSES, create_generator
from rag.indexes import (
    IndexManager,
    SearchableIndex,
    build_index,
    load_searchable_index,
)
from rag.models import Answer
from rag.reranking import create_reranker
from rag.retrieval import RETRIEVER_STRATEGIES, BaseRetriever, RerankedRetriever


# First entry of the index menu: chunk and embed from .env into a temp folder.
FRESH_INDEX_CHOICE = "Build a fresh index from .env (temporary)"


class QuitRequested(Exception):
    """Raised when the user presses Ctrl-C in a menu."""


def main() -> None:
    """Pick an index and settings from menus, then answer questions until quit."""
    configure_logging()
    try:
        run()
    except QuitRequested:
        print("Bye.")
    except ValueError as error:
        sys.exit(f"Error: {error}")


def run() -> None:
    """The whole session: choices, then loading, then the question loop."""
    # 1. Choose what to search and how.
    manager = IndexManager(settings.index_root)
    index_choices = [FRESH_INDEX_CHOICE] + manager.list_index_names()
    index_name = choose("Index:", index_choices, settings.index_name)
    build_fresh = index_name == FRESH_INDEX_CHOICE
    embed_model = None if build_fresh else choose_embedding_model(manager, index_name)
    strategy = choose(
        "Retriever:", list(RETRIEVER_STRATEGIES), settings.retriever_strategy
    )
    rerank = confirm("Rerank the retrieved chunks?")
    llm_provider = choose("LLM:", list(GENERATOR_CLASSES), settings.llm_provider)

    # 2. Load the chosen index, or build a fresh one that is deleted on quit.
    if not build_fresh:
        print("Loading index and models...", flush=True)
        searchable_index = load_searchable_index(index_name, embed_model)
        ask_questions(searchable_index, strategy, rerank, llm_provider)
        return

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_folder:
        temp_root = Path(temp_folder)
        print(
            f"Building index from {settings.policy_docs_dir} (chunk, then embed)...",
            flush=True,
        )
        fresh_index_name = build_index(temp_root, settings.policy_docs_dir)
        searchable_index = load_searchable_index(fresh_index_name, index_root=temp_root)
        ask_questions(searchable_index, strategy, rerank, llm_provider)


def ask_questions(
    searchable_index: SearchableIndex, strategy: str, rerank: bool, llm_provider: str
) -> None:
    """Build the retriever and LLM, then answer questions until an empty one."""
    retriever = build_retriever(searchable_index, strategy, rerank)
    generator = create_generator(llm_provider, settings.llm_model)
    retriever_label = f"{strategy} + rerank" if rerank else strategy
    print(
        f"Index: {searchable_index.name} | Retriever: {retriever_label}"
        f" | LLM: {settings.llm_model}"
    )

    while True:
        question = questionary.text("Question (empty to quit):").ask()
        if not question:
            break
        print("Thinking...", flush=True)
        answer = answer_question(question, retriever, generator, settings.top_k)
        print_answer(answer)

    searchable_index.close()
    print("Bye.")


def configure_logging() -> None:
    """Show this project's INFO logs, but only warnings from libraries (HF, HTTP)."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("rag").setLevel(logging.CRITICAL)


def choose(prompt: str, choices: list[str], preferred: str) -> str:
    """Ask the user to pick one choice; `preferred` (from .env) is preselected if valid."""
    if not choices:
        raise ValueError(f"Nothing to choose for {prompt!r}; prepare an index first")
    default = preferred if preferred in choices else None
    selected = questionary.select(prompt, choices=choices, default=default).ask()
    if selected is None:  # questionary returns None on Ctrl-C
        raise QuitRequested()
    return selected


def confirm(prompt: str) -> bool:
    """Ask a yes/no question; Ctrl-C quits."""
    answer = questionary.confirm(prompt, default=False).ask()
    if answer is None:
        raise QuitRequested()
    return answer


def choose_embedding_model(manager: IndexManager, index_name: str) -> str | None:
    """Ask which model's vectors to search, only when the index has several.

    None means "the only one there"; load_searchable_index picks it, or explains
    that the index has no vectors yet.
    """
    model_names = [
        config.model_name for config in manager.read_embedder_configs(index_name)
    ]
    if len(model_names) <= 1:
        return None
    return choose("Embedding model:", model_names, settings.embed_model)


def build_retriever(
    searchable_index: SearchableIndex, strategy: str, rerank: bool
) -> BaseRetriever:
    """The chosen retriever, wrapped so a reranker re-scores its candidates if asked."""
    retriever = searchable_index.make_retriever(strategy)
    if not rerank:
        return retriever
    reranker = create_reranker(settings.rerank_provider, settings.rerank_model)
    return RerankedRetriever(
        retriever=retriever, reranker=reranker, candidates=settings.rerank_candidates
    )


def print_answer(answer: Answer) -> None:
    """Print the answer, then every chunk it was shown, marking the ones it cited."""
    print()
    print("Answer:")
    print(answer.text)
    if not answer.cited_sources:
        print("(Not found in the sources.)")

    print()
    print("Sources (* = cited):")
    cited_chunk_ids = {source.chunk.chunk_id for source in answer.cited_sources}
    for position in range(len(answer.retrieved)):
        result = answer.retrieved[position]
        metadata = result.chunk.metadata
        cited_mark = "*" if result.chunk.chunk_id in cited_chunk_ids else " "
        label = policy_label(metadata.policy_id, metadata.version)
        print(
            f" {cited_mark} [{position + 1}] {metadata.doc_id:<36} {label:<14}"
            f" {metadata.section:<30.30} score {result.score:7.3f}"
        )
    print()
