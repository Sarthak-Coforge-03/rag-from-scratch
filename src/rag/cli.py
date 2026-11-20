"""rag-from-scratch: ask questions about the policy documents from the terminal.

    uv run rag-from-scratch

Menus pick the index, retriever, reranking and LLM (the .env values preselected),
then it keeps asking questions until you enter an empty one. Indexes are prepared
by scripts/02_chunk.py and scripts/03_embed.py; this CLI only loads one and
answers from it.
"""

import logging
import sys

import questionary

from rag.answering import answer_question, policy_label
from rag.config import settings
from rag.generation import GENERATOR_CLASSES, create_generator
from rag.indexes import IndexManager, SearchableIndex, load_searchable_index
from rag.models import Answer
from rag.reranking import create_reranker
from rag.retrieval import RETRIEVER_STRATEGIES, BaseRetriever, RerankedRetriever


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
    """The whole session: choices, loading, then the question loop."""
    # 1. Choose what to search and how.
    manager = IndexManager(settings.index_root)
    index_name = choose("Index:", manager.list_index_names(), settings.index_name)
    embed_model = choose_embedding_model(manager, index_name)
    strategy = choose(
        "Retriever:", list(RETRIEVER_STRATEGIES), settings.retriever_strategy
    )
    rerank = confirm("Rerank the retrieved chunks?")
    llm_provider = choose("LLM:", list(GENERATOR_CLASSES), settings.llm_provider)

    # 2. Load everything once; loading the models takes a few seconds.
    print("Loading index and models...", flush=True)
    searchable_index = load_searchable_index(index_name, embed_model)
    retriever = build_retriever(searchable_index, strategy, rerank)
    generator = create_generator(llm_provider, settings.llm_model)
    retriever_label = f"{strategy} + rerank" if rerank else strategy
    print(
        f"Index: {index_name} | Retriever: {retriever_label} | LLM: {settings.llm_model}"
    )

    # 3. Keep answering until an empty question.
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
    logging.getLogger("rag").setLevel(logging.INFO)


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
            f" {cited_mark} [{position + 1}] {metadata.doc_id:<40} {label:<14}"
            f" score {result.score:7.3f}"
        )
    print()
