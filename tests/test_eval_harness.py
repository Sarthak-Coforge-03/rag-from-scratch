"""Eval harness (Part 5): retrieval recall and answer accuracy on the fixed eval set.

A fresh index is built once from the policy folder with the CHUNK_* / EMBED_*
settings, searched with RETRIEVER and TOP_K. Recall cases need no LLM and always
run (in CI too). Answer cases call the real LLM, so they are marked `llm` and
skipped when the Foundry endpoint or key is not configured.

    uv run pytest tests/test_eval_harness.py -v          # all cases
    uv run pytest tests/test_eval_harness.py -m "not llm" # recall only
"""

import pytest

from rag.answering import answer_question
from rag.config import settings
from rag.evaluation import (
    EVAL_SET_FILE,
    load_eval_cases,
    missing_key_facts,
    retrieval_hit,
)
from rag.generation import create_generator
from rag.indexes import build_index, load_searchable_index
from rag.models import EvalCase
from rag.retrieval import BaseRetriever

EVAL_CASES = load_eval_cases(EVAL_SET_FILE)
LLM_CONFIGURED = bool(settings.llm_endpoint and settings.llm_api_key)


@pytest.fixture(scope="module")
def retriever(tmp_path_factory) -> BaseRetriever:
    """Build a fresh index once for every case and return the RETRIEVER over it."""
    index_root = tmp_path_factory.mktemp("eval_index")
    index_name = build_index(index_root, settings.policy_docs_dir)
    searchable_index = load_searchable_index(index_name, index_root=index_root)
    yield searchable_index.make_retriever(settings.retriever_strategy)
    searchable_index.close()


@pytest.mark.parametrize("case", EVAL_CASES, ids=[case.id for case in EVAL_CASES])
def test_retrieval_recall(case: EvalCase, retriever: BaseRetriever):
    """An expected document is among the top TOP_K chunks (recall@k = passed / total)."""
    retrieved = retriever.retrieve(case.question, settings.top_k)
    retrieved_docs = [result.chunk.metadata.doc_id for result in retrieved]

    assert retrieval_hit(case, retrieved), (
        f"expected one of {case.expected_docs}, got {retrieved_docs}"
    )


@pytest.mark.llm
@pytest.mark.skipif(not LLM_CONFIGURED, reason="LLM_ENDPOINT / API key not set")
@pytest.mark.parametrize("case", EVAL_CASES, ids=[case.id for case in EVAL_CASES])
def test_answer_contains_key_facts(case: EvalCase, retriever: BaseRetriever):
    """The generated answer contains every key fact (accuracy = passed / total)."""
    generator = create_generator(settings.llm_provider, settings.llm_model)

    answer = answer_question(case.question, retriever, generator, settings.top_k)

    missing = missing_key_facts(case, answer.text)
    assert missing == [], f"answer lacks {missing}: {answer.text}"
