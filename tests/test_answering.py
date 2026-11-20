"""Tests for the answer step: numbered sources in, an answer with cited chunks out."""

from pydantic import BaseModel

from rag.answering import answer_question, build_user_message
from rag.generation import BaseGenerator
from rag.models import Chunk, ChunkMetadata, CitedAnswer, RetrievedChunk
from rag.retrieval import BaseRetriever


def make_result(doc_id: str, policy_id: str, version: str, text: str) -> RetrievedChunk:
    """A retrieved chunk from document `doc_id` with the given policy id and version."""
    metadata = ChunkMetadata(
        doc_id=doc_id,
        title=doc_id,
        policy_id=policy_id,
        version=version,
        effective_date="",
        source_path=f"{doc_id}.md",
        chunk_index=0,
        start_index=0,
        end_index=len(text),
        token_count=0,
    )
    chunk = Chunk(chunk_id=f"{doc_id}::0", text=text, metadata=metadata)
    return RetrievedChunk(chunk=chunk, score=1.0)


RESULTS = [
    make_result("pto_policy", "PTO-001", "4.0", "Employees get 20 days off."),
    make_result("expense_policy", "EXP-001", "3.0", "Directors approve over $2,500."),
    make_result("travel_policy", "TRV-001", "", "Book flights 14 days ahead."),
]


class FixedRetriever(BaseRetriever):
    """Fake retriever: always returns the same results."""

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """The first top_k fixed results."""
        return RESULTS[:top_k]


class RecordingGenerator(BaseGenerator):
    """Fake LLM: returns a fixed structured answer and remembers the prompt it got."""

    def __init__(self, reply: CitedAnswer):
        """Reply with `reply` to every question."""
        super().__init__(model_name="fake")
        self.reply = reply
        self.received_user_message = None

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Not used by the answer step."""
        raise NotImplementedError

    def generate_structured(
        self, system_prompt: str, user_message: str, output_type: type[BaseModel]
    ) -> BaseModel:
        """Record the prompt and return the fixed reply."""
        self.received_user_message = user_message
        return self.reply


def test_sources_are_numbered_in_retrieval_order_with_their_document():
    """[1] is the first retrieved chunk; each source names its doc and policy version."""
    message = build_user_message("Who approves expenses?", RESULTS)

    assert message.index("[1] pto_policy (PTO-001 v4.0)") < message.index(
        "[2] expense_policy (EXP-001 v3.0)"
    )
    assert "[3] travel_policy (TRV-001)" in message  # no version: none shown
    assert "Directors approve over $2,500." in message
    assert message.rstrip().endswith("Question: Who approves expenses?")


def test_answer_keeps_only_real_cited_sources():
    """Numbers map to retrieved chunks; repeats and numbers never shown are dropped."""
    reply = CitedAnswer(answer="A director approves it.", source_numbers=[2, 2, 9])
    generator = RecordingGenerator(reply)

    answer = answer_question(
        "Who approves expenses?", FixedRetriever(), generator, top_k=3
    )

    assert answer.text == "A director approves it."
    assert [source.chunk.metadata.doc_id for source in answer.cited_sources] == [
        "expense_policy"
    ]
    assert answer.retrieved == RESULTS  # everything the model was shown
    assert "[2] expense_policy" in generator.received_user_message
