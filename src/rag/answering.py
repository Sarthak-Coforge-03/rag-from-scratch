"""The answer step: retrieve chunks, ask the LLM, and return the answer with its sources.

The retrieved chunks go into the prompt as numbered sources [1], [2], ... The model
replies in the CitedAnswer shape (structured output): the answer text plus the
numbers of the sources it used. Those numbers are mapped back to the chunks, so
every answer comes with the documents it was based on.

    answer = answer_question("Who approves expenses?", retriever, generator, top_k=5)
    answer.text, answer.cited_sources
"""

from rag.generation import BaseGenerator
from rag.models import Answer, CitedAnswer, RetrievedChunk
from rag.retrieval import BaseRetriever

SYSTEM_PROMPT = """\
You answer employees' questions about company policy using only the numbered \
sources you are given.
- Base the answer only on the sources, not on general knowledge.
- In source_numbers, list the number of every source the answer relies on.
- If the sources don't contain the answer, say so in the answer and leave \
source_numbers empty.
- Keep the answer short and direct."""


def answer_question(
    question: str, retriever: BaseRetriever, generator: BaseGenerator, top_k: int
) -> Answer:
    """Retrieve top_k chunks, have the LLM answer from them, and attach its cited chunks."""
    # 1. Find the chunks most relevant to the question.
    retrieved = retriever.retrieve(question, top_k)

    # 2. Ask the LLM, showing it those chunks as numbered sources.
    user_message = build_user_message(question, retrieved)
    reply = generator.generate_structured(SYSTEM_PROMPT, user_message, CitedAnswer)

    # 3. Turn the source numbers it cited back into chunks.
    cited_sources = sources_for_numbers(reply.source_numbers, retrieved)
    return Answer(text=reply.answer, cited_sources=cited_sources, retrieved=retrieved)


def build_user_message(question: str, retrieved: list[RetrievedChunk]) -> str:
    """The sources, numbered from [1] in retrieval order, followed by the question."""
    source_blocks = []
    for position in range(len(retrieved)):
        source_number = position + 1
        chunk = retrieved[position].chunk
        label = policy_label(chunk.metadata.policy_id, chunk.metadata.version)
        heading = f"[{source_number}] {chunk.metadata.doc_id} ({label})"
        source_blocks.append(f"{heading}\n{chunk.text}")

    sources_text = "\n\n".join(source_blocks)
    return f"{sources_text}\n\nQuestion: {question}"


def policy_label(policy_id: str, version: str) -> str:
    """e.g. "EXP-001 v3.0"; just the id when the document has no version."""
    if version:
        return f"{policy_id} v{version}"
    return policy_id


def sources_for_numbers(
    source_numbers: list[int], retrieved: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    """Map cited numbers (1 = first retrieved chunk) to chunks, in the model's order.

    Repeats are kept once, and numbers that were never shown (0, or more than the
    number of sources) are ignored, so a made-up citation can't point at a chunk.
    """
    cited_sources = []
    numbers_already_used = set()
    for number in source_numbers:
        is_shown_source = 1 <= number <= len(retrieved)
        if is_shown_source and number not in numbers_already_used:
            cited_sources.append(retrieved[number - 1])
            numbers_already_used.add(number)
    return cited_sources
