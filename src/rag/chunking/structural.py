"""StructuralChunker: split before markdown headings first (strategy "structural").

How it works: the same chonkie RecursiveChunker as the "recursive" strategy, but with
one extra level in front: the text is first split just *before* every markdown
heading line ("# ", "## ", ... "###### "), so each section starts its own piece with
its heading attached. Sections that are still longer than chunk_size fall through to
chonkie's default levels (paragraphs, sentences, pauses, words).

Why headings first: with plain recursive splitting a heading often lands at the *end*
of the previous chunk, separated from the text it introduces. Starting each chunk
with its heading keeps the section's topic words in the same embedding as its facts.

Works on any markdown: documents without headings (memos, FAQs) simply fall back to
paragraph splitting. Known limitation: in Q&A-style files where a bold question and
its answer are separate paragraphs, a question can end one chunk and its answer start
the next.

Not using chonkie's RecursiveRules.from_recipe("markdown"): it downloads the recipe
from Hugging Face at runtime, which breaks offline runs and CI.

No overlap, as with "recursive": chunk_overlap is ignored and a warning is logged when
it is non-zero. chunk_size has the same meaning and rationale as in token_window.py.
"""

import logging

import chonkie
from chonkie.types import RecursiveLevel, RecursiveRules

from rag.chunking.base import BaseChunker

logger = logging.getLogger(__name__)

# Split just before a heading line of any level; include_delim="next" keeps the
# heading at the start of the piece that follows it.
HEADING_LEVEL = RecursiveLevel(
    delimiters=["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### "],
    include_delim="next",
)


class StructuralChunker(BaseChunker):
    """Split at markdown headings first, then paragraphs, sentences and words."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        """Configure chonkie's RecursiveChunker with a heading level before its defaults.

        Logs a warning if chunk_overlap is non-zero, because this strategy can't apply it.
        """
        super().__init__(chunk_size, chunk_overlap)
        if chunk_overlap > 0:
            logger.warning(
                "Chunk strategy 'structural' does not support overlap; "
                "chunk_overlap=%d is ignored",
                chunk_overlap,
            )
        default_levels = RecursiveRules().levels
        rules = RecursiveRules(levels=[HEADING_LEVEL] + default_levels)
        self.chonkie_chunker = chonkie.RecursiveChunker(
            tokenizer="character",
            chunk_size=chunk_size,
            rules=rules,
        )

    def split_text(self, text: str) -> list[chonkie.Chunk]:
        """Split the text into heading-led sections, breaking long ones down further."""
        return self.chonkie_chunker.chunk(text)
