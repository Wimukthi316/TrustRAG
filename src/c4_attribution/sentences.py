"""Sentence segmentation for C4, and the ablation that measures what it costs.

`evidence_index` is an index into `AnalysisResult.context_sentences`, so the
splitter is part of the frozen serving contract: change it and every index C4
has ever written moves. `tests/test_c3_grounding.py` already asserts that C3's
copy of the splitter is identical to the backend's, and C4 adds a third
assertion of the same equality here.

So the naive splitter stays. What C4 adds is a *variant* splitter used only in
the Layer 1c ablation, which reports how many evidence indices would move if the
segmentation were better. That turns an inherited weakness into a measurement
instead of a dependency negotiation over spaCy or pysbd.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

# Mirrors backend/app/services/detector.py and src/c3_explanation/grounding.py.
# tests/test_c4_sentences.py asserts all three stay identical.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Blank line, and RAGTruth's QA passage markers ("passage 1:", "passage 2:").
# Both are hard boundaries that the naive splitter walks straight through
# because neither ends in a full stop.
_BLANK_LINE = re.compile(r"\n\s*\n")
_PASSAGE_MARKER = re.compile(r"(?i)(?=passage\s*\d+\s*:)")


def split_sentences(text: str) -> List[str]:
    """The serving path's splitter, character for character.

    Deliberately naive. Do not improve it here -- see the module docstring.
    """
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def split_sentences_variant(text: str) -> List[str]:
    """The Layer 1c ablation splitter: also breaks on blank lines and passage markers.

    Not used in serving and not used for any reported evidence_index. Its only
    job is to be compared against `split_sentences` so the report can say how
    sensitive the numbers are to the segmentation, rather than admitting the
    weakness without quantifying it.
    """
    pieces: List[str] = []
    for block in _BLANK_LINE.split(text.strip()):
        for chunk in _PASSAGE_MARKER.split(block):
            pieces.extend(split_sentences(chunk))
    return [p for p in pieces if p]


def sentence_offsets(text: str, sentences: Sequence[str]) -> List[Tuple[int, int]]:
    """Character offsets of each sentence in the original text.

    The splitter strips whitespace, so a sentence is not `text[i:j]` for any
    obvious i and j; the offsets are recovered by scanning forward. A sentence
    that cannot be located (which should never happen for output of the two
    splitters above) gets (-1, -1) rather than a guess, so a caller can count
    the failures instead of silently mis-indexing.
    """
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        found = text.find(sentence, cursor)
        if found < 0:
            offsets.append((-1, -1))
            continue
        offsets.append((found, found + len(sentence)))
        cursor = found + len(sentence)
    return offsets


def index_of_offset(offsets: Sequence[Tuple[int, int]], position: int) -> int:
    """Which sentence contains this character position. -1 when none does."""
    for index, (start, end) in enumerate(offsets):
        if start <= position < end:
            return index
    return -1


def realign(
    sentences: Sequence[str], variant_sentences: Sequence[str], index: int
) -> int:
    """The variant-splitter index of the sentence holding `sentences[index]`.

    Used by the segmentation ablation: a naive sentence may be cut into several
    variant sentences, so the mapping is "the first variant sentence that is a
    substring of, or contains, the naive one". Returns -1 when nothing matches.
    """
    if not 0 <= index < len(sentences):
        return -1
    target = sentences[index]
    for position, candidate in enumerate(variant_sentences):
        if candidate in target or target in candidate:
            return position
    return -1
