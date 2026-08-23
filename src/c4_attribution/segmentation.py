"""Context segmentation for Component 4 (Retrieval-Attribution Alignment).

Splits retrieved context passages into sentence units while preserving exact
character offsets (start, end) into the original text, ensuring that
`context[segment.start:segment.end] == segment.text` holds true.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set


@dataclass(frozen=True)
class ContextSegment:
    """One segmented sentence from the retrieved context.

    Attributes:
        index: 0-based sentence index within the context (maps to evidence_index).
        text: The sentence string stripped of leading/trailing whitespace.
        start: Inclusive start character offset in original context string.
        end: Exclusive end character offset in original context string.
    """

    index: int
    text: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError(f"index must be >= 0, got {self.index}")
        if self.end <= self.start:
            raise ValueError(
                f"end ({self.end}) must be greater than start ({self.start})"
            )


# Known common abbreviations (lowercase without trailing dot for quick lookup)
_KNOWN_ABBREVIATIONS: Set[str] = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "vs",
    "etc",
    "al",
    "approx",
    "inc",
    "ltd",
    "u.s",
    "u.k",
    "e.u",
    "e.g",
    "i.e",
}

# Regex to find candidate split points (terminal punctuation followed by whitespace or double newlines)
_CANDIDATE_SPLIT_REGEX = re.compile(
    r"([.!?])(?:\s+|\n+)|\n{2,}",
    flags=re.IGNORECASE,
)


def _is_abbreviation_or_decimal(context: str, punct_pos: int) -> bool:
    """Checks if the punctuation at punct_pos is part of an abbreviation, acronym, or decimal."""
    # Check for decimal number: e.g. "3.14"
    if punct_pos > 0 and punct_pos + 1 < len(context):
        if context[punct_pos - 1].isdigit() and context[punct_pos + 1].isdigit():
            return True

    # Extract the word preceding the punctuation
    preceding_text = context[:punct_pos]
    match = re.search(r"([a-zA-Z0-9\.]+)$", preceding_text)
    if match:
        word = match.group(1).lower().rstrip(".")
        if word in _KNOWN_ABBREVIATIONS:
            return True
        # Check single letter initial, e.g. "John F. Kennedy"
        if len(word) == 1 and word.isalpha():
            return True

    return False


def segment_context(context: str) -> List[ContextSegment]:
    """Splits a context string into clean sentence segments with exact character offsets.

    Args:
        context: The raw retrieved context text.

    Returns:
        A list of ContextSegment objects. Returns an empty list if context is empty
        or whitespace only.
    """
    if not context or not context.strip():
        return []

    split_points = [0]

    for match in _CANDIDATE_SPLIT_REGEX.finditer(context):
        punct = match.group(1)
        if punct is not None:
            punct_pos = match.start(1)
            if _is_abbreviation_or_decimal(context, punct_pos):
                continue
        split_points.append(match.end())

    split_points.append(len(context))

    segments: List[ContextSegment] = []
    seen_ranges = set()

    for i in range(len(split_points) - 1):
        raw_start = split_points[i]
        raw_end = split_points[i + 1]
        raw_chunk = context[raw_start:raw_end]

        stripped = raw_chunk.strip()
        if not stripped:
            continue

        l_offset = len(raw_chunk) - len(raw_chunk.lstrip())
        r_offset = len(raw_chunk) - len(raw_chunk.rstrip())

        start = raw_start + l_offset
        end = raw_end - r_offset
        text = context[start:end]

        if (start, end) in seen_ranges:
            continue
        seen_ranges.add((start, end))

        segments.append(
            ContextSegment(
                index=len(segments),
                text=text,
                start=start,
                end=end,
            )
        )

    if not segments and context.strip():
        l_offset = len(context) - len(context.lstrip())
        r_offset = len(context) - len(context.rstrip())
        start = l_offset
        end = len(context) - r_offset
        segments.append(
            ContextSegment(
                index=0,
                text=context[start:end],
                start=start,
                end=end,
            )
        )

    return segments
