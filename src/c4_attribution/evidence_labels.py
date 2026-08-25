"""Layer 1a -- the evidence label hiding inside RAGTruth's annotator comments.

Every one of RAGTruth's 14,289 hallucination annotations carries a free-text
`meta` field. A large share of them contain a line beginning `Original:` in
which the annotator quoted the source text they judged the span against. That
quote is the only human evidence label the corpus has, and no published work
uses it. This module is the extractor, and it is deliberately separate from the
audit that reports over it.

Two things make the extraction non-trivial, and both were found by counting
rather than assumed:

1.  **The closing marker is not one string.** The quote runs until the next
    annotator marker, and annotators used four different ones. An extractor
    that looks only for `Generated:` reads past the end of the quote on most of
    the corpus. The marker list here is closed on purpose: a generic
    `\\n\\s*Word\\s*:` scan also fires on `Review:`, `Sunday:` and `Weekends:`
    from *inside* the quoted text, which silently truncates real quotes.

2.  **Not every `Original:` line is a quote.** In roughly a fifth of them the
    annotator wrote that the information is not in the source at all. Those are
    a gold "no evidence exists" label -- the one thing a nearest-neighbour
    baseline structurally cannot produce -- and they must never be pooled with
    the quotes.

`src/c3_explanation/labels_audit.original_snippet` reads the same field with a
different convention (first line only). Neither is wrong; they answer different
questions and give different numbers. C4 reports the disagreement rather than
picking a winner quietly -- see `evidence_audit.convention_disagreement`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

# The closed marker vocabulary. Anything not on this list is treated as part of
# the quote, which is the conservative direction: a missed marker lengthens a
# snippet, a false marker destroys one.
MARKERS: tuple[str, ...] = ("original", "generative", "generated", "aigc", "aogc")

_MARKER_RE = re.compile(
    r"(?im)^[ \t]*(" + "|".join(MARKERS) + r")[ \t]*:",
)
_ORIGINAL_RE = re.compile(r"(?im)^[ \t]*original[ \t]*:")

# The phrases annotators used when they meant "there is nothing to point at".
# This is a heuristic, not a validated classifier, and the audit says so in its
# own output. `evidence_audit` builds the hand-check sample that decides whether
# it may be quoted as a label at all.
NO_EVIDENCE_PHRASES: tuple[str, ...] = (
    "not mentioned",
    "no mention",
    "not in",
    "not found",
    "not present",
    "not stated",
    "not provided",
    "not specified",
    "not included",
    "nothing in",
    "nothing about",
    "does not mention",
    "does not state",
    "does not say",
    "absent from",
    "no such",
    "not supported",
    "cannot be found",
    "no information",
)

_NO_EVIDENCE_RE = re.compile(
    "(?i)(" + "|".join(re.escape(p) for p in NO_EVIDENCE_PHRASES) + ")"
)

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class EvidenceLabel:
    """What one annotator comment says about where the evidence is."""

    response_id: str
    start: int
    end: int
    span_text: str
    label_type: str
    raw_meta: str
    implicit_true: bool
    due_to_null: bool
    has_original: bool
    snippet: Optional[str]
    no_evidence: bool
    closing_marker: Optional[str]

    @property
    def attributable(self) -> bool:
        """True when the annotator quoted real source text to point at."""
        return bool(self.snippet) and not self.no_evidence


def normalise(text: Optional[str]) -> str:
    """Whitespace-collapsed, lowercased. The form both verbatim tests use."""
    return _WS.sub(" ", (text or "").strip()).lower()


def has_original_marker(meta: Optional[str]) -> bool:
    return bool(meta) and _ORIGINAL_RE.search(meta) is not None


def closing_marker(meta: Optional[str]) -> Optional[str]:
    """Which marker ends the `Original:` block, or None when nothing does.

    Reported as its own distribution because "no second marker at all" is the
    third most common case and an extractor that assumes one would mangle it.
    """
    if not meta:
        return None
    opening = _ORIGINAL_RE.search(meta)
    if opening is None:
        return None
    following = _MARKER_RE.search(meta, opening.end())
    return following.group(1).lower() if following else None


def original_snippet(meta: Optional[str]) -> Optional[str]:
    """The quoted source text: everything after `Original:` up to the next marker.

    This is C4's convention. It keeps multi-line quotes whole, which the
    first-line convention does not.
    """
    if not meta:
        return None
    opening = _ORIGINAL_RE.search(meta)
    if opening is None:
        return None
    tail = meta[opening.end():]
    following = _MARKER_RE.search(tail)
    body = tail[: following.start()] if following else tail
    body = body.strip()
    return body or None


def original_snippet_first_line(meta: Optional[str]) -> Optional[str]:
    """The other convention: `Original:` up to the first line break.

    Identical in intent to `src/c3_explanation/labels_audit.original_snippet`.
    Kept here so the two conventions can be compared without importing C3, which
    is not always on the branch C4 is running from.
    """
    if not meta:
        return None
    opening = _ORIGINAL_RE.search(meta)
    if opening is None:
        return None
    line = meta[opening.end():].split("\n", 1)[0].strip()
    return line or None


def is_no_evidence(snippet: Optional[str]) -> bool:
    """Does this `Original:` text say the evidence does not exist?

    A heuristic over a closed phrase list. Its agreement with a human is
    measured in Layer 1b and every number that depends on it is reported with
    that agreement beside it.
    """
    if not snippet:
        return False
    return _NO_EVIDENCE_RE.search(snippet) is not None


def label_from_annotation(
    response_id: str, annotation: Dict[str, object]
) -> EvidenceLabel:
    """Turn one raw RAGTruth label dict into an EvidenceLabel."""
    meta = annotation.get("meta")
    meta_text = str(meta) if meta is not None else None
    snippet = original_snippet(meta_text)
    return EvidenceLabel(
        response_id=str(response_id),
        start=int(annotation["start"]),
        end=int(annotation["end"]),
        span_text=str(annotation.get("text") or ""),
        label_type=str(annotation.get("label_type") or ""),
        raw_meta=meta_text or "",
        implicit_true=bool(annotation.get("implicit_true")),
        due_to_null=bool(annotation.get("due_to_null")),
        has_original=has_original_marker(meta_text),
        snippet=snippet,
        no_evidence=is_no_evidence(snippet),
        closing_marker=closing_marker(meta_text),
    )


def marker_counts(metas: Sequence[Optional[str]]) -> Dict[str, int]:
    """Distribution of closing markers over the comments that have `Original:`.

    `none` counts the comments where the quote runs to the end of the field.
    """
    counts: Dict[str, int] = {name: 0 for name in MARKERS if name != "original"}
    counts["original"] = 0
    counts["none"] = 0
    for meta in metas:
        if not has_original_marker(meta):
            continue
        marker = closing_marker(meta)
        counts[marker if marker else "none"] += 1
    return counts


def false_marker_candidates(meta: Optional[str]) -> List[str]:
    """Line-initial `Word:` tokens in a comment that are NOT annotator markers.

    Only used by the audit and by the test that pins the noise down. These are
    the strings a generic marker pattern would have mistaken for a boundary.
    """
    if not meta:
        return []
    generic = re.finditer(r"(?im)^[ \t]*([A-Za-z][A-Za-z ]{0,20})[ \t]*:", meta)
    return [
        m.group(1).strip()
        for m in generic
        if m.group(1).strip().lower() not in MARKERS
    ]
