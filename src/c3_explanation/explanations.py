"""Layer 5.1 of C3: the explanation a person actually reads.

Every prior "explainable hallucination detector" shows a few example
explanations and asks you to be impressed. C3 can do better than that because
RAGTruth ships a human-written rationale on every annotated span, in the `meta`
field, so the explanation can be scored instead of demoed. This module generates;
src/c3_explanation/explanation_eval.py scores.

**Why templates.** A generated explanation that is fluent and wrong is worse than
a blunt one that is right, and the project has no paid API and no GPU budget left
for a seq2seq model. A template built from the predicted type, the span text and
the best-matching context sentence is deterministic, instant, runs on CPU, and is
honest about being a template. It populates Span.explanation, which
frontend/src/components/HighlightedAnswer.tsx already renders behind a truthiness
guard labelled "Why (C3):", so it appears in the interface with no frontend
change at all.

**Why it speaks in sets.** Layer 4 does not return one type; at a chosen alpha it
returns a set of types with a coverage guarantee. When that set has one member the
explanation commits. When it has two members sharing a mechanism, the system knows
whether the span contradicts the source or is absent from it, and does not know
how severe it is -- so the sentence says exactly that, rather than picking the
more likely severity and sounding certain. That is the whole point of putting a
conformal layer under an explanation: the wording degrades with the evidence.

**What it deliberately does not do.** It does not write to Span.evidence_sentence
or Span.evidence_index. Those are C4's fields under the contract in
src/common/schema.py, and C4 is deferred; the implicated sentence is quoted inside
the explanation text instead. It also never states a coverage number -- the caller
decides whether the guarantee applies, and this module has no opinion it could
overstate.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from src.c3_explanation.grounding import best_matching_sentence
from src.c3_explanation.labels_audit import (
    MECHANISM_BASELESS,
    MECHANISM_CONFLICT,
    SEVERITY_SUBTLE,
    axes_of_error_type,
)
from src.common.schema import ErrorType

# A quoted sentence longer than this is cut, because the explanation is read in a
# tooltip next to a highlighted span and a 400-character quote is not read at all.
QUOTE_CHARS = 220
SPAN_CHARS = 120


def _clip(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _quote(text: str, limit: int) -> str:
    return f"“{_clip(text, limit)}”"


def _axes_of_set(type_set: Sequence[str]) -> Tuple[Optional[str], Optional[str]]:
    """The mechanism and severity a set of types agrees on, or None where it does not."""
    mechanisms = set()
    severities = set()
    for name in type_set:
        try:
            mechanism, severity = axes_of_error_type(ErrorType(name))
        except ValueError:
            continue
        mechanisms.add(mechanism)
        severities.add(severity)
    return (
        next(iter(mechanisms)) if len(mechanisms) == 1 else None,
        next(iter(severities)) if len(severities) == 1 else None,
    )


def explain(
    span_text: str,
    context_sentences: Sequence[str],
    error_type: Optional[str] = None,
    type_set: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """One explanation for one span.

    Pass `type_set` when Layer 4 produced one; it takes precedence over
    `error_type`, because a set is a statement about what is known and a single
    type is a guess dressed as one. Pass `error_type` alone when no conformal
    layer is attached.

    Returns the sentence, the index of the context sentence it quoted (or None),
    and what it committed to, so the caller can log or score it without parsing
    English back out.
    """
    members = list(type_set) if type_set else ([error_type] if error_type else [])
    members = [m for m in members if m]
    mechanism, severity = _axes_of_set(members)
    index, score = best_matching_sentence(span_text, context_sentences)
    sentence = context_sentences[index] if index is not None else None

    hedge = "" if severity != SEVERITY_SUBTLE else " The difference is a small one."
    span_quote = _quote(span_text, SPAN_CHARS)

    if not members:
        text = (
            f"{span_quote} was flagged, but no error type was assigned to it."
        )
        committed = "none"
    elif mechanism == MECHANISM_BASELESS:
        text = f"No sentence in the source supports {span_quote}."
        if sentence is not None and score > 0:
            text += (
                f" The closest the source comes is {_quote(sentence, QUOTE_CHARS)}."
            )
        text += hedge
        committed = "baseless"
    elif mechanism == MECHANISM_CONFLICT:
        if sentence is None:
            text = (
                f"{span_quote} contradicts the source, but the source could not "
                "be split into sentences to quote."
            )
        else:
            text = (
                f"The source says {_quote(sentence, QUOTE_CHARS)}, "
                f"which {span_quote} contradicts."
            )
        text += hedge
        committed = "conflict"
    else:
        # The set spans both mechanisms: the system knows the span is unsupported
        # and does not know in which of the two ways. Saying so is the honest
        # sentence; picking the likelier one would be the dishonest one.
        text = (
            f"The source does not back up {span_quote}. It is not decidable here "
            "whether the source contradicts it or simply does not mention it"
        )
        if sentence is not None and score > 0:
            text += f"; the nearest sentence is {_quote(sentence, QUOTE_CHARS)}"
        text += "."
        committed = "unsupported"

    if len(members) > 1 and severity is None:
        text += " How severe it is was not decidable at this confidence level."

    return {
        "explanation": text,
        "quoted_sentence": sentence,
        "quoted_sentence_index": index,
        "match_score": float(score),
        "committed_to": committed,
        "type_set": members,
        "mechanism": mechanism,
        "severity": severity,
        "template": True,
    }


def explain_many(
    spans: Sequence[Dict[str, object]], context_sentences: Sequence[str]
) -> List[Dict[str, object]]:
    """Explain a list of {text, error_type?, type_set?} dicts, in order."""
    return [
        explain(
            str(span.get("text", "")),
            context_sentences,
            error_type=span.get("error_type"),  # type: ignore[arg-type]
            type_set=span.get("type_set"),  # type: ignore[arg-type]
        )
        for span in spans
    ]
