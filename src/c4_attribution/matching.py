"""Layer 2 -- when does a returned sentence count as the annotator's evidence?

The annotator quoted a free-text *snippet*. The system returns a *sentence
index*. There is no published rule for when those two match, and the choice of
rule moves every number downstream of it. So C4 writes one, publishes two, and
measures the disagreement instead of quietly picking the flattering one.

    Convention A   the sentence holding the largest share of the snippet's
                   content tokens is gold, provided it clears theta.
    Convention B   gold only when the snippet appears verbatim inside exactly
                   one sentence. Everything else is excluded, not guessed.

A is the usable convention -- Layer 1 measured that only about half the quotes
are verbatim, so B throws away half the supervision. B is the conservative
control, and reporting both is what stops the operating point looking like a
result.

**theta is fixed at 0.9 before any model number exists.** That is stated here,
in the artefact, and in the report, because a matching threshold chosen after
seeing a score is not a threshold, it is a knob.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.c4_attribution.dataset import SpanRecord, load_responses, span_records
from src.c4_attribution.evidence_labels import normalise
from src.c4_attribution.lexical import containment_scores, content_tokens

OUT_DIR = Path("results/c4/analysis")

# Fixed before any model was run. Do not tune.
THETA = 0.9
THETA_SWEEP: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# What a span's evidence label resolved to.
KIND_VERBATIM_UNIQUE = "verbatim_unique"
KIND_VERBATIM_MULTI = "verbatim_multiple"
KIND_CONTAINMENT = "containment"
KIND_SPANS_SENTENCES = "spans_multiple_sentences"
KIND_UNMATCHED = "unmatched"
KIND_NO_EVIDENCE = "no_evidence"
KIND_NO_SUPERVISION = "no_evidence_label"

KINDS: Tuple[str, ...] = (
    KIND_VERBATIM_UNIQUE,
    KIND_VERBATIM_MULTI,
    KIND_CONTAINMENT,
    KIND_SPANS_SENTENCES,
    KIND_UNMATCHED,
    KIND_NO_EVIDENCE,
    KIND_NO_SUPERVISION,
)


@dataclass(frozen=True)
class Match:
    """The gold evidence for one span under both conventions."""

    kind: str
    gold_index: Optional[int]
    gold_indices: Tuple[int, ...]
    best_containment: float
    n_verbatim_hits: int
    n_sentences: int

    @property
    def convention_a(self) -> Optional[int]:
        """Argmax containment, when it clears theta. None means excluded."""
        if self.kind in (KIND_UNMATCHED, KIND_NO_EVIDENCE, KIND_NO_SUPERVISION):
            return None
        return self.gold_index

    @property
    def convention_b(self) -> Optional[int]:
        """Verbatim inside exactly one sentence, or nothing."""
        return self.gold_index if self.kind == KIND_VERBATIM_UNIQUE else None

    @property
    def scorable(self) -> bool:
        """Does this span have a single gold sentence to score a top-1 against?"""
        return self.convention_a is not None


def _cover(snippet: str, sentences: Sequence[str], scores: Sequence[float]) -> Tuple[int, ...]:
    """A small set of sentences that together hold the snippet's content tokens.

    Greedy set cover, best-first, stopping when nothing new is added. Used only
    for the spans whose quote genuinely straddles a sentence boundary; for every
    other span it degenerates to the single argmax sentence.
    """
    wanted = set(content_tokens(snippet))
    if not wanted:
        return ()
    remaining = set(wanted)
    chosen: List[int] = []
    order = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))
    for index in order:
        gain = remaining & set(content_tokens(sentences[index]))
        if not gain:
            continue
        chosen.append(index)
        remaining -= gain
        if not remaining:
            break
    return tuple(sorted(chosen))


def match(row: SpanRecord, theta: float = THETA) -> Match:
    """Resolve one annotated span to its gold evidence sentence(s)."""
    sentences = row.sentences
    n = len(sentences)
    snippet = row.label.snippet or ""

    if not row.label.has_original or not snippet:
        # No Original: line at all. The annotator left no evidence label, so
        # there is nothing to score this span against either way. Excluded and
        # counted, never folded into the no-evidence class -- "the annotator
        # said there is no evidence" and "the annotator said nothing" are
        # different claims and pooling them would inflate Layer 7.
        return Match(KIND_NO_SUPERVISION, None, (), 0.0, 0, n)

    if row.label.no_evidence:
        return Match(KIND_NO_EVIDENCE, None, (), 0.0, 0, n)

    normalised = normalise(snippet)
    hits = [i for i, s in enumerate(sentences) if normalised and normalised in normalise(s)]
    scores = containment_scores(snippet, sentences)
    best_index = min(range(n), key=lambda i: (-scores[i], i)) if n else None
    best_value = float(scores[best_index]) if best_index is not None else 0.0

    if len(hits) == 1:
        return Match(KIND_VERBATIM_UNIQUE, hits[0], (hits[0],), best_value, 1, n)
    if len(hits) > 1:
        # Ties to the lowest index, matching lexical.best_sentence.
        return Match(KIND_VERBATIM_MULTI, hits[0], tuple(hits), best_value, len(hits), n)

    joined = normalise(" ".join(sentences))
    if normalised and normalised in joined:
        cover = _cover(snippet, sentences, scores)
        return Match(KIND_SPANS_SENTENCES, best_index, cover or (best_index,), best_value, 0, n)

    if best_index is not None and best_value >= theta:
        return Match(KIND_CONTAINMENT, best_index, (best_index,), best_value, 0, n)

    return Match(KIND_UNMATCHED, None, (), best_value, 0, n)


def match_all(rows: Sequence[SpanRecord], theta: float = THETA) -> List[Match]:
    return [match(row, theta) for row in rows]


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def kind_counts(matches: Sequence[Match]) -> Dict[str, Any]:
    total = len(matches)
    counts = {kind: sum(1 for m in matches if m.kind == kind) for kind in KINDS}
    return {
        "n": total,
        "counts": counts,
        "shares": {k: (v / total if total else 0.0) for k, v in counts.items()},
    }


def theta_sweep(
    rows: Sequence[SpanRecord], thetas: Sequence[float] = THETA_SWEEP
) -> List[Dict[str, Any]]:
    """Matched rate against theta. The figure that shows how much the rule decides."""
    out: List[Dict[str, Any]] = []
    for theta in thetas:
        matches = match_all(rows, theta)
        scorable = sum(1 for m in matches if m.scorable)
        b_scorable = sum(1 for m in matches if m.convention_b is not None)
        out.append(
            {
                "theta": theta,
                "n": len(matches),
                "convention_a_matched": scorable,
                "convention_a_rate": scorable / len(matches) if matches else 0.0,
                "convention_b_matched": b_scorable,
                "convention_b_rate": b_scorable / len(matches) if matches else 0.0,
                "unmatched": sum(1 for m in matches if m.kind == KIND_UNMATCHED),
            }
        )
    return out


def agreement(matches: Sequence[Match]) -> Dict[str, Any]:
    """Where A and B both give an answer, how often is it the same sentence?"""
    both = [(m.convention_a, m.convention_b) for m in matches if m.convention_b is not None]
    same = sum(1 for a, b in both if a == b)
    return {
        "n_both_defined": len(both),
        "same_sentence": same,
        "agreement": same / len(both) if both else 0.0,
        "n_a_only": sum(
            1 for m in matches if m.convention_a is not None and m.convention_b is None
        ),
    }


def build_report(rows: Sequence[SpanRecord], theta: float = THETA) -> Dict[str, Any]:
    matches = match_all(rows, theta)
    by_task: Dict[str, Any] = {}
    for task in sorted({r.task for r in rows}):
        subset = [m for r, m in zip(rows, matches) if r.task == task]
        by_task[task] = kind_counts(subset)
    return {
        "theta": theta,
        "theta_fixed_before_any_model_result": True,
        "conventions": {
            "A": "argmax content-token containment, gold when it clears theta",
            "B": "verbatim inside exactly one sentence, otherwise excluded",
        },
        "overall": kind_counts(matches),
        "by_task": by_task,
        "agreement": agreement(matches),
        "theta_sweep": theta_sweep(rows),
        "note": (
            "Spans in the spans_multiple_sentences bucket have a gold sentence "
            "SET, not a gold sentence. They are kept in Convention A through "
            "their argmax sentence and are the reason Layer 6 returns a set at "
            "all. The rate is reported here so the choice is visible."
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    responses = load_responses()
    rows = span_records(responses)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("calib", "test"):
        subset = [r for r in rows if r.response.split == split]
        report = build_report(subset)
        path = OUT_DIR / f"attribution_matching_{split}.json"
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        overall = report["overall"]
        print(f"[{split}] n={overall['n']}")
        for kind in KINDS:
            print(f"  {kind:26s} {overall['counts'][kind]:5d}  {overall['shares'][kind]:.4f}")
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
