"""Layer 3's scoring harness -- one place that decides what an attribution number means.

Every backend in C4 (lexical, embedding, entailment, HHEM, and the four floors)
is scored through this module, so a comparison between them is a comparison of
the systems and not of two people's arithmetic.

Three rules are enforced here rather than remembered:

* **Every interval is a response-level cluster bootstrap.** Spans inside one
  response share a context, a generator and an annotator, so they are not
  independent. C2 measured what forgetting that costs: at token level its
  intervals were too narrow by 4.65x to 6.53x.
* **Only spans with a resolvable gold sentence are scored**, and the exclusions
  are reported in the same object as the scores rather than in a footnote.
* **A floor row is not optional.** `floor_rows` builds all four, and the report
  writers below refuse to emit a table without them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.c2_calibration.clustered import percentile_interval
from src.c4_attribution.dataset import SpanRecord
from src.c4_attribution.lexical import FLOOR_NAMES, containment, floor_prediction
from src.c4_attribution.matching import Match
from src.c4_attribution.evidence_labels import normalise

RESAMPLES = 2000
SEED = 42
RECALL_AT: Tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class Prediction:
    """One system's answer for one span.

    `ranking` is sentence indices best-first. `top1` is None when the system
    abstained, which is a different object from an empty ranking and is scored
    as such in Layer 7.
    """

    ranking: Tuple[int, ...]
    top1: Optional[int]
    score: float = 0.0


def rank_from_scores(scores: Sequence[float]) -> Tuple[int, ...]:
    """Sentence indices best-first, ties to the lowest index."""
    return tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], i)))


def prediction_from_scores(scores: Sequence[float]) -> Prediction:
    ranking = rank_from_scores(scores)
    if not ranking:
        return Prediction((), None, 0.0)
    return Prediction(ranking, ranking[0], float(scores[ranking[0]]))


# --------------------------------------------------------------------------
# The response-level cluster bootstrap
# --------------------------------------------------------------------------


def response_index(response_ids: Sequence[str]) -> List[np.ndarray]:
    """Row positions grouped by response, in first-seen order."""
    groups: Dict[str, List[int]] = {}
    for position, rid in enumerate(response_ids):
        groups.setdefault(str(rid), []).append(position)
    return [np.asarray(v, dtype=np.int64) for v in groups.values()]


def bootstrap_by_response(
    response_ids: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> Dict[str, float]:
    """Percentile interval for a statistic, resampling whole responses.

    Identical in construction to `src/c3_explanation/type_eval.bootstrap_by_response`;
    `tests/test_c4_attribution_eval.py` asserts the two agree on the same input
    when C3 is importable, so this copy exists to keep C4 runnable on its own
    rather than to be a second opinion.
    """
    index = response_index(response_ids)
    if not index:
        return percentile_interval([])
    rng = np.random.default_rng(seed)
    n = len(index)
    draws: List[float] = []
    for _ in range(resamples):
        picks = rng.integers(0, n, size=n)
        rows = np.concatenate([index[i] for i in picks])
        draws.append(float(statistic(rows)))
    return percentile_interval([v for v in draws if not math.isnan(v)])


# --------------------------------------------------------------------------
# The per-span measurements
# --------------------------------------------------------------------------


def score_row(
    row: SpanRecord, gold: Match, prediction: Prediction, convention: str = "A"
) -> Dict[str, Any]:
    """Everything the metrics need for one scored span.

    `convention` picks which of Layer 2's two gold rules is being scored against.
    B matters beyond completeness: Convention A resolves gold by taking the
    argmax of content-token containment, and a lexical system also takes an
    argmax of token overlap, so A structurally flatters a lexical baseline.
    B's gold comes from exact string containment and carries no such coupling.
    Reporting A alone would be scoring a rule against itself.
    """
    target = gold.convention_a if convention == "A" else gold.convention_b
    ranking = prediction.ranking
    position = ranking.index(target) if target is not None and target in ranking else -1

    top1 = prediction.top1
    sentences = row.sentences
    returned = sentences[top1] if top1 is not None and 0 <= top1 < len(sentences) else ""
    snippet = row.label.snippet or ""

    return {
        "response_id": row.response_id,
        "task": row.task,
        "label_type": row.label.label_type,
        "implicit_true": row.label.implicit_true,
        "gold_index": target,
        "gold_indices": gold.gold_indices,
        "match_kind": gold.kind,
        "n_sentences": gold.n_sentences,
        "abstained": top1 is None,
        "top1": top1,
        "correct": bool(target is not None and top1 == target),
        "rank": position,
        "reciprocal_rank": 1.0 / (position + 1) if position >= 0 else 0.0,
        "in_gold_set": bool(top1 is not None and top1 in gold.gold_indices),
        "returned_containment": containment(snippet, returned) if snippet else 0.0,
        "returned_exact": bool(
            snippet and normalise(snippet) in normalise(returned)
        ),
        # Not "self-faithfulness" -- that is Layer 9's measurement, which checks
        # the quoted STRING against the context. This only says the index was in
        # range, which is the cheap precondition for it.
        "index_in_range": bool(top1 is None or (0 <= top1 < len(sentences))),
        "score": prediction.score,
    }


def score_rows(
    rows: Sequence[SpanRecord],
    matches: Sequence[Match],
    predictions: Sequence[Prediction],
    convention: str = "A",
) -> List[Dict[str, Any]]:
    if convention not in ("A", "B"):
        raise ValueError(f"convention must be 'A' or 'B', got {convention!r}")
    if not (len(rows) == len(matches) == len(predictions)):
        raise ValueError("rows, matches and predictions must be the same length")
    keep = (
        (lambda m: m.scorable)
        if convention == "A"
        else (lambda m: m.convention_b is not None)
    )
    return [
        score_row(row, gold, prediction, convention)
        for row, gold, prediction in zip(rows, matches, predictions)
        if keep(gold)
    ]


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def aggregate(scored: Sequence[Dict[str, Any]], with_intervals: bool = True) -> Dict[str, Any]:
    """Headline metrics for one system on one slice, with cluster intervals."""
    n = len(scored)
    if n == 0:
        return {"n": 0}

    correct = np.asarray([r["correct"] for r in scored], dtype=np.float64)
    rr = np.asarray([r["reciprocal_rank"] for r in scored], dtype=np.float64)
    rank = np.asarray([r["rank"] for r in scored], dtype=np.int64)
    contained = np.asarray([r["returned_containment"] for r in scored], dtype=np.float64)
    exact = np.asarray([r["returned_exact"] for r in scored], dtype=np.float64)
    abstained = np.asarray([r["abstained"] for r in scored], dtype=np.float64)
    in_set = np.asarray([r["in_gold_set"] for r in scored], dtype=np.float64)
    response_ids = [r["response_id"] for r in scored]

    out: Dict[str, Any] = {
        "n": n,
        "top1_accuracy": float(correct.mean()),
        "mrr": float(rr.mean()),
        "in_gold_set_rate": float(in_set.mean()),
        "snippet_containment": float(contained.mean()),
        "exact_snippet_rate": float(exact.mean()),
        "abstention_rate": float(abstained.mean()),
        "index_in_range_rate": float(
            np.mean([r["index_in_range"] for r in scored])
        ),
    }
    for k in RECALL_AT:
        out[f"recall_at_{k}"] = float(np.mean((rank >= 0) & (rank < k)))

    if with_intervals:
        out["top1_accuracy_ci"] = bootstrap_by_response(
            response_ids, lambda idx: float(correct[idx].mean())
        )
        out["mrr_ci"] = bootstrap_by_response(
            response_ids, lambda idx: float(rr[idx].mean())
        )
        out["snippet_containment_ci"] = bootstrap_by_response(
            response_ids, lambda idx: float(contained[idx].mean())
        )
    return out


def by_group(
    scored: Sequence[Dict[str, Any]], key: str, with_intervals: bool = False
) -> Dict[str, Any]:
    """The same metrics per task or per error type. Never pooled away."""
    out: Dict[str, Any] = {}
    for value in sorted({str(r[key]) for r in scored}):
        subset = [r for r in scored if str(r[key]) == value]
        out[value] = aggregate(subset, with_intervals=with_intervals)
    return out


# --------------------------------------------------------------------------
# The floors
# --------------------------------------------------------------------------


def floor_predictions(
    rows: Sequence[SpanRecord], name: str, seed: int = SEED
) -> List[Prediction]:
    """One floor policy's predictions over a slice.

    The random floor is seeded from the row position so a rerun reproduces, and
    the abstain floor returns None for every row, which is exactly the policy
    that beats a guesser on QA.
    """
    out: List[Prediction] = []
    for position, row in enumerate(rows):
        sentences = row.sentences
        index = floor_prediction(name, sentences, seed + position)
        if index is None:
            out.append(Prediction((), None, 0.0))
        else:
            ranking = (index,) + tuple(i for i in range(len(sentences)) if i != index)
            out.append(Prediction(ranking, index, 0.0))
    return out


def floor_rows(
    rows: Sequence[SpanRecord],
    matches: Sequence[Match],
    with_intervals: bool = False,
    convention: str = "A",
) -> Dict[str, Any]:
    """All four floors, scored through the same harness as the systems.

    A table without these is not a result. Three of the four ignore the input
    entirely, and on the QA task the fourth -- always abstain -- is the
    majority-correct policy, which is the honest thing this section has to say.
    """
    out: Dict[str, Any] = {}
    for name in FLOOR_NAMES:
        predictions = floor_predictions(rows, name)
        out[name] = aggregate(
            score_rows(rows, matches, predictions, convention),
            with_intervals=with_intervals,
        )
        out[name]["ignores_input"] = name in ("first", "longest", "random", "abstain")
    return out


def exclusions(matches: Sequence[Match]) -> Dict[str, Any]:
    """What was left out, in the same object as what was scored."""
    total = len(matches)
    scorable = sum(1 for m in matches if m.scorable)
    return {
        "n_annotated_spans": total,
        "n_scored": scorable,
        "share_scored": scorable / total if total else 0.0,
        "excluded_no_evidence_label": sum(
            1 for m in matches if m.kind == "no_evidence_label"
        ),
        "excluded_no_evidence": sum(1 for m in matches if m.kind == "no_evidence"),
        "excluded_unmatched": sum(1 for m in matches if m.kind == "unmatched"),
    }


# --------------------------------------------------------------------------
# Layer 3's artefact
# --------------------------------------------------------------------------


def evaluate_system(
    rows: Sequence[SpanRecord],
    matches: Sequence[Match],
    predictions: Sequence[Prediction],
    convention: str = "A",
    with_intervals: bool = True,
) -> Dict[str, Any]:
    """One system, one convention: overall plus the two breakdowns that must never pool."""
    scored = score_rows(rows, matches, predictions, convention)
    return {
        "overall": aggregate(scored, with_intervals=with_intervals),
        "by_task": by_group(scored, "task", with_intervals=with_intervals),
        "by_error_type": by_group(scored, "label_type"),
    }


def floors_by_task(
    rows: Sequence[SpanRecord], matches: Sequence[Match], convention: str = "A"
) -> Dict[str, Any]:
    """Every floor, per task. This is the table that decides how C4 is read.

    Pooling the three tasks hides the finding, because data-to-text supplies
    most of the scorable spans and its context is a serialised JSON object
    rather than prose. Sentence boundaries there are an artefact of json.dumps
    and a regex, so a length-based policy can look like an attribution system.
    """
    out: Dict[str, Any] = {}
    for name in FLOOR_NAMES:
        predictions = floor_predictions(rows, name)
        scored = score_rows(rows, matches, predictions, convention)
        out[name] = by_group(scored, "task")
    return out


def build_report(
    rows: Sequence[SpanRecord],
    matches: Sequence[Match],
    predictions: Sequence[Prediction],
    system_name: str,
) -> Dict[str, Any]:
    return {
        "system": system_name,
        "n_annotated_spans": len(rows),
        "exclusions": exclusions(matches),
        "convention_a": evaluate_system(rows, matches, predictions, "A"),
        "convention_b": evaluate_system(rows, matches, predictions, "B"),
        "floors": {
            "convention_a": floor_rows(rows, matches, with_intervals=True, convention="A"),
            "convention_b": floor_rows(rows, matches, with_intervals=False, convention="B"),
            "by_task_convention_a": floors_by_task(rows, matches, "A"),
        },
        "reading_notes": [
            "Convention A resolves gold by argmax content-token containment. A "
            "lexical system also takes an argmax of token overlap, so A flatters "
            "it. Convention B's gold is exact string containment and carries no "
            "such coupling. Both are reported.",
            "Do not read the pooled row. Data-to-text supplies most scorable "
            "spans and its context is a serialised JSON object, so its sentence "
            "boundaries come from json.dumps and a regex rather than from prose.",
            "Every interval is a response-level cluster bootstrap, 2000 "
            "resamples, seed 42.",
        ],
    }
