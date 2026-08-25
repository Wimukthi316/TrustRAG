"""Layer 7 -- "no evidence exists", as a supervised and measured claim.

A fifth of the annotator comments carrying an `Original:` line are not quotes.
The annotator wrote that the information is not in the source at all. That is a
gold label for *silence*, and it is the one thing a nearest-neighbour baseline
structurally cannot produce: `best_matching_sentence` always returns a sentence,
so its recall on this class is zero by construction, not by bad luck. Saying
that in a sentence is the cleanest single argument that C4 is more than an NLI
call.

Everything here is scored against three floors -- always abstain, never abstain,
and the per-task base rate -- because on the QA task the majority-correct policy
really is to say nothing, and a system that cannot beat silence should be told
so in the same table.

The threshold is fitted on the frozen calibration split and applied to test
once. `tests/test_c4_no_evidence.py` asserts that discipline rather than
trusting it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.c4_attribution.attribution_eval import bootstrap_by_response
from src.c4_attribution.dataset import SpanRecord
from src.c4_attribution.matching import KIND_NO_EVIDENCE, Match

OUT = Path("results/c4/analysis/no_evidence.json")

# The grid the abstention threshold is chosen from. Fixed, so "the threshold sat
# on the edge of the grid" is a statement that can be checked.
THRESHOLD_GRID = tuple(float(t) for t in np.linspace(0.0, 1.0, 101))


@dataclass(frozen=True)
class AbstentionScores:
    """Per-span confidence that the top-1 sentence really is the evidence.

    Higher means "there is evidence and this is it". The abstention rule is
    `confidence < threshold -> say nothing`.
    """

    confidence: Tuple[float, ...]


def gold_abstain(matches: Sequence[Match]) -> np.ndarray:
    """1 where the annotator said the information is not in the source."""
    return np.asarray(
        [1 if m.kind == KIND_NO_EVIDENCE else 0 for m in matches], dtype=np.int64
    )


def in_scope(matches: Sequence[Match]) -> np.ndarray:
    """Spans carrying an evidence label at all: attributable or explicitly none.

    Spans with no `Original:` line are excluded -- the annotator said nothing,
    which is not the same claim as "there is nothing", and pooling the two would
    inflate the positive class.
    """
    return np.asarray(
        [
            1 if (m.kind == KIND_NO_EVIDENCE or m.scorable or m.kind == "unmatched") else 0
            for m in matches
        ],
        dtype=np.int64,
    )


def confusion(gold: np.ndarray, predicted: np.ndarray) -> Dict[str, Any]:
    """Precision, recall and F1 of the *abstain* decision, with its four cells."""
    tp = int(np.sum((gold == 1) & (predicted == 1)))
    fp = int(np.sum((gold == 0) & (predicted == 1)))
    fn = int(np.sum((gold == 1) & (predicted == 0)))
    tn = int(np.sum((gold == 0) & (predicted == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    n = tp + fp + fn + tn
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / n if n else 0.0,
        "abstention_rate": (tp + fp) / n if n else 0.0,
        "base_rate": (tp + fn) / n if n else 0.0,
    }


def choose_threshold(
    confidence: Sequence[float],
    gold: np.ndarray,
    grid: Sequence[float] = THRESHOLD_GRID,
) -> Dict[str, Any]:
    """The grid point maximising abstention F1 on the calibration split.

    Ties go to the *smallest* threshold, which is the least abstaining rule
    among the equally good ones. Reported with a flag when the winner sits on
    the edge of the grid, because then it was decided by where the grid stopped
    rather than by the data.
    """
    scores = np.asarray(confidence, dtype=np.float64)
    best = {"threshold": None, "f1": -1.0}
    rows: List[Dict[str, Any]] = []
    for index, threshold in enumerate(grid):
        predicted = (scores < threshold).astype(np.int64)
        block = confusion(gold, predicted)
        block["threshold"] = float(threshold)
        rows.append(block)
        if block["f1"] > best["f1"]:
            best = dict(block)
            best["grid_index"] = index
    best["on_grid_edge"] = best.get("grid_index") in (0, len(grid) - 1)
    best["grid_size"] = len(grid)
    best["curve"] = rows
    return best


def evaluate(
    rows: Sequence[SpanRecord],
    matches: Sequence[Match],
    confidence: Sequence[float],
    threshold: float,
) -> Dict[str, Any]:
    """Abstention scored on one split at a threshold fitted somewhere else."""
    keep = in_scope(matches).astype(bool)
    gold = gold_abstain(matches)[keep]
    scores = np.asarray(confidence, dtype=np.float64)[keep]
    kept_rows = [r for r, k in zip(rows, keep) if k]
    predicted = (scores < threshold).astype(np.int64)

    block = confusion(gold, predicted)
    block["threshold"] = float(threshold)
    response_ids = [r.response_id for r in kept_rows]

    def f1_of(index: np.ndarray) -> float:
        return confusion(gold[index], predicted[index])["f1"]

    def recall_of(index: np.ndarray) -> float:
        return confusion(gold[index], predicted[index])["recall"]

    block["f1_ci"] = bootstrap_by_response(response_ids, f1_of)
    block["recall_ci"] = bootstrap_by_response(response_ids, recall_of)

    block["by_task"] = {}
    for task in sorted({r.task for r in kept_rows}):
        index = np.asarray([i for i, r in enumerate(kept_rows) if r.task == task])
        if index.size:
            block["by_task"][task] = confusion(gold[index], predicted[index])

    block["by_error_type"] = {}
    for label_type in sorted({r.label.label_type for r in kept_rows}):
        index = np.asarray(
            [i for i, r in enumerate(kept_rows) if r.label.label_type == label_type]
        )
        if index.size:
            block["by_error_type"][label_type] = confusion(gold[index], predicted[index])

    implicit = np.asarray([r.label.implicit_true for r in kept_rows])
    block["implicit_true"] = {
        "n": int(implicit.sum()),
        "recall_on_implicit_true": (
            confusion(gold[implicit], predicted[implicit])["recall"]
            if implicit.any()
            else float("nan")
        ),
        "recall_on_rest": (
            confusion(gold[~implicit], predicted[~implicit])["recall"]
            if (~implicit).any()
            else float("nan")
        ),
        "note": (
            "implicit_true marks a span that is factually correct but absent "
            "from the context. There is genuinely no sentence to point at, so "
            "abstention recall should be highest exactly here. That is the "
            "mechanism, not a coincidence, and it is checked rather than "
            "asserted."
        ),
    }
    return block


def floors(rows: Sequence[SpanRecord], matches: Sequence[Match]) -> Dict[str, Any]:
    """Always abstain, never abstain, and the per-task base rate."""
    keep = in_scope(matches).astype(bool)
    gold = gold_abstain(matches)[keep]
    kept_rows = [r for r, k in zip(rows, keep) if k]
    ones = np.ones_like(gold)
    zeros = np.zeros_like(gold)

    # The floors have to be per task as well as pooled. The always-abstain floor
    # is a function of the base rate, and the base rate differs by an order of
    # magnitude between the tasks -- so a system that clears the pooled floor
    # comfortably can still lose to doing nothing on one of them. Reporting only
    # the pooled row would hide exactly that.
    per_task: Dict[str, Any] = {}
    for task in sorted({r.task for r in kept_rows}):
        index = np.asarray([i for i, r in enumerate(kept_rows) if r.task == task])
        if not index.size:
            continue
        task_gold = gold[index]
        per_task[task] = {
            "n": int(index.size),
            "base_rate": float(task_gold.mean()),
            "always_abstain": confusion(task_gold, np.ones_like(task_gold)),
            "never_abstain": confusion(task_gold, np.zeros_like(task_gold)),
        }

    return {
        "always_abstain": confusion(gold, ones),
        "never_abstain": confusion(gold, zeros),
        "base_rate": float(gold.mean()) if gold.size else 0.0,
        "base_rate_by_task": {k: v["base_rate"] for k, v in per_task.items()},
        "by_task": per_task,
        "lexical_without_a_threshold": {
            "recall": 0.0,
            "note": (
                "grounding.best_matching_sentence always returns a sentence, so "
                "a pure nearest-neighbour attributor has zero recall on this "
                "class by construction. It is not a weak result; it is a "
                "capability the baseline does not have."
            ),
        },
    }
