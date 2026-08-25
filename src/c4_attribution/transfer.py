"""Layer 8 -- cross-task transfer. The out-of-domain result C3 could not produce.

C3 reports `n/a` for out-of-domain because RAGBench carries no error-type
labels. C4 is not stuck the same way: evidence labels exist across all three
RAGTruth tasks, so the scorer, the abstention threshold and the conformal
quantile can be calibrated on two tasks and tested on the third. That is a real
generalisation result, and C4 is the only component in this project that can
report one.

**RAGBench is settled first, not assumed.** `check_ragbench` scans every field
of every record in `data/processed/ragbench/`. If no field carries an evidence
link -- which is the expected answer -- the row is `n/a` **with the reason**, the
same pattern as C3's un-evaluable type head and C1's HHEM row.

**Written down before measuring.** The QA fold should break worst. On these
splits QA's share of "no evidence exists" comments is far above data-to-text's,
so calibrating on data-to-text and summarization and testing on QA is a label
shift of exactly the kind C2 diagnosed on RAGBench -- except inside a single
corpus, at a much shorter distance. If the guarantee breaks here too, that is a
stronger warning than C2's finding alone. If it survives, the boundary has a
name: robust to task, not to corpus.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


from src.c4_attribution.attribution_eval import Prediction, aggregate, score_rows
from src.c4_attribution.dataset import SpanRecord
from src.c4_attribution.evidence_conformal import ConformalRow, evaluate as conformal_evaluate
from src.c4_attribution.matching import Match
from src.c4_attribution.no_evidence import (
    choose_threshold as choose_abstention,
    evaluate as evaluate_abstention,
    gold_abstain,
    in_scope,
)

RAGBENCH_DIR = Path("data/processed/ragbench")
TASKS: Tuple[str, ...] = ("qa", "data2text", "summarization")

EVIDENCE_FIELD_HINTS: Tuple[str, ...] = (
    "evidence",
    "original",
    "support",
    "cite",
    "citation",
    "attribution",
    "rationale",
    "key_sentence",
    "relevant",
    "provenance",
)


def check_ragbench(path: Path | str = RAGBENCH_DIR) -> Dict[str, Any]:
    """Does RAGBench carry anything usable as an evidence link? Answer, with the scan.

    Reported whichever way it comes out. An `n/a` with a reason is a result; an
    `n/a` with a shrug is not.
    """
    directory = Path(path)
    if not directory.exists():
        return {
            "checked": False,
            "available": False,
            "reason": (
                f"{directory} is not present in this working copy, so the scan "
                "could not run here. Run it where the processed RAGBench lives "
                "before quoting this row."
            ),
        }

    files = sorted(directory.glob("*.jsonl"))
    fields: Dict[str, int] = {}
    n_records = 0
    for file in files:
        with file.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                n_records += 1
                for key in json.loads(line):
                    fields[key] = fields.get(key, 0) + 1

    hits = sorted(
        key
        for key in fields
        if any(hint in key.lower() for hint in EVIDENCE_FIELD_HINTS)
    )
    return {
        "checked": True,
        "n_subsets": len(files),
        "n_records": n_records,
        "fields": sorted(fields),
        "evidence_like_fields": hits,
        "available": bool(hits),
        "reason": (
            "RAGBench as processed here carries spans and an adherence score "
            "and no field linking an answer to the source sentence it was "
            "judged against. There is nothing to score an evidence attribution "
            "against, so the out-of-domain evidence row is n/a. The transfer "
            "study below is the generalisation result C4 can actually run."
            if not hits
            else "Evidence-like fields found; the row can be attempted."
        ),
    }


def _conformal_rows(
    rows: Sequence[SpanRecord],
    matches: Sequence[Match],
    scores: Sequence[Sequence[float]],
) -> List[ConformalRow]:
    from src.c4_attribution.evidence_conformal import normalise_scores

    return [
        ConformalRow(
            response_id=row.response_id,
            task=row.task,
            probabilities=normalise_scores(score),
            gold_index=gold.convention_a,
        )
        for row, gold, score in zip(rows, matches, scores)
    ]


def fold(
    calib_rows: Sequence[SpanRecord],
    calib_matches: Sequence[Match],
    calib_scores: Sequence[Sequence[float]],
    calib_predictions: Sequence[Prediction],
    test_rows: Sequence[SpanRecord],
    test_matches: Sequence[Match],
    test_scores: Sequence[Sequence[float]],
    test_predictions: Sequence[Prediction],
    held_out: str,
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """One leave-one-task-out fold, with everything fitted on the two seen tasks."""
    attribution = aggregate(
        score_rows(test_rows, test_matches, test_predictions), with_intervals=False
    )

    conformal = conformal_evaluate(
        _conformal_rows(calib_rows, calib_matches, calib_scores),
        _conformal_rows(test_rows, test_matches, test_scores),
        alpha,
        with_intervals=False,
    )

    keep = in_scope(calib_matches).astype(bool)
    calib_confidence = [p.score for p, k in zip(calib_predictions, keep) if k]
    calib_gold = gold_abstain(calib_matches)[keep]
    chosen = choose_abstention(calib_confidence, calib_gold)
    threshold = chosen.get("threshold")
    abstention = (
        evaluate_abstention(
            test_rows, test_matches, [p.score for p in test_predictions], float(threshold)
        )
        if threshold is not None
        else {"threshold": None, "note": "no feasible abstention threshold on this fold"}
    )
    abstention.pop("f1_ci", None)
    abstention.pop("recall_ci", None)

    return {
        "held_out_task": held_out,
        "n_calibration_spans": len(calib_rows),
        "n_test_spans": len(test_rows),
        "attribution": {
            key: attribution.get(key)
            for key in ("n", "top1_accuracy", "mrr", "recall_at_5", "snippet_containment")
        },
        "conformal": {
            key: conformal.get(key)
            for key in (
                "alpha",
                "target_coverage",
                "empirical_coverage",
                "band",
                "inside_band",
                "mean_set_size",
                "singleton_rate",
                "n_calibration",
                "n_test",
            )
        },
        "abstention": {
            key: abstention.get(key)
            for key in ("threshold", "precision", "recall", "f1", "base_rate", "n")
        },
        "abstention_threshold_fitted_on": "the two tasks that were not held out",
    }
