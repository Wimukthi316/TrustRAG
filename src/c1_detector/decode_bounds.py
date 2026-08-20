"""How much any pure re-decoding of C1's output could buy, measured not argued.

Block B proposed Viterbi decoding over the BIO transition matrix, on the theory
that predicted spans fragment and run together. Building it is roughly four hours.
Before spending them, this module measures the ceiling of the idea.

Two bounds, both computed from artifacts that already exist:

    glue_sweep          join every pair of adjacent predictions closer than k
                        characters and rescore. This is the most generous version
                        of "stop fragmenting" available, so whatever Viterbi's
                        fragment-joining does, it cannot beat this by much.

    threshold_sweep     re-decode at a range of operating points and score
                        span-exact, which evaluate_c1.threshold_sweep does not
                        report. Shows whether the operating point is the lever.

What glue_sweep does NOT bound: Viterbi can also change which tokens are tagged,
not only how runs are joined. Quote it as a bound on the fragment-joining part of
the idea, which is the part the plan proposed it for, and say so.

Read the result against the measured run-to-run variation. Three runs at seeds
42, 7 and 13 put the standard deviation at 0.38 points on example F1 and 0.54 on
span-exact; the earlier "1.1 points" came from two runs at one seed and was
pessimistic. See results/c1/analysis/variance.json.
On the test split the best gluing gains 0.26 points, which is inside that band,
and it costs exact matches that were already correct -- both halves belong in the
report, because a fix that trades one metric for another is not a free win.

The threshold rows decode with spans_from_token_mask, which merges adjacent runs.
That is a different span population from the argmax BIO decode every reported
number uses, so they are a diagnostic and never a reportable operating point.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.c1_detector.evaluate_c1 import (
    span_char_metrics,
    span_overlap_metrics,
    spans_from_token_mask,
)
from src.c1_detector.localisation import load_probability_dump

Span = Tuple[int, int]

GLUE_GAPS: Tuple[int, ...] = (0, 1, 2, 3, 5, 10, 20, 40)
THRESHOLDS: Tuple[float, ...] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def glue_adjacent(spans: Sequence[Span], max_gap: int) -> List[Span]:
    """Merge predictions separated by at most `max_gap` characters."""
    merged: List[Span] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def glue_sweep(
    records: Sequence[Dict[str, Any]], gaps: Sequence[int] = GLUE_GAPS
) -> List[Dict[str, Any]]:
    """Score the corpus after gluing, for each gap width.

    `exact_tp` is reported beside the F1 on purpose: gluing recovers some spans
    and destroys others, and the count makes that visible where the F1 alone
    hides it.
    """
    rows: List[Dict[str, Any]] = []
    for gap in gaps:
        rescored = [
            {
                "gold_spans": record["gold_spans"],
                "pred_spans": glue_adjacent(record["pred_spans"], gap),
            }
            for record in records
        ]
        exact = span_char_metrics(rescored)
        overlap = span_overlap_metrics(rescored)
        rows.append(
            {
                "max_gap_glued": gap,
                "n_pred_spans": sum(len(r["pred_spans"]) for r in rescored),
                "span_exact_tp": exact["tp"],
                "span_exact_f1": exact["f1"],
                "span_overlap_f1": overlap["f1"],
            }
        )
    return rows


def threshold_sweep_span_exact(
    records: Sequence[Dict[str, Any]], thresholds: Sequence[float] = THRESHOLDS
) -> List[Dict[str, Any]]:
    """Span-exact and span-overlap at each operating point. Diagnostic only.

    Decoded from the token mask, so adjacent spans merge. Not comparable to a
    reported argmax BIO number; see the module docstring.
    """
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        rescored = []
        for record in records:
            mask = [p >= threshold for p in record["token_probs"]]
            rescored.append(
                {
                    "gold_spans": record["gold_spans"],
                    "pred_spans": spans_from_token_mask(
                        mask, record["answer_offsets"], record["answer"]
                    ),
                }
            )
        exact = span_char_metrics(rescored)
        rows.append(
            {
                "threshold": threshold,
                "n_pred_spans": sum(len(r["pred_spans"]) for r in rescored),
                "span_exact_precision": exact["precision"],
                "span_exact_recall": exact["recall"],
                "span_exact_f1": exact["f1"],
                "span_overlap_f1": span_overlap_metrics(rescored)["f1"],
            }
        )
    return rows


def build_report(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    glue = glue_sweep(records)
    best = max(glue, key=lambda row: row["span_exact_f1"])

    # The k=0 row is not the untouched baseline: a gap of zero still merges two
    # predictions that touch end to start. Score the predictions as they are.
    untouched = [
        {"gold_spans": r["gold_spans"], "pred_spans": list(r["pred_spans"])}
        for r in records
    ]
    baseline = span_char_metrics(untouched)

    return {
        "n_records": len(records),
        "baseline_n_pred_spans": sum(len(r["pred_spans"]) for r in records),
        "baseline_span_exact_f1": baseline["f1"],
        "best_glued_span_exact_f1": best["span_exact_f1"],
        "best_gain_points": round(100 * (best["span_exact_f1"] - baseline["f1"]), 3),
        "exact_matches_lost_at_best": baseline["tp"] - best["span_exact_tp"],
        "glue_sweep": glue,
        "threshold_sweep_span_exact": threshold_sweep_span_exact(records),
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        f"records {report['n_records']:,}",
        "",
        "--- glue adjacent predictions within k characters ---",
        f"{'k':>4}{'pred spans':>12}{'exact tp':>10}{'exact F1':>10}{'overlap F1':>12}",
    ]
    for row in report["glue_sweep"]:
        lines.append(
            f"{row['max_gap_glued']:>4}{row['n_pred_spans']:>12,}"
            f"{row['span_exact_tp']:>10,}{row['span_exact_f1']:>10.4f}"
            f"{row['span_overlap_f1']:>12.4f}"
        )
    lines += [
        "",
        f"best gain over the argmax baseline: {report['best_gain_points']:.2f} F1 points, "
        f"costing {report['exact_matches_lost_at_best']:,} exact matches that were "
        "already correct",
        "Judge that against the run-to-run standard deviation: 0.54 points on "
        "span-exact over three seeds, in variance.json.",
        "",
        "--- threshold sweep, span-exact (token-mask decode, DIAGNOSTIC ONLY) ---",
        f"{'thr':>6}{'pred spans':>12}{'exact P':>10}{'exact R':>10}"
        f"{'exact F1':>10}{'overlap F1':>12}",
    ]
    for row in report["threshold_sweep_span_exact"]:
        lines.append(
            f"{row['threshold']:>6.2f}{row['n_pred_spans']:>12,}"
            f"{row['span_exact_precision']:>10.4f}{row['span_exact_recall']:>10.4f}"
            f"{row['span_exact_f1']:>10.4f}{row['span_overlap_f1']:>12.4f}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Upper bound on what re-decoding C1's output can buy."
    )
    parser.add_argument("--probs", default="results/c1/test/probabilities.jsonl")
    parser.add_argument("--out-dir", default="results/c1/analysis")
    args = parser.parse_args(argv)

    records = load_probability_dump(args.probs)
    report = build_report(records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "decode_bounds.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(format_report(report))
    print()
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
