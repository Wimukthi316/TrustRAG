"""Adapt the detector's operating point to RAGBench, on a calibration split.

The strongest objection to the cross-domain table is not that the detector fails.
It is that nobody adjusted it. The detector decides by argmax over three labels,
which was fitted to RAGTruth's 43% hallucination rate, and RAGBench runs at 14.2%.
A panel is entitled to say "your model did not fail, you just never turned the
dial", and without this the answer is an opinion.

So: carve a calibration split out of RAGBench, choose the dial setting there,
apply it once to the held-out half, and report both rows. Either outcome is a
result. If the score recovers, the gap is partly recoverable without retraining
and we say by how much. If it stays under the do-nothing baseline, the negative
result becomes very hard to argue with.

The dial is a response-level threshold on the highest hallucination probability
any answer token reaches. Two reasons it is that and not a span-decoding change.
RAGBench annotates whole sentences where RAGTruth annotates phrases, so example
level is the only reportable metric here and a response-level decision is the only
one that matters. And leaving the span decoder alone keeps every predicted span
coming from the same shared function, which is the rule three silent train/serve
mismatches on this project were caught by.

The threshold is chosen on calibration and applied to test exactly once. Sweeping
on test and reporting the best number is how a table becomes fiction.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CALIB_FRACTION = 0.30
SEED = 42
THRESHOLDS = tuple(round(0.02 * i, 2) for i in range(1, 50))  # 0.02 .. 0.98


def load_scores(path: Path | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_by_subset(
    rows: Sequence[Dict[str, Any]],
    calib_fraction: float = CALIB_FRACTION,
    seed: int = SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified split, so both halves carry all twelve domains.

    The subsets differ enormously in size and in hallucination rate - 3.5% on
    tatqa against 53.2% on expertqa - so an unstratified draw could hand the
    calibration half a base rate the test half does not share, and the threshold
    would be tuned for the wrong corpus.
    """
    if not 0.0 < calib_fraction < 1.0:
        raise ValueError("calib_fraction must be strictly between 0 and 1")

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["subset"]].append(row)

    calib: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    rng = random.Random(seed)
    for subset in sorted(grouped):
        members = sorted(grouped[subset], key=lambda r: str(r["id"]))
        rng.shuffle(members)
        cut = int(round(calib_fraction * len(members)))
        calib.extend(members[:cut])
        test.extend(members[cut:])
    return calib, test


def trivial_f1(positive_rate: float) -> float:
    """F1 of calling every response hallucinated: 2p/(1+p).

    The floor any real detector has to clear. Printed beside every score because
    without it a rare-positive corpus makes a useless detector look merely weak.
    """
    return (2 * positive_rate) / (1 + positive_rate) if positive_rate else 0.0


def score_rule(
    rows: Sequence[Dict[str, Any]], decide: Callable[[Dict[str, Any]], bool]
) -> Dict[str, Any]:
    """Positive-class precision, recall and F1 for one decision rule."""
    tp = fp = fn = tn = 0
    for row in rows:
        gold, pred = bool(row["gold_positive"]), bool(decide(row))
        if gold and pred:
            tp += 1
        elif pred:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n = tp + fp + fn + tn
    rate = (tp + fn) / n if n else 0.0
    return {
        "n": n,
        "positive_rate": rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "trivial_f1": trivial_f1(rate),
        "clears_trivial": f1 > trivial_f1(rate),
    }


def argmax_rule(row: Dict[str, Any]) -> bool:
    """What was reported: the response is positive if any span was decoded."""
    return bool(row["argmax_positive"])


def threshold_rule(threshold: float) -> Callable[[Dict[str, Any]], bool]:
    return lambda row: row["max_token_prob"] >= threshold


def sweep(
    rows: Sequence[Dict[str, Any]], thresholds: Sequence[float] = THRESHOLDS
) -> List[Dict[str, Any]]:
    return [
        {"threshold": t, **score_rule(rows, threshold_rule(t))} for t in thresholds
    ]


def choose_threshold(
    calib: Sequence[Dict[str, Any]], thresholds: Sequence[float] = THRESHOLDS
) -> Dict[str, Any]:
    """Pick the threshold with the best calibration F1, breaking ties low.

    Breaking ties toward the lower threshold is deliberate: among settings that
    score the same it keeps the more sensitive one, and being told about a
    hallucination that is not there costs a reviewer a glance, while missing one
    costs them the thing the detector exists for.
    """
    rows = sweep(calib, thresholds)
    if not rows:
        raise ValueError("no thresholds to choose from")
    best = max(rows, key=lambda r: (r["f1"], -r["threshold"]))
    return {"chosen": best, "sweep": rows}


def per_subset(
    rows: Sequence[Dict[str, Any]], decide: Callable[[Dict[str, Any]], bool]
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["subset"]].append(row)
    return {subset: score_rule(grouped[subset], decide) for subset in sorted(grouped)}


def build_report(
    scores: Sequence[Dict[str, Any]],
    calib_fraction: float = CALIB_FRACTION,
    seed: int = SEED,
    thresholds: Sequence[float] = THRESHOLDS,
) -> Dict[str, Any]:
    calib, test = split_by_subset(scores, calib_fraction, seed)
    chosen = choose_threshold(calib, thresholds)
    threshold = chosen["chosen"]["threshold"]
    decide = threshold_rule(threshold)

    return {
        "seed": seed,
        "calib_fraction": calib_fraction,
        "n_calibration": len(calib),
        "n_test": len(test),
        "chosen_threshold": threshold,
        "calibration": {
            "argmax": score_rule(calib, argmax_rule),
            "adapted": chosen["chosen"],
            "sweep": chosen["sweep"],
        },
        "test": {
            "argmax": score_rule(test, argmax_rule),
            "adapted": score_rule(test, decide),
            "per_subset_argmax": per_subset(test, argmax_rule),
            "per_subset_adapted": per_subset(test, decide),
        },
        "whole_corpus_argmax": score_rule(scores, argmax_rule),
    }


def format_report(report: Dict[str, Any]) -> str:
    def line(label: str, row: Dict[str, Any]) -> str:
        return (
            f"{label:<34}{row['n']:>7,}{row['positive_rate']:>9.3f}"
            f"{row['precision']:>9.4f}{row['recall']:>9.4f}{row['f1']:>9.4f}"
            f"{row['trivial_f1']:>9.4f}{('yes' if row['clears_trivial'] else 'NO'):>8}"
        )

    header = (
        f"{'row':<34}{'n':>7}{'pos rate':>9}{'P':>9}{'R':>9}"
        f"{'F1':>9}{'trivial':>9}{'clears':>8}"
    )
    lines = [
        f"threshold chosen on calibration: {report['chosen_threshold']:.2f}   "
        f"(calibration {report['n_calibration']:,} responses, "
        f"test {report['n_test']:,})",
        "",
        header,
        "-" * len(header),
        line("whole corpus, argmax", report["whole_corpus_argmax"]),
        line("calibration half, argmax", report["calibration"]["argmax"]),
        line("calibration half, adapted", report["calibration"]["adapted"]),
        "-" * len(header),
        line("TEST half, argmax", report["test"]["argmax"]),
        line("TEST half, adapted", report["test"]["adapted"]),
        "",
    ]

    delta = report["test"]["adapted"]["f1"] - report["test"]["argmax"]["f1"]
    lines.append(f"adapting the operating point moves test F1 by {100 * delta:+.2f} points")
    if report["test"]["adapted"]["clears_trivial"]:
        lines.append(
            "the adapted detector clears the do-nothing baseline: the transfer gap "
            "is partly an operating-point problem, and the table must say so"
        )
    else:
        lines.append(
            "the adapted detector still does NOT clear the do-nothing baseline. "
            "The transfer failure survives a fairly chosen operating point, which "
            "is the strongest form this negative result can take"
        )

    lines += ["", f"{'subset':<14}{'n':>7}{'argmax F1':>11}{'adapted F1':>12}{'trivial':>10}{'clears':>8}"]
    argmax_by = report["test"]["per_subset_argmax"]
    adapted_by = report["test"]["per_subset_adapted"]
    for subset in sorted(adapted_by):
        a, b = argmax_by[subset], adapted_by[subset]
        lines.append(
            f"{subset:<14}{b['n']:>7,}{a['f1']:>11.4f}{b['f1']:>12.4f}"
            f"{b['trivial_f1']:>10.4f}{('yes' if b['clears_trivial'] else 'NO'):>8}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Choose the RAGBench operating point on calibration, apply to test."
    )
    parser.add_argument(
        "--scores", default="results/ood/ragbench-scores/response_scores.jsonl"
    )
    parser.add_argument("--calib-fraction", type=float, default=CALIB_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default="results/ood/ragbench-scores/operating_point.json")
    args = parser.parse_args(argv)

    scores = load_scores(args.scores)
    report = build_report(scores, args.calib_fraction, args.seed)
    print(format_report(report))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
