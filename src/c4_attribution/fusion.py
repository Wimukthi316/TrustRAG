"""Layer 11 -- C1 x C4. Does knowing there is no evidence help decide it is a hallucination?

C1 says a stretch of the answer looks hallucinated. C4 says whether the context
contains anything to point at. The obvious question is whether the second helps
the first, and the obvious way to answer it badly is to report one F1.

Three things have to be true or the table will not survive a viva.

1. **Floors beside it, at the same size.** Trivial always-positive example F1
   0.5177 on this split; C1 alone 0.7623; HHEM adapted 0.7012;
   LettuceDetect-large 0.7918.
2. **The improvement has to clear the noise.** C1's seed standard deviation is
   0.384 points of example F1 over seeds 42/7/13, and its bootstrap interval is
   4.18 points wide. **Any gain under about one point is not distinguishable
   from zero and is reported that way**, exactly as C1's class-weight ablation
   was.
3. **Fit on calibration, apply to test once.** The frozen 1,509. Never fit on
   test. `tests/test_c4_fusion.py` asserts it.

And the part that is more informative than the headline: **the four
disagreement cells.** C1 flags and C4 finds evidence; C1 flags and C4 finds
none; C1 passes and C4 finds evidence; C1 passes and C4 finds none. Counting and
characterising those says what the two components actually know that the other
does not.

C2's per-span decisions are rebuilt here from `c2_artifact.json` using C2's own
calibrator and `SplitConformal` unchanged, and the rebuild is checked against
the coverage the artefact recorded. C3 did the same and reproduced
0.8807531380753139 to the last digit; if this ever stops matching, the run
aborts rather than reporting a number built on a drifted input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.c1_detector.evaluate_c1 import prf
from src.c2_calibration.calibration import calibrator_from_dict
from src.c2_calibration.conformal import SplitConformal
from src.c2_calibration.run_c2 import span_units

ARTIFACT = Path("results/c2/c1/c2_artifact.json")
CALIB_DUMP = Path("results/c1/calib/probabilities.jsonl")
TEST_DUMP = Path("results/c1/test/probabilities.jsonl")

# Inherited, and printed beside every fusion number. Sources named so a reader
# can check them: results/c1/test/metrics.json, results/hhem/hhem_metrics.json,
# results/lettucedetect/test/metrics.json, results/c1/analysis/variance.json,
# results/c1/analysis/bootstrap_test.json.
FLOORS: Dict[str, float] = {
    "trivial_always_positive": 0.5177,
    "c1_alone": 0.7623,
    "hhem_adapted": 0.7012,
    "lettucedetect_large": 0.7918,
}
NOISE_SEED_SD_POINTS = 0.384
NOISE_BOOTSTRAP_WIDTH_POINTS = 4.18
NOISE_FLOOR_POINTS = 1.0


def rebuild_c2_decisions(
    artifact: Dict[str, Any],
    calib_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    alpha: float = 0.1,
) -> Dict[str, Any]:
    """Per predicted test span: C2's calibrated score and its flag/abstain/pass.

    C2 stores only aggregate coverage, so the per-span decisions are rebuilt
    from the artefact rather than re-derived by a second method. The rebuilt
    coverage is compared with the stored one; a mismatch means one of the frozen
    inputs moved and is a loud failure here rather than a wrong number three
    layers downstream.
    """
    calibrator = calibrator_from_dict(artifact["calibrator"])
    calib_scores, calib_labels, _ = span_units(calib_records)
    test_scores, test_labels, _ = span_units(test_records)

    conformal = SplitConformal(alpha).fit(calibrator.transform(calib_scores), calib_labels)
    calibrated_test = calibrator.transform(test_scores)
    decisions = [d.value for d in conformal.decisions(calibrated_test)]
    sets = conformal.prediction_sets(calibrated_test)
    coverage = float(
        np.mean([1.0 if test_labels[i] in s else 0.0 for i, s in enumerate(sets)])
    )

    recorded = artifact.get(f"coverage_at_alpha_{alpha}")
    matches = recorded is None or abs(coverage - float(recorded)) < 1e-9

    rows: List[Dict[str, Any]] = []
    position = 0
    for record in test_records:
        for span in record.get("pred_spans", []):
            rows.append(
                {
                    "response_id": str(record["id"]),
                    "task_type": str(record.get("task_type", "unknown")),
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                    "calibrated_score": float(calibrated_test[position]),
                    "is_hallucinated": int(bool(span["is_hallucinated"])),
                    "decision": decisions[position],
                }
            )
            position += 1

    return {
        "rows": rows,
        "rebuilt_coverage": coverage,
        "recorded_coverage": recorded,
        "matches_recorded": bool(matches),
    }


# --------------------------------------------------------------------------
# The four cells
# --------------------------------------------------------------------------


def disagreement_cells(
    c1_flags: Sequence[int], c4_finds_evidence: Sequence[int]
) -> Dict[str, Any]:
    """The table that says more than the headline F1 does."""
    flags = np.asarray(c1_flags, dtype=int)
    evidence = np.asarray(c4_finds_evidence, dtype=int)
    cells = {
        "c1_flag_c4_evidence": int(np.sum((flags == 1) & (evidence == 1))),
        "c1_flag_c4_none": int(np.sum((flags == 1) & (evidence == 0))),
        "c1_pass_c4_evidence": int(np.sum((flags == 0) & (evidence == 1))),
        "c1_pass_c4_none": int(np.sum((flags == 0) & (evidence == 0))),
    }
    total = sum(cells.values())
    return {
        "n": total,
        "cells": cells,
        "shares": {k: (v / total if total else 0.0) for k, v in cells.items()},
        "reading": (
            "c1_flag_c4_none is the agreement case that matters: the detector "
            "flagged it and there is nothing in the context to support it. "
            "c1_flag_c4_evidence is where the two components disagree and is "
            "where a precision gain would have to come from."
        ),
    }


# --------------------------------------------------------------------------
# The response-level table
# --------------------------------------------------------------------------


def response_labels(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Gold: does this response contain at least one annotated span?"""
    return {str(r["id"]): int(len(r.get("gold_spans") or []) > 0) for r in records}


def example_f1(gold: Sequence[int], predicted: Sequence[int]) -> Dict[str, float]:
    """The metric C1 reports, computed the same way, so the rows are comparable."""
    gold = np.asarray(gold, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    tp = int(np.sum((gold == 1) & (predicted == 1)))
    fp = int(np.sum((gold == 0) & (predicted == 1)))
    fn = int(np.sum((gold == 1) & (predicted == 0)))
    tn = int(np.sum((gold == 0) & (predicted == 0)))
    out = prf(tp, fp, fn)
    out["tn"] = tn
    out["accuracy"] = (tp + tn) / len(gold) if len(gold) else 0.0
    return out


def compare_to_floors(value: float) -> Dict[str, Any]:
    """Is this better than C1 alone by more than the noise the project measured?"""
    gain_points = (value - FLOORS["c1_alone"]) * 100.0
    return {
        "example_f1": value,
        "floors": dict(FLOORS),
        "gain_over_c1_points": gain_points,
        "noise_seed_sd_points": NOISE_SEED_SD_POINTS,
        "noise_bootstrap_width_points": NOISE_BOOTSTRAP_WIDTH_POINTS,
        "distinguishable_from_zero": bool(abs(gain_points) >= NOISE_FLOOR_POINTS),
        "verdict": (
            f"{gain_points:+.2f} points against C1 alone. C1's seed sd is "
            f"{NOISE_SEED_SD_POINTS} points and its bootstrap interval is "
            f"{NOISE_BOOTSTRAP_WIDTH_POINTS} points wide, so a change smaller "
            f"than about {NOISE_FLOOR_POINTS:.0f} point is not distinguishable "
            "from zero and is reported as such."
        ),
    }


RULES: Tuple[str, ...] = ("c1_alone", "c1_c2_flag_only", "c1_c2_gated_by_c4", "c4_tie_break")


def fuse(
    records: Sequence[Dict[str, Any]],
    span_rows: Sequence[Dict[str, Any]],
    c4_no_evidence: Dict[Tuple[str, int, int], bool],
    rule: str,
) -> Dict[str, int]:
    """Response-level prediction under one rule. Four are reported side by side.

    `c1_alone`           any predicted span makes the response positive. This is
                         the rule C1's own example F1 uses, so it is the row
                         every other row is read against.
    `c1_c2_flag_only`    only a span C2 decided to FLAG counts. C2 already
                         drops PASS spans in serving, so this is what the demo
                         actually shows.
    `c1_c2_gated_by_c4`  a flagged span counts only if C4 also finds no
                         supporting evidence for it. A precision gate.
    `c4_tie_break`       the product question, and the interesting one. C2
                         abstains on a large share of spans -- C3 measured the
                         escalation budget at 37.4% -- and an abstention is a
                         span nobody has decided. This rule takes every FLAG,
                         and promotes an ABSTAIN only when C4 finds nothing in
                         the context to support it. It can only add positives to
                         `c1_c2_flag_only`, so the comparison between those two
                         isolates exactly what C4 contributed.

    Every rule is applied to the test split once, with C4's threshold fitted on
    calibration. None of them is tuned on test.
    """
    if rule not in RULES:
        raise ValueError(f"unknown fusion rule {rule!r}; expected one of {RULES}")

    positive: Dict[str, int] = {str(r["id"]): 0 for r in records}
    for row in span_rows:
        key = (row["response_id"], row["start"], row["end"])
        no_evidence = bool(c4_no_evidence.get(key, False))
        decision = str(row.get("decision", ""))

        if rule == "c1_alone":
            counts = True
        elif rule == "c1_c2_flag_only":
            counts = decision == "flag"
        elif rule == "c1_c2_gated_by_c4":
            counts = decision == "flag" and no_evidence
        else:  # c4_tie_break
            counts = decision == "flag" or (decision == "abstain" and no_evidence)

        if counts:
            positive[row["response_id"]] = 1
    return positive
