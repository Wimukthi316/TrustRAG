"""Layer 5 -- is the attribution score a probability, or just a number in [0, 1]?

C2's thesis applied to a third problem. Take the scorer's top-1 score as its
confidence that the top-1 sentence really is the one the annotator quoted, and
ask whether that number means what it says. Identity, temperature, Platt and
isotonic are fitted on the frozen calibration split and evaluated on test once.

**The prediction, written down before the measurement.** In C2 the
constant-base-rate predictor -- which ignores its input entirely -- had a
*better* ECE than the calibrated detector (0.0601 against 0.0644) and an AUROC
of 0.5 against 0.7533. C3 reproduced the same pattern on a different problem.
The prediction here is that it happens a third time: the floor wins on ECE and
loses on AUROC, Brier and NLL. If it does, that is no longer an anecdote about
one detector; it is a finding about the metric, and the floor row is printed in
every table so it cannot hide.

Nothing here is allowed to be fitted on test. `tests/test_c4_score_calibration.py`
asserts the split discipline and asserts the property that makes AUROC the
honest companion to ECE: a monotone recalibration cannot change it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.c2_calibration.calibration import (
    all_calibrators,
    calibration_report,
)
from src.c2_calibration.clustered import auroc

OUT = Path("results/c4/calibration/score_calibration.json")

EPSILON = 1e-6

PREDICTION = (
    "Written before running: the constant-base-rate floor will have the best "
    "ECE and an AUROC of 0.5. C2 and C3 both showed it. If it holds a third "
    "time on a third problem, ECE alone is not evidence of a useful model."
)


def fit_unit_map(scores: Sequence[float]) -> Tuple[float, float]:
    """The (low, high) of a min-max map, fitted on the calibration split only.

    Fitting it separately on each split would be a leak in miniature: the
    calibrator would be fitted on one scale and evaluated on another, and the
    ECE it reported would be partly an artefact of the two ranges differing.
    One map, fitted on calib, applied to both.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return (0.0, 1.0)
    return (float(values.min()), float(values.max()))


def to_unit_interval(
    scores: Sequence[float], bounds: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """Squash a backend's score into (0, 1) so a calibrator can be fitted.

    Cosine similarity is in [-1, 1] and the entailment heads are already
    probabilities, so a single affine map keeps the ordering exactly and
    therefore leaves AUROC untouched, which is the point: recalibration is
    supposed to move ECE and not discrimination. Test values outside the
    calibration range are clipped rather than rescaled, which is what applying a
    fitted transform to unseen data means.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return values
    low, high = bounds if bounds is not None else (float(values.min()), float(values.max()))
    if high - low < EPSILON:
        return np.full_like(values, 0.5)
    scaled = (values - low) / (high - low)
    return np.clip(scaled, EPSILON, 1.0 - EPSILON)


def constant_floor(calib_labels: Sequence[int], n_test: int) -> np.ndarray:
    """The base rate of the calibration split, repeated. Ignores its input."""
    rate = float(np.mean(calib_labels)) if len(calib_labels) else 0.5
    return np.full(n_test, np.clip(rate, EPSILON, 1.0 - EPSILON))


def evaluate(
    calib_scores: Sequence[float],
    calib_labels: Sequence[int],
    test_scores: Sequence[float],
    test_labels: Sequence[int],
) -> Dict[str, Any]:
    """Every calibrator on one backend's top-1 confidence, with the floor beside them."""
    bounds = fit_unit_map(calib_scores)
    calib_probs = to_unit_interval(calib_scores, bounds)
    test_probs = to_unit_interval(test_scores, bounds)
    calib_y = np.asarray(calib_labels, dtype=int)
    test_y = np.asarray(test_labels, dtype=int)

    rows: List[Dict[str, Any]] = []
    for calibrator in all_calibrators():
        try:
            fitted = calibrator.fit(calib_probs, calib_y)
            transformed = np.clip(fitted.transform(test_probs), EPSILON, 1 - EPSILON)
        except Exception as error:  # pragma: no cover - a degenerate split
            rows.append({"method": calibrator.name, "error": str(error)})
            continue
        report = calibration_report(transformed, test_y)
        report["method"] = calibrator.name
        report["auroc"] = auroc(transformed, test_y)
        report["is_floor"] = False
        report["fitted_on"] = "calib"
        rows.append(report)

    floor_probs = constant_floor(calib_y, test_y.size)
    floor = calibration_report(floor_probs, test_y)
    floor["method"] = "constant_base_rate"
    floor["auroc"] = auroc(floor_probs, test_y)
    floor["is_floor"] = True
    floor["ignores_input"] = True
    floor["fitted_on"] = "calib"
    rows.append(floor)

    best_ece = min(rows, key=lambda r: r.get("ece", float("inf")))
    best_auroc = max(rows, key=lambda r: r.get("auroc", -1.0))
    return {
        "n_calibration": int(calib_y.size),
        "n_test": int(test_y.size),
        "positive_rate_calibration": float(calib_y.mean()) if calib_y.size else 0.0,
        "positive_rate_test": float(test_y.mean()) if test_y.size else 0.0,
        "rows": rows,
        "best_ece_method": best_ece.get("method"),
        "best_auroc_method": best_auroc.get("method"),
        "floor_has_best_ece": bool(best_ece.get("is_floor")),
        "unit_map_bounds_fitted_on_calibration": list(bounds),
        "prediction": PREDICTION,
        "verdict": (
            "The floor won on ECE, as predicted. ECE alone is not evidence of a "
            "useful model; AUROC is what separates them."
            if best_ece.get("is_floor")
            else
            "The floor did NOT win on ECE this time, which breaks the pattern C2 "
            "and C3 both showed. Reported as measured."
        ),
    }
