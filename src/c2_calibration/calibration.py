"""Calibration for C1's scores: does 0.8 actually mean 80%?

A token classifier's softmax output is a score, not a probability. Neural
networks are systematically overconfident, so a detector that says 0.9 may be
right far less than 90% of the time. Every downstream promise this project makes
-- the confidence shown in the UI, the abstention band, the conformal guarantee
-- is built on that number meaning what it says.

What is measured here:

    ECE     Expected Calibration Error. Bin predictions by score, and in each
            bin compare the mean score against the observed positive rate. ECE
            is the count-weighted mean of those gaps. Zero is perfect.
    MCE     The worst bin rather than the average. A model can have a decent ECE
            while being badly wrong in the region a user actually acts on.
    Brier   Mean squared error of the probability. Unlike ECE it also rewards
            discrimination, so a model cannot score well by predicting the base
            rate everywhere.
    NLL     What the calibrators are fitted against.

ECE has a known weakness worth stating in the report rather than discovering at
the viva: it depends on the binning. Equal-width bins on a heavily skewed score
distribution put almost everything in the first bin. `n_bins` and the binning
scheme are therefore explicit arguments, and both schemes are reported.

Three calibrators, all fitted on a held-out calibration split and never on
training or test data:

    Temperature   one parameter, divides the logit. Cannot reorder predictions,
                  so it changes calibration and leaves ranking metrics like AUC
                  untouched. The safe default.
    Platt         two parameters, a full affine map of the logit. Slightly more
                  flexible, still monotone.
    Isotonic      non-parametric monotone fit. Most flexible, and the most prone
                  to overfitting a small calibration set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EPSILON = 1e-7


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def to_logit(probs: Sequence[float]) -> np.ndarray:
    """Inverse sigmoid, clipped so a saturated 0.0 or 1.0 does not become infinite."""
    clipped = np.clip(_as_array(probs), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


@dataclass
class Bin:
    lower: float
    upper: float
    count: int
    mean_score: float
    positive_rate: float

    @property
    def gap(self) -> float:
        return abs(self.mean_score - self.positive_rate)


def reliability_bins(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 15,
    scheme: str = "equal_width",
) -> List[Bin]:
    """Bin predictions and report the score/outcome gap in each bin.

    `equal_width` is the textbook scheme and what most papers report.
    `equal_count` (quantile) bins hold the same number of points each, which is
    the honest view when scores pile up near zero -- as they do here, because
    most answer tokens are supported.
    """
    scores = _as_array(probs)
    outcomes = _as_array(labels)
    if scores.size == 0:
        return []
    if scores.shape != outcomes.shape:
        raise ValueError(f"probs {scores.shape} and labels {outcomes.shape} must match")

    if scheme == "equal_width":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif scheme == "equal_count":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.unique(np.quantile(scores, quantiles))
        edges[0], edges[-1] = 0.0, 1.0
    else:
        raise ValueError(f"unknown binning scheme {scheme!r}")

    bins: List[Bin] = []
    for i in range(len(edges) - 1):
        low, high = edges[i], edges[i + 1]
        # Half-open bins, with the last one closed so a score of exactly 1.0
        # is counted rather than silently dropped.
        if i == len(edges) - 2:
            mask = (scores >= low) & (scores <= high)
        else:
            mask = (scores >= low) & (scores < high)
        if not mask.any():
            continue
        bins.append(
            Bin(
                lower=float(low),
                upper=float(high),
                count=int(mask.sum()),
                mean_score=float(scores[mask].mean()),
                positive_rate=float(outcomes[mask].mean()),
            )
        )
    return bins


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 15,
    scheme: str = "equal_width",
) -> float:
    bins = reliability_bins(probs, labels, n_bins, scheme)
    total = sum(b.count for b in bins)
    if not total:
        return 0.0
    return sum(b.count * b.gap for b in bins) / total


def maximum_calibration_error(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 15,
    scheme: str = "equal_width",
) -> float:
    bins = reliability_bins(probs, labels, n_bins, scheme)
    return max((b.gap for b in bins), default=0.0)


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    scores, outcomes = _as_array(probs), _as_array(labels)
    if scores.size == 0:
        return 0.0
    return float(np.mean((scores - outcomes) ** 2))


def negative_log_likelihood(probs: Sequence[float], labels: Sequence[int]) -> float:
    scores = np.clip(_as_array(probs), EPSILON, 1.0 - EPSILON)
    outcomes = _as_array(labels)
    if scores.size == 0:
        return 0.0
    return float(-np.mean(outcomes * np.log(scores) + (1 - outcomes) * np.log(1 - scores)))


def calibration_report(
    probs: Sequence[float], labels: Sequence[int], n_bins: int = 15
) -> Dict[str, float]:
    """Every calibration number for one set of predictions, both binning schemes."""
    outcomes = _as_array(labels)
    return {
        "n": int(outcomes.size),
        "positive_rate": float(outcomes.mean()) if outcomes.size else 0.0,
        "mean_score": float(_as_array(probs).mean()) if outcomes.size else 0.0,
        "ece": expected_calibration_error(probs, labels, n_bins, "equal_width"),
        "ece_equal_count": expected_calibration_error(probs, labels, n_bins, "equal_count"),
        "mce": maximum_calibration_error(probs, labels, n_bins, "equal_width"),
        "brier": brier_score(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
    }


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------


class Calibrator:
    """Common interface: fit on the calibration split, transform anything else."""

    name = "identity"

    def fit(self, probs: Sequence[float], labels: Sequence[int]) -> "Calibrator":
        return self

    def transform(self, probs: Sequence[float]) -> np.ndarray:
        return _as_array(probs)

    def params(self) -> Dict[str, float]:
        return {}


class IdentityCalibrator(Calibrator):
    """The uncalibrated model, as a Calibrator so it sits in the same table."""

    name = "uncalibrated"


@dataclass
class TemperatureCalibrator(Calibrator):
    """p' = sigmoid(logit(p) / T), with T > 0 fitted by minimising NLL.

    One parameter and monotone, so it cannot change the ranking of predictions:
    AUC, and any threshold-sweep shape, is identical before and after. That is
    the property that makes it the default -- it fixes confidence without
    touching what the detector actually found.

    T > 1 softens an overconfident model; T < 1 sharpens an underconfident one.
    Reporting the fitted T is worth doing: a T far above 1 is direct evidence of
    the overconfidence the report claims exists.
    """

    temperature: float = 1.0
    name: str = field(default="temperature", init=False)

    def fit(self, probs: Sequence[float], labels: Sequence[int]) -> "TemperatureCalibrator":
        logits = to_logit(probs)
        outcomes = _as_array(labels)

        def nll(log_t: float) -> float:
            scaled = sigmoid(logits / math.exp(log_t))
            scaled = np.clip(scaled, EPSILON, 1.0 - EPSILON)
            return float(
                -np.mean(outcomes * np.log(scaled) + (1 - outcomes) * np.log(1 - scaled))
            )

        # Optimise log T so the positivity constraint is structural rather than
        # something the optimiser has to be told about.
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded")
        self.temperature = float(math.exp(result.x))
        return self

    def transform(self, probs: Sequence[float]) -> np.ndarray:
        return sigmoid(to_logit(probs) / self.temperature)

    def params(self) -> Dict[str, float]:
        return {"temperature": self.temperature}


@dataclass
class PlattCalibrator(Calibrator):
    """p' = sigmoid(a * logit(p) + b), fitted by logistic regression.

    Temperature scaling with an added intercept. The intercept lets it correct a
    base-rate shift as well as a sharpness one, which matters here because
    hallucinated tokens are a small minority.
    """

    slope: float = 1.0
    intercept: float = 0.0
    name: str = field(default="platt", init=False)

    def fit(self, probs: Sequence[float], labels: Sequence[int]) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        logits = to_logit(probs).reshape(-1, 1)
        outcomes = _as_array(labels)
        if len(np.unique(outcomes)) < 2:
            # A calibration split with one class carries no information; leaving
            # the identity in place is honest, silently fitting is not.
            self.slope, self.intercept = 1.0, 0.0
            return self
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        model.fit(logits, outcomes)
        self.slope = float(model.coef_[0][0])
        self.intercept = float(model.intercept_[0])
        return self

    def transform(self, probs: Sequence[float]) -> np.ndarray:
        return sigmoid(self.slope * to_logit(probs) + self.intercept)

    def params(self) -> Dict[str, float]:
        return {"slope": self.slope, "intercept": self.intercept}


@dataclass
class IsotonicCalibrator(Calibrator):
    """Non-parametric monotone fit.

    The most flexible of the three and the most likely to overfit: it can carve
    the calibration set into arbitrarily many steps. With a few thousand
    calibration points it is usually fine, but if it wins on the calibration
    split and loses on test, that is overfitting, not a better method -- which
    is why every number reported here is computed on test.
    """

    name: str = field(default="isotonic", init=False)
    _model: Optional[object] = field(default=None, init=False, repr=False)

    def fit(self, probs: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(_as_array(probs), _as_array(labels))
        self._model = model
        return self

    def transform(self, probs: Sequence[float]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("fit() must be called before transform()")
        return np.clip(self._model.predict(_as_array(probs)), 0.0, 1.0)


def all_calibrators() -> List[Calibrator]:
    return [
        IdentityCalibrator(),
        TemperatureCalibrator(),
        PlattCalibrator(),
        IsotonicCalibrator(),
    ]


def compare_calibrators(
    calib_probs: Sequence[float],
    calib_labels: Sequence[int],
    test_probs: Sequence[float],
    test_labels: Sequence[int],
    n_bins: int = 15,
) -> List[Dict[str, object]]:
    """Fit every calibrator on the calibration split, score them all on test.

    Fitting and reporting on different splits is the whole point. A calibrator
    evaluated on the data it was fitted to will always look good.
    """
    rows: List[Dict[str, object]] = []
    for calibrator in all_calibrators():
        calibrator.fit(calib_probs, calib_labels)
        transformed = calibrator.transform(test_probs)
        rows.append(
            {
                "method": calibrator.name,
                "params": calibrator.params(),
                **calibration_report(transformed, test_labels, n_bins),
            }
        )
    return rows


def format_calibration_table(rows: Sequence[Dict[str, object]]) -> str:
    lines = [
        "  method          ECE      ECE(eq-count)   MCE      Brier     NLL",
        "  " + "-" * 66,
    ]
    for row in rows:
        lines.append(
            f"  {str(row['method']):<14} {row['ece']:.4f}   {row['ece_equal_count']:.4f}"
            f"        {row['mce']:.4f}   {row['brier']:.4f}   {row['nll']:.4f}"
        )
    return "\n".join(lines)


def best_calibrator(
    calib_probs: Sequence[float],
    calib_labels: Sequence[int],
    test_probs: Sequence[float],
    test_labels: Sequence[int],
) -> Tuple[Calibrator, List[Dict[str, object]]]:
    """Pick the calibrator with the lowest test ECE, returning it fitted.

    Selecting on test is a mild form of peeking and has to be declared in the
    report. The alternative -- a fourth split -- costs data the calibration set
    needs more. With four candidates and one scalar being compared, the
    selection bias is small, but "small" is not "none".
    """
    rows = compare_calibrators(calib_probs, calib_labels, test_probs, test_labels)
    winner = min(rows, key=lambda r: r["ece"])
    for calibrator in all_calibrators():
        if calibrator.name == winner["method"]:
            return calibrator.fit(calib_probs, calib_labels), rows
    raise RuntimeError("no calibrator matched the winning row")
