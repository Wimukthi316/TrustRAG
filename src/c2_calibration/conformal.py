"""Split conformal prediction at span level, with abstention.

This is the project's novelty claim, so the maths is written out rather than
imported, and every step is stated plainly enough to be defended.

**The guarantee.** Given a calibration set drawn exchangeably with the test set,
split conformal produces prediction *sets* C(x) satisfying

        P( y_test in C(x_test) )  >=  1 - alpha

with no assumption whatsoever about the model that produced the scores. It holds
for a well-trained detector and for a badly-trained one; a bad detector simply
pays for it with wider sets. That distribution-free property is exactly what an
LLM judge's verbalised confidence cannot offer, and it is the heart of the
argument against "just use an LLM".

**The construction** (LAC -- the least-ambiguous set-valued classifier):

1. On the calibration split, score each example by how badly the model treated
   its *true* label:            s_i = 1 - p_hat(y_i | x_i)
2. Take the finite-sample-corrected empirical quantile

        k    = ceil( (n + 1) * (1 - alpha) )
        q    = the k-th smallest calibration score

   The `n + 1` is not a rounding detail. It is what converts an asymptotic
   statement into one that holds exactly at finite n. If k > n, no finite
   threshold can offer the guarantee at this alpha with this much calibration
   data, and the honest output is an infinite threshold and a loud warning
   rather than a number that looks fine.
3. At test time include every label whose score clears the same bar:

        C(x) = { y : 1 - p_hat(y | x) <= q }

**Why the sets are the useful part.** For binary hallucination detection each
set is one of four things, and they map onto the decisions the product already
speaks in `src/common/schema.py`:

    {1}     FLAG      confidently hallucinated
    {0}     PASS      confidently supported
    {0,1}   ABSTAIN   the guarantee cannot separate these -- send to a human
    {}      ABSTAIN   neither label clears the bar; rarer, treated the same way

The empty set is possible under LAC and is not a bug. It means both labels were
unusual relative to calibration. It is folded into abstention because that is
the safe reading, and reported separately so it is never hidden.

**Exchangeability is the assumption, and it is a real one.** Three ways it can
break here, all of which have to be stated rather than hoped away:

1. *Contamination.* If the detector trained on the calibration data it is
   in-sample there and out-of-sample on test, and the two are not exchangeable.
   Measured on this repository on 2026-08-11: the public LettuceDetect
   checkpoint scores example-F1 0.9267 on a calibration split carved out of
   RAGTruth train, against 0.7918 on the official test split. Calibrating on
   that gap drove empirical coverage to 0.769 against a 0.900 target. The fix is
   to calibrate on held-out data only -- see `halve_by_response` in run_c2.py.

2. *Distribution shift.* Coverage is not claimed across a change of domain,
   which is exactly why the RAGBench OOD run is expected to break it. Reporting
   that break is a result, not a failure.

3. *Clustering, and this one is subtle.* The unit RAGTruth samples
   independently is the **response**. Spans and tokens within one response are
   strongly correlated -- one bad sentence produces a run of hallucinated tokens
   together. Pooling tokens across responses and treating them as exchangeable
   is therefore an approximation. Marginal coverage over the pooled population
   is still a meaningful statement and is what the tables report, but the
   effective sample size is nearer the number of responses than the number of
   tokens, so the noise band computed below is optimistic at token level. Treat
   a small group-level shortfall at token level as unresolved rather than as
   either fine or broken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from src.common.schema import ConformalDecision

LABELS = (0, 1)


def _as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def conformal_quantile(scores: Sequence[float], alpha: float) -> float:
    """The finite-sample corrected (1 - alpha) quantile of calibration scores.

    Returns +inf when the calibration set is too small for the requested alpha,
    i.e. when ceil((n+1)(1-alpha)) > n. At that point every prediction set
    contains every label: coverage is trivially 1.0 and the method is telling
    you it has nothing useful to say, which is the correct behaviour and much
    better than quietly returning max(scores).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    ordered = np.sort(_as_array(scores))
    n = ordered.size
    if n == 0:
        raise ValueError("cannot calibrate on an empty score set")

    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(ordered[k - 1])


def minimum_calibration_size(alpha: float) -> int:
    """Smallest calibration set for which a finite threshold exists at this alpha.

    ceil((n+1)(1-alpha)) <= n  holds first at n = ceil(1/alpha) - 1. Useful as a
    guard before splitting the data, and as the answer to "why is alpha=0.01
    missing from the table".
    """
    return math.ceil(1.0 / alpha) - 1


@dataclass
class SplitConformal:
    """Split conformal for binary span classification, with abstention.

    `alpha` is the miscoverage level: alpha = 0.1 targets 90% coverage.
    """

    alpha: float = 0.1
    threshold: float = field(default=float("nan"), init=False)
    n_calibration: int = field(default=0, init=False)

    def fit(self, probs: Sequence[float], labels: Sequence[int]) -> "SplitConformal":
        """Calibrate on held-out (probability, true label) pairs.

        **Calibrating the scores first changes nothing here, and that is worth
        understanding rather than working around.** LAC thresholds the score, so
        any strictly monotone transform of the scores -- which is exactly what
        temperature and Platt scaling are -- moves the threshold by the same
        transform and leaves the partition of examples identical. Coverage,
        abstention rate and flag rate all come out bit-identical. Verified
        empirically in tests/test_c2_calibration.py.

        The two layers therefore do separate jobs, and the report should say so:

            calibration   makes the number shown to a human mean what it says.
                          Measured by ECE. Does not touch the guarantee.
            conformal     provides the guarantee. Invariant to monotone
                          recalibration, which is a robustness property worth
                          claiming rather than a redundancy to apologise for.

        Isotonic regression is only weakly monotone -- it can map distinct
        scores onto the same value -- so it can shift results very slightly at
        the tie boundary. Everything else leaves them alone.
        """
        scores = _as_array(probs)
        outcomes = np.asarray(labels, dtype=int)
        if scores.shape != outcomes.shape:
            raise ValueError(f"probs {scores.shape} and labels {outcomes.shape} must match")

        # 1 - p(true label): p when the truth is 0, 1 - p when the truth is 1.
        nonconformity = np.where(outcomes == 1, 1.0 - scores, scores)
        self.threshold = conformal_quantile(nonconformity, self.alpha)
        self.n_calibration = int(scores.size)
        return self

    def prediction_sets(self, probs: Sequence[float]) -> List[Tuple[int, ...]]:
        """The set of labels that clear the calibrated bar, per input."""
        if math.isnan(self.threshold):
            raise RuntimeError("fit() must be called before prediction_sets()")
        scores = _as_array(probs)
        keep_one = (1.0 - scores) <= self.threshold
        keep_zero = scores <= self.threshold
        return [
            tuple(label for label, keep in zip(LABELS, (zero, one)) if keep)
            for zero, one in zip(keep_zero, keep_one)
        ]

    def decisions(self, probs: Sequence[float]) -> List[ConformalDecision]:
        """Map prediction sets onto the decisions schema.py already speaks."""
        out: List[ConformalDecision] = []
        for candidate in self.prediction_sets(probs):
            if candidate == (1,):
                out.append(ConformalDecision.FLAG)
            elif candidate == (0,):
                out.append(ConformalDecision.PASS)
            else:
                out.append(ConformalDecision.ABSTAIN)
        return out

    def evaluate(
        self, probs: Sequence[float], labels: Sequence[int]
    ) -> Dict[str, float]:
        """Empirical coverage and the cost paid for it, on a test set."""
        sets = self.prediction_sets(probs)
        outcomes = np.asarray(labels, dtype=int)
        if len(sets) != outcomes.size:
            raise ValueError("probs and labels must be the same length")

        covered = sum(1 for s, y in zip(sets, outcomes) if y in s)
        sizes = [len(s) for s in sets]
        decisions = self.decisions(probs)
        n = len(sets)

        singleton = [(s, y) for s, y in zip(sets, outcomes) if len(s) == 1]
        singleton_correct = sum(1 for s, y in singleton if s[0] == y)

        return {
            "alpha": self.alpha,
            "target_coverage": 1.0 - self.alpha,
            "empirical_coverage": covered / n if n else 0.0,
            "threshold": self.threshold,
            "n_calibration": self.n_calibration,
            "n_test": n,
            "mean_set_size": float(np.mean(sizes)) if sizes else 0.0,
            "abstention_rate": sum(
                1 for d in decisions if d is ConformalDecision.ABSTAIN
            ) / n if n else 0.0,
            "empty_set_rate": sum(1 for s in sets if not s) / n if n else 0.0,
            "flag_rate": sum(1 for d in decisions if d is ConformalDecision.FLAG) / n
            if n
            else 0.0,
            # Accuracy on the decided subset. This is the number that says
            # whether abstaining bought anything.
            "selective_accuracy": singleton_correct / len(singleton) if singleton else 0.0,
            "selective_n": len(singleton),
        }


def coverage_table(
    calib_probs: Sequence[float],
    calib_labels: Sequence[int],
    test_probs: Sequence[float],
    test_labels: Sequence[int],
    alphas: Sequence[float] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40),
) -> List[Dict[str, float]]:
    """Coverage vs alpha -- the headline table of the whole project.

    Empirical coverage should sit at or just above 1 - alpha for every row. A
    row below target is either a bug or a broken exchangeability assumption, and
    either way it invalidates the claim. Slightly above target is expected and
    correct: the guarantee is one-sided, and the finite-sample correction is
    deliberately conservative.
    """
    rows: List[Dict[str, float]] = []
    for alpha in alphas:
        conformal = SplitConformal(alpha=alpha).fit(calib_probs, calib_labels)
        rows.append(conformal.evaluate(test_probs, test_labels))
    return rows


def group_conditional_coverage(
    calib_probs: Sequence[float],
    calib_labels: Sequence[int],
    calib_groups: Sequence[str],
    test_probs: Sequence[float],
    test_labels: Sequence[int],
    test_groups: Sequence[str],
    alpha: float = 0.1,
) -> Dict[str, Dict[str, float]]:
    """Calibrate a separate threshold per group, so coverage holds within each.

    Marginal coverage is an average, and an average can hide a group that is
    badly under-covered while another is over-covered. RAGTruth's three tasks
    have very different hallucination rates, so this is not a hypothetical: it
    is the difference between "90% coverage overall" and "90% coverage for every
    task", and only the second is a useful promise.

    A group whose calibration slice is smaller than minimum_calibration_size()
    gets an infinite threshold, reported rather than hidden.
    """
    calib_groups = list(calib_groups)
    test_groups = list(test_groups)
    calib_probs, calib_labels = _as_array(calib_probs), list(calib_labels)
    test_probs, test_labels = _as_array(test_probs), list(test_labels)

    out: Dict[str, Dict[str, float]] = {}
    for group in sorted(set(test_groups)):
        calib_index = [i for i, g in enumerate(calib_groups) if g == group]
        test_index = [i for i, g in enumerate(test_groups) if g == group]
        if not calib_index or not test_index:
            continue
        conformal = SplitConformal(alpha=alpha).fit(
            calib_probs[calib_index], [calib_labels[i] for i in calib_index]
        )
        result = conformal.evaluate(
            test_probs[test_index], [test_labels[i] for i in test_index]
        )
        result["min_calibration_size_for_alpha"] = minimum_calibration_size(alpha)
        out[group] = result
    return out


def risk_coverage_curve(
    probs: Sequence[float],
    labels: Sequence[int],
    n_points: int = 21,
) -> List[Dict[str, float]]:
    """Selective prediction: accept the most confident fraction, measure the error.

    Independent of conformal prediction, and reported alongside it because it
    answers the question a practitioner actually asks -- "if I only act on the
    predictions the system is sure about, how wrong is it?". Confidence here is
    max(p, 1 - p): distance from the decision boundary, not from zero.

    A useful detector's risk falls as coverage falls. A flat curve means the
    confidence score carries no information about correctness, and no amount of
    calibration or conformal machinery can fix that.
    """
    scores = _as_array(probs)
    outcomes = np.asarray(labels, dtype=int)
    if scores.size == 0:
        return []

    confidence = np.maximum(scores, 1.0 - scores)
    predicted = (scores >= 0.5).astype(int)
    wrong = (predicted != outcomes).astype(float)

    order = np.argsort(-confidence)
    wrong_sorted = wrong[order]
    cumulative_errors = np.cumsum(wrong_sorted)

    rows: List[Dict[str, float]] = []
    for coverage in np.linspace(1.0 / n_points, 1.0, n_points):
        take = max(1, int(round(coverage * scores.size)))
        rows.append(
            {
                "coverage": take / scores.size,
                "n_accepted": int(take),
                "risk": float(cumulative_errors[take - 1] / take),
                "confidence_threshold": float(confidence[order][take - 1]),
            }
        )
    return rows


def area_under_risk_coverage(curve: Sequence[Dict[str, float]]) -> float:
    """AURC: one number for the whole curve. Lower is better.

    Trapezoidal integration of risk over coverage. Useful for comparing an
    uncalibrated model against a calibrated one in a single cell of a table --
    though note that a monotone calibrator cannot change it, since it cannot
    change the ranking.
    """
    if len(curve) < 2:
        return 0.0
    coverage = np.array([row["coverage"] for row in curve])
    risk = np.array([row["risk"] for row in curve])
    return float(np.trapezoid(risk, coverage) / (coverage[-1] - coverage[0]))


def format_coverage_table(rows: Sequence[Dict[str, float]]) -> str:
    lines = [
        "  alpha   target    empirical   +/-noise   abstain   empty    flag     mean|C|   sel.acc",
        "  " + "-" * 88,
    ]
    for row in rows:
        threshold = row["threshold"]
        note = "  (no finite threshold)" if math.isinf(threshold) else ""
        noise = coverage_tolerance(
            row["alpha"], int(row.get("n_calibration", 0)), int(row.get("n_test", 0))
        )
        lines.append(
            f"  {row['alpha']:<7.2f} {row['target_coverage']:<9.3f} "
            f"{row['empirical_coverage']:<11.4f} {noise:<10.4f} "
            f"{row['abstention_rate']:<9.4f} {row['empty_set_rate']:<8.4f} "
            f"{row['flag_rate']:<8.4f} {row['mean_set_size']:<9.3f} "
            f"{row['selective_accuracy']:.4f}{note}"
        )
    lines.append(
        "\n  +/-noise is the 3-sigma band a single calibration draw can fall below"
    )
    lines.append(
        "  target by and still be correct. Coverage is guaranteed in expectation"
    )
    lines.append("  over calibration sets, not for every individual draw.")
    return "\n".join(lines)


def coverage_tolerance(alpha: float, n_calibration: int, n_test: int, sigmas: float = 3.0) -> float:
    """How far below target a single run may legitimately land, in coverage units.

    The guarantee is **marginal**: it holds in expectation over the draw of the
    calibration set, not for every draw. One calibration set is one sample, so
    roughly half of all honest runs come out below 1 - alpha. Measured over 200
    synthetic trials, the fraction below target sat at 0.475-0.525 for every
    alpha tested, exactly as the theory says.

    A fixed tolerance is therefore the wrong test. It cries wolf on small
    calibration sets and waves through real breakage on large ones. The
    sampling standard deviation has two independent parts:

        calibration draw   the conditional coverage is Beta-distributed with
                           sd approximately sqrt(a(1-a) / n_calibration)
        test draw          binomial, sd approximately sqrt(a(1-a) / n_test)

    where a = alpha. This returns `sigmas` times their combination.
    """
    variance = alpha * (1.0 - alpha) * (1.0 / max(n_calibration, 1) + 1.0 / max(n_test, 1))
    return sigmas * math.sqrt(variance)


def check_coverage(
    rows: Sequence[Dict[str, float]],
    sigmas: float = 3.0,
    floor: float = 0.005,
) -> List[str]:
    """Rows whose coverage shortfall is too large to be sampling noise.

    The Aug 17 trigger point in the plan is exactly this check. If it returns
    anything, stop and debug the maths before doing anything else -- an
    under-covering conformal layer is not a weak result, it is a wrong one.

    `floor` keeps a very large test set from flagging a shortfall so small it
    could not matter to anyone.
    """
    problems: List[str] = []
    for row in rows:
        shortfall = row["target_coverage"] - row["empirical_coverage"]
        allowed = max(
            floor,
            coverage_tolerance(
                row["alpha"],
                int(row.get("n_calibration", 0)),
                int(row.get("n_test", 0)),
                sigmas,
            ),
        )
        if shortfall > allowed:
            problems.append(
                f"alpha={row['alpha']:.2f}: empirical coverage "
                f"{row['empirical_coverage']:.4f} is {shortfall:.4f} below the "
                f"{row['target_coverage']:.3f} target, which exceeds the "
                f"{allowed:.4f} that sampling noise explains "
                f"({sigmas:.0f} sigma)"
            )
    return problems
