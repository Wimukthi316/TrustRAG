"""Tests for online (adaptive) conformal prediction.

The central claim these pin down is the one the report leans on: the adaptive
loop drives the realized long-run miscoverage to the target on ANY stream,
including a shifted one, which is exactly the property split conformal loses off
distribution. That is not an empirical accident here -- it follows from the
update telescoping -- so the tolerance below is derived, not tuned.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.c2_calibration.conformal import SplitConformal, conformal_quantile
from src.c2_calibration.online_conformal import (
    AdaptiveConformal,
    _effective_threshold,
    _prediction_set,
    nonconformity,
    run_stream,
)


def _synthetic_stream(n: int, positive_rate: float, seed: int):
    """Scores and labels with a deliberate mismatch to the calibration set."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < positive_rate).astype(int)
    # Positives score high, negatives low, with overlap -- an over-confident
    # detector on a corpus whose base rate differs from calibration.
    scores = np.where(
        labels == 1,
        rng.beta(6, 2, n),
        rng.beta(3, 3, n),  # negatives score higher than they should: a shift
    )
    return scores.astype(float), labels


def _uniform_calibration(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.random(n))


def test_nonconformity_matches_lac_definition():
    assert nonconformity(0.8, 1) == pytest.approx(0.2)
    assert nonconformity(0.8, 0) == pytest.approx(0.8)


def test_effective_threshold_edges():
    calib = _uniform_calibration(500, 0)
    # At or past 0 the loop covers everything; at or past 1 it covers nothing.
    assert _effective_threshold(calib, 0.0) == float("inf")
    assert _effective_threshold(calib, -0.3) == float("inf")
    assert _effective_threshold(calib, 1.0) == float("-inf")
    # In between it is the ordinary conformal quantile.
    assert _effective_threshold(calib, 0.2) == conformal_quantile(calib, 0.2)


def test_prediction_set_and_coverage():
    # threshold 0.5: a score of 0.9 keeps only label 1 (1 - 0.9 = 0.1 <= 0.5,
    # 0.9 > 0.5), a score of 0.1 keeps only label 0.
    assert _prediction_set(0.9, 0.5) == (1,)
    assert _prediction_set(0.1, 0.5) == (0,)
    # A very high threshold keeps both.
    assert _prediction_set(0.5, 0.9) == (0, 1)


def test_gamma_zero_reduces_to_the_static_method():
    calib = _uniform_calibration(1000, 1)
    scores, labels = _synthetic_stream(3000, 0.3, 2)
    out = run_stream(calib, scores, labels, alpha_target=0.1, gamma=0.0, shuffle=False)
    # With no learning rate the dial never moves, so the adaptive coverage must
    # equal the static coverage exactly.
    assert out["final_alpha_t"] == pytest.approx(0.1)
    assert out["long_run_coverage_adaptive"] == pytest.approx(
        out["long_run_coverage_static"]
    )


def test_alpha_t_stays_in_unit_interval():
    calib = _uniform_calibration(500, 3)
    scores, labels = _synthetic_stream(4000, 0.05, 4)
    aci = AdaptiveConformal(calib, alpha_target=0.1, gamma=0.1)
    for p, y in zip(scores, labels):
        aci.step(float(p), int(y))
        assert 0.0 <= aci.alpha_t <= 1.0


@pytest.mark.parametrize("target,gamma", [(0.90, 0.05), (0.80, 0.05), (0.95, 0.02)])
def test_long_run_coverage_converges_under_shift(target, gamma):
    """Realized coverage lands on target, and derivably so.

    alpha_{T+1} - alpha_1 = gamma * sum_t (alpha_target - err_t), and alpha_t is
    confined to [0, 1], so |mean(err_t) - alpha_target| <= 1 / (gamma * T). With
    T = 6000 and these gammas that bound is well under 0.01; the assertion allows
    0.02 for the finite-sample slack in the coverage itself.
    """
    alpha = 1.0 - target
    calib = _uniform_calibration(1000, 5)
    scores, labels = _synthetic_stream(6000, 0.08, 6)
    out = run_stream(calib, scores, labels, alpha_target=alpha, gamma=gamma, shuffle=True)

    bound = 1.0 / (gamma * out["n_stream"]) + 0.02
    assert abs(out["long_run_coverage_adaptive"] - target) <= bound
    assert out["meets_target_adaptive"] in (True, False)  # field exists and is bool


def test_adaptive_beats_static_when_static_undercovers():
    # A deterministic shift the static method cannot cover: almost every span is
    # a true negative scored high (an over-confident detector off distribution),
    # so the fixed threshold hands most of them {1} and excludes the truth. This
    # is the RAGBench failure in miniature, built rather than drawn so the test
    # cannot flake into a stream that happens to be in distribution.
    calib = _uniform_calibration(1000, 7)  # threshold at alpha 0.1 sits near 0.9
    n = 6000
    labels = np.zeros(n, dtype=int)
    labels[:: 10] = 1  # 10% genuinely hallucinated
    scores = np.full(n, 0.95, dtype=float)  # detector says "hallucinated" for all

    out = run_stream(calib, scores, labels, alpha_target=0.1, gamma=0.05, shuffle=True)

    # Static under-covers badly: it only covers the ~10% that really are positive.
    assert out["long_run_coverage_static"] < 0.90
    static_gap = 0.90 - out["long_run_coverage_static"]
    adaptive_gap = abs(0.90 - out["long_run_coverage_adaptive"])
    assert adaptive_gap < static_gap


def test_static_path_matches_split_conformal():
    """run_stream's static baseline is the same arithmetic as SplitConformal."""
    calib_scores = _uniform_calibration(800, 9)
    # Turn calibration nonconformity back into (prob, label) is not needed: feed
    # the static path the same nonconformity array via a fitted SplitConformal and
    # compare coverage on one stream.
    scores, labels = _synthetic_stream(2000, 0.2, 10)

    # Build a SplitConformal whose calibration nonconformity equals calib_scores
    # by calibrating on label-0 points with prob == score (nonconformity == p).
    sc = SplitConformal(alpha=0.1)
    sc.fit(calib_scores, np.zeros_like(calib_scores, dtype=int))
    reference = sc.evaluate(scores, labels)["empirical_coverage"]

    out = run_stream(calib_scores, scores, labels, alpha_target=0.1, gamma=0.0, shuffle=False)
    assert out["long_run_coverage_static"] == pytest.approx(reference)


def test_determinism():
    calib = _uniform_calibration(500, 11)
    scores, labels = _synthetic_stream(2000, 0.1, 12)
    a = run_stream(calib, scores, labels, gamma=0.03, shuffle=True, seed=42)
    b = run_stream(calib, scores, labels, gamma=0.03, shuffle=True, seed=42)
    assert a["long_run_coverage_adaptive"] == b["long_run_coverage_adaptive"]
    assert a["alpha_path"] == b["alpha_path"]


def test_review_only_updates_only_on_non_pass():
    calib = _uniform_calibration(500, 13)
    # A stream the static method mostly PASSes: low scores, low positive rate.
    rng = np.random.default_rng(14)
    n = 2000
    labels = (rng.random(n) < 0.02).astype(int)
    scores = rng.beta(1.2, 6, n).astype(float)  # nearly all low -> PASS
    full = run_stream(calib, scores, labels, gamma=0.05, shuffle=False, review_only=False)
    review = run_stream(calib, scores, labels, gamma=0.05, shuffle=False, review_only=True)
    # Review-only skips feedback on PASS spans, so it reviews strictly fewer of
    # them and its dial ends somewhere different from the full loop.
    assert review["review_fraction"] < full["review_fraction"]
