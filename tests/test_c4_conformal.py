"""Layer 6: coverage inside C2's own band on exchangeable data, and set-size monotonicity."""

from __future__ import annotations

import numpy as np
import pytest

from src.c2_calibration.conformal import coverage_tolerance
from src.c4_attribution import evidence_conformal as cf


def _synthetic(n=600, seed=0, pool_low=4, pool_high=30):
    """Exchangeable rows with per-instance pool sizes, which is the whole question."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        size = int(rng.integers(pool_low, pool_high))
        gold = int(rng.integers(0, size))
        scores = rng.random(size)
        scores[gold] += 1.2  # the gold sentence is usually, not always, on top
        rows.append(
            cf.ConformalRow(
                response_id=f"r{i // 3}",
                task="qa",
                probabilities=cf.normalise_scores(scores),
                gold_index=gold,
            )
        )
    return rows


def test_normalise_produces_a_distribution():
    probabilities = cf.normalise_scores([0.2, 0.5, 0.3])
    assert sum(probabilities) == pytest.approx(1.0)
    assert all(p >= 0 for p in probabilities)


def test_negative_scores_are_shifted_not_clipped_away():
    probabilities = cf.normalise_scores([-0.5, 0.0, 0.5])
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[2] > probabilities[0]


def test_an_all_zero_context_becomes_uniform_which_is_the_honest_statement():
    probabilities = cf.normalise_scores([0.0, 0.0, 0.0, 0.0])
    assert probabilities == pytest.approx((0.25, 0.25, 0.25, 0.25))


def test_coverage_lands_inside_c2s_own_band_on_exchangeable_data():
    calib = _synthetic(600, seed=1)
    test = _synthetic(600, seed=2)
    for alpha in (0.05, 0.10, 0.20):
        block = cf.evaluate(calib, test, alpha, with_intervals=False)
        band = coverage_tolerance(alpha, block["n_calibration"], block["n_test"])
        assert block["empirical_coverage"] >= (1 - alpha) - band
        assert block["band"] == pytest.approx(band)


def test_the_guarantee_survives_pools_of_very_different_size():
    # The panel question: 4 sentences here, 104 there. Validity needs only the
    # gold label's non-conformity to be exchangeable, not equal cardinality.
    calib = _synthetic(500, seed=3, pool_low=3, pool_high=8)
    test = _synthetic(500, seed=4, pool_low=3, pool_high=8)
    mixed = _synthetic(500, seed=5, pool_low=3, pool_high=100)
    tight = cf.evaluate(calib, test, 0.10, with_intervals=False)
    wide = cf.evaluate(
        _synthetic(500, seed=6, pool_low=3, pool_high=100), mixed, 0.10, with_intervals=False
    )
    assert tight["empirical_coverage"] >= 0.90 - tight["band"]
    assert wide["empirical_coverage"] >= 0.90 - wide["band"]
    # And the set size adapts to the pool rather than being fixed.
    assert wide["mean_set_size"] > tight["mean_set_size"]


def test_set_size_is_non_increasing_in_alpha():
    calib = _synthetic(400, seed=7)
    test = _synthetic(400, seed=8)
    sizes = [
        cf.evaluate(calib, test, alpha, with_intervals=False)["mean_set_size"]
        for alpha in (0.05, 0.10, 0.20, 0.30, 0.40)
    ]
    assert sizes == sorted(sizes, reverse=True)


def test_a_calibration_set_too_small_for_this_alpha_returns_every_sentence():
    calib = _synthetic(5, seed=9)
    threshold = cf.fit(calib, 0.01)
    assert threshold == float("inf")
    row = calib[0]
    assert len(cf.prediction_set(row, threshold)) == len(row.probabilities)


def test_the_empty_set_case_is_representable():
    row = cf.ConformalRow("r", "qa", (0.5, 0.5), 0)
    assert cf.prediction_set(row, -1.0) == ()


def test_rows_without_a_gold_sentence_are_excluded_not_counted_as_misses():
    calib = _synthetic(300, seed=10)
    test = _synthetic(300, seed=11) + [
        cf.ConformalRow("rx", "qa", (0.3, 0.7), None) for _ in range(50)
    ]
    block = cf.evaluate(calib, test, 0.10, with_intervals=False)
    assert block["n_test"] == 300


def test_risk_control_returns_a_real_answer_when_the_level_is_unreachable():
    calib = _synthetic(20, seed=12)
    test = _synthetic(20, seed=13)
    result = cf.risk_control_cross_check(calib, test, alpha=0.001)
    assert result["calibration"]["feasible"] is False
    assert "reason" in result["calibration"]
    comparison = cf.compare_formalisms(
        cf.evaluate(calib, test, 0.001, with_intervals=False), result
    )
    assert comparison["agree"] is None


def test_the_two_formalisms_are_compared_rather_than_one_being_dropped():
    calib = _synthetic(500, seed=14)
    test = _synthetic(500, seed=15)
    lac = cf.evaluate(calib, test, 0.10, with_intervals=False)
    crc = cf.risk_control_cross_check(calib, test, 0.10)
    comparison = cf.compare_formalisms(lac, crc)
    assert "lac_coverage" in comparison or comparison["agree"] is None


def test_set_shape_separates_adjacent_pairs_from_distant_ones():
    calib = _synthetic(400, seed=16)
    test = _synthetic(400, seed=17)
    shape = cf.set_shape(calib, test, 0.10)
    assert shape["n_test"] == 400
    assert set(shape).issuperset({"adjacent", "mean_index_distance", "size_histogram"})


def test_the_per_task_prediction_is_written_down_before_the_measurement():
    assert "Written before running" in cf.PREDICTION_PER_TASK
    calib = _synthetic(300, seed=18)
    test = _synthetic(300, seed=19)
    block = cf.per_task(calib, test, 0.10)
    assert block["prediction"] == cf.PREDICTION_PER_TASK
    assert "verdict" in block
