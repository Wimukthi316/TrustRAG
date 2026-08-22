"""Tests for C2: calibration metrics, calibrators, and split conformal.

The conformal tests matter more than the rest of this repository's tests put
together. The project's central claim is a coverage guarantee, and a guarantee
that is quietly off by a fencepost is worse than no guarantee at all -- it is a
wrong number presented with confidence.

So the quantile is checked against the formula by hand, and coverage is checked
empirically on synthetic exchangeable data, including the case that makes
conformal prediction interesting: a deliberately miscalibrated model, where
coverage must still hold.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from src.c2_calibration.calibration import (  # noqa: E402
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    brier_score,
    calibration_report,
    compare_calibrators,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    to_logit,
)
from src.c2_calibration.conformal import (  # noqa: E402
    SplitConformal,
    area_under_risk_coverage,
    check_coverage,
    conformal_quantile,
    coverage_table,
    coverage_tolerance,
    group_conditional_coverage,
    minimum_calibration_size,
    risk_coverage_curve,
)
from src.c2_calibration.run_c2 import span_units, token_units  # noqa: E402
from src.common.schema import ConformalDecision  # noqa: E402


# --------------------------------------------------------------------------
# Calibration metrics
# --------------------------------------------------------------------------


def test_ece_is_zero_for_a_perfectly_calibrated_set():
    # Two bins. In each, the mean score equals the observed positive rate.
    probs = [0.2] * 10 + [0.8] * 10
    labels = [1] * 2 + [0] * 8 + [1] * 8 + [0] * 2
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.0)


def test_ece_matches_a_hand_computed_value():
    # One bin, 4 points, mean score 0.9, observed rate 0.5 -> gap 0.4.
    probs = [0.9, 0.9, 0.9, 0.9]
    labels = [1, 1, 0, 0]
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.4)
    assert maximum_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.4)


def test_ece_weights_bins_by_count():
    # 90 points with no gap, 10 points with a 0.5 gap -> 0.1 * 0.5 = 0.05.
    probs = [0.05] * 90 + [0.95] * 10
    labels = [0] * 90 + [1] * 5 + [0] * 5
    # bin 0: mean 0.05, rate 0.0  -> gap 0.05, count 90
    # bin 9: mean 0.95, rate 0.5  -> gap 0.45, count 10
    expected = (90 * 0.05 + 10 * 0.45) / 100
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(expected)


def test_brier_matches_hand_computation():
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier_score([0.5, 0.5], [1, 0]) == pytest.approx(0.25)


def test_reliability_bins_include_a_score_of_exactly_one():
    """The last bin must be closed or a saturated 1.0 vanishes from the metric."""
    bins = reliability_bins([1.0, 1.0], [1, 0], n_bins=10)
    assert sum(b.count for b in bins) == 2


def test_calibration_report_covers_the_empty_case():
    report = calibration_report([], [])
    assert report["n"] == 0 and report["ece"] == 0.0


def test_to_logit_clips_saturated_probabilities():
    assert math.isfinite(float(to_logit([0.0])[0]))
    assert math.isfinite(float(to_logit([1.0])[0]))


# --------------------------------------------------------------------------
# Calibrators
# --------------------------------------------------------------------------


def _overconfident(n=4000, seed=0):
    """A model that pushes scores toward 0 and 1 harder than reality supports."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.05, 0.95, size=n)
    labels = (rng.uniform(size=n) < true_p).astype(int)
    # Sharpen the logit by 3x: the classic overconfidence failure mode.
    scores = 1.0 / (1.0 + np.exp(-3.0 * to_logit(true_p)))
    return list(scores), list(labels)


def test_temperature_scaling_finds_a_temperature_above_one_when_overconfident():
    scores, labels = _overconfident()
    calibrator = TemperatureCalibrator().fit(scores, labels)
    assert calibrator.temperature > 1.5, (
        "an overconfident model must be softened, so T should be well above 1"
    )


def test_temperature_scaling_reduces_ece():
    scores, labels = _overconfident()
    before = expected_calibration_error(scores, labels)
    after = expected_calibration_error(
        TemperatureCalibrator().fit(scores, labels).transform(scores), labels
    )
    assert after < before / 2, f"ECE {before:.4f} -> {after:.4f} is not much of a fix"


def test_temperature_scaling_cannot_reorder_predictions():
    """Monotone by construction. This is why it leaves AUC untouched."""
    scores, labels = _overconfident(n=500)
    transformed = TemperatureCalibrator().fit(scores, labels).transform(scores)
    assert list(np.argsort(scores)) == list(np.argsort(transformed))


def test_platt_and_isotonic_stay_in_range_and_reduce_ece():
    scores, labels = _overconfident()
    for calibrator in (PlattCalibrator(), IsotonicCalibrator()):
        transformed = calibrator.fit(scores, labels).transform(scores)
        assert transformed.min() >= 0.0 and transformed.max() <= 1.0
        assert expected_calibration_error(transformed, labels) < expected_calibration_error(
            scores, labels
        )


def test_platt_survives_a_single_class_calibration_set():
    calibrator = PlattCalibrator().fit([0.2, 0.4, 0.6], [0, 0, 0])
    assert calibrator.slope == 1.0 and calibrator.intercept == 0.0


def test_compare_calibrators_reports_every_method_on_test_data():
    scores, labels = _overconfident(n=2000)
    rows = compare_calibrators(scores[:1000], labels[:1000], scores[1000:], labels[1000:])
    assert {r["method"] for r in rows} == {
        "uncalibrated",
        "temperature",
        "platt",
        "isotonic",
    }
    uncalibrated = next(r for r in rows if r["method"] == "uncalibrated")
    temperature = next(r for r in rows if r["method"] == "temperature")
    assert temperature["ece"] < uncalibrated["ece"]


# --------------------------------------------------------------------------
# Conformal: the formula
# --------------------------------------------------------------------------


def test_conformal_quantile_is_the_kth_smallest_by_the_formula():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # n = 10
    # k = ceil((10 + 1) * 0.9) = ceil(9.9) = 10 -> the 10th smallest
    assert conformal_quantile(scores, 0.10) == pytest.approx(1.0)
    # k = ceil(11 * 0.8) = ceil(8.8) = 9 -> the 9th smallest
    assert conformal_quantile(scores, 0.20) == pytest.approx(0.9)
    # k = ceil(11 * 0.5) = 6 -> the 6th smallest
    assert conformal_quantile(scores, 0.50) == pytest.approx(0.6)


def test_conformal_quantile_is_infinite_when_calibration_is_too_small():
    """Nine points cannot support alpha = 0.05; saying so beats inventing a number."""
    assert math.isinf(conformal_quantile([0.1] * 9, 0.05))
    assert math.isfinite(conformal_quantile([0.1] * 19, 0.05))


def test_minimum_calibration_size_matches_when_the_quantile_becomes_finite():
    for alpha in (0.05, 0.1, 0.2, 0.33):
        n = minimum_calibration_size(alpha)
        assert math.isinf(conformal_quantile([0.5] * (n - 1), alpha))
        assert math.isfinite(conformal_quantile([0.5] * n, alpha))


def test_conformal_quantile_rejects_alpha_outside_the_open_unit_interval():
    for alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            conformal_quantile([0.1, 0.2], alpha)


# --------------------------------------------------------------------------
# Conformal: the guarantee
# --------------------------------------------------------------------------


def _exchangeable(n, seed, distort=None):
    """Synthetic exchangeable data. `distort` maps the true probability to a score."""
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.0, 1.0, size=n)
    labels = (rng.uniform(size=n) < true_p).astype(int)
    scores = true_p if distort is None else distort(true_p)
    return list(scores), list(labels)


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20, 0.30])
def test_coverage_holds_on_average_over_many_calibration_draws(alpha):
    """The correct statement of the guarantee, and the correct way to test it.

    Coverage is guaranteed **in expectation over the draw of the calibration
    set**, not for every individual draw. Asserting that one run clears
    1 - alpha is wrong and will fail roughly half the time: measured over 200
    trials the fraction below target sits at about 0.5 for every alpha. So the
    assertion is on the mean.
    """
    coverages = []
    for trial in range(60):
        calib = _exchangeable(1500, seed=500 + trial * 2)
        test = _exchangeable(1500, seed=501 + trial * 2)
        coverages.append(
            SplitConformal(alpha=alpha).fit(*calib).evaluate(*test)["empirical_coverage"]
        )
    mean = float(np.mean(coverages))
    # Standard error of the mean over 60 trials is well under 0.003 here.
    assert mean >= (1 - alpha) - 0.01, f"mean coverage {mean:.4f} for alpha {alpha}"
    assert mean <= (1 - alpha) + 0.03, "coverage far above target means it is too conservative"


@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_coverage_still_holds_for_a_badly_miscalibrated_model(alpha):
    """The property that makes conformal prediction worth the trouble.

    The scores here are squared, so they are systematically far too low and the
    model is badly miscalibrated. Coverage must hold anyway -- the guarantee is
    distribution-free and says nothing at all about model quality.
    """
    coverages = []
    for trial in range(40):
        calib = _exchangeable(1500, seed=700 + trial * 2, distort=lambda p: p**2)
        test = _exchangeable(1500, seed=701 + trial * 2, distort=lambda p: p**2)
        coverages.append(
            SplitConformal(alpha=alpha).fit(*calib).evaluate(*test)["empirical_coverage"]
        )
    assert float(np.mean(coverages)) >= (1 - alpha) - 0.01


def test_lac_is_invariant_to_strictly_monotone_score_transforms():
    """Recalibrating the scores does not move the conformal result at all.

    LAC thresholds the score, so a strictly monotone transform moves the
    threshold by the same map and leaves the partition of examples identical.
    Temperature and Platt scaling are exactly such transforms.

    This matters for how the report frames C2, and it is easy to get wrong:
    calibration is NOT what makes the conformal layer efficient. Calibration
    makes the number a human reads mean what it says; conformal supplies the
    guarantee. They are complementary, not sequential.
    """
    calib_scores, calib_labels = _exchangeable(3000, seed=5)
    test_scores, test_labels = _exchangeable(3000, seed=6)

    def squash(values):
        # Strictly increasing, and a drastic change in the actual numbers.
        return [0.5 + 0.02 * (v - 0.5) for v in values]

    plain = SplitConformal(0.1).fit(calib_scores, calib_labels).evaluate(
        test_scores, test_labels
    )
    squashed = SplitConformal(0.1).fit(squash(calib_scores), calib_labels).evaluate(
        squash(test_scores), test_labels
    )

    assert plain["threshold"] != pytest.approx(squashed["threshold"]), (
        "the fixture should have actually changed the scores"
    )
    for key in ("empirical_coverage", "abstention_rate", "flag_rate", "mean_set_size"):
        assert plain[key] == pytest.approx(squashed[key]), f"{key} moved"


def test_an_uninformative_model_abstains_far_more_than_a_good_one():
    """Conformal charges a bad detector in abstentions, not in coverage.

    'Bad' here means genuinely uninformative -- the score carries no signal
    about the label -- rather than merely miscalibrated, because a monotone
    miscalibration changes nothing (see the invariance test above).
    """
    rng = np.random.default_rng(21)
    good_calib, good_labels = _exchangeable(4000, seed=5)
    good_test, good_test_labels = _exchangeable(4000, seed=6)
    noise_calib = list(rng.uniform(size=4000))
    noise_test = list(rng.uniform(size=4000))

    good = SplitConformal(0.1).fit(good_calib, good_labels).evaluate(good_test, good_test_labels)
    bad = SplitConformal(0.1).fit(noise_calib, good_labels).evaluate(
        noise_test, good_test_labels
    )

    assert good["empirical_coverage"] >= 0.87 and bad["empirical_coverage"] >= 0.87
    assert bad["abstention_rate"] > good["abstention_rate"]


def test_prediction_sets_map_onto_the_schema_decisions():
    conformal = SplitConformal(alpha=0.1)
    conformal.threshold = 0.3
    conformal.n_calibration = 100
    # p = 0.95 -> 1-p = 0.05 <= 0.3 keeps label 1; p = 0.95 > 0.3 drops label 0.
    decisions = conformal.decisions([0.95, 0.05, 0.5])
    assert decisions[0] is ConformalDecision.FLAG
    assert decisions[1] is ConformalDecision.PASS
    assert decisions[2] is ConformalDecision.ABSTAIN


def test_an_infinite_threshold_covers_everything_and_abstains_everywhere():
    conformal = SplitConformal(alpha=0.05).fit([0.5] * 5, [0, 1, 0, 1, 0])
    assert math.isinf(conformal.threshold)
    result = conformal.evaluate([0.2, 0.8], [0, 1])
    assert result["empirical_coverage"] == 1.0
    assert result["abstention_rate"] == 1.0
    assert result["mean_set_size"] == 2.0


def test_coverage_table_and_check_coverage_agree_on_a_healthy_run():
    calib_scores, calib_labels = _exchangeable(4000, seed=7)
    test_scores, test_labels = _exchangeable(4000, seed=8)
    rows = coverage_table(calib_scores, calib_labels, test_scores, test_labels)
    assert len(rows) == 6
    # This exact seed pair lands 0.0247 below target at alpha=0.30 -- about
    # 1.7 sigma, which is ordinary sampling noise and must not be flagged.
    assert check_coverage(rows) == []


def test_check_coverage_reports_a_real_shortfall():
    rows = [
        {
            "alpha": 0.1,
            "target_coverage": 0.9,
            "empirical_coverage": 0.80,
            "n_calibration": 4000,
            "n_test": 4000,
        }
    ]
    problems = check_coverage(rows)
    assert len(problems) == 1 and "below" in problems[0]


def test_coverage_tolerance_shrinks_as_the_sample_grows():
    """A fixed tolerance is wrong in both directions; this is why it is computed."""
    small = coverage_tolerance(0.1, 200, 200)
    large = coverage_tolerance(0.1, 40000, 40000)
    assert small > large
    # A 4 point shortfall is noise at n=200 and a real failure at n=40,000.
    row = {"alpha": 0.1, "target_coverage": 0.9, "empirical_coverage": 0.86}
    assert check_coverage([{**row, "n_calibration": 200, "n_test": 200}]) == []
    assert check_coverage([{**row, "n_calibration": 40000, "n_test": 40000}])


def test_group_conditional_coverage_holds_within_every_group():
    # The seeds here were `seed + hash(group) % 100`, and Python randomises
    # string hashing per process, so this test drew a different dataset on every
    # run and its outcome was a coin toss. Measured over 300 fixed-seed repeats
    # of this exact scenario, coverage on the qa group has mean 0.8992 and sd
    # 0.0119 -- correct -- but a heavy left tail reaching 0.7665, and it lands
    # below the 3-sigma band about 0.3% of the time. That tail is real and is
    # worth understanding rather than seeding away: qa's scores sit in [0, 0.3],
    # so its non-conformity scores are bimodal -- a dense cluster near zero from
    # the negatives and a sparse one above 0.7 from the positives -- and the
    # 0.9 quantile falls in the empty region between them, where moving one
    # calibration point moves the threshold a long way.
    #
    # So: fixed seeds, chosen once and never swept, and an assertion against the
    # noise band the project uses everywhere else rather than a bare 0.86.
    rng = np.random.default_rng(11)

    def build(n, seed):
        scores, labels, groups = [], [], []
        for offset, (group, low, high) in enumerate(
            (("qa", 0.0, 0.3), ("data2text", 0.3, 0.9))
        ):
            probs = np.random.default_rng(seed + offset).uniform(low, high, n)
            outcomes = (rng.uniform(size=n) < probs).astype(int)
            scores.extend(probs)
            labels.extend(outcomes)
            groups.extend([group] * n)
        return scores, labels, groups

    calib = build(2000, 101)
    test = build(2000, 201)
    result = group_conditional_coverage(*calib, *test, alpha=0.1)
    assert set(result) == {"qa", "data2text"}
    for group, row in result.items():
        floor = 0.9 - coverage_tolerance(
            0.1, int(row["n_calibration"]), int(row["n_test"])
        )
        assert row["empirical_coverage"] >= floor, (
            f"{group} covered {row['empirical_coverage']:.4f}, below the "
            f"{floor:.4f} floor that sampling noise explains"
        )


# --------------------------------------------------------------------------
# Risk-coverage
# --------------------------------------------------------------------------


def test_risk_falls_as_coverage_falls_for_an_informative_score():
    scores, labels = _exchangeable(3000, seed=9)
    curve = risk_coverage_curve(scores, labels)
    assert curve[0]["risk"] < curve[-1]["risk"], (
        "the most confident predictions should be the most accurate"
    )
    assert curve[-1]["coverage"] == pytest.approx(1.0)


def test_aurc_is_lower_for_a_better_ranking():
    scores, labels = _exchangeable(3000, seed=10)
    informative = area_under_risk_coverage(risk_coverage_curve(scores, labels))
    shuffled = list(np.random.default_rng(0).permutation(scores))
    uninformative = area_under_risk_coverage(risk_coverage_curve(shuffled, labels))
    assert informative < uninformative


# --------------------------------------------------------------------------
# Unit extraction from probabilities.jsonl
# --------------------------------------------------------------------------


def _probability_record():
    return {
        "id": "1",
        "task_type": "qa",
        "answer": "abcdef",
        "gold_spans": [{"start": 0, "end": 3, "text": "abc"}],
        "pred_spans": [
            {"start": 0, "end": 3, "mean_prob": 0.8, "max_prob": 0.9, "is_hallucinated": True},
            {"start": 4, "end": 6, "mean_prob": 0.6, "max_prob": 0.7, "is_hallucinated": False},
        ],
        "token_probs": [0.9, 0.8, 0.7, 0.1, 0.6, 0.7],
        "answer_offsets": [[i, i + 1] for i in range(6)],
    }


def test_span_units_uses_the_requested_score_key():
    scores, labels, groups = span_units([_probability_record()], "mean_prob")
    assert scores == [0.8, 0.6] and labels == [1, 0] and groups == ["qa", "qa"]
    scores, _, _ = span_units([_probability_record()], "max_prob")
    assert scores == [0.9, 0.7]


def test_token_units_relabels_from_gold_spans_and_offsets():
    scores, labels, groups = token_units([_probability_record()])
    assert scores == [0.9, 0.8, 0.7, 0.1, 0.6, 0.7]
    # Gold span covers characters 0..2, so the first three tokens are positive.
    assert labels == [1, 1, 1, 0, 0, 0]
    assert set(groups) == {"qa"}
