"""Tests for C3's calibrated conformal type sets.

The claim this layer makes is a coverage guarantee, so the test that matters is
the guarantee itself: on data generated to be exchangeable, empirical coverage
must sit inside the same 3-sigma band C2 judges its own coverage against. That
band is imported from C2 rather than written here, so the two components cannot
drift into judging coverage by different rules.

Nothing asserts a stochastic quantity against a hand-picked constant. Where a
number comes from a draw it is checked against the band, and where it is
arithmetic it is hand-computed.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.c2_calibration.conformal import coverage_tolerance
from src.c3_explanation.type_conformal import (
    analyse,
    base_rate_floor,
    coverage_row,
    fit_temperature,
    format_report,
    main,
    multiclass_brier,
    multiclass_nll,
    nonconformity,
    per_class_coverage,
    prediction_sets,
    set_shape,
    temperature_scale,
    top_label_report,
    unpack,
)

CLASSES = [
    "evident_conflict",
    "subtle_conflict",
    "evident_baseless_info",
    "subtle_baseless_info",
]


# --------------------------------------------------------------------------
# Calibration arithmetic
# --------------------------------------------------------------------------


def test_temperature_one_changes_nothing():
    probs = np.asarray([[0.7, 0.2, 0.05, 0.05], [0.25, 0.25, 0.25, 0.25]])
    assert np.allclose(temperature_scale(probs, 1.0), probs)


def test_temperature_above_one_softens_and_below_one_sharpens():
    probs = np.asarray([[0.7, 0.2, 0.05, 0.05]])
    assert temperature_scale(probs, 3.0).max() < probs.max()
    assert temperature_scale(probs, 0.5).max() > probs.max()


def test_temperature_never_changes_which_class_wins():
    """Calibration adjusts what a number means, not the ranking. C2 says so."""
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(4), size=200)
    for temperature in (0.3, 0.8, 2.5, 7.0):
        assert np.array_equal(
            temperature_scale(probs, temperature).argmax(axis=1), probs.argmax(axis=1)
        )


def test_a_non_positive_temperature_is_a_loud_error():
    with pytest.raises(ValueError):
        temperature_scale(np.asarray([[0.5, 0.5]]), 0.0)


def test_nll_and_brier_are_hand_computable():
    probs = np.asarray([[0.5, 0.5], [0.25, 0.75]])
    truth = np.asarray([0, 1])
    assert multiclass_nll(probs, truth) == pytest.approx(
        -(math.log(0.5) + math.log(0.75)) / 2
    )
    # row 1: (0.5-1)^2 + (0.5-0)^2 = 0.5 ; row 2: (0.25)^2 + (0.75-1)^2 = 0.125
    assert multiclass_brier(probs, truth) == pytest.approx((0.5 + 0.125) / 2)


def test_fit_temperature_softens_an_overconfident_model():
    """Overconfident means confident BEYOND its accuracy, not merely confident.

    A model that is always right should be confident, and the fitted temperature
    should sharpen it. The interesting case is a model that says 0.99 and is
    right seven times in ten.
    """
    rng = np.random.default_rng(1)
    n = 800
    truth = rng.integers(0, 4, size=n)
    guess = np.where(
        rng.random(n) < 0.7, truth, (truth + 1 + rng.integers(0, 3, size=n)) % 4
    )
    honest = np.full((n, 4), 0.1)
    honest[np.arange(n), guess] = 0.7

    assert fit_temperature(temperature_scale(honest, 0.2), truth) > 1.0
    assert fit_temperature(temperature_scale(honest, 5.0), truth) < 1.0


def test_fit_temperature_is_deterministic():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(4), size=300)
    truth = rng.integers(0, 4, size=300)
    assert fit_temperature(probs, truth) == fit_temperature(probs, truth)


def test_base_rate_floor_ignores_the_input():
    calib_truth = np.asarray([0, 0, 0, 1])
    test_truth = np.asarray([0, 1])
    floor = base_rate_floor(calib_truth, (2, 2), test_truth)
    assert floor["ignores_the_input"] is True
    assert floor["base_rates"] == pytest.approx([0.75, 0.25])
    assert floor["accuracy"] == pytest.approx(0.5)


def test_top_label_report_on_an_empty_set_is_nan_not_zero():
    report = top_label_report(np.empty((0, 4)), np.empty(0, dtype=int))
    assert report["n"] == 0
    assert math.isnan(report["ece"])


# --------------------------------------------------------------------------
# Conformal
# --------------------------------------------------------------------------


def test_nonconformity_is_one_minus_the_true_class_probability():
    probs = np.asarray([[0.6, 0.4], [0.1, 0.9]])
    assert list(nonconformity(probs, np.asarray([0, 0]))) == pytest.approx([0.4, 0.9])


def test_prediction_sets_keep_every_class_that_clears_the_bar():
    probs = np.asarray([[0.6, 0.3, 0.05, 0.05]])
    assert prediction_sets(probs, 0.5) == [(0,)]
    assert prediction_sets(probs, 0.7) == [(0, 1)]
    assert prediction_sets(probs, 0.2) == [()]


@pytest.mark.parametrize(
    "members,expected",
    [
        ((), "empty"),
        ((0,), "singleton"),
        ((0, 1, 2, 3), "full"),
        ((0, 1), "pair_severity_undecided"),
        ((2, 3), "pair_severity_undecided"),
        ((0, 2), "pair_mechanism_undecided"),
        ((1, 3), "pair_mechanism_undecided"),
        ((0, 1, 2), "other"),
    ],
)
def test_set_shape_names_what_the_set_is_declining_to_decide(members, expected):
    assert set_shape(members, CLASSES) == expected


def test_coverage_row_is_hand_computable():
    calib = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    calib_truth = np.asarray([0, 0, 0, 0])
    test = np.asarray([[0.95, 0.05], [0.05, 0.95]])
    test_truth = np.asarray([0, 0])
    row = coverage_row(calib, calib_truth, test, test_truth, 0.2, ["a", "b"])
    # Nonconformity on calibration is 0.1, 0.2, 0.3, 0.4 and the (n+1) corrected
    # 80th percentile of four points is the largest of them, so the bar is 0.4:
    # a class is kept when its probability is at least 0.6.
    #   test row 1, p(a) = 0.95 -> set {a}, truth a, covered
    #   test row 2, p(b) = 0.95 -> set {b}, truth a, NOT covered
    assert row["threshold"] == pytest.approx(0.4)
    assert row["empirical_coverage"] == pytest.approx(0.5)
    assert row["average_set_size"] == pytest.approx(1.0)
    assert row["singleton_rate"] == pytest.approx(1.0)
    assert row["target_coverage"] == pytest.approx(0.8)


def test_too_few_calibration_points_widens_the_bar_to_everything():
    """Four points cannot support 95% coverage, and the quantile says so.

    conformal_quantile returns infinity rather than pretending, so every class
    is kept and coverage is trivially met at the cost of a useless set.
    """
    calib = np.asarray([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]])
    calib_truth = np.asarray([0, 0, 0, 0])
    test = np.asarray([[0.95, 0.05], [0.05, 0.95]])
    test_truth = np.asarray([0, 0])
    row = coverage_row(calib, calib_truth, test, test_truth, 0.05, ["a", "b"])
    assert math.isinf(row["threshold"])
    assert row["empirical_coverage"] == pytest.approx(1.0)
    assert row["average_set_size"] == pytest.approx(2.0)
    assert row["full_set_rate"] == pytest.approx(1.0)


def test_coverage_holds_inside_c2s_own_band_on_exchangeable_data():
    """The guarantee, tested as a guarantee.

    Calibration and test are drawn from the same generator, so exchangeability
    holds by construction and coverage must not fall further below target than
    the 3-sigma band C2 computes for that split size. No magic constant: the
    tolerance is the imported one.
    """
    rng = np.random.default_rng(7)
    n = 3000
    truth = rng.integers(0, 4, size=n)
    probs = rng.dirichlet(np.ones(4) * 0.7, size=n)
    # Give the true class a nudge so the model is informative but far from perfect.
    probs[np.arange(n), truth] += 0.35
    probs /= probs.sum(axis=1, keepdims=True)

    calib_probs, test_probs = probs[:1500], probs[1500:]
    calib_truth, test_truth = truth[:1500], truth[1500:]
    for alpha in (0.05, 0.1, 0.2):
        row = coverage_row(
            calib_probs, calib_truth, test_probs, test_truth, alpha, CLASSES
        )
        band = coverage_tolerance(alpha, len(calib_truth), len(test_truth))
        assert row["band"] == pytest.approx(band)
        assert row["empirical_coverage"] >= row["target_coverage"] - band
        assert row["inside_band"] is True


def test_a_smaller_alpha_never_buys_coverage_with_a_smaller_set():
    rng = np.random.default_rng(11)
    truth = rng.integers(0, 4, size=600)
    probs = rng.dirichlet(np.ones(4), size=600)
    rows = [
        coverage_row(probs[:300], truth[:300], probs[300:], truth[300:], a, CLASSES)
        for a in (0.05, 0.1, 0.2, 0.4)
    ]
    sizes = [r["average_set_size"] for r in rows]
    assert sizes == sorted(sizes, reverse=True)


def test_per_class_coverage_marks_a_class_it_cannot_measure():
    """subtle_conflict has no calibration examples on the predicted-span split.

    Reporting 0.0 for it without saying why would be a fabricated number.
    """
    probs = np.asarray([[0.7, 0.1, 0.1, 0.1]] * 8)
    calib_truth = np.asarray([0, 0, 2, 2, 3, 3, 0, 2])
    test_truth = np.asarray([0, 1, 2, 3, 0, 2, 3, 0])
    out = per_class_coverage(probs, calib_truth, probs, test_truth, 0.1, CLASSES)
    assert out["subtle_conflict"]["n_calibration"] == 0
    assert out["subtle_conflict"]["measurable"] is False
    assert out["evident_conflict"]["measurable"] is True


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def rows_for(n, seed):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 4, size=n)
    probs = rng.dirichlet(np.ones(4), size=n)
    probs[np.arange(n), truth] += 0.5
    probs /= probs.sum(axis=1, keepdims=True)
    return [
        {
            "response_id": f"r{i // 2}",
            "task_type": ["qa", "summarization"][i % 2],
            "error_type": CLASSES[truth[i]],
            "classes": CLASSES,
            "probs_flat": [float(v) for v in probs[i]],
            "probs_axis_product": [float(v) for v in probs[i]],
        }
        for i in range(n)
    ]


def test_unpack_puts_the_truth_in_class_index_order():
    rows = rows_for(6, 3)
    probs, truth, ids, tasks = unpack(rows, "probs_flat")
    assert probs.shape == (6, 4)
    assert all(CLASSES[t] == r["error_type"] for t, r in zip(truth, rows))
    assert len(set(ids)) == 3
    assert set(tasks) == {"qa", "summarization"}


def test_analyse_reports_the_floor_the_gate_and_the_missing_ood_number():
    report = analyse(
        rows_for(400, 4), rows_for(400, 5), "probs_flat", resamples=50
    )
    methods = [row["method"] for row in report["calibration"]]
    assert any(row["is_floor"] for row in report["calibration"])
    assert any(m.startswith("temperature") for m in methods)
    assert report["headline_alpha"] == 0.1
    for row in report["coverage"]:
        assert 0.0 <= row["empirical_coverage"] <= 1.0
        assert row["set_shape_rates"]
        assert "coverage_interval" in row
    assert "guarantee_applies" in report["guarantee_note"]
    assert "RAGBench carries no error-type" in report["ood_note"]


def test_set_shape_rates_sum_to_one():
    report = analyse(rows_for(200, 6), rows_for(200, 7), "probs_flat", resamples=20)
    for row in report["coverage"]:
        assert sum(row["set_shape_rates"].values()) == pytest.approx(1.0)


def test_format_report_prints_the_floor_row_and_the_ood_absence():
    report = analyse(rows_for(200, 8), rows_for(200, 9), "probs_flat", resamples=20)
    text = format_report(report)
    assert "ignores the input" in text
    assert "un-evaluable" in text
    assert "pair_severity_undecided" in text


def test_main_writes_the_artefact(tmp_path):
    calib = tmp_path / "calib.jsonl"
    test = tmp_path / "test.jsonl"
    calib.write_text(
        "\n".join(json.dumps(r) for r in rows_for(200, 10)) + "\n", encoding="utf-8"
    )
    test.write_text(
        "\n".join(json.dumps(r) for r in rows_for(200, 11)) + "\n", encoding="utf-8"
    )
    out_dir = tmp_path / "out"
    assert (
        main(
            [
                "--calib",
                str(calib),
                "--test",
                str(test),
                "--out-dir",
                str(out_dir),
                "--resamples",
                "20",
            ]
        )
        == 0
    )
    written = json.loads((out_dir / "c3_conformal.json").read_text(encoding="utf-8"))
    assert written["n_test"] == 200


def test_main_says_so_when_there_is_nothing_to_calibrate(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    assert main(["--calib", str(empty), "--test", str(empty)]) == 1
