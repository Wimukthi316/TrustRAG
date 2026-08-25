"""Layer 5: the floor row exists, and recalibration cannot change AUROC."""

from __future__ import annotations

import numpy as np
import pytest

from src.c4_attribution import score_calibration as sc


def _split(n=400, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    scores = np.clip(0.5 + 0.25 * labels + rng.normal(0, 0.15, size=n), 0.01, 0.99)
    return scores, labels


def test_the_constant_floor_row_is_always_present_and_tagged():
    calib_scores, calib_labels = _split(seed=1)
    test_scores, test_labels = _split(seed=2)
    report = sc.evaluate(calib_scores, calib_labels, test_scores, test_labels)
    floors = [r for r in report["rows"] if r.get("is_floor")]
    assert len(floors) == 1
    assert floors[0]["method"] == "constant_base_rate"
    assert floors[0]["ignores_input"] is True


def test_the_floor_has_an_auroc_of_one_half_because_it_ignores_its_input():
    calib_scores, calib_labels = _split(seed=1)
    test_scores, test_labels = _split(seed=2)
    report = sc.evaluate(calib_scores, calib_labels, test_scores, test_labels)
    floor = next(r for r in report["rows"] if r.get("is_floor"))
    assert floor["auroc"] == pytest.approx(0.5, abs=1e-9)


def test_a_monotone_recalibration_cannot_change_auroc():
    scores, labels = _split(seed=3)
    from src.c2_calibration.calibration import PlattCalibrator
    from src.c2_calibration.clustered import auroc

    before = auroc(np.asarray(scores), np.asarray(labels))
    after = auroc(
        np.asarray(PlattCalibrator().fit(scores, labels).transform(scores)),
        np.asarray(labels),
    )
    assert after == pytest.approx(before, abs=1e-9)


def test_the_unit_map_preserves_the_ordering_and_therefore_the_ranking():
    values = [-0.4, 0.0, 0.3, 0.9, 2.5]
    mapped = sc.to_unit_interval(values)
    assert list(np.argsort(values)) == list(np.argsort(mapped))
    assert mapped.min() > 0.0 and mapped.max() < 1.0


def test_a_constant_score_column_does_not_explode():
    mapped = sc.to_unit_interval([0.3] * 10)
    assert np.allclose(mapped, 0.5)


def test_the_calibrators_are_fitted_on_calibration_only():
    calib_scores, calib_labels = _split(n=300, seed=4)
    test_scores, test_labels = _split(n=300, seed=5)
    report = sc.evaluate(calib_scores, calib_labels, test_scores, test_labels)
    assert report["n_calibration"] == 300
    assert report["n_test"] == 300
    assert all(r.get("fitted_on") == "calib" for r in report["rows"])


def test_the_prediction_is_recorded_before_the_verdict():
    calib_scores, calib_labels = _split(seed=6)
    test_scores, test_labels = _split(seed=7)
    report = sc.evaluate(calib_scores, calib_labels, test_scores, test_labels)
    assert "Written before running" in report["prediction"]
    assert isinstance(report["floor_has_best_ece"], bool)
    assert report["verdict"]


def test_the_unit_map_is_fitted_on_calibration_and_applied_to_test():
    """Fitting the scale separately per split would be a leak in miniature."""
    calib = [0.0, 0.5, 1.0]
    test = [-3.0, 0.5, 9.0]
    bounds = sc.fit_unit_map(calib)
    assert bounds == (0.0, 1.0)
    mapped = sc.to_unit_interval(test, bounds)
    # Out-of-range test values are clipped, not rescaled: that is what applying
    # a fitted transform to unseen data means.
    assert mapped[0] == pytest.approx(sc.EPSILON)
    assert mapped[1] == pytest.approx(0.5)
    assert mapped[2] == pytest.approx(1.0 - sc.EPSILON)


def test_the_bounds_used_are_recorded_in_the_artefact():
    calib_scores, calib_labels = _split(seed=20)
    test_scores, test_labels = _split(seed=21)
    report = sc.evaluate(calib_scores, calib_labels, test_scores, test_labels)
    low, high = report["unit_map_bounds_fitted_on_calibration"]
    assert low == pytest.approx(float(min(calib_scores)))
    assert high == pytest.approx(float(max(calib_scores)))
