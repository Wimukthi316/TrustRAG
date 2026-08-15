"""Tests for the bootstrap intervals reported beside C1's numbers."""

from __future__ import annotations

import pytest

from src.c1_detector.uncertainty import (
    bootstrap_ci,
    pair_by_id,
    paired_bootstrap,
)

SMALL = 200  # resamples; enough for the properties below and fast


def record(rid, gold, pred):
    return {"id": rid, "task_type": "qa", "gold_spans": gold, "pred_spans": pred}


@pytest.fixture
def mixed():
    """Half the responses located exactly, half missed entirely."""
    rows = []
    for i in range(20):
        gold = [(0, 10)]
        pred = [(0, 10)] if i % 2 == 0 else []
        rows.append(record(str(i), gold, pred))
    return rows


def test_interval_contains_the_point_estimate(mixed):
    for row in bootstrap_ci(mixed, resamples=SMALL).values():
        assert row["ci_low"] <= row["point"] <= row["ci_high"]


def test_interval_is_reproducible_for_a_fixed_seed(mixed):
    first = bootstrap_ci(mixed, resamples=SMALL, seed=7)
    second = bootstrap_ci(mixed, resamples=SMALL, seed=7)
    assert first == second


def test_a_different_seed_moves_the_interval(mixed):
    first = bootstrap_ci(mixed, resamples=SMALL, seed=1)
    second = bootstrap_ci(mixed, resamples=SMALL, seed=2)
    assert first["example"]["ci_low"] != second["example"]["ci_low"]


def test_a_uniform_corpus_has_no_sampling_uncertainty():
    """Every response identical, so every resample scores the same."""
    rows = [record(str(i), [(0, 5)], [(0, 5)]) for i in range(10)]
    result = bootstrap_ci(rows, resamples=SMALL)
    assert result["example"]["width_points"] == pytest.approx(0.0)
    assert result["span_exact"]["point"] == 1.0


def test_more_responses_narrow_the_interval():
    def build(n):
        return [
            record(str(i), [(0, 10)], [(0, 10)] if i % 2 == 0 else [])
            for i in range(n)
        ]

    narrow = bootstrap_ci(build(400), resamples=SMALL)["example"]["width_points"]
    wide = bootstrap_ci(build(40), resamples=SMALL)["example"]["width_points"]
    assert narrow < wide


def test_pair_by_id_lines_the_dumps_up(mixed):
    shuffled = list(reversed(mixed))
    pairs = pair_by_id(mixed, shuffled)
    assert [left["id"] for left, _ in pairs] == [row["id"] for row in mixed]
    assert all(left["id"] == right["id"] for left, right in pairs)


def test_pair_by_id_refuses_a_mismatch(mixed):
    with pytest.raises(ValueError, match="absent from the variant"):
        pair_by_id(mixed, mixed[:-1])


def test_a_model_compared_against_itself_shows_no_difference(mixed):
    result = paired_bootstrap(mixed, mixed, resamples=SMALL)
    for row in result.values():
        assert row["delta_points"] == pytest.approx(0.0)
        assert row["ci_low_points"] == pytest.approx(0.0)
        assert row["ci_high_points"] == pytest.approx(0.0)
        assert row["crosses_zero"] is True
        assert row["verdict"] == "not distinguishable from zero"


def test_a_real_difference_is_called_real(mixed):
    """The variant finds every span the baseline missed."""
    better = [record(row["id"], row["gold_spans"], row["gold_spans"]) for row in mixed]
    result = paired_bootstrap(mixed, better, resamples=SMALL)["example"]
    assert result["delta_points"] > 0
    assert result["ci_low_points"] > 0
    assert result["crosses_zero"] is False
    assert result["verdict"] == "higher"


def test_a_real_degradation_is_called_lower(mixed):
    worse = [record(row["id"], row["gold_spans"], []) for row in mixed]
    result = paired_bootstrap(mixed, worse, resamples=SMALL)["example"]
    assert result["delta_points"] < 0
    assert result["ci_high_points"] < 0
    assert result["verdict"] == "lower"


def test_paired_intervals_are_narrower_than_unpaired_ones(mixed):
    """The point of pairing: shared sampling noise cancels.

    The variant here differs from the baseline only in span boundaries, so the
    example-level score is identical on every draw and the paired interval
    collapses, while each model's own interval stays wide.
    """
    nudged = [
        record(row["id"], row["gold_spans"], [(s + 1, e) for s, e in row["pred_spans"]])
        for row in mixed
    ]
    paired = paired_bootstrap(mixed, nudged, resamples=SMALL)["example"]
    solo = bootstrap_ci(mixed, resamples=SMALL)["example"]
    paired_width = paired["ci_high_points"] - paired["ci_low_points"]
    assert paired_width < solo["width_points"]
