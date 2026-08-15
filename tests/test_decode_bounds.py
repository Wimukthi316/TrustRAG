"""Tests for the decode-bound measurements that decided against Block B."""

from __future__ import annotations

import pytest

from src.c1_detector.decode_bounds import (
    build_report,
    glue_adjacent,
    glue_sweep,
    threshold_sweep_span_exact,
)


@pytest.fixture
def fragmented():
    """One gold span the model cut into two pieces with a two-character gap."""
    return [
        {
            "id": "1",
            "answer": "0123456789",
            "gold_spans": [(0, 9)],
            "pred_spans": [(0, 4), (6, 9)],
            "token_probs": [0.9] * 10,
            "answer_offsets": [(i, i + 1) for i in range(10)],
        }
    ]


def test_glue_leaves_distant_spans_alone():
    assert glue_adjacent([(0, 3), (10, 13)], max_gap=2) == [(0, 3), (10, 13)]


def test_glue_joins_spans_within_the_gap():
    assert glue_adjacent([(0, 3), (5, 8)], max_gap=2) == [(0, 8)]


def test_glue_is_order_independent():
    assert glue_adjacent([(5, 8), (0, 3)], max_gap=2) == [(0, 8)]


def test_glue_keeps_the_outer_end_when_spans_nest():
    assert glue_adjacent([(0, 10), (2, 4)], max_gap=0) == [(0, 10)]


def test_gap_zero_is_the_untouched_baseline(fragmented):
    row = glue_sweep(fragmented, gaps=[0])[0]
    assert row["n_pred_spans"] == 2
    assert row["span_exact_tp"] == 0


def test_gluing_recovers_a_fragmented_span(fragmented):
    row = glue_sweep(fragmented, gaps=[2])[0]
    assert row["n_pred_spans"] == 1
    assert row["span_exact_tp"] == 1
    assert row["span_exact_f1"] == 1.0


def test_gluing_can_destroy_correct_predictions():
    """The cost side of the trade, which an F1 column alone hides."""
    records = [
        {
            "answer": "0123456789",
            "gold_spans": [(0, 3), (5, 8)],
            "pred_spans": [(0, 3), (5, 8)],
            "token_probs": [0.9] * 10,
            "answer_offsets": [(i, i + 1) for i in range(10)],
        }
    ]
    assert glue_sweep(records, gaps=[0])[0]["span_exact_tp"] == 2
    assert glue_sweep(records, gaps=[5])[0]["span_exact_tp"] == 0


def test_threshold_sweep_predicts_nothing_above_the_scores(fragmented):
    rows = threshold_sweep_span_exact(fragmented, thresholds=[0.95])
    assert rows[0]["n_pred_spans"] == 0
    assert rows[0]["span_exact_f1"] == 0.0


def test_threshold_sweep_reports_both_levels(fragmented):
    row = threshold_sweep_span_exact(fragmented, thresholds=[0.5])[0]
    assert set(row) >= {
        "threshold",
        "n_pred_spans",
        "span_exact_precision",
        "span_exact_recall",
        "span_exact_f1",
        "span_overlap_f1",
    }


def test_report_states_the_gain_and_what_it_cost(fragmented):
    report = build_report(fragmented)
    assert report["best_glued_span_exact_f1"] >= report["baseline_span_exact_f1"]
    assert report["best_gain_points"] == pytest.approx(100.0)
    assert report["exact_matches_lost_at_best"] <= 0
