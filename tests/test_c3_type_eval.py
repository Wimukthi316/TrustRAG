"""Tests for C3's type scoring and its response-level intervals.

Two things here are worth more than the rest. The vectorised confusion-matrix
path is what the bootstrap actually calls two thousand times per axis, and it is
a second implementation of the readable one, so they are asserted to agree
rather than assumed to. And a class with no instances in a bootstrap draw must
be recorded as unmeasurable, not as an F1 of zero -- averaging those in would
quietly punish the model for a draw that said nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.c3_explanation.type_eval import (
    bootstrap_by_response,
    bootstrap_metrics,
    classification_report,
    codes,
    confusion_metrics,
    evaluate_axis,
    floor_report,
    format_axis,
    majority_label,
    response_index,
)

LABELS = ("a", "b", "c")


def test_classification_report_is_hand_computable():
    gold = ["a", "a", "b", "b", "c"]
    pred = ["a", "b", "b", "b", "a"]
    report = classification_report(gold, pred, LABELS)
    assert report["n"] == 5
    assert report["accuracy"] == pytest.approx(3 / 5)
    # a: tp 1, fp 1, fn 1 -> P 0.5, R 0.5, F1 0.5
    assert report["per_class"]["a"] == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2, "predicted": 2}
    )
    # b: tp 2, fp 1, fn 0 -> P 2/3, R 1, F1 0.8
    assert report["per_class"]["b"]["f1"] == pytest.approx(0.8)
    # c: never predicted -> all zero, but still present
    assert report["per_class"]["c"]["f1"] == pytest.approx(0.0)
    assert report["per_class"]["c"]["support"] == 1
    assert report["macro_f1"] == pytest.approx((0.5 + 0.8 + 0.0) / 3)


def test_declared_labels_keep_an_unpredicted_class_in_the_average():
    """Dropping an absent class is how a macro F1 improves by looking away."""
    gold = ["a", "a"]
    pred = ["a", "a"]
    with_all = classification_report(gold, pred, LABELS)
    without = classification_report(gold, pred)
    assert with_all["macro_f1"] == pytest.approx(1 / 3)
    assert without["macro_f1"] == pytest.approx(1.0)


def test_mismatched_lengths_are_a_loud_error():
    with pytest.raises(ValueError):
        classification_report(["a"], ["a", "b"], LABELS)


def test_majority_label_breaks_ties_on_the_sorted_name():
    assert majority_label(["b", "a"]) == "a"
    assert majority_label(["b", "b", "a"]) == "b"
    assert majority_label([]) is None


def test_floor_report_ignores_the_input():
    report = floor_report(["a", "a", "b"], "a", LABELS)
    assert report["predicted_class"] == "a"
    assert report["ignores_the_input"] is True
    assert report["accuracy"] == pytest.approx(2 / 3)
    assert report["per_class"]["b"]["f1"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "gold,pred",
    [
        (["a", "b", "c", "a"], ["a", "b", "c", "a"]),
        (["a", "a", "a", "a"], ["b", "b", "b", "b"]),
        (["a", "b", "a", "c"], ["c", "b", "a", "a"]),
        (["a"], ["a"]),
    ],
)
def test_vectorised_and_readable_paths_agree(gold, pred):
    """The docstring claims these are the same arithmetic. This is that claim."""
    readable = classification_report(gold, pred, LABELS)
    fast = confusion_metrics(codes(gold, LABELS), codes(pred, LABELS), len(LABELS))
    assert fast["accuracy"] == pytest.approx(readable["accuracy"])
    assert fast["macro_f1"] == pytest.approx(readable["macro_f1"])
    for column, label in enumerate(LABELS):
        assert fast["f1"][column] == pytest.approx(
            readable["per_class"][label]["f1"]
        )
        assert fast["support"][column] == readable["per_class"][label]["support"]


def test_codes_marks_an_unknown_label_rather_than_guessing():
    assert list(codes(["a", "zzz"], LABELS)) == [0, -1]


def test_response_index_groups_rows_by_response():
    index = response_index(["r1", "r2", "r1", "r3"])
    assert [list(rows) for rows in index] == [[0, 2], [1], [3]]


def test_bootstrap_is_deterministic_for_a_given_seed():
    ids = ["r1", "r1", "r2", "r3", "r4"]
    values = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])

    def statistic(rows):
        return float(values[rows].mean())

    first = bootstrap_by_response(ids, statistic, resamples=200, seed=7)
    second = bootstrap_by_response(ids, statistic, resamples=200, seed=7)
    assert first == second
    assert first["ci_low"] <= first["ci_high"]


def test_bootstrap_resamples_responses_not_rows():
    """Two rows inside one response must move together or the interval lies."""
    ids = ["r1", "r1"]
    values = np.asarray([0.0, 1.0])

    def statistic(rows):
        return float(values[rows].mean())

    # Every draw picks response r1 twice, so every draw contains both rows and
    # the mean is always 0.5 -- a span-level bootstrap would produce spread here.
    result = bootstrap_by_response(ids, statistic, resamples=50, seed=1)
    assert result["ci_low"] == pytest.approx(0.5)
    assert result["ci_high"] == pytest.approx(0.5)


def test_bootstrap_metrics_counts_draws_that_could_not_measure_a_class():
    gold = ["a"] * 20 + ["c"]
    pred = ["a"] * 21
    ids = [f"r{i}" for i in range(21)]
    out = bootstrap_metrics(ids, gold, pred, LABELS, resamples=200, seed=3)
    assert out["c.f1"]["n_draws_without_the_class"] > 0
    assert out["b.f1"]["n_draws_without_the_class"] == 200
    assert math.isnan(out["b.f1"]["ci_low"])
    assert 0.0 <= out["accuracy"]["ci_low"] <= out["accuracy"]["ci_high"] <= 1.0


def test_evaluate_axis_carries_model_floor_tasks_and_intervals():
    gold = ["a", "a", "b", "b", "c", "a"]
    pred = ["a", "b", "b", "b", "a", "a"]
    ids = ["r1", "r1", "r2", "r3", "r4", "r4"]
    tasks = ["qa", "qa", "summarization", "qa", "summarization", "qa"]
    block = evaluate_axis(
        ids, gold, pred, LABELS, tasks=tasks, resamples=100, seed=5
    )
    assert block["labels"] == list(LABELS)
    assert block["floor"]["predicted_class"] == "a"
    assert set(block["per_task"]) == {"qa", "summarization"}
    assert (
        block["per_task"]["qa"]["model"]["n"]
        + block["per_task"]["summarization"]["model"]["n"]
        == len(gold)
    )
    assert "accuracy" in block["intervals"]
    assert "resamples" not in block["interval_note"].split()[0]


def test_evaluate_axis_can_skip_the_intervals():
    block = evaluate_axis(["r1"], ["a"], ["a"], LABELS, with_intervals=False)
    assert "intervals" not in block


def test_format_axis_puts_the_floor_under_the_model():
    block = evaluate_axis(
        ["r1", "r2"], ["a", "b"], ["a", "a"], LABELS, with_intervals=False
    )
    text = format_axis("severity", block).splitlines()
    model_line = next(i for i, line in enumerate(text) if "model" in line)
    floor_line = next(i for i, line in enumerate(text) if "floor" in line)
    assert floor_line == model_line + 1
    assert "ignores the input" in text[floor_line]
