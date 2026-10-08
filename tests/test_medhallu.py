"""Tests for the MedHallu loader and the example-level evaluator's helpers.

Network-free: they do not download MedHallu. The loader's build_context and
record shape are checked directly, and the evaluator's metric helpers are checked
against hand-built inputs (the torch-dependent path lives inside their main()
functions and is not imported here).
"""

from __future__ import annotations

from src.c1_detector.medhallu import _record, build_context
from src.c1_detector.evaluate_medhallu import (
    example_block,
    prf,
    stratified_split,
    trivial_f1,
)


# --- loader ---------------------------------------------------------------


def test_build_context_joins_passages_like_ragtruth():
    out = build_context(["first passage.", "second passage."])
    assert out == "passage 1:first passage.\n\npassage 2:second passage."


def test_build_context_tolerates_a_bare_string():
    assert build_context("only one") == "passage 1:only one"


def test_build_context_empty_for_none():
    assert build_context(None) == ""


def test_record_positive_carries_gold_and_category_no_spans():
    r = _record("42", "hall", True, "Q?", "passage 1:ctx", "answer text",
                "Mechanism and Pathway Misattribution", "hard", "pqa_labeled")
    assert r["gold_positive"] is True
    assert r["category"] == "Mechanism and Pathway Misattribution"
    assert r["spans"] == []
    assert r["task_type"] == "qa"
    assert r["answer"] == "answer text"
    assert r["id"] == "medhallu:pqa_labeled:42:hall"


def test_record_negative_drops_category():
    r = _record("42", "true", False, "Q?", "ctx", "ground truth", "Some Category",
                "easy", "pqa_labeled")
    assert r["gold_positive"] is False
    assert r["category"] == ""


# --- evaluator helpers ----------------------------------------------------


def test_trivial_f1_at_balanced_base_rate():
    assert abs(trivial_f1(0.5) - 2 / 3) < 1e-9
    assert trivial_f1(0.0) == 0.0


def test_prf_basic():
    out = prf(tp=3, fp=1, fn=1)
    assert abs(out["precision"] - 0.75) < 1e-9
    assert abs(out["recall"] - 0.75) < 1e-9
    assert abs(out["f1"] - 0.75) < 1e-9


def test_example_block_perfect_detector():
    rows = [{"gold_positive": True, "pred_positive": True}] * 5 + \
           [{"gold_positive": False, "pred_positive": False}] * 5
    b = example_block(rows)
    assert b["f1"] == 1.0
    assert b["positive_rate"] == 0.5
    assert b["clears_trivial"] is True


def test_example_block_flag_everything_equals_trivial():
    rows = [{"gold_positive": i < 5, "pred_positive": True} for i in range(10)]
    b = example_block(rows)
    assert abs(b["f1"] - 2 / 3) < 1e-9
    assert b["clears_trivial"] is False


def test_stratified_split_partitions_and_keeps_rate():
    labels = [1] * 70 + [0] * 30
    calib, test = stratified_split(labels, 0.3, seed=13)
    assert sorted(calib + test) == list(range(100))
    assert set(calib).isdisjoint(test)
    # ~30% of each class in calibration
    assert abs(sum(labels[i] for i in calib) - 21) <= 1
    assert abs(sum(1 - labels[i] for i in calib) - 9) <= 1
