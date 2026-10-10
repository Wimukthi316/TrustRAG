"""Network-free tests for the MedHallu answer-level classifier helpers.

They cover the text normaliser, the first-sequence builder and the example-level
metric block. Training itself needs torch, a GPU and a dataset download, so it is
exercised on Kaggle, not here.
"""

from __future__ import annotations

from src.c1_detector.train_medhallu_classifier import (
    _first_sequence,
    _norm,
    example_block,
    prf,
    trivial_f1,
)


def test_norm_collapses_case_and_whitespace():
    assert _norm("  Aspirin   LOWERS\tglucose\n") == "aspirin lowers glucose"
    assert _norm("The dose was 5 mg.") == _norm("the  dose   was 5 mg.")


def test_first_sequence_joins_question_and_context():
    assert _first_sequence("Does X cause Y?", "passage 1:evidence") == "Does X cause Y?\n\npassage 1:evidence"
    assert _first_sequence("", "passage 1:evidence") == "passage 1:evidence"
    assert _first_sequence("  ", "ctx") == "ctx"


def test_prf_basic():
    out = prf(tp=3, fp=1, fn=1)
    assert abs(out["precision"] - 0.75) < 1e-9
    assert abs(out["recall"] - 0.75) < 1e-9
    assert abs(out["f1"] - 0.75) < 1e-9


def test_trivial_f1_balanced_floor():
    assert abs(trivial_f1(0.5) - (2 / 3)) < 1e-9
    assert trivial_f1(0.0) == 0.0


def test_example_block_confusion_and_flags():
    labels = [1, 1, 0, 0]
    preds = [1, 0, 0, 1]  # tp, fn, tn, fp
    b = example_block(labels, preds)
    assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (1, 1, 1, 1)
    assert abs(b["accuracy"] - 0.5) < 1e-9
    assert abs(b["positive_rate"] - 0.5) < 1e-9
    assert abs(b["trivial_f1"] - (2 / 3)) < 1e-9


def test_example_block_perfect_clears_trivial():
    labels = [1, 0, 1, 0, 1, 0]
    b = example_block(labels, labels)
    assert b["f1"] == 1.0
    assert b["clears_trivial"] is True


def test_example_block_all_positive_predictions_is_trivial():
    labels = [1, 0, 1, 0]
    preds = [1, 1, 1, 1]
    b = example_block(labels, preds)
    # flagging everything scores exactly the trivial floor, so it must NOT clear it
    assert abs(b["f1"] - b["trivial_f1"]) < 1e-9
    assert b["clears_trivial"] is False
