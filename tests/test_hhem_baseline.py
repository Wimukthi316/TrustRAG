"""Tests for the HHEM baseline.

The mistake that would matter here is the score direction. HHEM is confident the
answer is SUPPORTED when its score is high, and this project's positive class is
"contains a hallucination", so a missed inversion produces a table that looks
fine and says the opposite of the truth. Most of these tests are about that.
"""

from __future__ import annotations

import pytest

from src.c1_detector.hhem_baseline import (
    MODEL_ID,
    build_pairs,
    build_report,
    choose_threshold,
    decide,
    format_report,
    premise_of,
    read_records,
    sweep,
    to_rows,
)


def record(rid, gold, task="qa", context="the source passage", answer="the answer"):
    return {
        "id": rid,
        "task_type": task,
        "context": context,
        "answer": answer,
        "spans": [{"start": 0, "end": 3}] if gold else [],
    }


def test_a_high_consistency_score_means_not_hallucinated():
    rows = to_rows([record("a", False)], [0.9])
    assert rows[0]["consistency"] == pytest.approx(0.9)
    assert rows[0]["p_hallucinated"] == pytest.approx(0.1)


def test_a_low_consistency_score_means_hallucinated():
    rows = to_rows([record("a", True)], [0.05])
    assert rows[0]["p_hallucinated"] == pytest.approx(0.95)


def test_the_gold_label_comes_from_the_spans():
    assert to_rows([record("a", True)], [0.5])[0]["gold_positive"] is True
    assert to_rows([record("a", False)], [0.5])[0]["gold_positive"] is False


def test_a_length_mismatch_is_refused_rather_than_zipped_short():
    with pytest.raises(ValueError, match="against"):
        to_rows([record("a", True), record("b", False)], [0.5])


def test_pairs_are_premise_then_hypothesis():
    pairs = build_pairs([record("a", True, context="EVIDENCE", answer="CLAIM")])
    assert pairs == [("EVIDENCE", "CLAIM")]


def test_premise_is_the_context_and_never_the_question():
    row = {"context": "passage 1: facts", "question": "what happened?", "answer": "x"}
    assert premise_of(row) == "passage 1: facts"


def test_a_record_with_no_context_fails_loudly():
    with pytest.raises(KeyError, match="no context field"):
        premise_of({"question": "q", "answer": "a"})


def test_records_load_from_jsonl(tmp_path):
    import json

    path = tmp_path / "r.jsonl"
    path.write_text(
        "\n".join(json.dumps(record(str(i), i % 2 == 0)) for i in range(4)) + "\n",
        encoding="utf-8",
    )
    assert len(read_records(path)) == 4
    assert len(read_records(path, limit=2)) == 2


@pytest.fixture
def separable():
    """Hallucinated responses score low on consistency, as they should."""
    records = [record(str(i), i % 2 == 0) for i in range(40)]
    scores = [0.05 if i % 2 == 0 else 0.95 for i in range(40)]
    return to_rows(records, scores)


def test_a_perfect_detector_is_found_by_the_sweep(separable):
    chosen = choose_threshold(separable, thresholds=[0.2, 0.5, 0.9])
    assert chosen["chosen"]["f1"] == pytest.approx(1.0)


def test_ties_break_toward_the_more_sensitive_threshold(separable):
    chosen = choose_threshold(separable, thresholds=[0.2, 0.5])
    assert chosen["chosen"]["threshold"] == 0.2


def test_lowering_the_threshold_never_lowers_recall(separable):
    recalls = [row["recall"] for row in sweep(separable, [0.1, 0.5, 0.9])]
    assert recalls == sorted(recalls, reverse=True)


def test_decide_reads_the_hallucination_probability(separable):
    positives = [row for row in separable if decide(0.5)(row)]
    assert all(row["p_hallucinated"] >= 0.5 for row in positives)
    assert len(positives) == 20


def test_the_report_marks_the_span_columns_not_applicable(separable):
    report = build_report(separable, separable)
    assert "n/a" in report["span_level"]
    assert "n/a" in format_report(report)
    assert report["model"] == MODEL_ID


def test_the_report_keeps_both_the_default_and_the_chosen_row(separable):
    report = build_report(separable, separable)
    assert "at_half" in report["test"]
    assert "adapted" in report["test"]
    assert report["test"]["at_half"]["n"] == report["test"]["adapted"]["n"]


def test_the_report_states_where_the_threshold_came_from(separable):
    report = build_report(separable, separable)
    assert "calibration" in report["threshold_source"]
    assert "1 - score" in report["score_direction"]
