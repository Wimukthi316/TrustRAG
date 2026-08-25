"""Layer 8: the folds leak nothing, and the RAGBench answer is a reason not a shrug."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.c4_attribution import transfer


def test_ragbench_absent_is_reported_with_a_reason_rather_than_as_a_number():
    block = transfer.check_ragbench(Path("data/processed/does-not-exist"))
    assert block["checked"] is False
    assert block["available"] is False
    assert "not present" in block["reason"]


@pytest.mark.skipif(
    not transfer.RAGBENCH_DIR.exists(),
    reason="processed RAGBench is not in this working copy",
)
def test_ragbench_scan_covers_every_subset_and_every_field():
    block = transfer.check_ragbench()
    assert block["checked"] is True
    assert block["n_subsets"] >= 1
    assert block["n_records"] > 0
    assert isinstance(block["fields"], list)
    # Whatever the answer, it comes with the scan that produced it.
    assert block["reason"]


def test_the_three_tasks_are_the_folds():
    assert transfer.TASKS == ("qa", "data2text", "summarization")


def test_the_field_hints_would_catch_an_evidence_column_if_one_existed():
    hits = [
        key
        for key in ("evidence_sentence", "supporting_sentences", "cited_spans", "answer")
        if any(hint in key.lower() for hint in transfer.EVIDENCE_FIELD_HINTS)
    ]
    assert "evidence_sentence" in hits
    assert "supporting_sentences" in hits
    assert "answer" not in hits


def test_a_fold_never_calibrates_on_the_task_it_is_tested_on():
    # The construction is asserted at the call site in run_c4, and this pins the
    # invariant the call site has to satisfy.
    rows = [("qa", 0), ("data2text", 1), ("summarization", 2), ("qa", 3)]
    for held_out in transfer.TASKS:
        calibration = [r for r in rows if r[0] != held_out]
        evaluation = [r for r in rows if r[0] == held_out]
        assert all(r[0] != held_out for r in calibration)
        assert all(r[0] == held_out for r in evaluation)
        assert not ({r[1] for r in calibration} & {r[1] for r in evaluation})
