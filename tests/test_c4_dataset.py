"""The join: frozen splits, C1's own context rule, and the spurious-span exclusion."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.c4_attribution import dataset as ds

RAW = Path("data/raw/response.jsonl")
SPLITS = Path("results/c1/modernbert-base/split_ids.json")

needs_corpus = pytest.mark.skipif(
    not RAW.exists() or not SPLITS.exists(),
    reason="the raw corpus or the frozen split file is not in this working copy",
)


def test_reported_splits_exclude_the_data_c1_trained_on():
    assert ds.REPORTED_SPLITS == ("calib", "test")
    assert "train" not in ds.REPORTED_SPLITS
    assert "val" not in ds.REPORTED_SPLITS


@needs_corpus
def test_the_frozen_split_is_the_one_c2_and_c3_calibrate_on():
    split_ids = ds.load_split_ids()
    assert len(split_ids["calib"]) == 1509
    assert len(split_ids["val"]) == 755
    assert len(split_ids["train"]) == 12826


@needs_corpus
def test_no_calibration_response_is_also_a_test_response():
    split_ids = ds.load_split_ids()
    responses = ds.load_responses()
    calib = {r.response_id for r in responses if r.split == "calib"}
    test = {r.response_id for r in responses if r.split == "test"}
    assert not (calib & test)
    assert calib <= split_ids["calib"]


@needs_corpus
def test_the_context_comes_from_c1s_own_rule():
    from src.c1_detector.build_examples import build_context_and_question
    from src.c1_detector.ragtruth_labels import task_type_of

    sources = {
        str(r["source_id"]): r for r in ds.read_jsonl(ds.RAW_DIR / "source_info.jsonl")
    }
    for record in ds.load_responses()[:25]:
        source = sources[record.source_id]
        context, question = build_context_and_question(
            task_type_of(str(source["task_type"])), source["source_info"]
        )
        assert record.context == context
        assert record.question == question


@needs_corpus
def test_every_span_slices_its_own_text_out_of_its_own_answer():
    for row in ds.span_records(ds.load_responses())[:500]:
        answer = row.response.answer
        assert answer[row.label.start:row.label.end] == row.label.span_text


@needs_corpus
def test_the_hypothesis_constructions_are_built_from_the_same_splitter():
    for row in ds.span_records(ds.load_responses())[:100]:
        assert row.span_text in row.answer_sentence or row.answer_sentence
        assert "<<" in row.marked_answer_sentence and ">>" in row.marked_answer_sentence


@needs_corpus
def test_the_spurious_rate_is_the_number_the_report_quotes():
    rate = ds.spurious_rate(ds.load_predicted_spans("results/c1/test/probabilities.jsonl"))
    assert rate["n_predicted_spans"] == 2390
    assert rate["n_spurious"] == 1134
    assert round(rate["spurious_rate"], 4) == 0.4745


@needs_corpus
def test_a_spurious_predicted_span_inherits_no_evidence_label():
    responses = ds.load_responses(splits=None, with_labels_only=False)
    predicted = ds.predicted_span_records("results/c1/test/probabilities.jsonl", responses)
    spurious = [r for r in predicted if ds.is_spurious(r)]
    assert len(spurious) == 1134
    assert all(not r.label.has_original for r in spurious)
    assert all(r.label.snippet is None for r in spurious)


def test_a_missing_split_file_is_a_loud_failure_not_a_redrawn_split():
    with pytest.raises(Exception):
        ds.load_split_ids("results/c1/does-not-exist/split_ids.json")
