"""Tests for the RAGBench out-of-distribution path.

The fixtures are shaped exactly like the real parquet rows, which were read
directly from `galileo-ai/ragbench` before this file was written: `response` is
the raw text with newlines, `response_sentences` is a list of [key, text] pairs
whose whitespace has been collapsed, and `unsupported_response_sentence_keys`
names the sentences the GPT-4o annotator called unsupported.

The whitespace mismatch in test_locates_sentence_across_a_newline is not
invented. Matching `response_sentences` against `response` with a plain
`str.find` locates 1,346 of pubmedqa's 2,450 test records; the whitespace
tolerant matcher locates 2,449. That measurement is why `sentence_pattern`
exists at all, so it gets a test.

Nothing here touches the network, a tokenizer or a checkpoint.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.schema import AnalysisResult, Span  # noqa: E402
from src.c1_detector.evaluate_ood import (  # noqa: E402
    REPORTABLE_LEVELS,
    format_ood_table,
    pooled_row,
    trivial_f1,
)
from src.c1_detector.ragbench import (  # noqa: E402
    SUBSETS,
    BuildProblems,
    build_context,
    build_record,
    locate_sentences,
    normalise_key,
    sentence_pattern,
    subset_stats,
)


# --------------------------------------------------------------------------
# Fixtures shaped like the real rows
# --------------------------------------------------------------------------

SUPPORTED_ROW = {
    "id": "covidqa_1",
    "question": "What causes prolonged inflammation?",
    "documents": ["First document text.", "Second document text."],
    "response": "Prolonged inflammation follows ineffective clearance. It is documented.",
    "generation_model_name": "gpt-3.5-turbo-1106",
    "annotating_model_name": "gpt-4o",
    "response_sentences": [
        ["a", "Prolonged inflammation follows ineffective clearance."],
        ["b", "It is documented."],
    ],
    "unsupported_response_sentence_keys": [],
    "adherence_score": True,
}

UNSUPPORTED_ROW = {
    "id": "pubmedqa_2",
    "question": "Does the covering alter healing?",
    "documents": ["Only document."],
    # A real newline in the response, collapsed to a space in the sentence list.
    "response": "Based on the context:\n\nYes, the covering alters healing.\nNo further detail exists.",
    "generation_model_name": "gpt-3.5-turbo-1106",
    "annotating_model_name": "gpt-4o",
    "response_sentences": [
        ["a", "Based on the context: Yes, the covering alters healing."],
        ["b", "No further detail exists."],
    ],
    "unsupported_response_sentence_keys": ["a"],
    "adherence_score": False,
}


# --------------------------------------------------------------------------
# Sentence localisation
# --------------------------------------------------------------------------


def test_sentence_pattern_is_none_for_blank():
    assert sentence_pattern("   \n ") is None


def test_sentence_pattern_tolerates_different_whitespace():
    pattern = sentence_pattern("one two three")
    assert pattern is not None
    assert pattern.search("one\n\ntwo   three") is not None


def test_locates_sentence_across_a_newline():
    response = UNSUPPORTED_ROW["response"]
    found, missing = locate_sentences(response, UNSUPPORTED_ROW["response_sentences"])
    assert missing == []
    start, end = found["a"]
    # The offsets index the ORIGINAL response, newline and all.
    assert response[start:end] == "Based on the context:\n\nYes, the covering alters healing."


def test_offsets_are_ordered_and_non_overlapping():
    response = UNSUPPORTED_ROW["response"]
    found, _ = locate_sentences(response, UNSUPPORTED_ROW["response_sentences"])
    a_start, a_end = found["a"]
    b_start, b_end = found["b"]
    assert a_start < a_end <= b_start < b_end


def test_repeated_sentence_resolves_to_successive_positions():
    response = "Same line. Same line."
    sentences = [["a", "Same line."], ["b", "Same line."]]
    found, missing = locate_sentences(response, sentences)
    assert missing == []
    assert found["a"] == (0, 10)
    assert found["b"] == (11, 21)


def test_unlocatable_sentence_is_reported_not_guessed():
    found, missing = locate_sentences("Real text.", [["a", "Text that is absent."]])
    assert found == {}
    assert missing == ["a"]


# --------------------------------------------------------------------------
# Context formatting
# --------------------------------------------------------------------------


def test_context_uses_ragtruth_passage_layout():
    # C1 trained on RAGTruth QA contexts, which look exactly like this. Changing
    # the layout would measure prompt-format shift instead of domain shift.
    assert build_context(["A", "B"]) == "passage 1:A\n\npassage 2:B"


def test_empty_documents_give_empty_context():
    assert build_context([]) == ""


# --------------------------------------------------------------------------
# Record building
# --------------------------------------------------------------------------


def test_supported_response_has_no_spans():
    problems = BuildProblems()
    record = build_record(SUPPORTED_ROW, "covidqa", problems)
    assert record is not None
    assert record["spans"] == []
    assert record["adherence_score"] is True
    assert problems.total() == 0


def test_unsupported_response_carries_the_example_label_as_a_span():
    problems = BuildProblems()
    record = build_record(UNSUPPORTED_ROW, "pubmedqa", problems)
    assert record is not None
    # example_metrics reads len(gold_spans) > 0, so the span list is what carries
    # the adherence label into the evaluator.
    assert len(record["spans"]) == 1
    assert record["adherence_score"] is False
    assert problems.total() == 0


def test_span_text_matches_the_answer_slice():
    problems = BuildProblems()
    record = build_record(UNSUPPORTED_ROW, "pubmedqa", problems)
    answer = record["answer"]
    for span in record["spans"]:
        assert answer[span["start"]:span["end"]] == span["text"]


def test_record_satisfies_the_frozen_contract():
    # schema.py validates that answer[start:end] == text. If the offsets were
    # computed on a whitespace-normalised copy of the response this would fail.
    problems = BuildProblems()
    record = build_record(UNSUPPORTED_ROW, "pubmedqa", problems)
    result = AnalysisResult(
        question=record["question"],
        context=record["context"],
        answer=record["answer"],
        spans=[Span(**{k: s[k] for k in ("start", "end", "text")}) for s in record["spans"]],
        model_version="test",
    )
    assert len(result.spans) == 1


def test_record_id_is_namespaced_by_subset():
    record = build_record(SUPPORTED_ROW, "covidqa", BuildProblems())
    assert record["id"] == "covidqa:covidqa_1"
    assert record["source"] == "covidqa"


def test_annotator_is_carried_through():
    # The report has to be able to say these labels came from a model.
    record = build_record(SUPPORTED_ROW, "covidqa", BuildProblems())
    assert record["annotator"] == "gpt-4o"


def test_empty_response_is_dropped_and_counted():
    problems = BuildProblems()
    row = dict(SUPPORTED_ROW, response="   ")
    assert build_record(row, "covidqa", problems) is None
    assert problems.empty_response == 1


def test_trailing_punctuation_in_a_key_still_resolves():
    # RAGBench writes "a." in unsupported_response_sentence_keys and "a" in
    # response_sentences on 303 keys across the test set. Without this, 175
    # hallucinated responses -- 10.4% of the positive class -- silently became
    # negatives and the measured positive rate moved from 14.2% to 12.9%.
    problems = BuildProblems()
    row = dict(UNSUPPORTED_ROW, unsupported_response_sentence_keys=["a."])
    record = build_record(row, "finqa", problems)
    assert record is not None
    assert len(record["spans"]) == 1
    assert problems.total() == 0


def test_key_normalisation_strips_only_punctuation_and_case():
    assert normalise_key("a.") == normalise_key("A") == "a"
    assert normalise_key(" 1b, ") == "1b"
    # Distinct keys must stay distinct -- checked against all 11,802 real
    # records, where this never collides.
    assert normalise_key("a") != normalise_key("b")


def test_duplicate_keys_produce_one_span():
    problems = BuildProblems()
    row = dict(UNSUPPORTED_ROW, unsupported_response_sentence_keys=["a", "a."])
    record = build_record(row, "finqa", problems)
    assert len(record["spans"]) == 1


def test_missing_sentence_key_is_counted():
    # 2 keys in the real test set are genuinely corrupt and survive
    # normalisation. They are skipped, never invented.
    problems = BuildProblems()
    row = dict(UNSUPPORTED_ROW, unsupported_response_sentence_keys=["a", "zz"])
    record = build_record(row, "pubmedqa", problems)
    assert record is not None
    assert len(record["spans"]) == 1
    assert problems.missing_sentence_key == 1


def test_positive_row_whose_spans_all_fail_is_dropped_not_silently_negated():
    # A non-adherent response with no resolvable span would otherwise enter the
    # evaluation as a negative and quietly weaken the positive class.
    problems = BuildProblems()
    row = dict(UNSUPPORTED_ROW, unsupported_response_sentence_keys=["zz"])
    assert build_record(row, "pubmedqa", problems) is None
    assert problems.positive_without_spans == 1


def test_null_adherence_is_dropped():
    problems = BuildProblems()
    row = dict(SUPPORTED_ROW, adherence_score=None)
    assert build_record(row, "covidqa", problems) is None
    assert problems.null_adherence == 1


def test_subset_stats_counts_positives_by_span_presence():
    records = [
        build_record(SUPPORTED_ROW, "covidqa", BuildProblems()),
        build_record(UNSUPPORTED_ROW, "covidqa", BuildProblems()),
    ]
    stats = subset_stats(records)
    assert stats["n"] == 2
    assert stats["positive"] == 1
    assert stats["positive_rate"] == pytest.approx(0.5)


def test_there_are_twelve_subsets():
    assert len(SUBSETS) == 12
    assert len(set(SUBSETS)) == 12


# --------------------------------------------------------------------------
# OOD reporting
# --------------------------------------------------------------------------


def _row(subset, n, f1, positive_rate=0.2):
    return {
        "subset": subset,
        "n": n,
        "positive_rate": positive_rate,
        "precision": f1,
        "recall": f1,
        "f1": f1,
        "accuracy": f1,
        "trivial_f1": trivial_f1(positive_rate),
        "n_gold_spans": 10,
        "n_pred_spans": 10,
    }


def test_pooled_row_is_size_weighted():
    rows = [_row("a", 100, 0.8), _row("b", 300, 0.4)]
    pooled = pooled_row(rows)
    assert pooled["n"] == 400
    assert pooled["f1"] == pytest.approx((0.8 * 100 + 0.4 * 300) / 400)


def test_trivial_f1_is_the_always_positive_baseline():
    # Precision p, recall 1 -> F1 = 2p/(1+p). At RAGTruth's 43.1% that is 0.602;
    # at RAGBench's 14.2% it is 0.249. The same detector scores lower on the
    # second corpus for free, which is why this column exists.
    assert trivial_f1(0.431) == pytest.approx(0.60238, abs=1e-4)
    assert trivial_f1(0.142) == pytest.approx(0.24869, abs=1e-4)
    assert trivial_f1(0.0) == 0.0


def test_table_marks_a_row_that_fails_to_beat_the_trivial_baseline():
    # expertqa really does score 0.6919 against a trivial 0.6945. If that row
    # ever printed without being marked, the table would read as a success.
    table = format_ood_table([_row("expertqa", 203, 0.6919, positive_rate=0.532)], None)
    assert "NO" in table


def test_table_marks_a_row_that_does_beat_it():
    table = format_ood_table([_row("delucionqa", 184, 0.3000, positive_rate=0.065)], None)
    assert "yes" in table


def test_table_has_no_truncation_column():
    # n_answer_truncated is always zero here because only the context is cut.
    # Printing it would say "nothing was truncated" about cuad and techqa, which
    # lose 73.5% and 92.4% of their contexts.
    table = format_ood_table([_row("cuad", 510, 0.1579)], None)
    assert "trunc" not in table.split("\n")[-3].lower()


def test_only_example_level_is_reportable():
    # Guard against someone adding span_overlap to this table later. RAGBench
    # marks sentences and RAGTruth marks phrases; the two do not compare.
    assert REPORTABLE_LEVELS == ("example",)


def test_table_states_the_two_caveats_that_make_the_numbers_readable():
    table = format_ood_table([_row("covidqa", 246, 0.5)], None)
    lowered = table.lower()
    # The labels are LLM-written, and the positive rate differs from RAGTruth's.
    # Neither number is readable without both facts on the page.
    assert "llm judge" in lowered
    assert "sentence" in lowered
    assert "pos rate" in lowered
    # Both RAGTruth rates appear: 34.9% is the test split we compare against,
    # 43.1% is the whole-corpus figure the paper quotes. Conflating them would
    # overstate the base-rate gap.
    assert "34.9" in table
    assert "43.1" in table


def test_gap_column_is_relative_to_the_reference():
    reference = _row("RAGTruth test", 2700, 0.7623, positive_rate=0.431)
    table = format_ood_table([_row("covidqa", 246, 0.6623)], reference)
    assert "-0.1000" in table
