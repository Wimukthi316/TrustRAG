"""Layer 4: the pre-filter's bookkeeping, the three hypotheses, and the score direction."""

from __future__ import annotations

import pytest

from src.c4_attribution import scorer
from src.c4_attribution.dataset import ResponseRecord, SpanRecord
from src.c4_attribution.evidence_labels import EvidenceLabel


def _row() -> SpanRecord:
    answer = "Alpha runs fast. Beta walks slowly. Gamma sits."
    span = "walks slowly"
    start = answer.index(span)
    response = ResponseRecord(
        response_id="r1", source_id="s1", split="test", task="qa", model="m",
        question="q", context="One. Two. Three.", answer=answer,
    )
    label = EvidenceLabel(
        response_id="r1", start=start, end=start + len(span), span_text=span,
        label_type="Evident Conflict", raw_meta="", implicit_true=False,
        due_to_null=False, has_original=False, snippet=None, no_evidence=False,
        closing_marker=None,
    )
    response.labels = [label]
    return SpanRecord(response=response, label=label)


def test_the_three_hypothesis_constructions_are_distinct():
    row = _row()
    bare = scorer.hypothesis_text(row, "span")
    sentence = scorer.hypothesis_text(row, "sentence")
    marked = scorer.hypothesis_text(row, "marked")
    assert bare == "walks slowly"
    assert sentence == "Beta walks slowly."
    assert marked == "Beta << walks slowly >>."
    assert len({bare, sentence, marked}) == 3


def test_an_unknown_hypothesis_is_a_loud_failure():
    with pytest.raises(ValueError):
        scorer.hypothesis_text(_row(), "invented")


def test_prefilter_keeps_the_highest_scoring_sentences_ties_to_the_lowest_index():
    assert scorer.prefilter_indices([0.1, 0.9, 0.5, 0.9], 2) == (1, 3)
    assert scorer.prefilter_indices([0.5, 0.5, 0.5], 2) == (0, 1)


def test_prefilter_recall_counts_the_error_floor_it_creates():
    kept = [(0, 1), (2, 3), (4, 5)]
    gold = [1, 9, 4]
    block = scorer.prefilter_recall(kept, gold, 2)
    assert block["n"] == 3
    assert block["recall"] == pytest.approx(2 / 3)
    assert block["error_floor"] == pytest.approx(1 / 3)


def test_prefilter_recall_ignores_rows_with_no_gold_sentence():
    block = scorer.prefilter_recall([(0,), (0,)], [None, None], 1)
    assert block["n"] == 0


def test_recall_is_non_decreasing_in_k():
    kept = [tuple(range(10))] * 5
    gold = [0, 2, 4, 6, 9]
    recalls = [scorer.prefilter_recall(kept, gold, k)["recall"] for k in (1, 3, 5, 10)]
    assert recalls == sorted(recalls)


def test_backend_scores_zero_fill_the_sentences_the_prefilter_dropped():
    record = {
        "n_sentences": 5,
        "topk": [1, 3],
        "lexical": [0.1, 0.2, 0.3, 0.4, 0.5],
        "embed": [0.5, 0.4, 0.3, 0.2, 0.1],
        "nli": {"sentence": [[1, 0.8, 0.1], [3, 0.2, 0.7]]},
        "hhem": [[1, 0.9], [3, 0.05]],
    }
    entail = scorer.backend_scores(record, "nli_entail")
    assert entail == [0.0, 0.8, 0.0, 0.2, 0.0]
    assert scorer.backend_scores(record, "lexical") == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_the_signed_score_is_max_of_the_two_heads_not_entailment_alone():
    record = {
        "n_sentences": 2,
        "topk": [0, 1],
        "nli": {"sentence": [[0, 0.05, 0.90], [1, 0.30, 0.10]]},
    }
    entail = scorer.backend_scores(record, "nli_entail")
    signed = scorer.backend_scores(record, "nli_signed")
    # A contradicting sentence is the one a human would cite for a hallucinated
    # span, and the entailment head alone ranks it last.
    assert entail.index(max(entail)) == 1
    assert signed.index(max(signed)) == 0


def test_hhem_decisive_is_distance_from_no_opinion():
    record = {"n_sentences": 3, "topk": [0, 1, 2], "hhem": [[0, 0.5], [1, 0.99], [2, 0.01]]}
    decisive = scorer.backend_scores(record, "hhem_decisive")
    assert decisive[0] == pytest.approx(0.0)
    assert decisive[1] == pytest.approx(0.98)
    assert decisive[2] == pytest.approx(0.98)


def test_an_unknown_backend_is_a_loud_failure():
    with pytest.raises(ValueError):
        scorer.backend_scores({"n_sentences": 1}, "gpt5")


def test_the_non_commercial_model_is_named_and_rejected():
    # LO5 depends on every dependency being licence-clean for commercial use.
    # The rejection is the evidence the check was made.
    assert "Bespoke-MiniCheck-7B" in scorer.__doc__
    assert "CC BY-NC" in scorer.__doc__
    assert scorer.NLI_MODEL.startswith("MoritzLaurer/")


def test_only_the_nli_backend_varies_its_hypothesis():
    """Listing a backend three times with identical numbers would read as an
    ablation that found no difference. Only NLI was actually run three ways."""
    assert scorer.hypotheses_for("nli_entail") == scorer.HYPOTHESES
    assert scorer.hypotheses_for("nli_signed") == scorer.HYPOTHESES
    assert scorer.hypotheses_for("lexical") == ("sentence",)
    assert scorer.hypotheses_for("embed") == ("sentence",)
    assert scorer.hypotheses_for("hhem_support") == ("sentence",)
    assert scorer.hypotheses_for("hhem_decisive") == ("sentence",)
