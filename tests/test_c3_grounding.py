"""Tests for the C3 grounding features.

The features are the honest baseline the transformer has to beat, so what
matters is that they mean what their names say and that they are computable
identically for a gold span and a predicted one.
"""

from __future__ import annotations

import pytest

from backend.app.services.detector import split_sentences as backend_split_sentences
from src.c3_explanation.grounding import (
    FEATURE_NAMES,
    best_matching_sentence,
    capitalised_tokens,
    content_tokens,
    feature_vector,
    idf_weights,
    number_tokens,
    sentence_scores,
    span_features,
    split_sentences,
    tokenise,
)

CONTEXT = (
    "Anne Frank died in Bergen-Belsen in 1945. "
    "The Red Cross archives were re-examined by researchers. "
    "Witnesses reported symptoms before February 7."
)


@pytest.mark.parametrize(
    "text",
    [
        CONTEXT,
        "One sentence only",
        "Ends with a question? Then more! And a stop.",
        "   leading and trailing   ",
        "",
    ],
)
def test_sentence_splitter_matches_the_serving_path(text):
    """One splitter in two files. If this fails, one of them was edited alone."""
    assert split_sentences(text) == backend_split_sentences(text)


def test_tokenise_and_content_tokens():
    assert tokenise("The Red Cross, in 1945.") == ["the", "red", "cross", "in", "1945"]
    assert "the" not in content_tokens("The Red Cross")
    assert "red" in content_tokens("The Red Cross")


def test_number_and_entity_tokens():
    assert number_tokens("died in 1945 aged 15") == ["1945", "15"]
    # The first token is skipped because a sentence-initial capital says nothing.
    assert capitalised_tokens("Anne Frank died in Bergen") == ["frank", "bergen"]


def test_idf_gives_a_rare_token_more_weight_than_a_common_one():
    sentences = ["the cat sat", "the cat ran", "the zebra sang"]
    weights = idf_weights(sentences)
    assert weights["zebra"] > weights["cat"]


def test_best_matching_sentence_picks_the_sentence_that_shares_content():
    sentences = split_sentences(CONTEXT)
    index, score = best_matching_sentence("the Red Cross archives", sentences)
    assert index == 1
    assert score > 0


def test_best_matching_sentence_breaks_ties_on_the_lowest_index():
    sentences = ["alpha beta", "alpha beta"]
    assert best_matching_sentence("alpha beta", sentences)[0] == 0


def test_best_matching_sentence_handles_an_empty_context():
    assert best_matching_sentence("anything", []) == (None, 0.0)


def test_sentence_scores_are_bounded():
    sentences = split_sentences(CONTEXT)
    for row in sentence_scores("Anne Frank died", sentences):
        for key in ("containment", "jaccard", "idf_overlap"):
            assert 0.0 <= row[key] <= 1.0


def test_unseen_token_rate_separates_baseless_from_conflict():
    """The one feature axis A is supposed to lean on.

    A span made of words that appear nowhere in the context is the baseless case;
    a span that contradicts a sentence still shares that sentence's vocabulary.
    """
    sentences = split_sentences(CONTEXT)
    baseless = span_features(
        "the museum issued a corrected commemorative postage stamp",
        "answer text",
        0,
        sentences,
        "summarization",
    )
    conflict = span_features(
        "Anne Frank died in Bergen-Belsen in 1943",
        "answer text",
        0,
        sentences,
        "summarization",
    )
    assert baseless["unseen_token_rate"] > conflict["unseen_token_rate"]
    assert conflict["max_containment"] > baseless["max_containment"]


def test_number_present_rate_catches_a_changed_figure():
    sentences = split_sentences(CONTEXT)
    wrong = span_features("died in 2022", "a", 0, sentences, "qa")
    right = span_features("died in 1945", "a", 0, sentences, "qa")
    assert wrong["number_present_rate"] == pytest.approx(0.0)
    assert right["number_present_rate"] == pytest.approx(1.0)
    assert wrong["has_number"] == 1.0


def test_a_span_with_no_numbers_is_not_penalised():
    """Absent evidence is not evidence of a mismatch, so the rate defaults to 1."""
    sentences = split_sentences(CONTEXT)
    features = span_features("the researchers", "a", 0, sentences, "qa")
    assert features["has_number"] == 0.0
    assert features["number_present_rate"] == pytest.approx(1.0)


def test_task_one_hot_is_exclusive():
    sentences = split_sentences(CONTEXT)
    for task, key in (
        ("qa", "task_qa"),
        ("data2text", "task_data2text"),
        ("summarization", "task_summarization"),
    ):
        features = span_features("anything", "a", 0, sentences, task)
        hot = [
            features[name]
            for name in ("task_qa", "task_data2text", "task_summarization")
        ]
        assert sum(hot) == 1.0
        assert features[key] == 1.0
    unknown = span_features("anything", "a", 0, sentences, "other")
    assert (
        unknown["task_qa"]
        + unknown["task_data2text"]
        + unknown["task_summarization"]
        == 0.0
    )


def test_relative_position_is_a_fraction_of_the_answer():
    sentences = split_sentences(CONTEXT)
    features = span_features("x", "0123456789", 5, sentences, "qa")
    assert features["span_relative_position"] == pytest.approx(0.5)


def test_feature_vector_matches_the_declared_order():
    sentences = split_sentences(CONTEXT)
    features = span_features("Anne Frank", "answer", 0, sentences, "qa")
    vector = feature_vector(features)
    assert len(vector) == len(FEATURE_NAMES)
    assert vector[FEATURE_NAMES.index("task_qa")] == 1.0


def test_features_are_computable_with_no_context_at_all():
    features = span_features("orphan span", "orphan span", 0, [], "qa")
    assert set(features) == set(FEATURE_NAMES)
    assert features["unseen_token_rate"] == pytest.approx(1.0)
    assert features["n_context_sentences"] == 0.0
