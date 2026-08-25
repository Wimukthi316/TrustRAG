"""The lexical baseline, its four floors, and parity with C3's original."""

from __future__ import annotations

import pytest

from src.c4_attribution import lexical


SENTENCES = [
    "The cat sat on the mat.",
    "A dog barked loudly at midnight near the river.",
    "Nothing at all happened afterwards.",
]


def test_parity_with_c3_grounding_when_it_is_importable():
    try:
        from src.c3_explanation.grounding import (
            best_matching_sentence,
            content_tokens as c3_content_tokens,
        )
    except Exception:  # pragma: no cover - C3 not on this branch
        pytest.skip("src.c3_explanation is not importable on this branch")

    for span in ("dog barked", "the cat", "river midnight", "unrelated words"):
        assert lexical.content_tokens(span) == c3_content_tokens(span)
        assert lexical.best_sentence(span, SENTENCES) == best_matching_sentence(
            span, SENTENCES
        )


def test_stopwords_match_c3_exactly_when_importable():
    try:
        from src.c3_explanation.grounding import STOPWORDS as c3_stopwords
    except Exception:  # pragma: no cover
        pytest.skip("src.c3_explanation is not importable on this branch")
    assert lexical.STOPWORDS == c3_stopwords


def test_best_sentence_ties_break_on_the_lowest_index():
    sentences = ["alpha beta", "alpha beta"]
    index, _ = lexical.best_sentence("alpha beta", sentences)
    assert index == 0


def test_best_sentence_handles_an_empty_pool():
    assert lexical.best_sentence("anything", []) == (None, 0.0)


def test_containment_is_over_content_tokens_not_strings():
    # The quote is paraphrased and re-punctuated; a string match scores zero,
    # containment does not. That difference is the whole reason Layer 1 exists.
    assert lexical.containment("dog barked loudly", SENTENCES[1]) == 1.0
    assert lexical.containment("dog barked loudly", SENTENCES[0]) == 0.0


def test_containment_of_an_empty_quote_is_zero_not_one():
    assert lexical.containment("", SENTENCES[0]) == 0.0
    assert lexical.containment("the of and", SENTENCES[0]) == 0.0


def test_the_four_floors_exist_and_behave_as_named():
    assert lexical.FLOOR_NAMES == ("first", "longest", "random", "abstain")
    assert lexical.floor_prediction("first", SENTENCES, 0) == 0
    assert lexical.floor_prediction("longest", SENTENCES, 0) == 1
    assert lexical.floor_prediction("abstain", SENTENCES, 0) is None
    picked = lexical.floor_prediction("random", SENTENCES, 7)
    assert 0 <= picked < len(SENTENCES)


def test_the_random_floor_is_seeded_and_reproduces():
    a = [lexical.floor_prediction("random", SENTENCES, s) for s in range(20)]
    b = [lexical.floor_prediction("random", SENTENCES, s) for s in range(20)]
    assert a == b


def test_an_unknown_floor_is_a_loud_failure():
    with pytest.raises(ValueError):
        lexical.floor_prediction("cheat", SENTENCES, 0)


def test_idf_is_local_to_the_record_so_nothing_leaks_across_records():
    weights_a = lexical.idf_weights(SENTENCES)
    weights_b = lexical.idf_weights(SENTENCES[:1])
    assert weights_a != weights_b
