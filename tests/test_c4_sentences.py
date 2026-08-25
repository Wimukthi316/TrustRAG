"""The splitter is part of the serving contract. This is the third assertion of that."""

from __future__ import annotations

from backend.app.services.detector import split_sentences as backend_split
from src.c4_attribution import sentences as c4


def test_c4_splitter_is_identical_to_the_serving_path():
    samples = [
        "One. Two! Three? Four.",
        "  leading and trailing  ",
        "No terminal punctuation",
        "Dr. Smith went home. Then he slept.",
        "",
    ]
    for text in samples:
        assert c4.split_sentences(text) == backend_split(text)


def test_c3_splitter_is_identical_too_when_it_is_importable():
    try:
        from src.c3_explanation.grounding import split_sentences as c3_split
    except Exception:  # pragma: no cover - C3 not on this branch
        import pytest

        pytest.skip("src.c3_explanation is not importable on this branch")
    text = "Alpha runs. Beta walks! Gamma sits?"
    assert c4.split_sentences(text) == c3_split(text)


def test_offsets_round_trip():
    text = "Alpha runs. Beta walks. Gamma sits."
    pieces = c4.split_sentences(text)
    offsets = c4.sentence_offsets(text, pieces)
    assert [text[a:b] for a, b in offsets] == pieces


def test_index_of_offset_finds_the_containing_sentence():
    text = "Alpha runs. Beta walks. Gamma sits."
    pieces = c4.split_sentences(text)
    offsets = c4.sentence_offsets(text, pieces)
    assert c4.index_of_offset(offsets, 0) == 0
    assert c4.index_of_offset(offsets, text.index("Beta")) == 1
    assert c4.index_of_offset(offsets, len(text) + 5) == -1


def test_the_variant_splitter_breaks_where_the_naive_one_cannot():
    # No whitespace after the full stop, so the naive splitter walks through it.
    text = "passage 1:Alpha runs.passage 2:Beta walks."
    assert len(c4.split_sentences(text)) == 1
    assert len(c4.split_sentences_variant(text)) == 2


def test_the_variant_splitter_also_breaks_on_a_blank_line():
    text = "A heading with no full stop\n\nAlpha runs."
    assert len(c4.split_sentences_variant(text)) == 2


def test_realign_maps_a_naive_sentence_into_the_variant_pool():
    naive = ["passage 1:Alpha runs.passage 2:Beta walks."]
    variant = ["passage 1:Alpha runs.", "passage 2:Beta walks."]
    assert c4.realign(naive, variant, 0) == 0
    assert c4.realign(naive, variant, 5) == -1
