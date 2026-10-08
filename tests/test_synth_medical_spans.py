"""Tests for the synthetic medical span generator's injectors.

Network-free: they call the injectors on hand-written strings, so they need no
datasets download. The one property that must always hold is that every injected
span slices back to its own text in the NEW answer.
"""

from __future__ import annotations

import random

from src.c1_detector.synth_medical_spans import (
    inject_one,
    _inject_direction,
    _inject_fabricated,
    _inject_number,
)


def _roundtrips(result):
    assert result is not None
    new_text, start, end, span_text, etype = result
    assert end > start
    assert new_text[start:end] == span_text
    assert etype in ("evident_conflict", "evident_baseless_info")
    return result


def test_number_injector_changes_a_number_and_roundtrips():
    rng = random.Random(0)
    new_text, start, end, span, etype = _roundtrips(
        _inject_number("The recommended dose was 5 mg given 3 times daily.", rng)
    )
    assert etype == "evident_conflict"
    assert span != "5" or new_text != "The recommended dose was 5 mg given 3 times daily."


def test_number_injector_handles_percent_and_decimal():
    rng = random.Random(1)
    _roundtrips(_inject_number("Response rate improved by 12.5% over baseline.", rng))


def test_number_injector_returns_none_without_numbers():
    assert _inject_number("No numeric content in this clinical sentence.", random.Random(0)) is None


def test_direction_injector_flips_a_word_and_roundtrips():
    rng = random.Random(0)
    new_text, start, end, span, etype = _roundtrips(
        _inject_direction("Serum levels increased significantly after treatment.", rng)
    )
    assert etype == "evident_conflict"
    # the flipped word is an antonym, not the original
    assert span.lower() not in ("increased", "significantly")


def test_direction_injector_preserves_capitalisation():
    rng = random.Random(0)
    new_text, start, end, span, _ = _roundtrips(
        _inject_direction("Increased expression was observed.", rng)
    )
    assert span[0].isupper()


def test_fabricated_injector_inserts_and_roundtrips():
    rng = random.Random(0)
    _roundtrips(
        _inject_fabricated("The cohort was followed for one year. Outcomes were recorded.", rng)
    )


def test_inject_one_prefers_in_place_conflict_over_fabrication():
    # Has both a number and a direction word, so the fabricated fallback must not fire.
    rng = random.Random(0)
    result = _roundtrips(inject_one("Risk decreased by 40% in the treated group.", rng))
    assert result[4] == "evident_conflict"


def test_inject_one_falls_back_to_fabrication():
    rng = random.Random(0)
    result = _roundtrips(inject_one("The patients were enrolled from several clinics.", rng))
    assert result[4] == "evident_baseless_info"


def test_inject_one_always_finds_something_for_real_text():
    # The fabricated fallback can insert anywhere, so any non-empty answer yields
    # exactly one span. build_records guards empty answers upstream, so None is
    # never reached in practice.
    _roundtrips(inject_one("Patients recovered within the follow-up window.", random.Random(0)))
