"""Tests for the synthetic medical span generator's injectors.

Network-free: they call the injectors on hand-written strings, so they need no
datasets download. The property that must always hold is that every injected span
slices back to its own text in the NEW answer.
"""

from __future__ import annotations

import random

from src.c1_detector.synth_medical_spans import (
    inject_one,
    _inject_direction,
    _inject_entity,
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
    original = "The recommended dose was 5 mg given 3 times daily."
    new_text, start, end, span, etype = _roundtrips(_inject_number(original, "", rng))
    assert etype == "evident_conflict"
    assert new_text != original


def test_number_injector_handles_percent_and_decimal():
    _roundtrips(_inject_number("Response rate improved by 12.5% over baseline.", "", random.Random(1)))


def test_number_injector_returns_none_without_numbers():
    assert _inject_number("No numeric content in this clinical sentence.", "", random.Random(0)) is None


def test_direction_injector_flips_a_word_and_roundtrips():
    rng = random.Random(0)
    _new, _s, _e, span, etype = _roundtrips(
        _inject_direction("Serum levels increased significantly after treatment.", "", rng)
    )
    assert etype == "evident_conflict"
    assert span.lower() not in ("increased", "significantly")


def test_direction_injector_preserves_capitalisation():
    _new, _s, _e, span, _ = _roundtrips(
        _inject_direction("Increased expression was observed.", "", random.Random(0))
    )
    assert span[0].isupper()


def test_entity_injector_swaps_for_a_context_entity():
    context = "passage 1:Metformin and Sitagliptin were compared in diabetic patients."
    answer = "Metformin reduced fasting glucose in the cohort."
    new_text, start, end, span, etype = _roundtrips(
        _inject_entity(answer, context, random.Random(0))
    )
    assert etype == "evident_conflict"
    # the swapped-in entity is one the context actually uses, and not the original
    assert span in ("Sitagliptin",) and span != "Metformin"


def test_entity_injector_none_when_no_shared_entity():
    # The answer has no capitalised entity that appears in the context.
    assert _inject_entity("the drug reduced glucose", "passage 1:Aspirin and Warfarin.", random.Random(0)) is None


def test_fabricated_injector_inserts_and_roundtrips():
    _roundtrips(
        _inject_fabricated("The cohort was followed for one year. Outcomes were recorded.", "", random.Random(0))
    )


def test_inject_one_prefers_in_place_conflict_over_fabrication():
    rng = random.Random(0)
    result = _roundtrips(inject_one("Risk decreased by 40% in the treated group.", "", rng))
    assert result[4] == "evident_conflict"


def test_inject_one_uses_entity_before_fabrication():
    # No number and no direction word, but a shared context entity, so the entity
    # injector should fire rather than the fabricated fallback.
    context = "passage 1:Ibuprofen and Naproxen were studied."
    result = _roundtrips(inject_one("Ibuprofen was administered to participants.", context, random.Random(0)))
    assert result[4] == "evident_conflict"


def test_inject_one_falls_back_to_fabrication():
    result = _roundtrips(inject_one("The participants were enrolled from several clinics.", "", random.Random(0)))
    assert result[4] == "evident_baseless_info"
