"""Layer 9: an off-by-one evidence_index has to be caught here or nowhere."""

from __future__ import annotations

import pytest

from src.c4_attribution.attribution_eval import Prediction
from src.c4_attribution.dataset import ResponseRecord, SpanRecord
from src.c4_attribution.evidence_labels import EvidenceLabel
from src.c4_attribution.faithfulness import check
from src.c4_attribution.matching import KIND_VERBATIM_UNIQUE, Match


def _row(context="Alpha runs. Beta walks. Gamma sits."):
    answer = "The answer mentions Beta."
    response = ResponseRecord(
        response_id="r1", source_id="s1", split="test", task="qa", model="m",
        question="q", context=context, answer=answer,
    )
    label = EvidenceLabel(
        response_id="r1", start=20, end=24, span_text="Beta",
        label_type="Evident Conflict", raw_meta="", implicit_true=False,
        due_to_null=False, has_original=True, snippet="Beta walks",
        no_evidence=False, closing_marker=None,
    )
    response.labels = [label]
    return SpanRecord(response=response, label=label)


def _match(gold=1):
    return Match(KIND_VERBATIM_UNIQUE, gold, (gold,), 1.0, 1, 3)


def test_a_correct_index_is_perfectly_self_faithful():
    block = check([_row()], [_match(1)], [Prediction((1, 0, 2), 1, 0.9)])
    assert block["self_faithfulness"] == pytest.approx(1.0)
    assert block["index_out_of_range"] == 0
    assert block["n_errors"] == 0


def test_an_off_by_one_shows_up_as_an_error_at_distance_one():
    block = check([_row()], [_match(1)], [Prediction((2, 1, 0), 2, 0.9)])
    assert block["n_errors"] == 1
    assert block["near_miss"] == 1
    assert block["mean_index_distance"] == pytest.approx(1.0)


def test_an_index_past_the_end_of_the_pool_is_counted_and_never_silent():
    block = check([_row()], [_match(1)], [Prediction((9,), 9, 0.9)])
    assert block["index_out_of_range"] == 1
    assert block["self_faithfulness"] < 1.0


def test_abstained_rows_are_not_counted_as_returned():
    block = check([_row()], [_match(1)], [Prediction((), None, 0.0)])
    assert block["n_returned"] == 0


def test_far_misses_and_near_misses_are_separated():
    rows = [_row(), _row()]
    matches = [_match(0), _match(0)]
    predictions = [Prediction((1,), 1, 0.5), Prediction((2,), 2, 0.5)]
    block = check(rows, matches, predictions)
    assert block["near_miss"] == 1
    assert block["far_miss"] == 1
    assert block["share_near_miss"] == pytest.approx(0.5)
