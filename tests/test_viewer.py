"""Tests for the Localisation Viewer.

The colours are the whole point of the page, so what is tested here is that a
character gets the colour that honestly describes it - especially that gold text
the model failed to cover comes out red rather than being quietly absorbed into
the neighbouring highlight.
"""

from __future__ import annotations

import json

import pytest

from src.c1_detector.viewer import (
    CATEGORIES,
    IDEAL_CHARS,
    annotate,
    build_payload,
    pick_examples,
    render_page,
    segments,
)

ANSWER = "0123456789abcdefghij"


def record(rid, gold, pred, answer=ANSWER, task="qa"):
    return {"id": rid, "task_type": task, "answer": answer,
            "gold_spans": gold, "pred_spans": pred}


def classes(gold, pred, answer=ANSWER):
    return annotate(record("x", gold, pred, answer))["classes"]


def test_an_exact_match_is_green_throughout():
    result = classes([(0, 5)], [(0, 5)])
    assert result[:5] == ["exact"] * 5
    assert set(result[5:]) == {"plain"}


def test_gold_text_no_prediction_covers_is_red():
    assert classes([(0, 5)], [])[:5] == ["missed"] * 5


def test_a_prediction_touching_no_gold_is_grey():
    assert classes([], [(0, 5)])[:5] == ["spurious"] * 5


def test_a_short_prediction_leaves_the_uncovered_gold_red():
    """The under-coverage case, which is the finding the page exists to show."""
    result = classes([(0, 10)], [(3, 10)])
    assert result[0:3] == ["missed"] * 3
    assert result[3:10] == ["overlap"] * 7


def test_a_long_prediction_marks_the_overshoot_separately():
    result = classes([(3, 10)], [(0, 10)])
    assert result[0:3] == ["overreach"] * 3
    assert result[3:10] == ["overlap"] * 7


def test_every_character_gets_exactly_one_class():
    result = classes([(0, 4), (12, 16)], [(2, 8), (14, 20)])
    assert len(result) == len(ANSWER)
    assert set(result) <= {"exact", "overlap", "missed", "spurious", "overreach", "plain"}


def test_groups_carry_the_boundary_deltas():
    groups = annotate(record("x", [(0, 10)], [(3, 9)]))["groups"]
    assert groups[0]["category"] == "boundary"
    assert groups[0]["start_delta"] == 3
    assert groups[0]["end_delta"] == -1
    assert groups[0]["gold_text"] == ["0123456789"]
    assert groups[0]["pred_text"] == ["345678"]


def test_groups_omit_deltas_where_they_would_be_meaningless():
    groups = annotate(record("x", [(0, 4), (6, 10)], [(0, 10)]))["groups"]
    assert groups[0]["category"] == "merge"
    assert "start_delta" not in groups[0]


def test_segments_collapse_runs_and_preserve_the_text():
    runs = segments(ANSWER, classes([(0, 5)], [(0, 5)]))
    assert runs[0] == ("exact", "01234")
    assert "".join(text for _, text in runs) == ANSWER


@pytest.fixture
def corpus():
    """Enough variety that every category has a candidate."""
    filler = "x" * (IDEAL_CHARS - 20)
    long_answer = ANSWER + filler
    return [
        record("exact-1", [(0, 5)], [(0, 5)], long_answer),
        record("boundary-1", [(0, 10)], [(3, 10)], long_answer, "data2text"),
        record("split-1", [(0, 10)], [(0, 4), (6, 10)], long_answer),
        record("merge-1", [(0, 4), (6, 10)], [(0, 10)], long_answer),
        record("missed-1", [(0, 5)], [], long_answer),
        record("spurious-1", [], [(0, 5)], long_answer),
        record("too-short", [(0, 5)], [(0, 5)], "tiny"),
    ]


def test_every_category_gets_an_example(corpus):
    featured = {row["featured"] for row in pick_examples(corpus)}
    assert featured == set(CATEGORIES)


def test_answers_outside_the_readable_window_are_skipped(corpus):
    assert all(row["id"] != "too-short" for row in pick_examples(corpus))


def test_the_choice_is_reproducible(corpus):
    first = [row["id"] for row in pick_examples(corpus)]
    second = [row["id"] for row in pick_examples(corpus)]
    assert first == second


def test_no_example_appears_twice(corpus):
    ids = [row["id"] for row in pick_examples(corpus)]
    assert len(ids) == len(set(ids))


METRICS = {
    "overall": {
        "n_examples": 2700,
        "n_gold_spans": 1517,
        "n_pred_spans": 2390,
        "example": {"f1": 0.7623},
        "span_overlap": {"f1": 0.5063},
        "span_exact": {"f1": 0.1485},
    }
}
LOCALISATION = {
    "tokenisation_ceiling": {"span_exact": {"f1": 0.9967}},
    "overall": {"buckets": {"exact": {"components": 290, "gold": 290, "pred": 290}}},
}


def test_payload_carries_the_headline_and_the_ceiling(corpus):
    payload = build_payload(corpus, METRICS, LOCALISATION)
    assert payload["headline"]["span_exact_f1"] == 0.1485
    assert payload["headline"]["ceiling_span_exact_f1"] == 0.9967
    assert payload["examples"]


def test_payload_survives_a_missing_localisation_report(corpus):
    payload = build_payload(corpus, METRICS, None)
    assert "ceiling_span_exact_f1" not in payload["headline"]


def test_the_page_is_one_self_contained_file(corpus):
    page = render_page(build_payload(corpus, METRICS, LOCALISATION))
    assert page.startswith("<!doctype html>")
    assert "__PAYLOAD__" not in page
    assert "<script src=" not in page and "<link " not in page
    assert "0.9967" in page or "ceiling_span_exact_f1" in page


def test_the_payload_cannot_close_the_script_tag_early():
    """An answer containing </script> would otherwise break the page open."""
    hostile = [record("h", [(0, 9)], [(0, 9)], "</script> and the rest of it" + "y" * 200)]
    page = render_page(build_payload(hostile, METRICS, None))
    assert "</script>" not in page.split('id="payload"')[1].split("</script")[0]
    blob = page.split('type="application/json">')[1].split("</script>")[0]
    assert json.loads(blob.replace("<\\/", "</"))["examples"][0]["id"] == "h"
