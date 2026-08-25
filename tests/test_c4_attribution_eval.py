"""The scoring harness: the bootstrap, the exclusions, and the floors."""

from __future__ import annotations

import numpy as np
import pytest

from src.c4_attribution import attribution_eval as ev
from src.c4_attribution.matching import KIND_UNMATCHED, KIND_VERBATIM_UNIQUE, Match


def _match(gold, kind=KIND_VERBATIM_UNIQUE, n=5):
    return Match(kind, gold, (gold,) if gold is not None else (), 1.0, 1, n)


def test_bootstrap_agrees_with_c3s_when_it_is_importable():
    try:
        from src.c3_explanation.type_eval import bootstrap_by_response as c3_bootstrap
    except Exception:  # pragma: no cover - C3 not on this branch
        pytest.skip("src.c3_explanation is not importable on this branch")

    rng = np.random.default_rng(0)
    values = rng.random(200)
    ids = [f"r{i // 3}" for i in range(200)]

    def statistic(index):
        return float(values[index].mean())

    ours = ev.bootstrap_by_response(ids, statistic, resamples=200, seed=1)
    theirs = c3_bootstrap(ids, statistic, resamples=200, seed=1)
    assert ours["ci_low"] == pytest.approx(theirs["ci_low"])
    assert ours["ci_high"] == pytest.approx(theirs["ci_high"])


def test_the_bootstrap_resamples_responses_not_spans():
    # Every span in a response carries the same value, so resampling responses
    # must produce a WIDER interval than resampling rows would.
    values = np.asarray([0.0] * 50 + [1.0] * 50)
    clustered_ids = ["a"] * 50 + ["b"] * 50
    independent_ids = [f"r{i}" for i in range(100)]

    def statistic(index):
        return float(values[index].mean())

    clustered = ev.bootstrap_by_response(clustered_ids, statistic, resamples=500, seed=3)
    independent = ev.bootstrap_by_response(independent_ids, statistic, resamples=500, seed=3)
    assert clustered["sd"] > independent["sd"]


def test_rank_from_scores_ties_to_the_lowest_index():
    assert ev.rank_from_scores([0.5, 0.5, 0.1]) == (0, 1, 2)


def test_prediction_from_empty_scores_abstains():
    prediction = ev.prediction_from_scores([])
    assert prediction.top1 is None
    assert prediction.ranking == ()


def test_only_scorable_rows_are_scored():
    class Row:
        response_id = "r1"
        task = "qa"
        sentences = ["a", "b", "c"]

        class label:
            label_type = "Evident Conflict"
            implicit_true = False
            snippet = "b"

        response = type("R", (), {"context": "a b c"})()

    rows = [Row(), Row()]
    matches = [_match(1), _match(None, KIND_UNMATCHED)]
    predictions = [ev.Prediction((1, 0, 2), 1, 0.9), ev.Prediction((0, 1, 2), 0, 0.1)]
    scored = ev.score_rows(rows, matches, predictions)
    assert len(scored) == 1
    assert scored[0]["correct"] is True


def test_convention_b_scores_fewer_rows_than_convention_a():
    class Row:
        response_id = "r1"
        task = "qa"
        sentences = ["a", "b", "c"]

        class label:
            label_type = "Evident Conflict"
            implicit_true = False
            snippet = "b"

        response = type("R", (), {"context": "a b c"})()

    rows = [Row(), Row()]
    matches = [_match(1), Match("containment", 0, (0,), 0.95, 0, 3)]
    predictions = [ev.Prediction((1,), 1, 0.9), ev.Prediction((0,), 0, 0.9)]
    assert len(ev.score_rows(rows, matches, predictions, "A")) == 2
    assert len(ev.score_rows(rows, matches, predictions, "B")) == 1


def test_an_unknown_convention_is_a_loud_failure():
    with pytest.raises(ValueError):
        ev.score_rows([], [], [], convention="C")


def test_exclusions_add_up():
    matches = [
        _match(1),
        _match(None, KIND_UNMATCHED),
        _match(None, "no_evidence"),
        _match(None, "no_evidence_label"),
    ]
    block = ev.exclusions(matches)
    assert block["n_annotated_spans"] == 4
    assert block["n_scored"] == 1
    assert (
        block["excluded_unmatched"]
        + block["excluded_no_evidence"]
        + block["excluded_no_evidence_label"]
        + block["n_scored"]
        == 4
    )


def test_recall_at_k_is_non_decreasing_in_k():
    scored = [
        {"response_id": "r", "task": "qa", "label_type": "x", "correct": False,
         "rank": 4, "reciprocal_rank": 0.2, "returned_containment": 0.0,
         "returned_exact": False, "abstained": False, "in_gold_set": False,
         "index_in_range": True, "score": 0.1}
    ]
    block = ev.aggregate(scored, with_intervals=False)
    values = [block[f"recall_at_{k}"] for k in ev.RECALL_AT]
    assert values == sorted(values)
