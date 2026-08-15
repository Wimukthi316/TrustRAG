"""Tests for the Block A localisation decomposition.

The fixture below is deliberately hand-built rather than sampled from the real
probability dump: that file is gitignored, and a test that needs 12 MB of
artifacts to run is a test that stops running. Every expected count here was
worked out by hand, so a failure means the code changed, not that the data did.
"""

from __future__ import annotations

import json

import pytest

from src.c1_detector.evaluate_c1 import span_char_metrics, span_overlap_metrics
from src.c1_detector.localisation import (
    BUCKETS,
    boundary_shape,
    build_report,
    classify,
    decompose,
    edge_delta_bucket,
    load_probability_dump,
    overlap_components,
    reconcile,
    span_lengths,
    tokenisation_ceiling,
)


def record(rid, task, gold, pred, answer="0123456789"):
    return {
        "id": rid,
        "task_type": task,
        "answer": answer,
        "gold_spans": gold,
        "pred_spans": pred,
        "answer_offsets": [(i, i + 1) for i in range(len(answer))],
    }


@pytest.fixture
def cases():
    """One record per bucket, so every branch of classify() is exercised."""
    return [
        record("exact", "qa", [(0, 3)], [(0, 3)]),
        record("boundary", "qa", [(0, 5)], [(1, 5)]),
        record("merge", "qa", [(0, 3), (5, 8)], [(0, 8)]),
        record("split", "summarization", [(0, 8)], [(0, 3), (5, 8)]),
        record("missed", "summarization", [(0, 3)], []),
        record("spurious", "data2text", [], [(0, 3)]),
        record("tangled", "data2text", [(0, 4), (6, 10)], [(0, 7), (5, 10)]),
    ]


def test_disjoint_spans_form_separate_components():
    components = overlap_components([(0, 3), (10, 13)], [(0, 3), (10, 13)])
    assert sorted(components) == [([0], [0]), ([1], [1])]


def test_untouched_spans_get_their_own_component():
    assert overlap_components([(0, 3)], []) == [([0], [])]
    assert overlap_components([], [(0, 3)]) == [([], [0])]


def test_one_prediction_spanning_two_gold_spans_is_one_component():
    assert overlap_components([(0, 3), (5, 8)], [(0, 8)]) == [([0, 1], [0])]


def test_every_span_lands_in_exactly_one_component():
    gold = [(0, 4), (6, 10), (20, 24)]
    pred = [(0, 7), (5, 10), (30, 33)]
    components = overlap_components(gold, pred)
    assert sorted(i for c, _ in components for i in c) == [0, 1, 2]
    assert sorted(i for _, p in components for i in p) == [0, 1, 2]


@pytest.mark.parametrize(
    "n_gold,n_pred,identical,expected",
    [
        (1, 1, True, "exact"),
        (1, 1, False, "boundary"),
        (2, 1, False, "merge"),
        (3, 1, False, "merge"),
        (1, 2, False, "split"),
        (2, 2, False, "tangled"),
        (1, 0, False, "missed"),
        (0, 1, False, "spurious"),
    ],
)
def test_classify(n_gold, n_pred, identical, expected):
    assert classify(n_gold, n_pred, identical) == expected


def test_classify_rejects_an_empty_component():
    with pytest.raises(ValueError):
        classify(0, 0, False)


@pytest.mark.parametrize(
    "delta,expected",
    [(0, "0"), (1, "1"), (-1, "1"), (2, "2"), (4, "3-5"), (-5, "3-5"), (9, "6-10"), (40, ">10")],
)
def test_edge_delta_bucket_ignores_sign(delta, expected):
    assert edge_delta_bucket(delta) == expected


@pytest.mark.parametrize(
    "dstart,dend,expected",
    [
        (0, 0, "near"),
        (2, -2, "near"),
        (-30, 0, "overrun"),
        (0, 30, "overrun"),
        (30, 0, "undershoot"),
        (0, -30, "undershoot"),
        (-30, -30, "shifted"),
    ],
)
def test_boundary_shape(dstart, dend, expected):
    assert boundary_shape(dstart, dend, near=10) == expected


def test_decompose_counts_every_bucket(cases):
    buckets = decompose(cases)["buckets"]
    expected = {
        "exact": (1, 1, 1),
        "boundary": (1, 1, 1),
        "merge": (1, 2, 1),
        "split": (1, 1, 2),
        "tangled": (1, 2, 2),
        "missed": (1, 1, 0),
        "spurious": (1, 0, 1),
    }
    for name, (components, gold, pred) in expected.items():
        assert buckets[name] == {"components": components, "gold": gold, "pred": pred}, name


def test_buckets_account_for_every_span(cases):
    buckets = decompose(cases)["buckets"]
    assert sum(buckets[b]["gold"] for b in BUCKETS) == sum(len(r["gold_spans"]) for r in cases)
    assert sum(buckets[b]["pred"] for b in BUCKETS) == sum(len(r["pred_spans"]) for r in cases)


def test_reconcile_reproduces_the_greedy_overlap_metric(cases):
    derived = reconcile(decompose(cases)["buckets"])
    overlap = span_overlap_metrics(cases)
    assert derived["overlap_tp"] == overlap["tp"]
    assert derived["overlap_fp"] == overlap["fp"]
    assert derived["overlap_fn"] == overlap["fn"]


def test_exact_bucket_equals_the_exact_metric(cases):
    derived = reconcile(decompose(cases)["buckets"])
    assert derived["exact_tp"] == span_char_metrics(cases)["tp"]


def test_greedy_counts_overstate_misses_and_spurious_predictions(cases):
    """The distinction the plan had wrong, pinned so it cannot regress.

    span_overlap's fn is not "gold spans nothing touched" and its fp is not
    "predictions touching no gold": merges and splits inflate both.
    """
    derived = reconcile(decompose(cases)["buckets"])
    overlap = span_overlap_metrics(cases)
    assert derived["gold_never_touched"] < overlap["fn"]
    assert derived["pred_touching_no_gold"] < overlap["fp"]
    assert derived["surplus_gold_from_merges"] == 1
    assert derived["surplus_pred_from_splits"] == 1


def test_boundary_detail_is_only_collected_for_boundary_groups(cases):
    boundary = decompose(cases)["boundary"]
    assert boundary["n_groups"] == 1
    # gold (0, 5) against predicted (1, 5): start moved by 1, end unchanged.
    assert boundary["start_edge_delta"] == {"1": 1}
    assert boundary["end_edge_delta"] == {"0": 1}
    assert boundary["which_edge_wrong"] == {"start only": 1}
    assert boundary["pred_strictly_inside_gold"] == 1


def test_examples_are_kept_for_the_viewer(cases):
    examples = decompose(cases)["examples"]
    assert examples["merge"][0]["id"] == "merge"
    assert examples["merge"][0]["gold"] == [[0, 3], [5, 8]]
    assert examples["merge"][0]["pred"] == [[0, 8]]


def test_ceiling_is_perfect_when_gold_lands_on_token_boundaries():
    rows = [
        {
            "task_type": "qa",
            "answer": "hello world",
            "gold_spans": [(0, 5)],
            "answer_offsets": [(0, 5), (5, 11)],
        }
    ]
    ceiling = tokenisation_ceiling(rows)
    assert ceiling["span_exact"]["f1"] == 1.0
    assert ceiling["gold_width_delta_chars"] == {"0": 1}


def test_ceiling_records_the_widening_when_gold_ends_inside_a_token():
    rows = [
        {
            "task_type": "qa",
            "answer": "hello world",
            "gold_spans": [(0, 3)],
            "answer_offsets": [(0, 5), (5, 11)],
        }
    ]
    ceiling = tokenisation_ceiling(rows)
    assert ceiling["span_exact"]["tp"] == 0
    # the decoded span is (0, 5): two characters wider than the gold (0, 3)
    assert ceiling["gold_width_delta_chars"] == {"2": 1}


def test_span_lengths_separates_gold_from_predictions(cases):
    lengths = span_lengths(cases)
    assert lengths["ALL"]["gold_chars"]["n"] == sum(len(r["gold_spans"]) for r in cases)
    assert lengths["ALL"]["pred_chars"]["n"] == sum(len(r["pred_spans"]) for r in cases)
    assert set(lengths) == {"ALL", "data2text", "qa", "summarization"}


def test_load_probability_dump_reads_the_dict_span_format(tmp_path):
    path = tmp_path / "probabilities.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "1",
                "task_type": "qa",
                "answer": "abc",
                "gold_spans": [{"start": 0, "end": 3, "text": "abc"}],
                "pred_spans": [
                    {"start": 0, "end": 2, "text": "ab", "is_hallucinated": False}
                ],
                "answer_offsets": [[0, 1], [1, 2], [2, 3]],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_probability_dump(path)
    assert rows[0]["gold_spans"] == [(0, 3)]
    # is_hallucinated is C2's gold label, not a filter: the span is still a prediction
    assert rows[0]["pred_spans"] == [(0, 2)]


def test_build_report_confirms_it_reconciles(cases):
    report = build_report(cases)
    assert report["reconciles_with_measured_metrics"] is True
    assert report["n_gold_spans"] == 8
    assert report["n_pred_spans"] == 8
    assert set(report["per_task"]) == {"data2text", "qa", "summarization"}
