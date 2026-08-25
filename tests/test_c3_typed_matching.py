"""Tests for C3 Layer 2, the typed span matching protocol.

The protocol's only job is to be defensible, so the tests check the three things
a panel would poke at: that the hard case is handled by a stated rule rather than
by whichever gold span happened to come first, that the rule is deterministic,
and that a prediction with no gold span to inherit from is excluded rather than
guessed at.
"""

from __future__ import annotations

import json

import pytest

from src.c3_explanation.typed_matching import (
    FAILURE,
    MAJORITY,
    UNASSIGNED,
    assign_types_for_record,
    build_report,
    build_rows,
    format_report,
    gold_types_by_record,
    main,
    majority_type_by_characters,
    overlap_chars,
)

CONFLICT = "evident_conflict"
BASELESS = "evident_baseless_info"
SUBTLE = "subtle_baseless_info"


def by_offsets(rows):
    return {(r["start"], r["end"]): r for r in rows}


def test_overlap_chars_is_half_open():
    assert overlap_chars((0, 10), (5, 15)) == 5
    assert overlap_chars((0, 10), (10, 20)) == 0
    assert overlap_chars((0, 10), (20, 30)) == 0


def test_exact_and_boundary_inherit_the_single_gold_type():
    out = assign_types_for_record(
        gold=[(0, 10), (20, 30)],
        gold_types=[[CONFLICT], [BASELESS]],
        pred=[(0, 10), (22, 30)],
    )
    preds = by_offsets(out["predictions"])
    assert preds[(0, 10)]["bucket"] == "exact"
    assert preds[(0, 10)][MAJORITY] == CONFLICT
    assert preds[(22, 30)]["bucket"] == "boundary"
    assert preds[(22, 30)][MAJORITY] == BASELESS
    assert all(p["typeable"] for p in out["predictions"])
    assert not any(p["crosses_type_boundary"] for p in out["predictions"])


def test_split_gives_every_fragment_the_same_gold_type():
    out = assign_types_for_record(
        gold=[(0, 20)], gold_types=[[SUBTLE]], pred=[(0, 8), (12, 20)]
    )
    assert {p["bucket"] for p in out["predictions"]} == {"split"}
    assert {p[MAJORITY] for p in out["predictions"]} == {SUBTLE}
    assert {p[FAILURE] for p in out["predictions"]} == {SUBTLE}


def test_merge_within_one_type_is_not_the_hard_case():
    out = assign_types_for_record(
        gold=[(0, 10), (12, 20)],
        gold_types=[[CONFLICT], [CONFLICT]],
        pred=[(0, 20)],
    )
    prediction = out["predictions"][0]
    assert prediction["bucket"] == "merge"
    assert prediction["crosses_type_boundary"] is False
    assert prediction[MAJORITY] == CONFLICT
    assert prediction[FAILURE] == CONFLICT


def test_merge_across_types_reports_both_conventions():
    """The footnote-6 case. Neither convention is applied silently."""
    out = assign_types_for_record(
        gold=[(0, 10), (12, 20)],
        gold_types=[[CONFLICT], [BASELESS]],
        pred=[(0, 20)],
    )
    prediction = out["predictions"][0]
    assert prediction["bucket"] == "merge"
    assert prediction["crosses_type_boundary"] is True
    # 10 characters of conflict against 8 of baseless.
    assert prediction[MAJORITY] == CONFLICT
    assert prediction[FAILURE] == UNASSIGNED


def test_character_majority_breaks_ties_on_the_sorted_name():
    """A tie must not depend on gold order, dict order or insertion order."""
    forward = majority_type_by_characters(
        (0, 21), [(0, 10), (11, 21)], ["z_conflict", "a_baseless"]
    )
    backward = majority_type_by_characters(
        (0, 21), [(11, 21), (0, 10)], ["a_baseless", "z_conflict"]
    )
    assert forward == backward == "a_baseless"


def test_a_spurious_prediction_is_excluded_not_guessed():
    out = assign_types_for_record(
        gold=[(0, 10)], gold_types=[[CONFLICT]], pred=[(0, 10), (50, 60)]
    )
    preds = by_offsets(out["predictions"])
    spurious = preds[(50, 60)]
    assert spurious["bucket"] == "spurious"
    assert spurious["typeable"] is False
    assert spurious[MAJORITY] is None
    assert spurious[FAILURE] is None


def test_a_missed_gold_span_is_recorded_with_no_prediction():
    out = assign_types_for_record(
        gold=[(0, 10), (50, 60)],
        gold_types=[[CONFLICT], [BASELESS]],
        pred=[(0, 10)],
    )
    gold = by_offsets(out["gold"])
    assert gold[(50, 60)]["bucket"] == "missed"
    assert gold[(50, 60)]["n_pred_in_component"] == 0


def test_a_gold_span_holding_two_types_is_flagged_separately():
    out = assign_types_for_record(
        gold=[(0, 10)], gold_types=[[CONFLICT, BASELESS]], pred=[(0, 10)]
    )
    assert out["gold"][0]["gold_multi_type"] is True
    # sorted()[0] of the two, so the primary is deterministic
    assert out["gold"][0]["error_type"] == sorted([CONFLICT, BASELESS])[0]


def test_mismatched_type_lists_are_a_loud_error():
    with pytest.raises(ValueError):
        assign_types_for_record(gold=[(0, 10)], gold_types=[], pred=[])


# --------------------------------------------------------------------------
# Report level
# --------------------------------------------------------------------------

RECORDS = [
    {
        "id": "1",
        "task_type": "qa",
        "model": "m1",
        "gold_spans": [(0, 10), (12, 20)],
        "pred_spans": [(0, 20)],
    },
    {
        "id": "2",
        "task_type": "summarization",
        "model": "m2",
        "gold_spans": [(0, 10)],
        "pred_spans": [(0, 10), (50, 60)],
    },
    {
        "id": "3",
        "task_type": "qa",
        "model": "m1",
        "gold_spans": [(0, 10)],
        "pred_spans": [],
    },
]

TYPES = {
    "1": {(0, 10): [CONFLICT], (12, 20): [BASELESS]},
    "2": {(0, 10): [SUBTLE]},
    "3": {(0, 10): [CONFLICT]},
}


@pytest.fixture()
def report():
    return build_report(build_rows(RECORDS, TYPES))


def test_bucket_counts_reconcile_with_the_span_totals(report):
    assert sum(report["predicted_by_bucket"].values()) == report["n_predicted_spans"]
    assert sum(report["gold_by_bucket"].values()) == report["n_gold_spans"]
    assert report["n_predicted_spans"] == 3
    assert report["n_gold_spans"] == 4
    assert report["n_records_with_spans"] == 3


def test_typeable_and_excluded_partition_the_predictions(report):
    block = report["typeable"]
    assert block["n"] + block["n_excluded_spurious"] == report["n_predicted_spans"]
    assert block["n"] == 2
    assert block["n_excluded_spurious"] == 1


def test_footnote_6_counts_are_hand_computed(report):
    f6 = report["footnote_6"]
    assert f6["n_multi_gold_components"] == 1
    assert f6["n_multi_gold_crossing"] == 1
    assert f6["crossing_rate_of_multi_gold"] == pytest.approx(1.0)
    assert f6["n_typeable_predictions_affected"] == 1
    assert f6["affected_rate_of_typeable"] == pytest.approx(0.5)
    assert report["conventions_disagree_on"] == 1


def test_type_mix_is_reported_under_both_conventions(report):
    mix = report["type_mix"]
    assert mix[MAJORITY] == {CONFLICT: 1, SUBTLE: 1}
    assert mix[FAILURE] == {SUBTLE: 1, UNASSIGNED: 1}


def test_unmatched_gold_is_counted_not_silently_dropped():
    rows = build_rows(RECORDS, {"1": TYPES["1"]})
    report = build_report(rows)
    assert report["unmatched_gold_spans"] == 2
    assert "WARNING" in format_report(report)


def test_gold_types_by_record_prefers_the_full_type_list():
    processed = [
        {
            "id": "9",
            "spans": [
                {"start": 0, "end": 5, "error_type": CONFLICT, "error_types": [CONFLICT, BASELESS]},
                {"start": 6, "end": 9, "error_type": SUBTLE},
            ],
        }
    ]
    table = gold_types_by_record(processed)
    assert table["9"][(0, 5)] == [CONFLICT, BASELESS]
    assert table["9"][(6, 9)] == [SUBTLE]


def test_main_writes_all_three_artefacts(tmp_path):
    # load_probability_dump reads the on-disk shape, where a span is an object
    # with start and end rather than a pair.
    def as_dump(record):
        return {
            **record,
            "gold_spans": [{"start": s, "end": e} for s, e in record["gold_spans"]],
            "pred_spans": [{"start": s, "end": e} for s, e in record["pred_spans"]],
        }

    probs = tmp_path / "probabilities.jsonl"
    probs.write_text(
        "\n".join(json.dumps(as_dump(r)) for r in RECORDS) + "\n", encoding="utf-8"
    )
    processed = tmp_path / "processed.jsonl"
    processed.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": rid,
                    "spans": [
                        {"start": s, "end": e, "error_types": t}
                        for (s, e), t in table.items()
                    ],
                }
            )
            for rid, table in TYPES.items()
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    assert (
        main(
            [
                "--probs",
                str(probs),
                "--processed",
                str(processed),
                "--split",
                "unit",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )
    written = json.loads(
        (out_dir / "typed_matching_unit.json").read_text(encoding="utf-8")
    )
    assert written["n_predicted_spans"] == 3
    assert (out_dir / "typed_predictions_unit.jsonl").exists()
    assert (out_dir / "typed_gold_unit.jsonl").exists()
