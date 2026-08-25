"""Layer 11: fit on calib, apply to test once, and the four cells add up."""

from __future__ import annotations

import json

import pytest

from src.c4_attribution import fusion


ARTIFACT = fusion.ARTIFACT
DUMPS = (fusion.CALIB_DUMP, fusion.TEST_DUMP)
needs_c1 = pytest.mark.skipif(
    not ARTIFACT.exists() or not all(p.exists() for p in DUMPS),
    reason="C1's probability dumps or C2's artefact are not in this working copy",
)


def test_the_floors_are_the_ones_the_project_measured():
    assert fusion.FLOORS["trivial_always_positive"] == 0.5177
    assert fusion.FLOORS["c1_alone"] == 0.7623
    assert fusion.FLOORS["lettucedetect_large"] == 0.7918
    assert fusion.FLOORS["hhem_adapted"] == 0.7012


def test_a_gain_smaller_than_the_noise_is_reported_as_not_distinguishable():
    tiny = fusion.compare_to_floors(fusion.FLOORS["c1_alone"] + 0.005)
    assert tiny["distinguishable_from_zero"] is False
    assert "not distinguishable" in tiny["verdict"]

    real = fusion.compare_to_floors(fusion.FLOORS["c1_alone"] + 0.03)
    assert real["distinguishable_from_zero"] is True


def test_the_noise_band_is_c1s_own_measured_noise():
    assert fusion.NOISE_SEED_SD_POINTS == 0.384
    assert fusion.NOISE_BOOTSTRAP_WIDTH_POINTS == 4.18


def test_the_four_cells_partition_the_predicted_spans():
    cells = fusion.disagreement_cells([1, 1, 0, 0, 1], [1, 0, 1, 0, 0])
    assert cells["n"] == 5
    assert sum(cells["cells"].values()) == 5
    assert cells["cells"]["c1_flag_c4_none"] == 2
    assert cells["cells"]["c1_pass_c4_evidence"] == 1


def test_example_f1_matches_c1s_own_definition():
    from src.c1_detector.evaluate_c1 import prf

    gold = [1, 1, 0, 0]
    predicted = [1, 0, 1, 0]
    ours = fusion.example_f1(gold, predicted)
    theirs = prf(1, 1, 1)
    assert ours["f1"] == pytest.approx(theirs["f1"])
    assert ours["tn"] == 1


def _records_and_spans():
    records = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    spans = [
        {"response_id": "a", "start": 0, "end": 3, "decision": "flag"},
        {"response_id": "b", "start": 0, "end": 3, "decision": "flag"},
        {"response_id": "c", "start": 0, "end": 3, "decision": "abstain"},
        {"response_id": "d", "start": 0, "end": 3, "decision": "pass"},
    ]
    # C4 finds NO evidence for a and c; it finds evidence for b and d.
    flags = {
        ("a", 0, 3): True,
        ("b", 0, 3): False,
        ("c", 0, 3): True,
        ("d", 0, 3): False,
    }
    return records, spans, flags


def test_c1_alone_counts_every_predicted_span():
    records, spans, flags = _records_and_spans()
    assert fusion.fuse(records, spans, flags, "c1_alone") == {
        "a": 1, "b": 1, "c": 1, "d": 1
    }


def test_flag_only_drops_the_spans_c2_did_not_flag():
    records, spans, flags = _records_and_spans()
    assert fusion.fuse(records, spans, flags, "c1_c2_flag_only") == {
        "a": 1, "b": 1, "c": 0, "d": 0
    }


def test_the_c4_gate_removes_a_flag_that_has_supporting_evidence():
    records, spans, flags = _records_and_spans()
    assert fusion.fuse(records, spans, flags, "c1_c2_gated_by_c4") == {
        "a": 1, "b": 0, "c": 0, "d": 0
    }


def test_the_tie_break_only_ever_adds_to_flag_only():
    records, spans, flags = _records_and_spans()
    flag_only = fusion.fuse(records, spans, flags, "c1_c2_flag_only")
    tie_break = fusion.fuse(records, spans, flags, "c4_tie_break")
    assert all(tie_break[k] >= flag_only[k] for k in flag_only)
    # c was an abstention nobody had decided; C4 found nothing to support it.
    assert tie_break["c"] == 1 and flag_only["c"] == 0
    # d is a PASS and stays a PASS whatever C4 thinks.
    assert tie_break["d"] == 0


def test_an_unknown_fusion_rule_is_a_loud_failure():
    records, spans, flags = _records_and_spans()
    with pytest.raises(ValueError):
        fusion.fuse(records, spans, flags, "whatever_scores_best")


@needs_c1
def test_the_c2_rebuild_reproduces_the_stored_coverage_to_the_last_digit():
    from src.c4_attribution.dataset import read_jsonl

    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    rebuild = fusion.rebuild_c2_decisions(
        artifact, list(read_jsonl(fusion.CALIB_DUMP)), list(read_jsonl(fusion.TEST_DUMP))
    )
    assert rebuild["matches_recorded"] is True
    assert rebuild["rebuilt_coverage"] == pytest.approx(
        rebuild["recorded_coverage"], abs=1e-12
    )


@needs_c1
def test_c1_alone_recomputed_here_reproduces_its_reported_example_f1():
    from src.c4_attribution.dataset import read_jsonl

    records = list(read_jsonl(fusion.TEST_DUMP))
    gold = fusion.response_labels(records)
    predicted = {
        str(r["id"]): int(len(r.get("pred_spans") or []) > 0) for r in records
    }
    ids = list(gold)
    scores = fusion.example_f1([gold[i] for i in ids], [predicted[i] for i in ids])
    assert scores["f1"] == pytest.approx(0.7623, abs=5e-4)
