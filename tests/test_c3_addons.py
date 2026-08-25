"""Tests for C3's Layer 6 escalation count and the ranked add-ons.

Two of these are guards rather than unit tests. The C2 rebuild must reproduce the
coverage stored in C2's own serving artefact, because everything in Layer 6 is
computed from decisions that are not stored anywhere; and the risk-control loss
must be monotone in the threshold, because conformal risk control is only valid
for a monotone loss and a silently non-monotone one would produce a bound that
means nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.c3_explanation.addons import (
    IMPLICIT_TRUE_WEIGHT,
    SEVERITY_WEIGHTS,
    escalation_budget,
    generator_cross_tab,
    implicit_true_ablation,
    offsets,
    span_weight,
    type_conditional_calibration,
    type_conditional_detectability,
    weighted_loss_matrix,
)

CLASSES = [
    "evident_conflict",
    "subtle_conflict",
    "evident_baseless_info",
    "subtle_baseless_info",
]


def test_offsets_accepts_both_on_disk_shapes():
    assert offsets({"start": 3, "end": 9}) == (3, 9)
    assert offsets((3, 9)) == (3, 9)
    assert offsets([3, 9]) == (3, 9)


# --------------------------------------------------------------------------
# Layer 6
# --------------------------------------------------------------------------

C2_ROWS = [
    {"response_id": "r1", "start": 0, "end": 5, "decision": "abstain"},
    {"response_id": "r1", "start": 6, "end": 9, "decision": "abstain"},
    {"response_id": "r2", "start": 0, "end": 5, "decision": "flag"},
    {"response_id": "r2", "start": 6, "end": 9, "decision": "pass"},
]
SIZES = {
    ("r1", 0, 5): 3,
    ("r1", 6, 9): 1,
    ("r2", 0, 5): 4,
    ("r2", 6, 9): 2,
}


def test_escalation_needs_both_conditions():
    """Abstaining alone is not enough, and an ambiguous type alone is not either."""
    budget = escalation_budget(C2_ROWS, SIZES, alphas=(0.1,))
    cell = budget["per_alpha"]["0.1"]
    # r1/0-5 abstains with a set of 3 -> escalates.
    # r1/6-9 abstains but the set is a singleton -> does not.
    # r2/0-5 has a set of 4 but C2 flagged it -> does not.
    assert cell["n_escalated"] == 1
    assert budget["n_abstained"] == 2
    assert cell["share_of_abstained"] == pytest.approx(0.5)
    assert cell["share_of_all_spans"] == pytest.approx(0.25)


def test_escalation_budget_uses_the_set_sizes_for_each_alpha():
    per_alpha_sizes = {
        0.1: SIZES,
        0.4: {key: 1 for key in SIZES},
    }
    budget = escalation_budget(
        C2_ROWS, SIZES, alphas=(0.1, 0.4), set_sizes_by_alpha=per_alpha_sizes
    )
    assert budget["per_alpha"]["0.1"]["n_escalated"] == 1
    assert budget["per_alpha"]["0.4"]["n_escalated"] == 0


def test_escalation_budget_says_the_loop_was_not_run():
    budget = escalation_budget(C2_ROWS, SIZES, alphas=(0.1,))
    assert "not run" in budget["loop_status"]


# --------------------------------------------------------------------------
# add-on 1
# --------------------------------------------------------------------------


def test_type_conditional_calibration_splits_a_hidden_bad_class():
    """A model can be calibrated on average and confidently wrong inside a class.

    Class A is answered correctly at 0.9 confidence; class D is answered wrongly
    at the same confidence. The pooled ECE understates D on its own.
    """
    rows = []
    for i in range(50):
        rows.append(
            {
                "response_id": f"a{i}",
                "task_type": "qa",
                "classes": CLASSES,
                "error_type": CLASSES[0],
                "probs_flat": [0.9, 0.04, 0.03, 0.03],
            }
        )
    for i in range(50):
        rows.append(
            {
                "response_id": f"d{i}",
                "task_type": "qa",
                "classes": CLASSES,
                "error_type": CLASSES[3],
                "probs_flat": [0.9, 0.04, 0.03, 0.03],
            }
        )
    out = type_conditional_calibration(rows, "probs_flat", 1.0)
    good = out["by_error_type"][CLASSES[0]]
    bad = out["by_error_type"][CLASSES[3]]
    assert good["accuracy"] == pytest.approx(1.0)
    assert bad["accuracy"] == pytest.approx(0.0)
    assert bad["ece"] > out["overall"]["ece"]
    assert good["n"] == 50 and bad["n"] == 50


def test_type_conditional_calibration_on_no_rows_is_empty_not_zero():
    assert type_conditional_calibration([], "probs_flat", 1.0) == {}


# --------------------------------------------------------------------------
# add-on 2
# --------------------------------------------------------------------------


def test_span_weight_encodes_the_stated_value_judgement():
    assert span_weight("evident_conflict", False) == SEVERITY_WEIGHTS["evident"]
    assert span_weight("subtle_baseless_info", False) == SEVERITY_WEIGHTS["subtle"]
    assert span_weight("evident_conflict", True) == IMPLICIT_TRUE_WEIGHT
    assert span_weight(None, False) == 1.0
    assert span_weight("not a type", False) == 1.0


RECORDS = [
    {
        "id": "r1",
        "gold_spans": [{"start": 0, "end": 10}, {"start": 20, "end": 30}],
        "pred_spans": [{"start": 0, "end": 10}, {"start": 20, "end": 30}],
    },
    {
        "id": "r2",
        "gold_spans": [{"start": 0, "end": 10}],
        "pred_spans": [],
    },
]
GOLD = {
    "r1": {
        (0, 10): {"error_type": "evident_conflict", "implicit_true": False},
        (20, 30): {"error_type": "subtle_baseless_info", "implicit_true": False},
    },
    "r2": {(0, 10): {"error_type": "evident_conflict", "implicit_true": False}},
}
SCORES = {("r1", 0, 10): 0.9, ("r1", 20, 30): 0.4}


def test_weighted_loss_is_monotone_in_the_threshold():
    """Conformal risk control is only valid for a monotone loss."""
    thresholds = np.linspace(0.0, 1.0, 21)
    losses, kept = weighted_loss_matrix(RECORDS, SCORES, GOLD, thresholds)
    assert kept == ["r1", "r2"]
    for row in losses:
        assert list(row) == sorted(row)


def test_weighted_loss_is_hand_computable():
    thresholds = np.asarray([0.0, 0.5, 0.95])
    losses, _ = weighted_loss_matrix(RECORDS, SCORES, GOLD, thresholds)
    # r1 weights: evident 1.0 (score 0.9), subtle 0.5 (score 0.4); total 1.5
    #   t = 0.00 -> nothing missed                      -> 0
    #   t = 0.50 -> the subtle span is missed           -> 0.5 / 1.5
    #   t = 0.95 -> both missed                         -> 1.0
    assert losses[0][0] == pytest.approx(0.0)
    assert losses[0][1] == pytest.approx(0.5 / 1.5)
    assert losses[0][2] == pytest.approx(1.0)
    # r2 has a gold span nothing predicted, so it is missed at every threshold.
    assert list(losses[1]) == pytest.approx([1.0, 1.0, 1.0])


def test_an_implicit_true_span_costs_nothing_to_miss():
    gold = {"r1": {(0, 10): {"error_type": "evident_conflict", "implicit_true": True}}}
    records = [{"id": "r1", "gold_spans": [{"start": 0, "end": 10}], "pred_spans": []}]
    losses, _ = weighted_loss_matrix(records, {}, gold, np.asarray([0.0, 1.0]))
    assert list(losses[0]) == pytest.approx([0.0, 0.0])


def test_responses_with_no_gold_span_can_be_kept_or_dropped():
    records = [{"id": "r0", "gold_spans": [], "pred_spans": []}] + RECORDS
    with_gold, kept = weighted_loss_matrix(
        records, SCORES, GOLD, np.asarray([0.5]), only_responses_with_gold=True
    )
    everything, kept_all = weighted_loss_matrix(
        records, SCORES, GOLD, np.asarray([0.5]), only_responses_with_gold=False
    )
    assert kept == ["r1", "r2"]
    assert kept_all == ["r0", "r1", "r2"]
    assert with_gold.shape[0] == 2 and everything.shape[0] == 3


# --------------------------------------------------------------------------
# add-ons 3, 4, 5
# --------------------------------------------------------------------------


def test_generator_cross_tab_normalises_within_each_model():
    from collections import Counter

    rows = [
        {"model": "gpt-4", "error_type": "evident_baseless_info"},
        {"model": "gpt-4", "error_type": "evident_baseless_info"},
        {"model": "gpt-4", "error_type": "evident_conflict"},
        {"model": "llama", "error_type": "evident_conflict"},
    ]
    out = generator_cross_tab(rows, Counter({"gpt-4": 6, "llama": 2}))
    gpt = out["by_model"]["gpt-4"]
    assert gpt["n_spans"] == 3
    assert gpt["spans_per_response"] == pytest.approx(0.5)
    assert sum(gpt["rates"].values()) == pytest.approx(1.0)
    assert gpt["rates"]["evident_baseless_info"] == pytest.approx(2 / 3)


def test_type_conditional_detectability_pairs_recall_with_set_size():
    records = [
        {
            "id": "r1",
            "gold_spans": [{"start": 0, "end": 10}, {"start": 50, "end": 60}],
            "pred_spans": [{"start": 2, "end": 8}],
        }
    ]
    gold = {
        "r1": {
            (0, 10): {"error_type": "evident_conflict"},
            (50, 60): {"error_type": "subtle_conflict"},
        }
    }
    out = type_conditional_detectability(records, gold, {("r1", 2, 8): 3})
    assert out["by_error_type"]["evident_conflict"]["overlap_recall"] == pytest.approx(1.0)
    assert out["by_error_type"]["evident_conflict"]["mean_conformal_set_size"] == (
        pytest.approx(3.0)
    )
    assert out["by_error_type"]["subtle_conflict"]["overlap_recall"] == pytest.approx(0.0)


def test_implicit_true_ablation_only_changes_responses_it_empties():
    records = [
        {"id": "a", "gold_spans": [{"start": 0, "end": 5}], "pred_spans": [{"start": 0, "end": 5}]},
        {"id": "b", "gold_spans": [{"start": 0, "end": 5}], "pred_spans": []},
        {"id": "c", "gold_spans": [], "pred_spans": [{"start": 0, "end": 5}]},
    ]
    gold = {
        "a": {(0, 5): {"implicit_true": True}},
        "b": {(0, 5): {"implicit_true": False}},
        "c": {},
    }
    out = implicit_true_ablation(records, gold, "implicit_true")
    assert out["n_spans_flagged"] == 1
    assert out["n_responses_that_change_label"] == 1
    # Before: a is tp, b is fn, c is fp -> P 1/2, R 1/2, F1 1/2
    assert out["example_before"]["f1"] == pytest.approx(0.5)
    # After: a becomes fp, so tp 0 -> F1 0
    assert out["example_after"]["f1"] == pytest.approx(0.0)
    assert out["delta_f1"] == pytest.approx(-0.5)
