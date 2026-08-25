"""Layer 7: the abstention arithmetic, the floors, and the calibration discipline."""

from __future__ import annotations

import numpy as np
import pytest

from src.c4_attribution import no_evidence as ne
from src.c4_attribution.matching import (
    KIND_NO_EVIDENCE,
    KIND_NO_SUPERVISION,
    KIND_UNMATCHED,
    KIND_VERBATIM_UNIQUE,
    Match,
)


def _match(kind, gold=None):
    return Match(kind, gold, (gold,) if gold is not None else (), 0.0, 0, 5)


def test_the_scope_excludes_spans_the_annotator_said_nothing_about():
    matches = [
        _match(KIND_VERBATIM_UNIQUE, 1),
        _match(KIND_NO_EVIDENCE),
        _match(KIND_UNMATCHED),
        _match(KIND_NO_SUPERVISION),
    ]
    scope = ne.in_scope(matches)
    assert list(scope) == [1, 1, 1, 0]


def test_gold_marks_only_the_explicit_denials():
    matches = [_match(KIND_NO_EVIDENCE), _match(KIND_VERBATIM_UNIQUE, 0)]
    assert list(ne.gold_abstain(matches)) == [1, 0]


def test_confusion_cells_add_up_and_the_rates_are_of_abstaining():
    gold = np.asarray([1, 1, 0, 0])
    predicted = np.asarray([1, 0, 1, 0])
    block = ne.confusion(gold, predicted)
    assert (block["tp"], block["fp"], block["fn"], block["tn"]) == (1, 1, 1, 1)
    assert block["precision"] == pytest.approx(0.5)
    assert block["recall"] == pytest.approx(0.5)
    assert block["f1"] == pytest.approx(0.5)
    assert block["base_rate"] == pytest.approx(0.5)
    assert block["abstention_rate"] == pytest.approx(0.5)


def test_always_abstain_has_recall_one_and_precision_equal_to_the_base_rate():
    gold = np.asarray([1, 0, 0, 0])
    block = ne.confusion(gold, np.ones_like(gold))
    assert block["recall"] == pytest.approx(1.0)
    assert block["precision"] == pytest.approx(0.25)


def test_never_abstain_has_recall_zero_which_is_the_lexical_baselines_ceiling():
    gold = np.asarray([1, 1, 0, 0])
    block = ne.confusion(gold, np.zeros_like(gold))
    assert block["recall"] == pytest.approx(0.0)
    assert block["f1"] == pytest.approx(0.0)


def test_the_threshold_is_chosen_on_the_split_it_is_handed_and_nowhere_else():
    rng = np.random.default_rng(0)
    gold = rng.integers(0, 2, size=300)
    confidence = np.where(gold == 1, rng.random(300) * 0.4, 0.6 + rng.random(300) * 0.4)
    chosen = ne.choose_threshold(confidence, gold)
    assert chosen["threshold"] is not None
    assert chosen["f1"] > 0.8
    # Applying it elsewhere must not refit: same threshold, different data.
    other = rng.random(300)
    predicted = (other < chosen["threshold"]).astype(int)
    assert set(np.unique(predicted)) <= {0, 1}


def test_ties_in_f1_go_to_the_least_abstaining_threshold():
    gold = np.zeros(10, dtype=int)
    confidence = np.linspace(0, 1, 10)
    chosen = ne.choose_threshold(confidence, gold)
    assert chosen["threshold"] == pytest.approx(0.0)


def test_a_threshold_on_the_edge_of_the_grid_is_flagged_not_hidden():
    gold = np.ones(10, dtype=int)
    confidence = np.zeros(10)
    chosen = ne.choose_threshold(confidence, gold)
    assert "on_grid_edge" in chosen
    assert chosen["grid_size"] == len(ne.THRESHOLD_GRID)


def test_the_structural_zero_recall_of_the_lexical_baseline_is_stated_as_a_floor():
    class Row:
        response_id = "r"
        task = "qa"

        class label:
            label_type = "Evident Baseless Info"
            implicit_true = False

    rows = [Row(), Row()]
    matches = [_match(KIND_NO_EVIDENCE), _match(KIND_VERBATIM_UNIQUE, 0)]
    floors = ne.floors(rows, matches)
    assert floors["lexical_without_a_threshold"]["recall"] == 0.0
    assert "always_abstain" in floors and "never_abstain" in floors


def test_the_floors_are_reported_per_task_not_only_pooled():
    """The always-abstain floor is a function of the base rate, and the base
    rates differ by an order of magnitude between tasks. A system can clear the
    pooled floor and still lose to doing nothing on one of them, so the per-task
    floor has to be in the artefact for the comparison to be the right one."""

    class Row:
        def __init__(self, task):
            self.task = task
            self.response_id = "r"

            class label:
                label_type = "Evident Baseless Info"
                implicit_true = False

            self.label = label

    rows = [Row("qa"), Row("qa"), Row("qa"), Row("data2text")]
    matches = [
        _match(KIND_NO_EVIDENCE),
        _match(KIND_NO_EVIDENCE),
        _match(KIND_VERBATIM_UNIQUE, 0),
        _match(KIND_VERBATIM_UNIQUE, 0),
    ]
    block = ne.floors(rows, matches)
    assert set(block["by_task"]) == {"qa", "data2text"}
    assert block["by_task"]["qa"]["base_rate"] == pytest.approx(2 / 3)
    assert block["by_task"]["data2text"]["base_rate"] == pytest.approx(0.0)
    # The pooled floor is not the same number as either task's floor.
    assert block["by_task"]["qa"]["always_abstain"]["f1"] != block["always_abstain"]["f1"]
