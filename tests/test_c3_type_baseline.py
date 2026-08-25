"""Tests for C3's CPU feature baseline.

The baseline is also the shipped serving model, so the test that matters most is
the round trip: a model stored as JSON coefficients must score a span exactly as
scikit-learn would, because the FastAPI path scores with the JSON and every
reported number comes from scikit-learn.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.c3_explanation.type_baseline import (
    ERROR_TYPES,
    MECHANISMS,
    SEVERITIES,
    axis_product,
    fit_logistic,
    gold_rows,
    matrix,
    predict_labels,
    predict_proba,
    predicted_rows,
    primary_type,
    probability_rows,
    reorder,
    score,
    train,
)

CONTEXT = (
    "Anne Frank died in Bergen-Belsen in 1945. "
    "The Red Cross archives were re-examined. "
    "Witnesses reported symptoms before February 7."
)


def record(rid, split, task, spans):
    answer = "Anne Frank died in 2022 and the museum issued a stamp about it."
    return {
        "id": rid,
        "split": split,
        "task_type": task,
        "model": "gpt-4-0613",
        "context": CONTEXT,
        "answer": answer,
        "spans": spans,
    }


def span(start, end, error_types, **extra):
    answer = "Anne Frank died in 2022 and the museum issued a stamp about it."
    return {
        "start": start,
        "end": end,
        "text": answer[start:end],
        "error_types": error_types,
        **extra,
    }


CORPUS = [
    record("1", "train", "qa", [span(0, 23, ["evident_conflict"])]),
    record("2", "train", "summarization", [span(28, 62, ["evident_baseless_info"])]),
    record("3", "calib", "qa", [span(0, 23, ["subtle_baseless_info"])]),
    record("4", "test", "data2text", [span(28, 62, ["subtle_conflict"])]),
]


def test_primary_type_is_deterministic_when_preprocessing_merged_two():
    assert primary_type({"error_types": ["subtle_conflict", "evident_conflict"]}) == (
        "evident_conflict"
    )
    assert primary_type({"error_type": "subtle_conflict"}) == "subtle_conflict"
    assert primary_type({}) == "unknown"


def test_gold_rows_respects_the_frozen_split_ids():
    split_of = {"1": "train", "2": "calib", "3": "calib", "4": "test"}
    train_rows = gold_rows(CORPUS, split_of, keep_splits=("train",))
    calib_rows = gold_rows(CORPUS, split_of, keep_splits=("calib",))
    assert [r["response_id"] for r in train_rows] == ["1"]
    assert sorted(r["response_id"] for r in calib_rows) == ["2", "3"]


def test_gold_rows_carry_both_axes_and_the_confound_flags():
    rows = gold_rows(
        [record("9", "train", "qa", [span(0, 23, ["subtle_baseless_info"], implicit_true=True)])]
    )
    assert rows[0]["mechanism"] == "baseless"
    assert rows[0]["severity"] == "subtle"
    assert rows[0]["implicit_true"] is True
    assert rows[0]["source"] == "gold"


def test_gold_rows_skip_a_span_with_no_usable_type():
    assert gold_rows([record("9", "train", "qa", [span(0, 5, [])])]) == []


# --------------------------------------------------------------------------
# The stored model
# --------------------------------------------------------------------------


@pytest.fixture()
def fitted():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 16))
    # A label that depends on one column, so the fit is not degenerate.
    labels = ["conflict" if row[0] > 0 else "baseless" for row in features]
    return features, labels, fit_logistic(features, labels, MECHANISMS)


def test_stored_model_reproduces_sklearn_exactly(fitted):
    """The serving path scores with this JSON. It must not drift from the fit."""
    from sklearn.linear_model import LogisticRegression

    features, labels, model = fitted
    mean = np.asarray(model["mean"])
    scale = np.asarray(model["scale"])
    reference = LogisticRegression(max_iter=2000, random_state=42)
    reference.fit((features - mean) / scale, labels)

    ours = predict_proba(model, features)
    theirs = reference.predict_proba((features - mean) / scale)
    order = [model["classes"].index(c) for c in reference.classes_]
    assert np.allclose(ours[:, order], theirs, atol=1e-9)
    assert predict_labels(model, features) == list(reference.predict((features - mean) / scale))


def test_stored_model_is_json_round_trippable(fitted):
    features, _, model = fitted
    revived = json.loads(json.dumps(model))
    assert np.allclose(predict_proba(revived, features), predict_proba(model, features))


def test_binary_fit_is_stored_with_one_row_per_class(fitted):
    _, _, model = fitted
    assert len(model["coef"]) == 2
    assert len(model["intercept"]) == 2
    assert sorted(model["classes"]) == sorted(MECHANISMS)


def test_probabilities_sum_to_one(fitted):
    features, _, model = fitted
    assert np.allclose(predict_proba(model, features).sum(axis=1), 1.0)


def test_predict_proba_on_an_empty_matrix_returns_an_empty_matrix(fitted):
    _, _, model = fitted
    assert predict_proba(model, np.empty((0, 16))).shape == (0, 2)


# --------------------------------------------------------------------------
# The two-axis product
# --------------------------------------------------------------------------


def test_axis_product_is_the_outer_product_in_error_type_order():
    mechanism = np.asarray([[0.25, 0.75]])  # baseless, conflict
    severity = np.asarray([[0.4, 0.6]])  # evident, subtle
    out = axis_product(mechanism, list(MECHANISMS), severity, list(SEVERITIES))
    assert out.sum() == pytest.approx(1.0)
    expected = {
        "evident_conflict": 0.75 * 0.4,
        "subtle_conflict": 0.75 * 0.6,
        "evident_baseless_info": 0.25 * 0.4,
        "subtle_baseless_info": 0.25 * 0.6,
    }
    for column, name in enumerate(ERROR_TYPES):
        assert out[0, column] == pytest.approx(expected[name])


def test_reorder_puts_columns_in_the_declared_order():
    probs = np.asarray([[0.1, 0.9]])
    out = reorder(probs, ["conflict", "baseless"], ["baseless", "conflict", "other"])
    assert list(out[0]) == pytest.approx([0.9, 0.1, 0.0])


# --------------------------------------------------------------------------
# Predicted spans
# --------------------------------------------------------------------------

TYPED = [
    {
        "response_id": "4",
        "start": 28,
        "end": 62,
        "typeable": True,
        "bucket": "boundary",
        "crosses_type_boundary": False,
        "majority_by_characters": "evident_baseless_info",
    },
    {
        "response_id": "4",
        "start": 0,
        "end": 5,
        "typeable": False,
        "bucket": "spurious",
        "crosses_type_boundary": False,
        "majority_by_characters": None,
    },
]


def test_predicted_rows_drop_spurious_spans():
    """A spurious prediction has no true type, so scoring it would score C1."""
    rows = predicted_rows(TYPED, CORPUS)
    assert len(rows) == 1
    assert rows[0]["source"] == "predicted"
    assert rows[0]["error_type"] == "evident_baseless_info"
    assert rows[0]["bucket"] == "boundary"


def test_predicted_rows_take_the_named_convention():
    typed = [
        {
            **TYPED[0],
            "majority_by_characters": "evident_conflict",
            "typed_merge_failure": "unassigned",
        }
    ]
    assert predicted_rows(typed, CORPUS)[0]["error_type"] == "evident_conflict"
    assert predicted_rows(typed, CORPUS, convention="typed_merge_failure") == []


# --------------------------------------------------------------------------
# End to end on a tiny corpus
# --------------------------------------------------------------------------


def test_train_and_score_produce_every_axis():
    rows = gold_rows(CORPUS * 8)
    models = train(rows)
    assert set(models) == {"mechanism", "severity", "flat", "majority"}
    block = score(models, rows, with_intervals=False)
    assert block["n"] == len(rows)
    for axis in ("mechanism", "severity", "error_type_flat", "error_type_axis_product"):
        assert block[axis]["model"]["n"] == len(rows)
        assert block[axis]["floor"]["ignores_the_input"] is True


def test_probability_rows_expose_both_distributions_and_the_truth():
    rows = gold_rows(CORPUS * 8)
    models = train(rows)
    out = probability_rows(models, rows)
    assert len(out) == len(rows)
    first = out[0]
    assert first["classes"] == list(ERROR_TYPES)
    assert sum(first["probs_flat"]) == pytest.approx(1.0)
    assert sum(first["probs_axis_product"]) == pytest.approx(1.0)
    assert first["error_type"] in ERROR_TYPES


def test_matrix_has_one_column_per_declared_feature():
    from src.c3_explanation.grounding import FEATURE_NAMES

    rows = gold_rows(CORPUS)
    assert matrix(rows).shape == (len(rows), len(FEATURE_NAMES))
    assert matrix([]).shape == (0, len(FEATURE_NAMES))
