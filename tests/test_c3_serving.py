"""Tests for C3 on the serving path.

The serving layer scores a span from stored JSON coefficients rather than from
scikit-learn, so the test that matters is that it agrees with the offline model
it was exported from. The rest guard the two rules C3 inherits: escalation fires
only on the conjunction, and nothing C3 puts on screen carries a coverage number
when the guarantee has been withdrawn.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.app.services.c3_typer import C3Typer
from src.c3_explanation.grounding import FEATURE_NAMES, span_features, split_sentences
from src.c3_explanation.type_baseline import ERROR_TYPES, fit_logistic, predict_proba
from src.common.schema import ConformalDecision, ErrorType

CONTEXT = (
    "Anne Frank died in Bergen-Belsen in 1945. "
    "The Red Cross archives were re-examined by researchers. "
    "Witnesses reported symptoms before February 7."
)
SENTENCES = split_sentences(CONTEXT)
ANSWER = "Anne Frank died in 2022 and the museum issued a commemorative stamp."


@pytest.fixture()
def stored(tmp_path):
    """A real fitted model, exported the way type_baseline exports one."""
    rng = np.random.default_rng(0)
    features = rng.normal(size=(120, len(FEATURE_NAMES)))
    labels = [ERROR_TYPES[i % len(ERROR_TYPES)] for i in range(120)]
    # Make the label depend on a column so the fit is not degenerate.
    labels = [
        ERROR_TYPES[min(len(ERROR_TYPES) - 1, int((row[0] + 2) // 1))]
        for row in features
    ]
    flat = fit_logistic(features, labels, ERROR_TYPES)
    model_path = tmp_path / "baseline_model.json"
    model_path.write_text(json.dumps({"flat": flat}), encoding="utf-8")

    conformal_path = tmp_path / "c3_conformal.json"
    conformal_path.write_text(
        json.dumps(
            {
                "classes": flat["classes"],
                "temperature": 1.0,
                "coverage": [
                    {"alpha": 0.1, "threshold": 0.9},
                    {"alpha": 0.4, "threshold": 0.3},
                ],
            }
        ),
        encoding="utf-8",
    )
    return features, flat, model_path, conformal_path


def test_serving_scores_match_the_offline_model(stored):
    """The API must not disagree with the numbers the report was built from."""
    features, flat, model_path, conformal_path = stored
    typer = C3Typer.load(model_path, conformal_path)
    offline = predict_proba(flat, features)
    for row in range(0, features.shape[0], 17):
        values = dict(zip(FEATURE_NAMES, features[row]))
        served = typer._probabilities(values)
        expected = [
            offline[row][flat["classes"].index(name)] for name in typer.classes
        ]
        assert served == pytest.approx(expected, abs=1e-9)


def test_missing_artefacts_turn_c3_off_instead_of_crashing(tmp_path):
    assert C3Typer.load(tmp_path / "nothing.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert C3Typer.load(broken) is None
    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    assert C3Typer.load(empty) is None


def test_a_model_without_a_conformal_file_still_types(stored):
    _, _, model_path, _ = stored
    typer = C3Typer.load(model_path, model_path.parent / "absent.json")
    assert typer.available
    assert not typer.has_conformal
    members, alpha = typer.type_set([0.9, 0.05, 0.03, 0.02], 0.1)
    assert members == []
    assert alpha is None


def test_type_set_uses_the_nearest_calibrated_alpha(stored):
    _, _, model_path, conformal_path = stored
    typer = C3Typer.load(model_path, conformal_path)
    _, used = typer.type_set([0.5, 0.3, 0.1, 0.1], 0.12)
    assert used == 0.1
    _, used_far = typer.type_set([0.5, 0.3, 0.1, 0.1], 0.38)
    assert used_far == 0.4


def test_a_looser_threshold_never_shrinks_the_set(stored):
    _, _, model_path, conformal_path = stored
    typer = C3Typer.load(model_path, conformal_path)
    tight, _ = typer.type_set([0.5, 0.3, 0.1, 0.1], 0.4)
    loose, _ = typer.type_set([0.5, 0.3, 0.1, 0.1], 0.1)
    assert set(tight) <= set(loose)


def test_escalation_requires_abstention_and_an_ambiguous_set(stored):
    _, _, model_path, conformal_path = stored
    typer = C3Typer.load(model_path, conformal_path)
    features = span_features(ANSWER[0:23], ANSWER, 0, SENTENCES, "summarization")

    ambiguous = typer.annotate(
        features, ANSWER[0:23], SENTENCES, 0.1, ConformalDecision.ABSTAIN
    )
    assert (len(ambiguous["type_set"]) >= 2) == ambiguous["escalated"]

    flagged = typer.annotate(
        features, ANSWER[0:23], SENTENCES, 0.1, ConformalDecision.FLAG
    )
    assert flagged["escalated"] is False

    committed = typer.annotate(
        features, ANSWER[0:23], SENTENCES, 0.4, ConformalDecision.ABSTAIN
    )
    if len(committed["type_set"]) < 2:
        assert committed["escalated"] is False


def test_annotate_returns_a_schema_error_type_and_a_sentence(stored):
    _, _, model_path, conformal_path = stored
    typer = C3Typer.load(model_path, conformal_path)
    features = span_features(ANSWER[0:23], ANSWER, 0, SENTENCES, "summarization")
    out = typer.annotate(features, ANSWER[0:23], SENTENCES, 0.1, ConformalDecision.FLAG)
    assert isinstance(out["error_type"], ErrorType)
    assert out["explanation"]
    assert sum(out["type_probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_nothing_c3_shows_carries_a_coverage_number(stored):
    """The guarantee is C2's to give and to withdraw. C3 never restates it."""
    _, _, model_path, conformal_path = stored
    typer = C3Typer.load(model_path, conformal_path)
    for start, end in ((0, 23), (28, 60)):
        features = span_features(
            ANSWER[start:end], ANSWER, start, SENTENCES, "summarization"
        )
        for decision in (
            ConformalDecision.FLAG,
            ConformalDecision.ABSTAIN,
            ConformalDecision.PASS,
        ):
            text = typer.annotate(
                features, ANSWER[start:end], SENTENCES, 0.1, decision
            )["explanation"].lower()
            for banned in ("coverage", "%", "guarantee", "confidence level of"):
                assert banned not in text


def test_temperature_changes_the_probabilities_but_not_the_winner(stored):
    features, flat, model_path, conformal_path = stored
    cold = C3Typer(flat and {"flat": flat}, flat["classes"], 1.0, {})
    warm = C3Typer(flat and {"flat": flat}, flat["classes"], 3.0, {})
    values = dict(zip(FEATURE_NAMES, features[0]))
    sharp = cold._probabilities(values)
    soft = warm._probabilities(values)
    assert max(soft) < max(sharp)
    assert sharp.index(max(sharp)) == soft.index(max(soft))
