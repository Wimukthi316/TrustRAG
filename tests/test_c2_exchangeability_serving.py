"""The three ways the serving path can answer "does the guarantee apply here?"

None of these load a model. `_distribution_check` is pure arithmetic over the
token probabilities and spans the detector already produced, so it can be
exercised directly -- which is the point: the branch that decides whether a
coverage promise is shown to a user should not need a GPU to test.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("fastapi")

from backend.app.services.detector import StubDetector  # noqa: E402
from backend.app.services.lettucedetect_detector import (  # noqa: E402
    LettuceDetectDetector,
)
from src.c2_calibration.exchangeability import ExchangeabilityReference  # noqa: E402
from src.common.schema import AnalyzeRequest  # noqa: E402

from tests.test_c2_exchangeability import make_record  # noqa: E402


@pytest.fixture()
def reference_path(tmp_path):
    """A small reference built from ordinary-looking synthetic responses."""
    records = [
        make_record(seed=i, n_tokens=40 + (i % 20), spans=((0, 12, 3),), answer_chars=200)
        for i in range(300)
    ]
    reference = ExchangeabilityReference.build(records)
    path = tmp_path / "c2_ood_reference.json"
    path.write_text(json.dumps(reference.to_dict()), encoding="utf-8")
    return path


@pytest.fixture()
def artifact_path(tmp_path):
    """A minimal C2 artifact, enough for the detector to consider itself calibrated."""
    path = tmp_path / "c2_artifact.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "detector": "test",
                "unit": "span",
                "score_key": "mean_prob",
                "calibrator": {"name": "uncalibrated"},
                "n_calibration": 200,
                "nonconformity_sorted": sorted(np.linspace(0.0, 0.9, 200).tolist()),
            }
        ),
        encoding="utf-8",
    )
    return path


def _ordinary_call():
    """Arguments matching a response the reference was built from."""
    record = make_record(seed=7, n_tokens=45, spans=((0, 12, 3),), answer_chars=200)
    return (
        record["token_probs"],
        [(12, 24)],
        [[0.05] * 3],
        record["answer"],
    )


def test_no_calibration_layer_means_no_promise_to_check(reference_path):
    detector = LettuceDetectDetector(
        model_id="unused", artifact_path=None, reference_path=reference_path
    )
    check, applies = detector._distribution_check(*_ordinary_call())
    assert applies is False
    assert check.checked is False
    assert "no coverage guarantee" in check.message.lower()


def test_no_reference_keeps_the_old_behaviour_but_says_so(artifact_path):
    """A missing reference must degrade to the previous behaviour, not to a 500."""
    detector = LettuceDetectDetector(
        model_id="unused", artifact_path=artifact_path, reference_path=None
    )
    check, applies = detector._distribution_check(*_ordinary_call())
    assert applies is True
    assert check.checked is False
    assert "exchangeability" in check.message.lower()


def test_an_ordinary_input_keeps_the_promise(artifact_path, reference_path):
    detector = LettuceDetectDetector(
        model_id="unused", artifact_path=artifact_path, reference_path=reference_path
    )
    check, applies = detector._distribution_check(*_ordinary_call())
    assert applies is True
    assert check.checked is True
    assert check.in_distribution is True
    assert check.p_value >= check.threshold
    # The green case must not claim the input is in distribution. The check is
    # one-sided and the wording has to stay one-sided with it.
    assert "not proof" in check.message


def test_an_absurd_input_loses_the_promise_but_keeps_its_spans(
    artifact_path, reference_path
):
    detector = LettuceDetectDetector(
        model_id="unused", artifact_path=artifact_path, reference_path=reference_path
    )
    answer = "x" * 20000
    check, applies = detector._distribution_check(
        [0.9] * 4000, [(0, 20000)], [[0.9] * 4000], answer
    )
    assert applies is False
    assert check.checked is True
    assert check.in_distribution is False
    assert check.most_unusual is not None
    assert "does not apply" in check.message
    # The message names the offending quantity in words a person can read.
    assert any(row.unusual for row in check.features)
    assert all(row.label for row in check.features)


def test_the_check_counts_spans_before_any_are_dropped(artifact_path, reference_path):
    """PASS spans are dropped from the response but must still be counted here.

    The reference was built from unfiltered decoder output. Counting filtered
    spans would compare a filtered number against an unfiltered distribution and
    shift three features at once, which is exactly the kind of silent mismatch
    this project keeps finding.
    """
    detector = LettuceDetectDetector(
        model_id="unused", artifact_path=artifact_path, reference_path=reference_path
    )
    token_probs, _, _, answer = _ordinary_call()
    few = detector._distribution_check(token_probs, [(0, 12)], [[0.1] * 3], answer)[0]
    many = detector._distribution_check(
        token_probs,
        [(i, i + 4) for i in range(0, 120, 6)],
        [[0.1] * 2] * 20,
        answer,
    )[0]
    span_count = next(
        row for row in few.features if row.name == "log_candidate_spans"
    ).value
    span_count_many = next(
        row for row in many.features if row.name == "log_candidate_spans"
    ).value
    assert span_count_many > span_count


def test_the_stub_never_claims_a_guarantee():
    """The placeholder must not look like the real thing, here as elsewhere."""
    result = StubDetector().analyze(
        AnalyzeRequest(
            question="q",
            context="the context mentions apples",
            answer="the answer mentions zebras",
            alpha=0.1,
        )
    )
    assert result.guarantee_applies is False
