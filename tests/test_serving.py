"""Tests for the serving path: C2 artifact round-trip and the real detector's wiring.

None of these download a model. The transformer forward pass is the one part
that cannot be unit-tested cheaply, so it is stubbed and everything around it --
prompt construction, candidate spans, calibration, the conformal decision, the
schema contract -- is tested for real.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from backend.app.services.detector import StubDetector, get_detector, set_detector  # noqa: E402
from backend.app.services.lettucedetect_detector import (  # noqa: E402
    QA_TEMPLATE,
    C2Layer,
    LettuceDetectDetector,
    build_from_env,
    format_prompt,
)
from src.c2_calibration.calibration import (  # noqa: E402
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    calibrator_from_dict,
)
from src.common.schema import AnalyzeRequest, ConformalDecision, TaskType  # noqa: E402


# --------------------------------------------------------------------------
# Calibrator serialisation
# --------------------------------------------------------------------------


def _fitted_pairs(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    return list(p), list(y)


@pytest.mark.parametrize(
    "calibrator",
    [IdentityCalibrator(), TemperatureCalibrator(), PlattCalibrator(), IsotonicCalibrator()],
)
def test_calibrator_survives_a_json_round_trip(calibrator):
    """The artifact is JSON, not a pickle, so this has to hold exactly."""
    probs, labels = _fitted_pairs()
    calibrator.fit(probs, labels)
    before = calibrator.transform(probs)

    payload = json.loads(json.dumps(calibrator.to_dict()))
    after = calibrator_from_dict(payload).transform(probs)

    assert np.allclose(before, after), f"{calibrator.name} did not round-trip"


def test_isotonic_breakpoints_reproduce_sklearn_exactly():
    """np.interp over the stored thresholds must equal sklearn's own predict."""
    from sklearn.isotonic import IsotonicRegression

    probs, labels = _fitted_pairs(seed=3)
    reference = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    reference.fit(probs, labels)

    ours = IsotonicCalibrator().fit(probs, labels)
    query = list(np.linspace(0.0, 1.0, 101))
    assert np.allclose(reference.predict(query), ours.transform(query))


def test_calibrator_from_dict_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="unknown calibrator"):
        calibrator_from_dict({"name": "magic"})


# --------------------------------------------------------------------------
# The C2 serving layer
# --------------------------------------------------------------------------


def _artifact(nonconformity=None):
    return {
        "schema_version": "1.0.0",
        "detector": "test",
        "unit": "span",
        "score_key": "mean_prob",
        "calibrator": {"name": "uncalibrated"},
        "n_calibration": 100,
        "nonconformity_sorted": sorted(
            nonconformity if nonconformity is not None else list(np.linspace(0.0, 1.0, 100))
        ),
    }


def test_c2_layer_recomputes_the_quantile_for_any_alpha():
    """The slider can land anywhere, so thresholds are computed, never interpolated."""
    layer = C2Layer(_artifact())
    thresholds = [layer.threshold(a) for a in (0.05, 0.1, 0.2, 0.3, 0.4)]
    assert thresholds == sorted(thresholds, reverse=True), (
        "a larger alpha is a weaker promise and must give a smaller threshold"
    )


def test_c2_layer_decisions_split_flag_abstain_and_pass():
    layer = C2Layer(_artifact(nonconformity=[0.3] * 100))
    # threshold ends up at 0.3: p=0.95 keeps only label 1, p=0.05 keeps only 0.
    flag, nonconformity = layer.decide(0.95, alpha=0.1)
    assert flag is ConformalDecision.FLAG
    assert nonconformity == pytest.approx(0.05)
    assert layer.decide(0.05, alpha=0.1)[0] is ConformalDecision.PASS
    assert layer.decide(0.50, alpha=0.1)[0] is ConformalDecision.ABSTAIN


def test_c2_layer_loads_from_a_file(tmp_path):
    path = tmp_path / "c2_artifact.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    assert C2Layer.load(path).n_calibration == 100


# --------------------------------------------------------------------------
# Prompt construction
# --------------------------------------------------------------------------


def test_qa_prompt_matches_the_lettucedetect_template():
    prompt = format_prompt("Who built it?", "passage 1: A. passage 2: B.", TaskType.QA)
    assert prompt.startswith("Briefly answer the following question:")
    assert prompt.endswith("output:")
    assert "following 2 passages" in prompt, "passage count is read off the context"
    assert QA_TEMPLATE.split("\n")[0] in prompt


def test_summary_prompt_is_used_when_there_is_no_question():
    for task in (TaskType.SUMMARIZATION, TaskType.DATA2TEXT):
        assert format_prompt("", "some text", task).startswith("Summarize the following text:")
    # An empty question falls back to the summary template even for QA.
    assert format_prompt("   ", "some text", TaskType.QA).startswith("Summarize")


# --------------------------------------------------------------------------
# The detector, with the forward pass stubbed
# --------------------------------------------------------------------------


class _ScriptedDetector(LettuceDetectDetector):
    """Real logic, scripted token probabilities."""

    def __init__(self, probs, offsets, **kwargs):
        super().__init__(**kwargs)
        self._probs, self._offsets = probs, offsets

    def token_probabilities(self, prompt, answer):
        return self._probs, self._offsets


def _request(answer="alpha beta gamma delta", alpha=0.1):
    return AnalyzeRequest(
        question="a question",
        context="passage 1: some context",
        answer=answer,
        alpha=alpha,
        task_type=TaskType.QA,
    )


def _char_offsets(text):
    return [(i, i + 1) for i in range(len(text))]


def test_detector_produces_a_schema_valid_result_with_real_span_offsets():
    answer = "alpha beta gamma delta"
    start, end = answer.index("beta"), answer.index("beta") + len("beta gamma")
    probs = [0.9 if start <= i < end else 0.01 for i in range(len(answer))]
    detector = _ScriptedDetector(
        probs,
        _char_offsets(answer),
        artifact_path=None,
        candidate_threshold=0.3,
    )
    result = detector.analyze(_request(answer))

    assert len(result.spans) == 1
    span = result.spans[0]
    # The schema validator already enforces this, but assert it explicitly:
    # a silent offset drift is the bug class that ruins the demo.
    assert result.answer[span.start : span.end] == span.text == "beta gamma"
    assert span.span_score == pytest.approx(0.9)
    assert span.token_probs and len(span.token_probs) == end - start


def test_conformal_decisions_reach_the_spans_and_follow_alpha(tmp_path):
    answer = "alpha beta gamma delta"
    probs = [0.55 if 6 <= i < 15 else 0.01 for i in range(len(answer))]
    path = tmp_path / "c2_artifact.json"
    path.write_text(
        json.dumps(_artifact(nonconformity=list(np.linspace(0.0, 0.9, 200)))),
        encoding="utf-8",
    )
    detector = _ScriptedDetector(
        probs, _char_offsets(answer), artifact_path=path, candidate_threshold=0.3
    )

    tight = detector.analyze(_request(answer, alpha=0.05)).spans
    loose = detector.analyze(_request(answer, alpha=0.40)).spans

    assert tight and tight[0].conformal_decision is ConformalDecision.ABSTAIN
    assert tight[0].alpha == pytest.approx(0.05)
    # A weaker promise gives a smaller threshold, so the same span is now decided.
    assert loose and loose[0].conformal_decision is ConformalDecision.FLAG


def test_pass_spans_are_not_returned(tmp_path):
    """PASS means 'this is the rest of the answer'. Drawing it is noise.

    PASS needs the score to clear the bar for label 0 but not for label 1:
    p <= q and 1 - p > q. With p = 0.35 that means q in [0.35, 0.65).
    """
    answer = "alpha beta gamma delta"
    probs = [0.35 if 6 <= i < 15 else 0.01 for i in range(len(answer))]
    path = tmp_path / "c2_artifact.json"
    path.write_text(
        json.dumps(_artifact(nonconformity=[0.5] * 200)), encoding="utf-8"
    )
    detector = _ScriptedDetector(
        probs, _char_offsets(answer), artifact_path=path, candidate_threshold=0.3
    )
    assert detector.analyze(_request(answer)).spans == []


def test_a_threshold_that_cannot_discriminate_abstains_rather_than_passing(tmp_path):
    """A weak detector must abstain, never quietly clear a span.

    With a large conformal threshold both labels stay in the set, which is the
    honest answer: the guarantee cannot separate them. Getting this backwards
    would mean an uncertain span silently disappearing from the UI.
    """
    answer = "alpha beta gamma delta"
    probs = [0.35 if 6 <= i < 15 else 0.01 for i in range(len(answer))]
    path = tmp_path / "c2_artifact.json"
    path.write_text(
        json.dumps(_artifact(nonconformity=[0.9] * 200)), encoding="utf-8"
    )
    detector = _ScriptedDetector(
        probs, _char_offsets(answer), artifact_path=path, candidate_threshold=0.3
    )
    spans = detector.analyze(_request(answer)).spans
    assert len(spans) == 1 and spans[0].conformal_decision is ConformalDecision.ABSTAIN


def test_model_version_says_whether_c2_is_attached(tmp_path):
    assert LettuceDetectDetector(artifact_path=None).model_version.endswith("+uncalibrated")
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    assert LettuceDetectDetector(artifact_path=path).model_version.endswith("+c2")


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


def test_build_from_env_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("TRUSTRAG_DETECTOR", raising=False)
    assert build_from_env() is None
    monkeypatch.setenv("TRUSTRAG_DETECTOR", "stub")
    assert build_from_env() is None


def test_build_from_env_refuses_a_missing_artifact(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUSTRAG_DETECTOR", "lettucedetect")
    monkeypatch.setenv("TRUSTRAG_C2_ARTIFACT", str(tmp_path / "nope.json"))
    with pytest.raises(SystemExit, match="does not exist"):
        build_from_env()


def test_get_detector_defaults_to_the_stub(monkeypatch):
    """The default must never be a 1.6GB download, and must never silently lie."""
    monkeypatch.delenv("TRUSTRAG_DETECTOR", raising=False)
    import backend.app.services.detector as module

    monkeypatch.setattr(module, "_detector", None)
    detector = get_detector()
    assert isinstance(detector, StubDetector)
    assert detector.model_version == "stub-v0"


def test_set_detector_overrides_the_singleton():
    original = get_detector()
    try:
        sentinel = StubDetector()
        set_detector(sentinel)
        assert get_detector() is sentinel
    finally:
        set_detector(original)
