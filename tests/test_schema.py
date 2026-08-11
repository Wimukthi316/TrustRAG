"""Guards on the JSON contract.

These are cheap and they catch the class of bug that silently corrupts a demo:
span offsets that do not line up with the answer text. Run with:
    .venv\\Scripts\\python.exe -m pytest -q
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.schema import (  # noqa: E402
    AnalysisResult,
    AnalyzeRequest,
    ConformalDecision,
    Span,
)

ANSWER = "Sigiriya was built in 477 CE by 25,000 workers."


def test_span_text_must_match_answer_slice():
    """A span whose text disagrees with its offsets must be rejected."""
    bad = Span(start=0, end=8, text="Polonnaruwa")
    with pytest.raises(ValueError, match="does not match"):
        AnalysisResult(
            question="q", context="c", answer=ANSWER, spans=[bad], model_version="test"
        )


def test_span_cannot_run_past_the_answer():
    far = Span(start=0, end=999, text="x" * 999)
    with pytest.raises(ValueError, match="runs past"):
        AnalysisResult(
            question="q", context="c", answer=ANSWER, spans=[far], model_version="test"
        )


def test_end_must_exceed_start():
    with pytest.raises(ValueError, match="greater than start"):
        Span(start=10, end=10, text="")


def test_valid_result_round_trips_through_json():
    span = Span(
        start=32,
        end=38,
        text=ANSWER[32:38],
        span_score=0.81,
        calibrated_score=0.69,
        conformal_decision=ConformalDecision.ABSTAIN,
        alpha=0.1,
    )
    result = AnalysisResult(
        question="How was Sigiriya built?",
        context="Sigiriya is an ancient rock fortress.",
        answer=ANSWER,
        spans=[span],
        model_version="test",
    )
    revived = AnalysisResult.model_validate_json(result.model_dump_json())
    assert revived.spans[0].text == ANSWER[32:38]
    assert revived.spans[0].conformal_decision is ConformalDecision.ABSTAIN


def test_c3_and_c4_fields_default_to_absent():
    """PP2 ships without C3/C4. The contract must not require them."""
    span = Span(start=0, end=8, text=ANSWER[0:8])
    assert span.explanation is None
    assert span.error_type is None
    assert span.evidence_sentence is None
    assert span.escalated is False


def test_alpha_is_bounded():
    with pytest.raises(ValueError):
        AnalyzeRequest(question="q", context="c", answer="a", alpha=1.0)
    with pytest.raises(ValueError):
        AnalyzeRequest(question="q", context="c", answer="a", alpha=0.0)
