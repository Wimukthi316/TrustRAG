"""End-to-end check that the API returns contract-valid payloads.

Uses FastAPI's TestClient, so no server needs to be running.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402
from src.common.schema import SCHEMA_VERSION, AnalysisResult  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["schema_version"] == SCHEMA_VERSION


def test_example_is_analysable():
    """The canned example must survive a full round trip through /analyze."""
    ex = client.get("/api/example").json()
    r = client.post("/api/analyze", json=ex)
    assert r.status_code == 200, r.text

    result = AnalysisResult.model_validate(r.json())
    assert result.spans, "stub detector found nothing in the example answer"

    # Every span must slice back out of the answer exactly.
    for s in result.spans:
        assert result.answer[s.start:s.end] == s.text


def test_answer_fully_grounded_in_context_yields_no_spans():
    r = client.post(
        "/api/analyze",
        json={
            "question": "Where is it?",
            "context": "Sigiriya is located in the Matale District of Sri Lanka.",
            "answer": "Sigiriya is located in the Matale District.",
            "alpha": 0.1,
            "task_type": "qa",
        },
    )
    assert r.status_code == 200
    assert r.json()["spans"] == []


def test_alpha_out_of_range_is_rejected():
    r = client.post(
        "/api/analyze",
        json={"question": "q", "context": "c", "answer": "a", "alpha": 1.5},
    )
    assert r.status_code == 422
