"""The history store and the routes over it.

Two halves. The pure functions -- preview, filter building, limit clamping,
entry derivation -- are tested directly, because they are where the bugs that
matter live: a filter that silently matches nothing, a search box that hands a
regex to the query planner, a page size a caller can set to a million.

The routes are tested against a fake store defined here. It exists only in this
file, on purpose: an in-memory store that shipped in the application would make
every screen look right while losing every check on restart, which is the exact
kind of quiet lie this project is about.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.main import app  # noqa: E402
from backend.app.services import history as history_module  # noqa: E402
from backend.app.services.history import (  # noqa: E402
    DEFAULT_LIMIT,
    MAX_LIMIT,
    UnavailableStore,
    build_filter,
    clamp_limit,
    entry_from_result,
    preview,
)
from src.common.schema import (  # noqa: E402
    AnalysisResult,
    ConformalDecision,
    HistoryDetail,
    HistoryEntry,
    Span,
)


# --- fixtures ---------------------------------------------------------------


def make_span(start: int, end: int, text: str, decision: str) -> Span:
    return Span(
        start=start,
        end=end,
        text=text,
        token_probs=None,
        span_score=0.5,
        calibrated_score=0.5,
        nonconformity=0.5,
        conformal_decision=ConformalDecision(decision),
        alpha=0.1,
        error_type=None,
        explanation=None,
        escalated=False,
        evidence_sentence=None,
        evidence_index=None,
        entailment_score=None,
    )


def make_result(
    answer: str = "The sky is green and the grass is blue.",
    decisions: Tuple[str, ...] = ("flag", "abstain", "pass"),
    question: str = "what colour is the sky",
    task_type: str = "qa",
    guarantee: bool = True,
) -> AnalysisResult:
    spans = [make_span(i * 3, i * 3 + 2, answer[i * 3 : i * 3 + 2], d) for i, d in enumerate(decisions)]
    return AnalysisResult(
        question=question,
        context="The sky is blue.",
        answer=answer,
        context_sentences=["The sky is blue."],
        spans=spans,
        task_type=task_type,
        model_version="test-v0",
        alpha=0.1,
        distribution_check=None,
        guarantee_applies=guarantee,
        latency_ms=12.0,
        timestamp="2026-08-25T00:00:00+00:00",
    )


class FakeStore:
    """A store that keeps checks in a list. Test-only -- see the module docstring."""

    backend = "fake"

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self._next = 1
        self.up = True

    def available(self) -> bool:
        return self.up

    def note(self) -> str:
        return "" if self.up else "the fake store is down"

    def save(self, result: AnalysisResult) -> HistoryEntry:
        entry_id = str(self._next)
        self._next += 1
        created = f"2026-08-25T00:00:{int(entry_id):02d}+00:00"
        entry = entry_from_result(result, entry_id, created)
        self.rows.append({"entry": entry, "result": result})
        return entry

    def list(self, q=None, task=None, only_flagged=False, limit=None, offset=0):
        build_filter(q, task, only_flagged)  # raises on an unknown task, as Mongo's does
        rows = [r["entry"] for r in self.rows]
        if q:
            needle = q.lower()
            rows = [
                e
                for e in rows
                if needle in e.question.lower() or needle in e.answer_preview.lower()
            ]
        if task and task != "all":
            rows = [e for e in rows if e.task_type == task]
        if only_flagged:
            rows = [e for e in rows if e.n_flagged > 0]
        rows.sort(key=lambda e: e.created_at, reverse=True)
        total = len(rows)
        return rows[offset : offset + clamp_limit(limit)], total

    def get(self, entry_id: str) -> Optional[HistoryDetail]:
        for row in self.rows:
            if row["entry"].id == entry_id:
                return HistoryDetail(entry=row["entry"], result=row["result"])
        return None

    def delete(self, entry_id: str) -> bool:
        before = len(self.rows)
        self.rows = [r for r in self.rows if r["entry"].id != entry_id]
        return len(self.rows) < before

    def clear(self) -> int:
        n = len(self.rows)
        self.rows = []
        return n

    def export(self):
        return [{"id": r["entry"].id, "result": r["result"].model_dump(mode="json")} for r in self.rows]


@pytest.fixture()
def store():
    fake = FakeStore()
    history_module.set_store(fake)
    yield fake
    history_module.set_store(None)


@pytest.fixture()
def client(store):
    return TestClient(app)


# --- the pure parts ---------------------------------------------------------


def test_preview_collapses_whitespace():
    """An answer starting with a newline must not render as an empty row."""
    assert preview("\n\n  one   two\n three ") == "one two three"


def test_preview_truncates_and_marks_it():
    out = preview("x" * 500, limit=20)
    assert len(out) == 20
    assert out.endswith("…")


def test_search_term_cannot_smuggle_a_regex():
    """Someone typing `c++` or `.*` into a search box is not writing a query."""
    query = build_filter(q="c++ (.*)")
    pattern = query["$or"][0]["question"]["$regex"]
    assert ".*" not in pattern
    assert r"\(" in pattern


def test_empty_filter_is_empty():
    """No filters must mean no clauses, not a clause that matches everything."""
    assert build_filter() == {}
    assert build_filter(q="   ") == {}


def test_unknown_task_is_rejected_not_ignored():
    """A silently-empty result reads to the user as 'you have no history'."""
    with pytest.raises(ValueError):
        build_filter(task="nonsense")


def test_task_all_is_not_a_filter():
    assert build_filter(task="all") == {}


def test_limit_is_clamped_at_both_ends():
    assert clamp_limit(None) == DEFAULT_LIMIT
    assert clamp_limit(0) == 1
    assert clamp_limit(10_000) == MAX_LIMIT


def test_entry_counts_the_decisions():
    entry = entry_from_result(
        make_result(decisions=("flag", "flag", "abstain", "pass")), "id", "when"
    )
    assert (entry.n_spans, entry.n_flagged, entry.n_unsure) == (4, 2, 1)


def test_unavailable_store_is_quiet_not_broken():
    """Absent is a normal state; only writing raises."""
    store = UnavailableStore("nothing configured")
    assert store.available() is False
    assert store.list() == ([], 0)
    assert store.get("x") is None
    assert store.delete("x") is False
    assert store.clear() == 0
    assert store.export() == []
    with pytest.raises(RuntimeError):
        store.save(make_result())


# --- the routes -------------------------------------------------------------


def test_save_then_open_returns_the_same_result(client, store):
    result = make_result()
    created = client.post("/api/history", json=result.model_dump(mode="json"))
    assert created.status_code == 201
    entry_id = created.json()["id"]

    opened = client.get(f"/api/history/{entry_id}")
    assert opened.status_code == 200
    assert opened.json()["result"]["answer"] == result.answer
    assert len(opened.json()["result"]["spans"]) == len(result.spans)


def test_list_is_newest_first(client, store):
    for i in range(3):
        client.post(
            "/api/history",
            json=make_result(question=f"question {i}").model_dump(mode="json"),
        )
    entries = client.get("/api/history").json()["entries"]
    assert [e["question"] for e in entries] == ["question 2", "question 1", "question 0"]


def test_list_carries_the_filtered_total_not_the_page_size(client, store):
    for i in range(7):
        client.post("/api/history", json=make_result(question=f"q{i}").model_dump(mode="json"))
    page = client.get("/api/history", params={"limit": 2}).json()
    assert len(page["entries"]) == 2
    assert page["total"] == 7


def test_filters_narrow_the_list(client, store):
    client.post("/api/history", json=make_result(question="about mice").model_dump(mode="json"))
    client.post(
        "/api/history",
        json=make_result(question="about carbon", decisions=("pass",)).model_dump(mode="json"),
    )

    assert client.get("/api/history", params={"q": "mice"}).json()["total"] == 1
    assert client.get("/api/history", params={"only_flagged": True}).json()["total"] == 1
    assert client.get("/api/history", params={"task": "summarization"}).json()["total"] == 0


def test_unknown_task_filter_is_a_422(client, store):
    assert client.get("/api/history", params={"task": "nonsense"}).status_code == 422


def test_delete_removes_one_and_404s_the_second_time(client, store):
    entry_id = client.post(
        "/api/history", json=make_result().model_dump(mode="json")
    ).json()["id"]
    assert client.delete(f"/api/history/{entry_id}").status_code == 204
    assert client.delete(f"/api/history/{entry_id}").status_code == 404
    assert client.get(f"/api/history/{entry_id}").status_code == 404


def test_clear_removes_everything(client, store):
    for _ in range(4):
        client.post("/api/history", json=make_result().model_dump(mode="json"))
    assert client.delete("/api/history").json() == {"deleted": 4}
    assert client.get("/api/history").json()["total"] == 0


def test_export_route_is_not_read_as_an_id(client, store):
    """/api/history/export must not be matched by /api/history/{entry_id}."""
    client.post("/api/history", json=make_result().model_dump(mode="json"))
    body = client.get("/api/history/export").json()
    assert body["count"] == 1
    assert body["checks"][0]["result"]["answer"] == make_result().answer


def test_an_unreachable_store_is_a_page_that_loads(client, store):
    """Not a 500. The interface has a real fallback and needs to be told why."""
    store.up = False
    page = client.get("/api/history")
    assert page.status_code == 200
    body = page.json()
    assert body["available"] is False
    assert body["note"]
    assert body["entries"] == []


def test_writing_to_an_unreachable_store_is_a_503(client, store):
    """Reads degrade quietly; a write that did would be losing someone's data."""
    store.up = False
    assert client.post(
        "/api/history", json=make_result().model_dump(mode="json")
    ).status_code == 503
    assert client.delete("/api/history").status_code == 503
    assert client.get("/api/history/export").status_code == 503
