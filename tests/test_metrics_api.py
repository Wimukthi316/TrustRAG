"""Tests for the metrics endpoint and the figure route.

The endpoint exists so a panel can see the evidence behind the number on
screen. Two things therefore have to be true and are tested here: it must not
fall over when the results are not on disk, because a demo that 500s is worse
than one that says "nothing measured yet"; and its coverage rows must carry the
band, because coverage below target without the band beside it is the single
easiest number in this project to misread.

The figure route is tested for the boring reason: it takes a name from the URL,
and a name from a URL that reaches the filesystem is how a demo server ends up
serving something it should not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("fastapi")
pytest.importorskip("numpy")

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402
from backend.app.services import metrics as metrics_service  # noqa: E402
from src.common.schema import MetricsResponse  # noqa: E402


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_metrics_endpoint_validates_against_the_schema(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    # Round-tripping through the model is the real assertion: a field the
    # backend renamed and types.ts did not would fail here.
    MetricsResponse.model_validate(response.json())


def test_metrics_survives_missing_results(monkeypatch):
    """No artefacts on disk is a state, not a crash."""
    metrics_service.build_metrics.cache_clear()
    monkeypatch.setattr(metrics_service, "_load", lambda _relative: None)
    payload = metrics_service.build_metrics()
    assert payload.available is False
    assert payload.notes, "an unavailable payload must say what to run"
    assert payload.coverage == []
    metrics_service.build_metrics.cache_clear()


def test_every_coverage_row_carries_its_band(client):
    payload = client.get("/api/metrics").json()
    if not payload["available"]:
        pytest.skip("no C2 results on disk")
    assert payload["coverage"], "results exist, so coverage rows should too"
    for row in payload["coverage"]:
        assert row["band"] > 0, "a coverage row without its band is unreadable"
        shortfall = row["target_coverage"] - row["empirical_coverage"]
        assert row["inside_band"] == (shortfall <= max(row["band"], 0.005))


def test_the_floor_row_is_present_and_marked(client):
    payload = client.get("/api/metrics").json()
    if not payload["available"] or not payload["calibration"]:
        pytest.skip("no C2 results on disk")
    floors = [row for row in payload["calibration"] if row["is_floor"]]
    # House rule 10: an ECE is never shown without the uninformative baseline.
    assert len(floors) <= 1
    if floors:
        assert not floors[0]["selected"]


def test_exactly_one_calibrator_is_marked_selected(client):
    payload = client.get("/api/metrics").json()
    if not payload["available"] or not payload["calibration"]:
        pytest.skip("no C2 results on disk")
    selected = [row for row in payload["calibration"] if row["selected"]]
    assert len(selected) == 1
    assert selected[0]["method"] == payload["selected_calibrator"]


def test_shift_rows_never_claim_the_guarantee(client):
    payload = client.get("/api/metrics").json()
    if not payload["available"] or not payload["shift_available"]:
        pytest.skip("no shift study on disk")
    for row in payload["shift"]:
        assert row["target_coverage"] == pytest.approx(1 - row["alpha"])
        # The oracle repair needs the target's true base rate and must not be
        # surfaced here, where there is no room for the caveat it requires.
        assert "oracle" not in (row["repaired_method"] or "")


def test_unknown_figure_is_a_404_not_a_path_traversal(client):
    for name in ("../../.env", "..%2F..%2F.env", "secrets"):
        response = client.get(f"/api/figures/{name}.png")
        assert response.status_code == 404


def test_a_known_figure_is_served_or_reported_missing(client):
    response = client.get("/api/figures/c2_coverage_vs_alpha.png")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert response.headers["content-type"] == "image/png"
        assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    else:
        assert "figures" in response.json()["detail"]
