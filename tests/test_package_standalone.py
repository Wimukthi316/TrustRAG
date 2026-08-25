"""The packaged single-file copy must carry real artefacts, and never diverge."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common import package_standalone as pkg


def test_the_payload_escapes_a_closing_script_tag(tmp_path):
    """`</script>` inside inlined JSON would end the element early and break the
    page in a way that only shows up in a browser."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "assets" / "app.js").write_text("console.log('hi');", encoding="utf-8")
    (dist / "assets" / "app.css").write_text("body{}", encoding="utf-8")
    (dist / "index.html").write_text(
        '<link rel="stylesheet" href="/assets/app.css">'
        '<script type="module" src="/assets/app.js"></script>',
        encoding="utf-8",
    )
    out = tmp_path / "out.html"
    monkey = {"metrics": {"note": "</script><script>alert(1)</script>"}, "figures": {}}
    original = pkg.build_bundle
    pkg.build_bundle = lambda names: monkey  # noqa: E731
    try:
        pkg.package(dist=dist, out=out)
    finally:
        pkg.build_bundle = original
    text = out.read_text(encoding="utf-8")
    assert "<\\/script>" in text
    assert "</script><script>alert(1)" not in text


def test_a_missing_build_is_a_loud_failure_not_a_broken_page(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<div id=\"root\"></div>", encoding="utf-8")
    with pytest.raises(SystemExit):
        pkg.package(dist=dist, out=tmp_path / "out.html")


@pytest.mark.skipif(
    not (pkg.DIST / "index.html").exists(),
    reason="the frontend has not been built in this working copy",
)
def test_the_packaged_copy_is_self_contained():
    out = pkg.package()
    text = out.read_text(encoding="utf-8")
    # No reference to a file it would have to fetch to work.
    assert 'src="/assets' not in text
    assert 'href="/assets' not in text
    assert '<script id="trustrag-offline"' in text
    bundle = json.loads(
        text.split('<script id="trustrag-offline" type="application/json">')[1]
        .split("</script>")[0]
        .replace("<\\/", "</")
    )
    assert bundle["metrics"]["available"] is True
    # Figures travel as data URIs, because there is no backend to serve them.
    assert all(uri.startswith("data:image/png;base64,") for uri in bundle["figures"].values())


@pytest.mark.skipif(
    not Path("results/c2/c1/c2_results.json").exists(),
    reason="C2 results are not in this working copy",
)
def test_the_bundle_is_the_same_object_the_api_serves():
    """Two sources for the same table is how a demo and a report drift apart."""
    from backend.app.services.metrics import build_metrics

    bundle = pkg.build_bundle([])
    assert bundle["metrics"] == json.loads(build_metrics().model_dump_json())
