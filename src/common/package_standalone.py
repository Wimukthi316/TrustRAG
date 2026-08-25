"""Package the built frontend as one self-contained HTML file.

The same application serves two situations. Normally it talks to the FastAPI
backend and shows live results. Packaged by this module it carries everything it
needs in one file: the recorded runs, the measurements, and the figures, all
inlined. That copy opens from a filesystem or a hosted page with no install, no
model download and no backend.

It is the same build either way. Nothing is re-implemented for the offline case
and no number is written by hand — the metrics come from the same
`build_metrics()` the API serves, the recorded runs from `recorded_analysis`
and the three proof figures from `proof_charts`, so the packaged copy cannot
say something the live one would not.

    python -m src.common.package_standalone

Run `npm run build` in frontend/ first; this reads frontend/dist/.
"""

from __future__ import annotations

import base64
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

DIST = Path("frontend/dist")
RECORDED = Path("frontend/public/recorded-runs.json")
PROOF = Path("frontend/public/proof-charts.json")
FIGURE_DIR = Path("paper/figures")
OUT = Path("paper/trustrag_app.html")

_SCRIPT = re.compile(r'<script[^>]+src="([^"]+)"[^>]*></script>')
_STYLE = re.compile(r'<link[^>]+rel="stylesheet"[^>]+href="([^"]+)"[^>]*>')


def _asset(dist: Path, reference: str) -> Path:
    """Path for a reference like /assets/index-abc.js, relative to this build."""
    return dist / reference.lstrip("/")


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def build_bundle(figure_names: Sequence[str]) -> Dict[str, Any]:
    """The measurements and figures the packaged copy carries with it."""
    from backend.app.services.metrics import build_metrics

    metrics = build_metrics()
    figures: Dict[str, str] = {}
    for name in figure_names:
        path = FIGURE_DIR / f"{name}.png"
        if path.exists():
            figures[name] = data_uri(path)
    return {
        "metrics": json.loads(metrics.model_dump_json()),
        "figures": figures,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }


def package(dist: Path = DIST, out: Path = OUT) -> Path:
    index = (dist / "index.html").read_text(encoding="utf-8")

    script_match = _SCRIPT.search(index)
    style_match = _STYLE.search(index)
    if not script_match:
        raise SystemExit(f"no bundled script found in {dist / 'index.html'}; run npm run build")

    js = _asset(dist, script_match.group(1)).read_text(encoding="utf-8")
    css = _asset(dist, style_match.group(1)).read_text(encoding="utf-8") if style_match else ""

    from backend.app.services.metrics import build_metrics

    bundle = build_bundle(build_metrics().figures)
    recorded = json.loads(RECORDED.read_text(encoding="utf-8")) if RECORDED.exists() else None
    proof = json.loads(PROOF.read_text(encoding="utf-8")) if PROOF.exists() else None

    def payload(value: Any) -> str:
        # `</` inside a script element would close it early.
        return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")

    parts = [
        "<title>TrustRAG</title>",
        f"<style>{css}</style>" if css else "",
        '<div id="root"></div>',
    ]
    if recorded is not None:
        parts.append(
            f'<script id="trustrag-recorded" type="application/json">{payload(recorded)}</script>'
        )
    if proof is not None:
        parts.append(
            f'<script id="trustrag-proof" type="application/json">{payload(proof)}</script>'
        )
    parts.append(
        f'<script id="trustrag-offline" type="application/json">{payload(bundle)}</script>'
    )
    parts.append(f"<script type=\"module\">{js}</script>")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(p for p in parts if p), encoding="utf-8")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    out = package()
    size = out.stat().st_size / 1024
    print(f"wrote {out} ({size:.0f} kB)")
    print(
        "Self-contained: recorded runs, measurements and figures are inlined. "
        "The API is still tried first, so a live install never shows this copy "
        "instead of a real call."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
