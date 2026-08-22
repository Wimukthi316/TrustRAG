"""Serve C2's offline evidence to the demo, without recomputing any of it.

The demo shows one span with a calibrated score on it. A panel is entitled to
ask where that calibration came from and whether the promise attached to it was
ever checked. This service answers that from the same JSON the report is built
from, so the tab and the report cannot disagree.

Three rules it follows.

**It computes nothing.** It selects, renames and reshapes. The one exception is
the noise band, which is derived from `coverage_tolerance` -- a pure function of
alpha and the two split sizes -- because storing it would mean storing the same
number in two places and letting them drift.

**A missing artefact is a state, not an error.** `available` comes back false
and the tab says so. A demo that 500s because a results file has not been
regenerated is worse than a demo that says "no metrics on disk".

**It is loaded once and cached.** These files do not change while the server
runs, and re-reading a megabyte of JSON per request to render a static table
would be a strange thing to do on the machine that is also holding a model.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.c2_calibration.conformal import coverage_tolerance
from src.common.schema import (
    CalibrationRow,
    CoverageRow,
    GroupCoverageRow,
    MetricsResponse,
    RiskControlRow,
    ShiftRow,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "results"
FIGURE_DIR = REPO_ROOT / "paper" / "figures"

# The order they appear in the tab. Names only; the files are served separately.
FIGURE_ORDER = (
    "c2_coverage_vs_alpha",
    "c2_abstention_vs_alpha",
    "c2_reliability",
    "c2_shift",
    "c2_risk_control",
)


def _load(relative: str) -> Optional[Dict[str, Any]]:
    path = RESULTS / relative
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A truncated file mid-regeneration is a real possibility on a laptop
        # that is running the demo and a rerun at the same time. Treat it as
        # absent rather than taking the server down.
        return None


def _inside_band(target: float, empirical: float, band: float) -> bool:
    """Same rule as conformal.check_coverage, floor included."""
    return (target - empirical) <= max(band, 0.005)


def _calibration_rows(block: Dict[str, Any], floor: Optional[Dict[str, Any]]) -> List[CalibrationRow]:
    selected = block["calibration"]["selected"]
    rows = [
        CalibrationRow(
            method=str(row["method"]),
            ece=float(row["ece"]),
            mce=float(row["mce"]),
            brier=float(row["brier"]),
            selected=str(row["method"]) == selected,
        )
        for row in block["calibration"]["rows"]
    ]
    if floor:
        constant = floor["constant_base_rate"]
        rows.append(
            CalibrationRow(
                method="constant base rate",
                ece=float(constant["ece"]),
                mce=float(constant["mce"]),
                brier=float(constant["brier"]),
                is_floor=True,
            )
        )
    return rows


def _coverage_rows(block: Dict[str, Any]) -> List[CoverageRow]:
    out: List[CoverageRow] = []
    for row in block["conformal"]["coverage"]:
        band = coverage_tolerance(
            row["alpha"], int(row["n_calibration"]), int(row["n_test"])
        )
        out.append(
            CoverageRow(
                alpha=float(row["alpha"]),
                target_coverage=float(row["target_coverage"]),
                empirical_coverage=float(row["empirical_coverage"]),
                band=band,
                inside_band=_inside_band(
                    row["target_coverage"], row["empirical_coverage"], band
                ),
                abstention_rate=float(row["abstention_rate"]),
                empty_set_rate=float(row["empty_set_rate"]),
                flag_rate=float(row["flag_rate"]),
            )
        )
    return out


def _group_rows(block: Dict[str, Any]) -> List[GroupCoverageRow]:
    out: List[GroupCoverageRow] = []
    for group, row in sorted(
        block["conformal"]["group_conditional_alpha_0.1"].items()
    ):
        band = coverage_tolerance(
            0.1, int(row["n_calibration"]), int(row["n_test"])
        )
        out.append(
            GroupCoverageRow(
                group=group,
                n_test=int(row["n_test"]),
                n_calibration=int(row["n_calibration"]),
                empirical_coverage=float(row["empirical_coverage"]),
                band=band,
                inside_band=_inside_band(0.9, row["empirical_coverage"], band),
                abstention_rate=float(row["abstention_rate"]),
            )
        )
    return out


def _shift_rows(
    in_domain: Optional[Dict[str, Any]], repairs: Optional[Dict[str, Any]]
) -> List[ShiftRow]:
    """One row per alpha: in-domain, shifted, and the best deployable repair.

    The oracle-prior repair is deliberately not surfaced here. It needs the
    target's true base rate, which no deployment has, and a tab is the wrong
    place for a number that only makes sense with a paragraph of caveat beside
    it. It stays in the report and in `repair.json`.
    """
    if not repairs:
        return []
    reference = (
        {
            row["alpha"]: row["empirical_coverage"]
            for row in in_domain["span"]["conformal"]["coverage"]
        }
        if in_domain and "span" in in_domain
        else {}
    )

    out: List[ShiftRow] = []
    for entry in repairs["verdict"]:
        alpha = float(entry["alpha"])
        unrepaired = entry.get("unrepaired") or {}
        candidates = [
            (name, entry[name])
            for name in ("label_shift_estimated", "covariate_shift")
            if entry.get(name)
        ]
        best_name, best = (
            max(candidates, key=lambda pair: pair[1]["coverage"])
            if candidates
            else (None, None)
        )
        out.append(
            ShiftRow(
                alpha=alpha,
                target_coverage=1.0 - alpha,
                in_domain=reference.get(alpha),
                shifted=unrepaired.get("coverage"),
                repaired=best["coverage"] if best else None,
                repaired_method=(
                    best_name.replace("_", " ") if best_name else None
                ),
                shifted_meets_target=unrepaired.get("meets_target"),
                repaired_meets_target=best["meets_target"] if best else None,
            )
        )
    return out


def _risk_rows(risk: Optional[Dict[str, Any]]) -> List[RiskControlRow]:
    if not risk:
        return []
    out: List[RiskControlRow] = []
    for row in risk["rows"]:
        chosen = row["chosen"]
        out.append(
            RiskControlRow(
                alpha=float(chosen["alpha"]),
                threshold=chosen.get("threshold"),
                test_risk=row.get("test_risk"),
                token_flag_rate=(
                    row["test"]["token_flag_rate"] if row.get("test") else None
                ),
                bound_held=row.get("bound_held"),
                on_grid_edge=bool(chosen.get("on_grid_edge", False)),
            )
        )
    return out


def available_figures() -> List[str]:
    return [name for name in FIGURE_ORDER if (FIGURE_DIR / f"{name}.png").exists()]


@lru_cache(maxsize=1)
def build_metrics() -> MetricsResponse:
    """The whole payload, read once and cached for the life of the process."""
    results = _load("c2/c1/c2_results.json")
    if not results or "span" not in results:
        return MetricsResponse(
            available=False,
            notes=[
                "No C2 results on disk. Run "
                "`python -m src.c2_calibration.run_c2` and reload."
            ],
        )

    block = results["span"]
    uncertainty = _load("c2/c1/c2_uncertainty.json")
    floor = (uncertainty or {}).get("span", {}).get("floor")
    repairs = _load("c2/ragbench/repair.json")
    risk = _load("c2/c1/c2_risk_control.json")

    notes = [
        "Coverage below target is only a shortfall when it falls outside the "
        "band. The guarantee holds in expectation over the draw of the "
        "calibration set, so roughly half of honest runs land below target.",
        "Read every ECE against the constant-base-rate row, not against zero. "
        "That predictor ignores its input and cannot rank anything.",
        "Span level: a span is analysed only if the detector proposed it, so "
        "what is calibrated and guaranteed here is precision. Recall is C1's.",
    ]
    if repairs:
        notes.append(
            "The shifted column is VOID. Calibration and test data from "
            "different corpora are not exchangeable, so the coverage guarantee "
            "does not apply to those numbers; they measure the break."
        )

    return MetricsResponse(
        available=True,
        detector=str(results.get("test_file", "")),
        unit="span",
        n_calibration=int(block["n_calibration"]),
        n_test=int(block["n_test"]),
        positive_rate_test=float(block["positive_rate_test"]),
        ece_before=float(block["calibration"]["before"]["ece"]),
        ece_after=float(block["calibration"]["after"]["ece"]),
        auroc=(floor["detector"]["auroc_raw"] if floor else None),
        selected_calibrator=str(block["calibration"]["selected"]),
        calibration=_calibration_rows(block, floor),
        coverage=_coverage_rows(block),
        per_task=_group_rows(block),
        shift=_shift_rows(results, repairs),
        shift_available=bool(repairs),
        risk_control=_risk_rows(risk),
        figures=available_figures(),
        notes=notes,
    )
