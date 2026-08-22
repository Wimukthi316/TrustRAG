"""Checks that frontend/src/types.ts still matches src/common/schema.py.

The TypeScript mirror is maintained by hand, so it drifts silently. A drifted
field shows up as `undefined` in the UI rather than an error, which is the worst
kind of bug to find during a demo.

This is a textual check, not a real type check -- it only verifies that every
Python field name appears somewhere in the TS interface. That is enough to catch
a forgotten field.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.schema import (  # noqa: E402
    AnalysisResult,
    AnalyzeRequest,
    CalibrationRow,
    CoverageRow,
    GroupCoverageRow,
    HealthResponse,
    MetricsResponse,
    RiskControlRow,
    ShiftRow,
    Span,
)

TYPES_TS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "types.ts"


def _ts_block(name: str) -> str:
    """Pull one `export interface Name { ... }` body out of types.ts."""
    source = TYPES_TS.read_text(encoding="utf-8")
    match = re.search(rf"export interface {name} \{{(.*?)\n\}}", source, re.S)
    assert match, f"interface {name} not found in types.ts"
    return match.group(1)


def _assert_fields_present(model, ts_interface: str) -> None:
    block = _ts_block(ts_interface)
    missing = [f for f in model.model_fields if not re.search(rf"\b{f}\b", block)]
    assert not missing, (
        f"{ts_interface} in types.ts is missing {missing}. "
        f"schema.py and types.ts must change in the same commit."
    )


def test_span_fields_mirrored():
    _assert_fields_present(Span, "Span")


def test_analysis_result_fields_mirrored():
    _assert_fields_present(AnalysisResult, "AnalysisResult")


def test_analyze_request_fields_mirrored():
    _assert_fields_present(AnalyzeRequest, "AnalyzeRequest")


def test_health_response_fields_mirrored():
    _assert_fields_present(HealthResponse, "HealthResponse")


def test_metrics_models_mirrored():
    """The C2 metrics tab speaks the same contract as everything else.

    It is read-only and it is only a demo tab, which is exactly why it would be
    the first place a field quietly stops existing. A missing field there shows
    up as a blank cell in front of a panel rather than as an error.
    """
    for model, interface in (
        (CalibrationRow, "CalibrationRow"),
        (CoverageRow, "CoverageRow"),
        (GroupCoverageRow, "GroupCoverageRow"),
        (ShiftRow, "ShiftRow"),
        (RiskControlRow, "RiskControlRow"),
        (MetricsResponse, "MetricsResponse"),
    ):
        _assert_fields_present(model, interface)


def test_no_extra_ts_fields():
    """Catch the reverse drift: a field in TS that no longer exists in Python."""
    for model, interface in (
        (Span, "Span"),
        (CoverageRow, "CoverageRow"),
        (ShiftRow, "ShiftRow"),
        (MetricsResponse, "MetricsResponse"),
    ):
        block = _ts_block(interface)
        ts_fields = set(re.findall(r"^\s{2}(\w+)[?:]", block, re.M))
        extra = ts_fields - set(model.model_fields)
        assert not extra, f"types.ts {interface} has fields Python does not: {extra}"
