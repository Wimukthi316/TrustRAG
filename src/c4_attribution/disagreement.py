"""Detector-Attribution Disagreement Analysis for Component 4.

Quantifies the statistical divergence between discriminative sequence taggers (C1)
and deductive cross-encoders (C4), isolating lexical variations and false-positive
penalizations from genuine factual hallucinations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from src.c4_attribution.shoa import OriginType, SHOADiagnosis


@dataclass
class DisagreementReport:
    """Statistical summary of detector-attribution agreements and failure modes."""

    total_evaluated_spans: int
    c1_flagged_count: int
    disagreement_count: int
    disagreement_rate: float
    type_c_count: int
    type_s_count: int
    type_p_count: int
    type_c_pct: float
    type_s_pct: float
    type_p_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_evaluated_spans": self.total_evaluated_spans,
            "c1_flagged_count": self.c1_flagged_count,
            "disagreement_count": self.disagreement_count,
            "disagreement_rate_pct": round(self.disagreement_rate * 100, 2),
            "breakdown": {
                "type_c_conflict": {
                    "count": self.type_c_count,
                    "pct": round(self.type_c_pct * 100, 2),
                },
                "type_s_sufficiency": {
                    "count": self.type_s_count,
                    "pct": round(self.type_s_pct * 100, 2),
                },
                "type_p_parametric": {
                    "count": self.type_p_count,
                    "pct": round(self.type_p_pct * 100, 2),
                },
            },
        }


class DisagreementAnalyzer:
    """Analyzes diagnostic outputs to compute origin distributions and disagreement metrics."""

    @staticmethod
    def analyze(diagnoses: List[SHOADiagnosis]) -> DisagreementReport:
        """Computes statistical report from a collection of span diagnoses.

        Args:
            diagnoses: List of SHOADiagnosis objects.

        Returns:
            DisagreementReport with counts, rates, and percentage breakdowns.
        """
        total = len(diagnoses)
        if total == 0:
            return DisagreementReport(
                total_evaluated_spans=0,
                c1_flagged_count=0,
                disagreement_count=0,
                disagreement_rate=0.0,
                type_c_count=0,
                type_s_count=0,
                type_p_count=0,
                type_c_pct=0.0,
                type_s_pct=0.0,
                type_p_pct=0.0,
            )

        disagreements = sum(1 for d in diagnoses if d.is_detector_disagreement)
        type_c = sum(1 for d in diagnoses if d.origin_type == OriginType.TYPE_C)
        type_s = sum(1 for d in diagnoses if d.origin_type == OriginType.TYPE_S)
        type_p = sum(1 for d in diagnoses if d.origin_type == OriginType.TYPE_P)

        true_hallucinations = type_c + type_s + type_p
        denom = true_hallucinations if true_hallucinations > 0 else 1

        return DisagreementReport(
            total_evaluated_spans=total,
            c1_flagged_count=total,
            disagreement_count=disagreements,
            disagreement_rate=disagreements / total,
            type_c_count=type_c,
            type_s_count=type_s,
            type_p_count=type_p,
            type_c_pct=type_c / denom,
            type_s_pct=type_s / denom,
            type_p_pct=type_p / denom,
        )
