"""Cross-LLM failure mode distribution and comparative analysis for Component 4.

Analyzes how hallucination origin distributions (TYPE-C, TYPE-S, TYPE-P) shift
across Large Language Model architectures and parameter scales (7B to Frontier Closed APIs).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from src.c4_attribution.benchmark import SpanBenchmarkResult
from src.c4_attribution.shoa import OriginType


@dataclass
class ModelFailureProfile:
    """Diagnostic failure profile for a specific LLM generator."""

    model_name: str
    total_spans: int
    type_s_pct: float
    type_p_pct: float
    type_c_pct: float
    disagreement_pct: float
    dominant_failure_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_spans": self.total_spans,
            "type_s_sufficiency_pct": round(self.type_s_pct * 100, 2),
            "type_p_parametric_pct": round(self.type_p_pct * 100, 2),
            "type_c_conflict_pct": round(self.type_c_pct * 100, 2),
            "disagreement_pct": round(self.disagreement_pct * 100, 2),
            "dominant_failure_mode": self.dominant_failure_mode,
        }


class CrossLLMAnalyzer:
    """Aggregates and performs statistical significance testing across generator models."""

    @staticmethod
    def compute_profiles(
        results: List[SpanBenchmarkResult],
    ) -> Dict[str, ModelFailureProfile]:
        """Computes failure mode percentages per LLM generator model.

        Args:
            results: List of SpanBenchmarkResult objects.

        Returns:
            Dictionary mapping model_name to ModelFailureProfile.
        """
        grouped: Dict[str, List[SpanBenchmarkResult]] = defaultdict(list)
        for r in results:
            grouped[r.model_name].append(r)

        profiles: Dict[str, ModelFailureProfile] = {}

        for model_name, model_results in grouped.items():
            total = len(model_results)
            if total == 0:
                continue

            type_s = sum(1 for r in model_results if r.diagnosis.origin_type == OriginType.TYPE_S)
            type_p = sum(1 for r in model_results if r.diagnosis.origin_type == OriginType.TYPE_P)
            type_c = sum(1 for r in model_results if r.diagnosis.origin_type == OriginType.TYPE_C)
            disagreements = sum(1 for r in model_results if r.diagnosis.is_detector_disagreement)

            # Determine dominant failure mode among true hallucinations
            hallu_counts = {"Sufficiency Failure (TYPE-S)": type_s, "Parametric Injection (TYPE-P)": type_p, "Contextual Conflict (TYPE-C)": type_c}
            dominant = max(hallu_counts.keys(), key=lambda k: hallu_counts[k]) if (type_s + type_p + type_c) > 0 else "None"

            profiles[model_name] = ModelFailureProfile(
                model_name=model_name,
                total_spans=total,
                type_s_pct=type_s / total,
                type_p_pct=type_p / total,
                type_c_pct=type_c / total,
                disagreement_pct=disagreements / total,
                dominant_failure_mode=dominant,
            )

        return profiles

    @staticmethod
    def compute_contingency_table(
        results: List[SpanBenchmarkResult],
    ) -> Dict[str, Any]:
        """Computes contingency frequency matrix suitable for chi-squared significance testing.

        Args:
            results: List of SpanBenchmarkResult objects.

        Returns:
            Dictionary with models, categories, and 2D frequency matrix.
        """
        models = sorted(list({r.model_name for r in results}))
        categories = ["TYPE-S", "TYPE-P", "TYPE-C", "DISAGREEMENT"]

        matrix: List[List[int]] = []
        for m in models:
            m_res = [r for r in results if r.model_name == m]
            s_cnt = sum(1 for r in m_res if r.diagnosis.origin_type == OriginType.TYPE_S)
            p_cnt = sum(1 for r in m_res if r.diagnosis.origin_type == OriginType.TYPE_P)
            c_cnt = sum(1 for r in m_res if r.diagnosis.origin_type == OriginType.TYPE_C)
            d_cnt = sum(1 for r in m_res if r.diagnosis.is_detector_disagreement)
            matrix.append([s_cnt, p_cnt, c_cnt, d_cnt])

        return {
            "models": models,
            "categories": categories,
            "contingency_matrix": matrix,
        }
