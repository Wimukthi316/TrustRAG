"""RAGTruth benchmark evaluation harness for Component 4 (SHOA).

Evaluates the SHOA origin attribution framework against human-annotated RAGTruth spans,
empirically decomposing 'Baseless Information' into Retriever Sufficiency Failures (TYPE-S)
versus Parametric Injections (TYPE-P).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from src.c4_attribution.shoa import OriginType, SHOADiagnosis, SHOAEngine
from src.common.schema import Span


@dataclass
class RAGTruthRecord:
    """A single evaluation record containing context, response, model, and human span labels."""

    source_id: str
    task_type: str
    source_context: str
    response_text: str
    model_name: str
    human_spans: List[Dict[str, Any]]  # List of {"start": int, "end": int, "label": str, "text": Optional[str]}


@dataclass
class SpanBenchmarkResult:
    """Benchmark outcome linking human annotation to SHOA diagnostic origin."""

    source_id: str
    model_name: str
    task_type: str
    span_text: str
    human_label: str
    diagnosis: SHOADiagnosis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "span_text": self.span_text,
            "human_label": self.human_label,
            "shoa_origin": self.diagnosis.origin_type.value if self.diagnosis.origin_type else None,
            "shoa_origin_name": self.diagnosis.origin_name,
            "pipeline_blame": self.diagnosis.pipeline_blame.value,
            "is_disagreement": self.diagnosis.is_detector_disagreement,
            "metrics": {
                "semantic_similarity": self.diagnosis.semantic_similarity,
                "max_entailment": self.diagnosis.max_entailment,
                "max_contradiction": self.diagnosis.max_contradiction,
            },
        }


class RAGTruthEvaluator:
    """Evaluates SHOA diagnoses across RAGTruth records."""

    def __init__(self, shoa_engine: Optional[SHOAEngine] = None) -> None:
        """Initializes the benchmark evaluator.

        Args:
            shoa_engine: Pre-configured SHOAEngine, or None for default.
        """
        self.shoa = shoa_engine if shoa_engine is not None else SHOAEngine()

    def evaluate_record(self, record: RAGTruthRecord) -> List[SpanBenchmarkResult]:
        """Runs SHOA diagnosis on all human-annotated spans in a single record.

        Args:
            record: RAGTruthRecord instance.

        Returns:
            List of SpanBenchmarkResult objects.
        """
        if not record.human_spans:
            return []

        spans_to_diagnose: List[Span] = []
        human_labels: List[str] = []

        for span_meta in record.human_spans:
            start = int(span_meta["start"])
            end = int(span_meta["end"])
            label = str(span_meta.get("label", "unknown"))

            # Derive text from offsets if not explicitly provided
            text = span_meta.get("text")
            if not text:
                if start < len(record.response_text) and end <= len(record.response_text):
                    text = record.response_text[start:end]
                else:
                    text = f"span_{start}_{end}"

            spans_to_diagnose.append(Span(start=start, end=end, text=text))
            human_labels.append(label)

        diagnoses = self.shoa.diagnose_spans(record.source_context, spans_to_diagnose)

        results: List[SpanBenchmarkResult] = []
        for i, diag in enumerate(diagnoses):
            results.append(
                SpanBenchmarkResult(
                    source_id=record.source_id,
                    model_name=record.model_name,
                    task_type=record.task_type,
                    span_text=diag.span_text,
                    human_label=human_labels[i],
                    diagnosis=diag,
                )
            )

        return results

    def evaluate_dataset(self, records: List[RAGTruthRecord]) -> List[SpanBenchmarkResult]:
        """Evaluates an entire collection of RAGTruth records in batch.

        Args:
            records: List of RAGTruthRecord instances.

        Returns:
            Flat list of all SpanBenchmarkResult objects.
        """
        all_results: List[SpanBenchmarkResult] = []
        for rec in records:
            all_results.extend(self.evaluate_record(rec))
        return all_results

    @staticmethod
    def compute_baseless_decomposition(
        results: List[SpanBenchmarkResult],
    ) -> Dict[str, Any]:
        """Empirically decomposes RAGTruth 'Baseless Information' spans into TYPE-S vs TYPE-P.

        Args:
            results: List of SpanBenchmarkResult objects.

        Returns:
            Dictionary containing exact counts and percentage breakdowns.
        """
        baseless_spans = [
            r for r in results if "baseless" in r.human_label.lower()
        ]

        total = len(baseless_spans)
        if total == 0:
            return {
                "total_baseless_spans": 0,
                "type_s_retriever_omission_pct": 0.0,
                "type_p_parametric_injection_pct": 0.0,
                "type_c_conflict_pct": 0.0,
                "disagreement_pct": 0.0,
                "breakdown_counts": {"type_s": 0, "type_p": 0, "type_c": 0, "disagreement": 0},
            }

        type_s = sum(1 for r in baseless_spans if r.diagnosis.origin_type == OriginType.TYPE_S)
        type_p = sum(1 for r in baseless_spans if r.diagnosis.origin_type == OriginType.TYPE_P)
        type_c = sum(1 for r in baseless_spans if r.diagnosis.origin_type == OriginType.TYPE_C)
        disagreements = sum(1 for r in baseless_spans if r.diagnosis.is_detector_disagreement)

        return {
            "total_baseless_spans": total,
            "type_s_retriever_omission_pct": round((type_s / total) * 100, 2),
            "type_p_parametric_injection_pct": round((type_p / total) * 100, 2),
            "type_c_conflict_pct": round((type_c / total) * 100, 2),
            "disagreement_pct": round((disagreements / total) * 100, 2),
            "breakdown_counts": {
                "type_s": type_s,
                "type_p": type_p,
                "type_c": type_c,
                "disagreement": disagreements,
            },
        }
