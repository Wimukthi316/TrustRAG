"""Span-Level Hallucination Origin Attribution (SHOA) diagnostic engine.

Classifies hallucinated spans into mechanistic RAG pipeline failure modes:
    1. Contextual Conflict (TYPE-C): Generator generated factual contradiction.
    2. Sufficiency Failure (TYPE-S): Retriever failed to retrieve relevant context.
    3. Parametric Injection (TYPE-P): Context exists, but LLM injected parametric prior.
And isolates Detector-NLI Disagreements (abstractive paraphrases / lexical mismatches).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from src.c4_attribution.scoring import EvidenceScorer, SpanEvidenceScore
from src.common.schema import Span

DEFAULT_THETA_SIM = 0.50
DEFAULT_THETA_ENTAIL = 0.70
DEFAULT_THETA_CONTRA = 0.60


class OriginType(str, Enum):
    """The three mechanistic failure origins in RAG hallucination."""

    TYPE_C = "type_c"  # Contextual Conflict (Generator contradiction)
    TYPE_S = "type_s"  # Sufficiency Failure (Retriever omission)
    TYPE_P = "type_p"  # Parametric Injection (Parametric prior override)


class PipelineBlame(str, Enum):
    """Which component of the RAG pipeline caused the failure."""

    GENERATOR = "generator"
    RETRIEVER = "retriever"
    PARAMETRIC_GUARDRAIL = "parametric_guardrail"
    DETECTOR_MISMATCH = "detector_mismatch"
    NONE = "none"


@dataclass
class SHOADiagnosis:
    """The full diagnostic attribution result for one answer span."""

    span_text: str
    start: int
    end: int
    is_hallucination: bool
    origin_type: Optional[OriginType]
    origin_name: str
    is_detector_disagreement: bool
    pipeline_blame: PipelineBlame
    actionable_remediation: str
    evidence_sentence: Optional[str] = None
    evidence_index: Optional[int] = None
    semantic_similarity: float = 0.0
    max_entailment: float = 0.0
    max_contradiction: float = 0.0
    c1_prob: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "span_text": self.span_text,
            "start": self.start,
            "end": self.end,
            "is_hallucination": self.is_hallucination,
            "origin_type": self.origin_type.value if self.origin_type else None,
            "origin_name": self.origin_name,
            "is_detector_disagreement": self.is_detector_disagreement,
            "pipeline_blame": self.pipeline_blame.value,
            "actionable_remediation": self.actionable_remediation,
            "evidence_sentence": self.evidence_sentence,
            "evidence_index": self.evidence_index,
            "metrics": {
                "semantic_similarity": self.semantic_similarity,
                "max_entailment": self.max_entailment,
                "max_contradiction": self.max_contradiction,
                "c1_prob": self.c1_prob,
            },
        }


class SHOAEngine:
    """Diagnostic attribution engine executing the SHOA decision tree."""

    def __init__(
        self,
        theta_sim: float = DEFAULT_THETA_SIM,
        theta_entail: float = DEFAULT_THETA_ENTAIL,
        theta_contra: float = DEFAULT_THETA_CONTRA,
        evidence_scorer: Optional[EvidenceScorer] = None,
    ) -> None:
        """Initializes the SHOA diagnostic engine.

        Args:
            theta_sim: Semantic similarity cutoff for retrieval sufficiency.
            theta_entail: NLI entailment threshold.
            theta_contra: NLI contradiction threshold.
            evidence_scorer: Pre-configured EvidenceScorer instance.
        """
        self.theta_sim = theta_sim
        self.theta_entail = theta_entail
        self.theta_contra = theta_contra
        self.evidence_scorer = (
            evidence_scorer if evidence_scorer is not None else EvidenceScorer()
        )

    def diagnose_evidence(
        self,
        evidence: SpanEvidenceScore,
        c1_prob: Optional[float] = None,
    ) -> SHOADiagnosis:
        """Applies the 3-way decision tree to a single span's evidence scores.

        Args:
            evidence: Precomputed SpanEvidenceScore for the span.
            c1_prob: Optional C1 detector probability.

        Returns:
            SHOADiagnosis object with origin type, blame, and remediation advice.
        """
        sim = evidence.semantic_similarity
        entail = evidence.max_entailment
        contra = evidence.max_contradiction

        # Branch 1: Detector Disagreement Check (Paraphrase / False Positive)
        # If NLI finds strong entailment, the context legitimately supports this span.
        if entail >= self.theta_entail:
            return SHOADiagnosis(
                span_text=evidence.span_text,
                start=evidence.start,
                end=evidence.end,
                is_hallucination=False,
                origin_type=None,
                origin_name="Legitimate Contextual Support (Detector Paraphrase Mismatch)",
                is_detector_disagreement=True,
                pipeline_blame=PipelineBlame.DETECTOR_MISMATCH,
                actionable_remediation="None: Span is semantically supported; detector false-positive due to lexical variation.",
                evidence_sentence=evidence.best_supporting_sentence,
                evidence_index=evidence.evidence_index,
                semantic_similarity=sim,
                max_entailment=entail,
                max_contradiction=contra,
                c1_prob=c1_prob,
            )

        # Branch 2: Sufficiency Failure (TYPE-S)
        # No semantically relevant sentence was retrieved in the context.
        if not evidence.is_topically_related or sim < self.theta_sim:
            return SHOADiagnosis(
                span_text=evidence.span_text,
                start=evidence.start,
                end=evidence.end,
                is_hallucination=True,
                origin_type=OriginType.TYPE_S,
                origin_name="Sufficiency Failure (Retriever Omission)",
                is_detector_disagreement=False,
                pipeline_blame=PipelineBlame.RETRIEVER,
                actionable_remediation="Retriever: Increase top-k chunks, adjust chunk size, or implement dense-sparse hybrid search.",
                evidence_sentence=None,
                evidence_index=None,
                semantic_similarity=sim,
                max_entailment=entail,
                max_contradiction=contra,
                c1_prob=c1_prob,
            )

        # Branch 3: Contextual Conflict (TYPE-C)
        # Related context exists, but directly contradicts the claim.
        if contra >= self.theta_contra:
            return SHOADiagnosis(
                span_text=evidence.span_text,
                start=evidence.start,
                end=evidence.end,
                is_hallucination=True,
                origin_type=OriginType.TYPE_C,
                origin_name="Contextual Conflict (Generator Contradiction)",
                is_detector_disagreement=False,
                pipeline_blame=PipelineBlame.GENERATOR,
                actionable_remediation="Generator: Enforce strict faithful decoding or fine-tune generator with DPO against contradictions.",
                evidence_sentence=evidence.best_contradicting_sentence,
                evidence_index=evidence.evidence_index,
                semantic_similarity=sim,
                max_entailment=entail,
                max_contradiction=contra,
                c1_prob=c1_prob,
            )

        # Branch 4: Parametric Injection (TYPE-P)
        # Related context exists, but neither entails nor contradicts the claim.
        return SHOADiagnosis(
            span_text=evidence.span_text,
            start=evidence.start,
            end=evidence.end,
            is_hallucination=True,
            origin_type=OriginType.TYPE_P,
            origin_name="Parametric Memory Injection (LLM Prior Bleed-Through)",
            is_detector_disagreement=False,
            pipeline_blame=PipelineBlame.PARAMETRIC_GUARDRAIL,
            actionable_remediation="Guardrails: Add explicit system prompt instruction forbidding ungrounded world knowledge extrapolation.",
            evidence_sentence=evidence.best_supporting_sentence,
            evidence_index=evidence.evidence_index,
            semantic_similarity=sim,
            max_entailment=entail,
            max_contradiction=contra,
            c1_prob=c1_prob,
        )

    def diagnose_spans(
        self,
        context: str,
        spans: Union[List[Span], List[Tuple[int, int, str]], List[str]],
    ) -> List[SHOADiagnosis]:
        """Runs the full diagnostic pipeline on a context and a list of spans.

        Args:
            context: The raw retrieved context text.
            spans: List of Span objects, (start, end, text) tuples, or strings.

        Returns:
            List of SHOADiagnosis objects in matching order.
        """
        evidence_scores = self.evidence_scorer.score_spans(context, spans)
        diagnoses: List[SHOADiagnosis] = []

        for i, ev in enumerate(evidence_scores):
            c1_prob = None
            if spans and isinstance(spans[i], Span):
                c1_prob = spans[i].span_score
            diagnoses.append(self.diagnose_evidence(ev, c1_prob=c1_prob))

        return diagnoses
