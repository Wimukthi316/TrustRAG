"""Detector service.

Runs the span detection pipeline and enriches candidate spans with C4
Retrieval-Attribution evidence sentences, indices, and NLI entailment scores.
"""

from __future__ import annotations

import re
import time
from typing import List, Optional, Protocol

from src.c4_attribution.scoring import EvidenceScorer
from src.c4_attribution.segmentation import segment_context
from src.common.schema import (
    AnalysisResult,
    AnalyzeRequest,
    ConformalDecision,
    Span,
)


def split_sentences(text: str) -> List[str]:
    """Extracts clean context sentences using C4's sentence segmentation engine."""
    segments = segment_context(text)
    if not segments:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return [s.text for s in segments]


class Detector(Protocol):
    """The interface C1 must satisfy. Keep this tiny."""

    model_version: str

    def analyze(self, req: AnalyzeRequest) -> AnalysisResult: ...


class StubDetector:
    """Flags answer words that never appear in the context.

    Enriched with real C4 sentence segmentation and evidence attribution.
    """

    model_version = "stub-v0"

    # Words that carry no factual weight -- ignore them entirely.
    STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
        "at", "by", "for", "from", "in", "into", "of", "off", "on", "onto",
        "out", "over", "to", "up", "with", "as", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "it", "its", "they", "them", "their", "this", "that", "these", "those",
        "i", "you", "he", "she", "we", "my", "your", "his", "her", "our",
    }

    _WORD_REGEX = re.compile(r"\b[A-Za-z0-9'-]+\b")

    def __init__(self) -> None:
        self._evidence_scorer: Optional[EvidenceScorer] = None

    @property
    def evidence_scorer(self) -> EvidenceScorer:
        if self._evidence_scorer is None:
            self._evidence_scorer = EvidenceScorer()
        return self._evidence_scorer

    def analyze(self, req: AnalyzeRequest) -> AnalysisResult:
        t0 = time.perf_counter()

        context_words = {
            m.group(0).lower() for m in self._WORD_REGEX.finditer(req.context)
        }

        spans: List[Span] = []
        for match in self._WORD_REGEX.finditer(req.answer):
            word = match.group(0)
            if word.lower() in self.STOPWORDS:
                continue

            in_context = word.lower() in context_words

            # Placeholder score formula
            raw = 0.15 if in_context else 0.85
            calibrated = min(1.0, max(0.0, raw * 0.85 - 0.05 if in_context else raw * 0.85 - 0.02))

            spans.append(
                Span(
                    start=match.start(),
                    end=match.end(),
                    text=word,
                    span_score=round(raw, 4),
                    calibrated_score=round(calibrated, 4),
                    nonconformity=round(1.0 - calibrated, 4),
                    conformal_decision=self._decide(calibrated, req.alpha),
                    alpha=req.alpha,
                )
            )

        spans = self._merge_adjacent(spans, req.answer)

        # Enrich spans with real C4 evidence sentences and entailment scores
        try:
            spans = self.evidence_scorer.enrich_spans(req.context, spans)
        except Exception as e:
            print(f"C4 evidence scoring fallback: {e}")

        return AnalysisResult(
            question=req.question,
            context=req.context,
            answer=req.answer,
            context_sentences=split_sentences(req.context),
            spans=spans,
            task_type=req.task_type,
            model_version=self.model_version,
            alpha=req.alpha,
            guarantee_applies=False,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    @staticmethod
    def _decide(score: float, alpha: float) -> ConformalDecision:
        upper = 1.0 - alpha / 2
        lower = 0.5
        if score >= upper:
            return ConformalDecision.FLAG
        if score >= lower:
            return ConformalDecision.ABSTAIN
        return ConformalDecision.PASS

    @staticmethod
    def _merge_adjacent(spans: List[Span], answer: str) -> List[Span]:
        """Join spans separated only by whitespace, so we highlight phrases not words."""
        if not spans:
            return []

        merged = [spans[0]]
        for nxt in spans[1:]:
            prev = merged[-1]
            gap = answer[prev.end:nxt.start]
            same_call = prev.conformal_decision == nxt.conformal_decision
            if gap.strip() == "" and len(gap) <= 2 and same_call:
                scores = [s for s in (prev.calibrated_score, nxt.calibrated_score) if s is not None]
                raws = [s for s in (prev.span_score, nxt.span_score) if s is not None]
                merged[-1] = Span(
                    start=prev.start,
                    end=nxt.end,
                    text=answer[prev.start:nxt.end],
                    span_score=round(max(raws), 4) if raws else None,
                    calibrated_score=round(max(scores), 4) if scores else None,
                    nonconformity=round(1.0 - max(scores), 4) if scores else None,
                    conformal_decision=prev.conformal_decision,
                    alpha=prev.alpha,
                )
            else:
                merged.append(nxt)
        return merged


# Module-level singleton the API depends on.
_detector: Detector | None = None


def get_detector() -> Detector:
    global _detector
    if _detector is not None:
        return _detector

    try:
        from backend.app.services.lettucedetect_detector import build_from_env

        configured = build_from_env()
    except Exception as exc:
        print(f"real detector could not be configured ({exc}); serving the stub")
        configured = None

    _detector = configured or StubDetector()
    return _detector


def set_detector(detector: Detector) -> None:
    global _detector
    _detector = detector
