"""Detector service.

Right now this is a STUB. It produces structurally valid AnalysisResult objects
so the React frontend can be built and demoed before C1 exists.

⚠️  THE NUMBERS THIS FILE PRODUCES ARE FAKE. They are deterministic keyword
matches with hard-coded probabilities. Never put them in a slide, a table or the
report. Every real number must come from a trained model evaluated on RAGTruth.

Swap plan (this is the whole point of the JSON contract):
    Step 1 (now)      -> StubDetector, fake scores, frontend gets built
    Step 2 (~Aug 13)  -> LettuceDetectAdapter, real probs from the public
                         KRLabsOrg/lettucedect-large-modernbert-en-v1 checkpoint
    Step 3 (~Aug 14)  -> C1Detector, our own fine-tuned ModernBERT

Nothing upstream of `analyze()` changes across those three steps.
"""

from __future__ import annotations

import re
import time
from typing import List, Protocol

from src.common.schema import (
    AnalysisResult,
    AnalyzeRequest,
    ConformalDecision,
    Span,
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    """Deliberately naive sentence splitter.

    Good enough for the demo. If C4 ever ships, replace this with a real
    segmenter (spaCy / pysbd) because evidence_index depends on it.
    """
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


class Detector(Protocol):
    """The interface C1 must satisfy. Keep this tiny."""

    model_version: str

    def analyze(self, req: AnalyzeRequest) -> AnalysisResult: ...


class StubDetector:
    """Flags answer words that never appear in the context.

    This is a crude "baseless information" heuristic, NOT a research baseline.
    It exists purely so the UI has something to render.
    """

    model_version = "stub-v0"

    # Words that carry no factual weight -- ignore them entirely.
    _STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
        "been", "being", "to", "of", "in", "on", "at", "for", "with", "as",
        "by", "from", "that", "this", "these", "those", "it", "its", "has",
        "have", "had", "will", "would", "can", "could", "should", "may",
        "might", "must", "not", "no", "which", "who", "whom", "there", "their",
        "they", "he", "she", "his", "her", "you", "your", "we", "our", "i",
    }

    def analyze(self, req: AnalyzeRequest) -> AnalysisResult:
        t0 = time.perf_counter()

        context_vocab = {w.lower().strip(".,;:!?()\"'") for w in req.context.split()}
        spans: List[Span] = []

        for match in re.finditer(r"\b[\w'-]+\b", req.answer):
            word = match.group(0)
            key = word.lower()
            if key in self._STOPWORDS or len(key) <= 2 or key in context_vocab:
                continue

            # Fake, deterministic "confidence": longer unseen words look more
            # suspicious. Again -- this is theatre, not science.
            raw = min(0.55 + 0.03 * len(key), 0.97)
            calibrated = raw * 0.85  # pretend temperature scaling cooled it down

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

        return AnalysisResult(
            question=req.question,
            context=req.context,
            answer=req.answer,
            context_sentences=split_sentences(req.context),
            spans=spans,
            task_type=req.task_type,
            model_version=self.model_version,
            alpha=req.alpha,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    @staticmethod
    def _decide(score: float, alpha: float) -> ConformalDecision:
        """Placeholder decision rule.

        ⚠️  THIS IS NOT CONFORMAL PREDICTION. Real C2 derives the threshold as
        the ceil((n+1)(1-alpha))/n empirical quantile of non-conformity scores
        on a held-out calibration split. This is two magic numbers. It will be
        replaced wholesale by src/c2_calibration/conformal.py.
        """
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
_detector: Detector = StubDetector()


def get_detector() -> Detector:
    return _detector
