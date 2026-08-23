"""Evidence scoring pipeline combining semantic alignment and NLI inference.

Extracts maximum entailment, maximum contradiction, and supporting context sentences
for candidate spans, populating the frozen schema fields in `src.common.schema.Span`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from src.c4_attribution.alignment import SemanticAligner, SpanAlignment
from src.c4_attribution.nli import NLIPrediction, NLIScorer
from src.c4_attribution.segmentation import ContextSegment, segment_context
from src.common.schema import Span


@dataclass
class SpanEvidenceScore:
    """Consolidated evidence evaluation for one answer span."""

    span_text: str
    start: int
    end: int
    max_entailment: float = 0.0
    max_contradiction: float = 0.0
    best_supporting_sentence: Optional[str] = None
    best_contradicting_sentence: Optional[str] = None
    evidence_index: Optional[int] = None
    semantic_similarity: float = 0.0
    is_topically_related: bool = False
    predictions: List[NLIPrediction] = field(default_factory=list)


class EvidenceScorer:
    """Scores candidate spans against retrieved context using semantic search and NLI."""

    def __init__(
        self,
        aligner: Optional[SemanticAligner] = None,
        nli_scorer: Optional[NLIScorer] = None,
        top_k_candidates: int = 5,
    ) -> None:
        """Initializes the evidence scorer.

        Args:
            aligner: SemanticAligner instance (Day 1) or None for default.
            nli_scorer: NLIScorer instance (Day 2) or None for default.
            top_k_candidates: Number of top candidate sentences to score per span via NLI.
        """
        self.aligner = aligner if aligner is not None else SemanticAligner()
        self.nli_scorer = nli_scorer if nli_scorer is not None else NLIScorer()
        self.top_k = top_k_candidates

    def score_spans(
        self,
        context: str,
        spans: Union[List[Span], List[Tuple[int, int, str]], List[str]],
    ) -> List[SpanEvidenceScore]:
        """Evaluates evidence entailment and contradiction for all spans in context.

        Args:
            context: Raw retrieved context text.
            spans: List of Span objects, or (start, end, text) tuples, or span text strings.

        Returns:
            List of SpanEvidenceScore objects.
        """
        if not spans:
            return []

        # Standardize span inputs into (start, end, text) tuples
        span_items: List[Tuple[int, int, str]] = []
        for s in spans:
            if isinstance(s, Span):
                span_items.append((s.start, s.end, s.text))
            elif isinstance(s, tuple) and len(s) == 3:
                span_items.append(s)
            elif isinstance(s, str):
                span_items.append((0, len(s), s))
            else:
                raise TypeError(f"Unsupported span type: {type(s)}")

        segments = segment_context(context)
        if not segments or not context.strip():
            return [
                SpanEvidenceScore(
                    span_text=text,
                    start=start,
                    end=end,
                    max_entailment=0.0,
                    max_contradiction=0.0,
                    best_supporting_sentence=None,
                    best_contradicting_sentence=None,
                    evidence_index=None,
                    semantic_similarity=0.0,
                    is_topically_related=False,
                    predictions=[],
                )
                for start, end, text in span_items
            ]

        # 1. Semantic alignment to find candidate sentences
        alignments = self.aligner.align_spans(context, spans, top_k=self.top_k)

        # 2. Build NLI candidate pairs
        all_pairs: List[Tuple[str, str]] = []
        pair_metadata: List[Tuple[int, ContextSegment]] = []  # (span_idx, segment)

        for i, align in enumerate(alignments):
            for cand in align.candidates:
                all_pairs.append((cand.segment.text, align.span_text))
                pair_metadata.append((i, cand.segment))

        # 3. Batch NLI inference
        nli_preds = self.nli_scorer.predict_pairs(all_pairs)

        # 4. Group results by span
        evidence_scores: List[SpanEvidenceScore] = []
        for i, (start, end, text) in enumerate(span_items):
            align = alignments[i]
            
            # Filter NLI predictions belonging to span i
            span_preds: List[NLIPrediction] = []
            span_segments: List[ContextSegment] = []

            for (span_idx, seg), pred in zip(pair_metadata, nli_preds):
                if span_idx == i:
                    span_preds.append(pred)
                    span_segments.append(seg)

            if not span_preds:
                evidence_scores.append(
                    SpanEvidenceScore(
                        span_text=text,
                        start=start,
                        end=end,
                        max_entailment=0.0,
                        max_contradiction=0.0,
                        best_supporting_sentence=None,
                        best_contradicting_sentence=None,
                        evidence_index=None,
                        semantic_similarity=align.top_similarity,
                        is_topically_related=align.is_topically_related,
                        predictions=[],
                    )
                )
                continue

            entail_scores = [p.entailment for p in span_preds]
            contra_scores = [p.contradiction for p in span_preds]

            best_entail_idx = int(max(range(len(entail_scores)), key=lambda k: entail_scores[k]))
            best_contra_idx = int(max(range(len(contra_scores)), key=lambda k: contra_scores[k]))

            max_entail = float(entail_scores[best_entail_idx])
            max_contra = float(contra_scores[best_contra_idx])

            best_support_seg = span_segments[best_entail_idx]
            best_contra_seg = span_segments[best_contra_idx]

            evidence_scores.append(
                SpanEvidenceScore(
                    span_text=text,
                    start=start,
                    end=end,
                    max_entailment=max_entail,
                    max_contradiction=max_contra,
                    best_supporting_sentence=best_support_seg.text,
                    best_contradicting_sentence=best_contra_seg.text,
                    evidence_index=best_support_seg.index,
                    semantic_similarity=align.top_similarity,
                    is_topically_related=align.is_topically_related,
                    predictions=span_preds,
                )
            )

        return evidence_scores

    def enrich_spans(self, context: str, spans: List[Span]) -> List[Span]:
        """Enriches existing Span schema objects with C4 evidence fields.

        Populates:
            span.evidence_sentence
            span.evidence_index
            span.entailment_score

        Returns:
            The modified list of Span objects.
        """
        if not spans:
            return spans

        evidence_scores = self.score_spans(context, spans)
        for span, ev in zip(spans, evidence_scores):
            span.evidence_sentence = ev.best_supporting_sentence
            span.evidence_index = ev.evidence_index
            span.entailment_score = ev.max_entailment

        return spans
