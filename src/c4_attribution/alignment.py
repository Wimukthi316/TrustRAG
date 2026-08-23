"""Semantic alignment between answer spans and context sentences for Component 4.

Maps each generated answer span to the most semantically relevant sentence(s)
in the retrieved context and establishes the initial topical relevance decision
boundary for Sufficiency Failure (TYPE-S) attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from src.c4_attribution.embeddings import EmbeddingEncoder
from src.c4_attribution.segmentation import ContextSegment, segment_context
from src.common.schema import Span

DEFAULT_SIMILARITY_THRESHOLD = 0.50


@dataclass
class CandidateSentence:
    """A candidate context sentence with its semantic similarity score."""

    segment: ContextSegment
    similarity: float


@dataclass
class SpanAlignment:
    """Semantic alignment results for one answer span against the retrieved context."""

    span_text: str
    start: int
    end: int
    top_sentence: Optional[ContextSegment] = None
    top_similarity: float = 0.0
    top_sentence_index: Optional[int] = None
    is_topically_related: bool = False
    candidates: List[CandidateSentence] = field(default_factory=list)


class SemanticAligner:
    """Aligns answer spans to retrieved context sentences via dense semantic search."""

    def __init__(
        self,
        encoder: Optional[EmbeddingEncoder] = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        """Initializes the semantic aligner.

        Args:
            encoder: Pre-instantiated EmbeddingEncoder, or None to initialize default.
            similarity_threshold: Cosine similarity cutoff (theta_sim) below which
                a span is considered to have no related context retrieved (Sufficiency Failure).
        """
        self.encoder = encoder if encoder is not None else EmbeddingEncoder()
        self.similarity_threshold = similarity_threshold

    def align_spans(
        self,
        context: str,
        spans: Union[List[Span], List[Tuple[int, int, str]], List[str]],
        top_k: int = 3,
    ) -> List[SpanAlignment]:
        """Aligns a list of spans against a context string.

        Args:
            context: The raw retrieved context text.
            spans: List of Span objects, or (start, end, text) tuples, or span text strings.
            top_k: Maximum number of candidate sentences to return per span.

        Returns:
            List of SpanAlignment objects in corresponding order.
        """
        segments = segment_context(context)
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

        if not segments:
            # Context is empty or could not be segmented
            return [
                SpanAlignment(
                    span_text=text,
                    start=start,
                    end=end,
                    top_sentence=None,
                    top_similarity=0.0,
                    top_sentence_index=None,
                    is_topically_related=False,
                    candidates=[],
                )
                for start, end, text in span_items
            ]

        span_texts = [text for _, _, text in span_items]
        segment_texts = [seg.text for seg in segments]

        sim_matrix = self.encoder.compute_similarity_matrix(span_texts, segment_texts)
        # sim_matrix shape: [len(spans), len(segments)]

        alignments: List[SpanAlignment] = []
        for i, (start, end, text) in enumerate(span_items):
            scores = sim_matrix[i]
            # Rank sentence indices by similarity descending
            ranked_indices = scores.argsort()[::-1][:top_k]

            candidates: List[CandidateSentence] = []
            for idx in ranked_indices:
                candidates.append(
                    CandidateSentence(
                        segment=segments[idx],
                        similarity=float(scores[idx]),
                    )
                )

            top_sim = float(scores[ranked_indices[0]])
            top_seg = segments[ranked_indices[0]]
            is_related = top_sim >= self.similarity_threshold

            alignments.append(
                SpanAlignment(
                    span_text=text,
                    start=start,
                    end=end,
                    top_sentence=top_seg,
                    top_similarity=top_sim,
                    top_sentence_index=top_seg.index,
                    is_topically_related=is_related,
                    candidates=candidates,
                )
            )

        return alignments
