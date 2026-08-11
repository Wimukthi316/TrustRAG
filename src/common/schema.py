"""TrustRAG shared data contract.

FROZEN CONTRACT. Every component (C1 detector, C2 calibration/conformal,
C3 explanation, C4 attribution), the FastAPI backend and the React frontend
all speak exactly these shapes. Changing a field here is a breaking change for
all four components -- discuss with the team before editing.

Field ownership:
    C1 -> Span.start, .end, .text, .token_probs, .span_score
    C2 -> Span.calibrated_score, .conformal_decision, .alpha, .nonconformity
    C3 -> Span.error_type, .explanation, .escalated
    C4 -> Span.evidence_sentence, .evidence_index, .entailment_score

Anything a component has not produced yet stays None. The frontend must
render gracefully when the C3/C4 fields are absent -- for PP2 they will be.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0.0"


class ConformalDecision(str, Enum):
    """What the C2 conformal layer decided about a span.

    FLAG    -- confidently hallucinated, show it in red.
    ABSTAIN -- inside the uncertainty band, route to a human reviewer.
    PASS    -- confidently supported, leave it alone.
    """

    FLAG = "flag"
    ABSTAIN = "abstain"
    PASS = "pass"


class ErrorType(str, Enum):
    """RAGTruth's four annotated hallucination categories (Niu et al., ACL 2024).

    Values are our own snake_case identifiers, not verbatim strings from the
    dataset. The mapping from RAGTruth's raw labels lives in
    src/c1_detector/ragtruth_labels.py and MUST be verified against the real
    data before any result is reported.
    """

    EVIDENT_CONFLICT = "evident_conflict"
    SUBTLE_CONFLICT = "subtle_conflict"
    EVIDENT_BASELESS = "evident_baseless_info"
    SUBTLE_BASELESS = "subtle_baseless_info"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    """The three RAGTruth task types. Results are reported per task."""

    QA = "qa"
    DATA2TEXT = "data2text"
    SUMMARIZATION = "summarization"
    OTHER = "other"


class Span(BaseModel):
    """One contiguous stretch of the answer, with everything we know about it.

    `start` and `end` are character offsets into `AnalysisResult.answer`,
    half-open (Python slice semantics): answer[start:end] == text.
    """

    # --- C1: detection -------------------------------------------------
    start: int = Field(..., ge=0, description="Char offset into answer, inclusive")
    end: int = Field(..., gt=0, description="Char offset into answer, exclusive")
    text: str = Field(..., description="answer[start:end], denormalised for convenience")
    token_probs: Optional[List[float]] = Field(
        default=None,
        description="Per-token P(hallucinated) from C1, one entry per token in this span",
    )
    span_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Raw uncalibrated P(hallucinated) for the span (C1 aggregate)",
    )

    # --- C2: calibration + conformal ----------------------------------
    calibrated_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Post temperature/Platt/isotonic scaling. This is the number the UI shows.",
    )
    nonconformity: Optional[float] = Field(
        default=None, description="Conformal non-conformity score for this span"
    )
    conformal_decision: Optional[ConformalDecision] = None
    alpha: Optional[float] = Field(
        default=None, gt=0.0, lt=1.0,
        description="Miscoverage level the decision was taken at (target coverage = 1 - alpha)",
    )

    # --- C3: explanation (not in PP2 scope) ---------------------------
    error_type: Optional[ErrorType] = None
    explanation: Optional[str] = None
    escalated: bool = Field(
        default=False, description="True if the bounded agent was invoked on this span"
    )

    # --- C4: attribution (not in PP2 scope) ---------------------------
    evidence_sentence: Optional[str] = None
    evidence_index: Optional[int] = Field(
        default=None, ge=0, description="Index into AnalysisResult.context_sentences"
    )
    entailment_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_offsets(self) -> "Span":
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must be greater than start ({self.start})")
        return self


class AnalysisResult(BaseModel):
    """The complete response for one (context, question, answer) triple.

    This is exactly what POST /api/analyze returns and what the React app parses.
    """

    question: str
    context: str
    answer: str
    context_sentences: List[str] = Field(
        default_factory=list,
        description="Context split into sentences; C4 evidence_index points in here",
    )
    spans: List[Span] = Field(
        default_factory=list, description="Only hallucinated/uncertain spans, never the whole answer"
    )

    task_type: TaskType = TaskType.OTHER
    model_version: str = Field(
        ..., description="Which detector produced this, e.g. 'stub-v0' or 'modernbert-base-e6'"
    )
    schema_version: str = SCHEMA_VERSION
    alpha: Optional[float] = Field(
        default=None, description="Miscoverage level applied across all spans in this result"
    )
    latency_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _spans_within_answer(self) -> "AnalysisResult":
        n = len(self.answer)
        for s in self.spans:
            if s.end > n:
                raise ValueError(
                    f"span [{s.start}:{s.end}] runs past the end of a {n}-char answer"
                )
            if self.answer[s.start:s.end] != s.text:
                raise ValueError(
                    f"span text {s.text!r} does not match answer[{s.start}:{s.end}] "
                    f"== {self.answer[s.start:s.end]!r}"
                )
        return self


class AnalyzeRequest(BaseModel):
    """Request body for POST /api/analyze."""

    question: str = Field(..., min_length=1)
    context: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    alpha: float = Field(
        default=0.1, gt=0.0, lt=1.0,
        description="Miscoverage level. 0.1 => 90% target coverage.",
    )
    task_type: TaskType = TaskType.OTHER


class HealthResponse(BaseModel):
    status: str
    schema_version: str
    model_version: str
    detector_loaded: bool
