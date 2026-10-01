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
    prediction_set: Optional[List[int]] = Field(
        default=None,
        description=(
            "The labels the calibrated threshold could not rule out, ascending: "
            "[], [0], [1] or [0, 1]. Carried separately because "
            "`conformal_decision` cannot express all four -- SplitConformal."
            "decisions() maps the two-label set AND the empty set both onto "
            "ABSTAIN, so a genuinely borderline span and a span the system will "
            "not stand behind either way arrive identical. The empty set is not "
            "hypothetical: 0.0% of spans at alpha 0.10, 19.4% at alpha 0.40 "
            "(results/c2/c1/c2_results.json). None means no conformal layer ran."
        ),
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
        if self.prediction_set is not None:
            labels = self.prediction_set
            if any(v not in (0, 1) for v in labels):
                raise ValueError(f"prediction_set may only contain 0 and 1, got {labels}")
            if labels != sorted(set(labels)):
                raise ValueError(
                    f"prediction_set must be ascending and without repeats, got {labels}"
                )
        return self


class FeatureCheck(BaseModel):
    """One quantity of the input, and where it sits in the calibration split."""

    name: str
    label: str = Field(..., description="Plain-English name for the interface")
    value: float
    percentile: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of calibration responses at or below this value",
    )
    unusual: bool


class DistributionCheck(BaseModel):
    """Is this input the kind of thing the coverage guarantee was calibrated on?

    Split conformal promises coverage for data exchangeable with the calibration
    split, and nothing in the serving path used to check that. Block C measured
    what the promise does when it is not: coverage falls from 0.8808 to 0.7410.
    This is the run-time counterpart -- it does not repair anything, it decides
    whether the promise may be shown at all.

    **One-sided.** A small p-value is evidence against exchangeability. A large
    one is not evidence for it: the check sees a handful of features and a shift
    can be real and invisible to all of them. `message` says so in the words the
    interface should print.
    """

    checked: bool = Field(
        default=False, description="False when no reference is loaded"
    )
    in_distribution: bool = True
    p_value: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="P(a calibration response is at least this unusual)",
    )
    threshold: float = Field(
        default=0.01, description="Warn below this. It is a false-alarm rate."
    )
    n_reference: int = 0
    most_unusual: Optional[str] = None
    features: List[FeatureCheck] = Field(default_factory=list)
    message: str = ""


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
    distribution_check: Optional[DistributionCheck] = None
    guarantee_applies: bool = Field(
        default=True,
        description=(
            "Whether the coverage guarantee may be claimed for THIS result. "
            "False when no conformal layer is attached, or when the "
            "distribution check found the input unlike the calibration split. "
            "The spans are still returned either way -- what is withdrawn is "
            "the promise, not the detection."
        ),
    )
    attribution_checked: bool = Field(
        default=False,
        description=(
            "Whether C4's document check ran on this result at all. Without it, "
            "'the check read every sentence and would cite none of them' and 'no "
            "check was attached' arrive as the same three empty fields on every "
            "span, and a reader cannot tell a finding from an absence. True "
            "means the spans without evidence are a result; False means they "
            "are silence."
        ),
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


# --------------------------------------------------------------------------
# C2 metrics, for the demo's metrics tab
# --------------------------------------------------------------------------
#
# These carry no per-request state. They are a read-only view of the JSON that
# C2's offline runs wrote, served so the demo can show the evidence behind the
# number on screen instead of asking a panel to take it on trust. Every field
# below exists in an artefact under results/; the backend does no arithmetic on
# them beyond selecting and renaming, so what the tab shows and what the report
# shows cannot drift apart.


class CalibrationRow(BaseModel):
    """One calibrator, scored on the test split."""

    method: str
    ece: float
    mce: float
    brier: float
    selected: bool = False
    is_floor: bool = Field(
        default=False,
        description=(
            "True for the constant-base-rate predictor, which ignores its input "
            "entirely. Every ECE must be read against this row rather than "
            "against zero."
        ),
    )


class CoverageRow(BaseModel):
    """Coverage at one miscoverage level, with the noise band it is judged against."""

    alpha: float
    target_coverage: float
    empirical_coverage: float
    band: float = Field(
        ..., description="3-sigma range a single honest calibration draw may fall short by"
    )
    inside_band: bool
    abstention_rate: float
    empty_set_rate: float
    flag_rate: float


class GroupCoverageRow(BaseModel):
    """Coverage within one group, at a single alpha."""

    group: str
    n_test: int
    n_calibration: int
    empirical_coverage: float
    band: float
    inside_band: bool
    abstention_rate: float


class ShiftRow(BaseModel):
    """In-domain, shifted and repaired coverage at one alpha.

    `shifted` is VOID: exchangeability does not hold between the corpora, so the
    guarantee does not apply to it. It is a measurement of the break.
    """

    alpha: float
    target_coverage: float
    in_domain: Optional[float] = None
    shifted: Optional[float] = None
    repaired: Optional[float] = None
    repaired_method: Optional[str] = None
    shifted_meets_target: Optional[bool] = None
    repaired_meets_target: Optional[bool] = None


class RiskControlRow(BaseModel):
    """A false-negative bound, the threshold that buys it, and what it costs."""

    alpha: float
    threshold: Optional[float] = None
    test_risk: Optional[float] = None
    token_flag_rate: Optional[float] = None
    bound_held: Optional[bool] = None
    on_grid_edge: bool = False


class MetricsResponse(BaseModel):
    """Everything the metrics tab shows. Absent artefacts leave lists empty."""

    schema_version: str = SCHEMA_VERSION
    available: bool = Field(
        ..., description="False when no C2 results are on disk; the tab says so"
    )
    detector: str = ""
    unit: str = "span"
    n_calibration: int = 0
    n_test: int = 0
    positive_rate_test: float = 0.0
    ece_before: Optional[float] = None
    ece_after: Optional[float] = None
    auroc: Optional[float] = None
    selected_calibrator: Optional[str] = None
    calibration: List[CalibrationRow] = Field(default_factory=list)
    coverage: List[CoverageRow] = Field(default_factory=list)
    per_task: List[GroupCoverageRow] = Field(default_factory=list)
    shift: List[ShiftRow] = Field(default_factory=list)
    shift_available: bool = False
    risk_control: List[RiskControlRow] = Field(default_factory=list)
    figures: List[str] = Field(
        default_factory=list, description="Figure names served under /api/figures/"
    )
    notes: List[str] = Field(default_factory=list)


# --- history ----------------------------------------------------------------
# A check a person ran, kept so they can open it again. The store behind it is
# swappable and may be absent; every response says which one answered, because
# "your history is empty" and "the history store is unreachable" are different
# sentences and only one of them is the reader's problem.


class HistoryEntry(BaseModel):
    """One saved check, as it appears in the list.

    Deliberately not the whole result. A list of fifty checks should not carry
    fifty documents and fifty span arrays across the wire; the full result is
    fetched only when someone opens one.
    """

    id: str
    # ISO 8601, UTC, seconds precision. A string rather than a datetime because
    # this crosses into TypeScript, where a Date is whatever the parser made of
    # it, and an ISO string is the same instant in both languages.
    created_at: str
    task_type: TaskType
    question: str
    answer_preview: str
    n_spans: int = 0
    n_flagged: int = 0
    n_unsure: int = 0
    guarantee_applies: bool = True
    alpha: Optional[float] = None
    model_version: str = ""


class HistoryDetail(BaseModel):
    """One saved check, opened."""

    entry: HistoryEntry
    result: AnalysisResult


class HistoryPage(BaseModel):
    """A page of saved checks, and the truth about where they came from."""

    schema_version: str = SCHEMA_VERSION
    available: bool = Field(
        ..., description="False when no store is configured or it is unreachable"
    )
    backend: str = Field(
        "none", description='Which store answered: "mongodb", or "none"'
    )
    note: str = Field(
        "", description="Why it is unavailable, in words a reader can act on"
    )
    # Matching the filter, not the page. A reader who filtered to three results
    # out of two hundred needs to be told both numbers.
    total: int = 0
    entries: List[HistoryEntry] = Field(default_factory=list)
