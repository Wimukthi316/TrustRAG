// Mirror of src/common/schema.py. If you change the Pydantic models, change this
// file in the same commit -- the whole point of the contract is that these agree.

export type ConformalDecision = "flag" | "abstain" | "pass";

export type ErrorType =
  | "evident_conflict"
  | "subtle_conflict"
  | "evident_baseless_info"
  | "subtle_baseless_info"
  | "unknown";

export type TaskType = "qa" | "data2text" | "summarization" | "other";

export interface Span {
  // C1
  start: number;
  end: number;
  text: string;
  token_probs: number[] | null;
  span_score: number | null;
  // C2
  calibrated_score: number | null;
  nonconformity: number | null;
  conformal_decision: ConformalDecision | null;
  alpha: number | null;
  // C3 (not in PP2 scope -- render only if present)
  error_type: ErrorType | null;
  explanation: string | null;
  escalated: boolean;
  // C4 (not in PP2 scope)
  evidence_sentence: string | null;
  evidence_index: number | null;
  entailment_score: number | null;
}

// Is this input the kind of thing the guarantee was calibrated on? One-sided:
// a small p_value is evidence against, a large one is not evidence for.
export interface FeatureCheck {
  name: string;
  label: string;
  value: number;
  // Fraction of calibration responses at or below this value.
  percentile: number;
  unusual: boolean;
}

export interface DistributionCheck {
  checked: boolean;
  in_distribution: boolean;
  p_value: number | null;
  // Warn below this. It is a false-alarm rate, not a magic number.
  threshold: number;
  n_reference: number;
  most_unusual: string | null;
  features: FeatureCheck[];
  message: string;
}

export interface AnalysisResult {
  question: string;
  context: string;
  answer: string;
  context_sentences: string[];
  spans: Span[];
  task_type: TaskType;
  model_version: string;
  schema_version: string;
  alpha: number | null;
  distribution_check: DistributionCheck | null;
  // False when no conformal layer is attached, or when the input is unlike the
  // calibration split. The spans still render; only the promise is withdrawn.
  guarantee_applies: boolean;
  latency_ms: number | null;
  timestamp: string;
}

export interface AnalyzeRequest {
  question: string;
  context: string;
  answer: string;
  alpha: number;
  task_type: TaskType;
}

export interface HealthResponse {
  status: string;
  schema_version: string;
  model_version: string;
  detector_loaded: boolean;
}

// --- C2 metrics -------------------------------------------------------------
// A read-only view of the JSON that C2's offline runs wrote. Nothing here is
// computed at request time, so what the metrics tab shows and what the report
// shows cannot drift apart.

export interface CalibrationRow {
  method: string;
  ece: number;
  mce: number;
  brier: number;
  selected: boolean;
  // The constant-base-rate predictor, which ignores its input. Every ECE is
  // read against this row rather than against zero.
  is_floor: boolean;
}

export interface CoverageRow {
  alpha: number;
  target_coverage: number;
  empirical_coverage: number;
  // 3-sigma range a single honest calibration draw may fall short by. Coverage
  // under target but inside this band is not a miss.
  band: number;
  inside_band: boolean;
  abstention_rate: number;
  empty_set_rate: number;
  flag_rate: number;
}

export interface GroupCoverageRow {
  group: string;
  n_test: number;
  n_calibration: number;
  empirical_coverage: number;
  band: number;
  inside_band: boolean;
  abstention_rate: number;
}

export interface ShiftRow {
  alpha: number;
  target_coverage: number;
  in_domain: number | null;
  // VOID: exchangeability does not hold across corpora, so the guarantee does
  // not apply to this number. It measures the break.
  shifted: number | null;
  repaired: number | null;
  repaired_method: string | null;
  shifted_meets_target: boolean | null;
  repaired_meets_target: boolean | null;
}

export interface RiskControlRow {
  alpha: number;
  threshold: number | null;
  test_risk: number | null;
  token_flag_rate: number | null;
  bound_held: boolean | null;
  on_grid_edge: boolean;
}

export interface MetricsResponse {
  schema_version: string;
  available: boolean;
  detector: string;
  unit: string;
  n_calibration: number;
  n_test: number;
  positive_rate_test: number;
  ece_before: number | null;
  ece_after: number | null;
  auroc: number | null;
  selected_calibrator: string | null;
  calibration: CalibrationRow[];
  coverage: CoverageRow[];
  per_task: GroupCoverageRow[];
  shift: ShiftRow[];
  shift_available: boolean;
  risk_control: RiskControlRow[];
  figures: string[];
  notes: string[];
}
