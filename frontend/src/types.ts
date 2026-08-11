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
