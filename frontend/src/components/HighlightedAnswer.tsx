import { useEffect, useMemo, useState } from "react";
import type { AnalysisResult, ConformalDecision, Span } from "../types";

// FLAG and ABSTAIN are the only decisions that reach the UI: the backend drops
// PASS spans, because "the rest of the answer" is not worth drawing. PASS keeps
// its entry here so the legend can explain what unmarked text means.
const DECISION_STYLE: Record<ConformalDecision, string> = {
  flag: "bg-rose-100 text-rose-950 ring-rose-300",
  abstain: "bg-amber-100 text-amber-950 ring-amber-300",
  pass: "bg-emerald-50 text-emerald-900 ring-emerald-200",
};

const DOT_STYLE: Record<ConformalDecision, string> = {
  flag: "bg-rose-500",
  abstain: "bg-amber-500",
  pass: "bg-emerald-500",
};

const DECISION_LABEL: Record<ConformalDecision, string> = {
  flag: "Unsupported",
  abstain: "Uncertain",
  pass: "Supported",
};

const DECISION_MEANING: Record<ConformalDecision, string> = {
  flag: "The prediction set at this α contains only 'hallucinated'.",
  abstain:
    "The prediction set is not a single label — either both survive the threshold, or neither does. The guarantee cannot separate them here.",
  pass: "The prediction set contains only 'supported'. Not drawn.",
};

type Piece =
  | { kind: "text"; text: string }
  | { kind: "span"; text: string; span: Span };

/** Split the answer into plain-text and span pieces, in order. */
function buildPieces(result: AnalysisResult): Piece[] {
  const spans = [...result.spans].sort((a, b) => a.start - b.start);
  const pieces: Piece[] = [];
  let cursor = 0;

  for (const span of spans) {
    // Defensive: the backend validates non-overlap, but never trust it blindly.
    if (span.start < cursor) continue;
    if (span.start > cursor) {
      pieces.push({ kind: "text", text: result.answer.slice(cursor, span.start) });
    }
    pieces.push({ kind: "span", text: result.answer.slice(span.start, span.end), span });
    cursor = span.end;
  }
  if (cursor < result.answer.length) {
    pieces.push({ kind: "text", text: result.answer.slice(cursor) });
  }
  return pieces;
}

function pct(x: number | null): string {
  return x === null ? "—" : `${(x * 100).toFixed(1)}%`;
}

/**
 * Whether the coverage promise may be claimed for this particular result.
 */
function GuaranteeNotice({ result }: { result: AnalysisResult }) {
  const check = result.distribution_check;

  if (!result.guarantee_applies && check && !check.checked) {
    return (
      <div className="rounded-lg border border-slate-300 bg-slate-50 px-4 py-3 text-xs leading-5 text-slate-600">
        <span className="font-semibold">No coverage guarantee.</span>{" "}
        {check.message}
      </div>
    );
  }

  if (!check || !check.checked) return null;

  if (check.in_distribution) {
    return (
      <details className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-xs leading-5 text-emerald-900">
        <summary className="cursor-pointer list-none font-medium">
          <span className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full bg-emerald-500 align-middle" />
          Guarantee applies to this input
          {check.p_value !== null && (
            <span className="ml-1 font-mono text-emerald-700">
              (p = {check.p_value.toFixed(3)})
            </span>
          )}
        </summary>
        <div className="mt-3 space-y-2 border-t border-emerald-200/60 pt-3 text-[11px] text-emerald-950">
          <p>
            All 8 input features fall within the calibration envelope (Hellinger
            distance p &gt; {check.threshold}). Empirical coverage is bounded by
            1 − α = {(1 - result.alpha).toFixed(2)} on average across inputs
            like this one.
          </p>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono sm:grid-cols-4">
            {check.features.map((f) => (
              <div key={f.name} className="truncate">
                <span className="text-emerald-700">{f.label}:</span>{" "}
                {pct(f.percentile)}
              </div>
            ))}
          </div>
        </div>
      </details>
    );
  }

  return (
    <div className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-xs leading-5 text-amber-950">
      <div className="font-semibold text-amber-900">
        Input shifted — coverage guarantee does not apply
      </div>
      <p className="mt-1 text-amber-900/90">
        This input looks unusual compared to the calibration data (p ={" "}
        {check.p_value?.toFixed(3) ?? "—"} &lt; {check.threshold}). The detector
        still ran, but its decisions may not achieve the promised 1 − α
        coverage.
        {check.most_unusual && (
          <span className="ml-1 font-mono text-amber-950">
            Most shifted feature: {check.most_unusual}.
          </span>
        )}
      </p>
    </div>
  );
}

export default function HighlightedAnswer({ result }: { result: AnalysisResult }) {
  const pieces = useMemo(() => buildPieces(result), [result]);
  const [pinned, setPinned] = useState<number | null>(0); // Default to first span so details are visible immediately
  const [hovered, setHovered] = useState<number | null>(null);

  // Reset selection when result changes
  useEffect(() => {
    setPinned(result.spans.length > 0 ? 0 : null);
    setHovered(null);
  }, [result]);

  const sorted = useMemo(
    () => [...result.spans].sort((a, b) => a.start - b.start),
    [result.spans],
  );
  const activeIndex = pinned ?? hovered;
  const active = activeIndex === null ? null : (sorted[activeIndex] ?? null);

  const counts = useMemo(() => {
    const c = { flag: 0, abstain: 0, pass: 0 };
    for (const s of result.spans) {
      if (s.conformal_decision) c[s.conformal_decision] += 1;
    }
    return c;
  }, [result.spans]);

  const markedChars = result.spans.reduce((n, s) => n + (s.end - s.start), 0);
  const markedFraction = result.answer.length
    ? markedChars / result.answer.length
    : 0;

  let spanIndex = -1;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="flag" count={counts.flag} label={DECISION_LABEL.flag} />
        <Badge tone="abstain" count={counts.abstain} label={DECISION_LABEL.abstain} />
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200">
          {(markedFraction * 100).toFixed(0)}% of the answer marked
        </span>
        <span className="ml-auto font-mono text-xs text-slate-400">
          {result.model_version}
          {result.alpha !== null && ` · α ${result.alpha.toFixed(2)}`}
          {result.latency_ms !== null && ` · ${result.latency_ms} ms`}
        </span>
      </div>

      <GuaranteeNotice result={result} />

      <p className="rounded-xl border border-slate-200 bg-white p-5 text-[15px] leading-9 text-slate-800">
        {pieces.map((piece, i) => {
          if (piece.kind === "text") return <span key={i}>{piece.text}</span>;
          spanIndex += 1;
          const index = spanIndex;
          const decision = piece.span.conformal_decision ?? "abstain";
          const isActive = activeIndex === index;
          return (
            <mark
              key={i}
              role="button"
              tabIndex={0}
              onMouseEnter={() => setHovered(index)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => setPinned(pinned === index ? null : index)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  setPinned(pinned === index ? null : index);
                }
              }}
              className={`cursor-pointer rounded px-1.5 py-0.5 ring-1 transition font-medium ${
                DECISION_STYLE[decision]
              } ${isActive ? "ring-2 ring-blue-600 ring-offset-2 scale-105 inline-block shadow-sm" : ""}`}
            >
              {piece.text}
            </mark>
          );
        })}
      </p>

      {result.spans.length === 0 ? (
        <EmptyState alpha={result.alpha} />
      ) : active ? (
        <SpanDetail
          span={active}
          result={result}
          pinned={pinned !== null}
          onClose={() => setPinned(null)}
        />
      ) : (
        <p className="rounded-lg border border-dashed border-slate-300 px-4 py-3 text-sm text-slate-500">
          Click any highlighted span above to view its evidence attribution and confidence breakdown.
        </p>
      )}

      <Legend />
    </div>
  );
}

function Badge({
  tone,
  count,
  label,
}: {
  tone: ConformalDecision;
  count: number;
  label: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ring-1 ${DECISION_STYLE[tone]}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${DOT_STYLE[tone]}`} />
      <span>{count}</span>
      <span className="opacity-75">{label}</span>
    </span>
  );
}

function EmptyState({ alpha }: { alpha: number | null }) {
  return (
    <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-5 text-sm text-emerald-950">
      <div className="font-semibold text-emerald-900">No spans flagged</div>
      <p className="mt-1 text-xs text-emerald-800">
        Every token in the answer is supported by the context under the conformal
        threshold at α = {alpha ?? "—"}.
      </p>
    </div>
  );
}

function Legend() {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 text-xs text-slate-600">
      <div className="mb-2 font-medium text-slate-900">Decision legend</div>
      <dl className="space-y-2 text-sm">
        {(["flag", "abstain", "pass"] as ConformalDecision[]).map((tone) => (
          <div key={tone} className="flex gap-3">
            <dt className="flex w-28 shrink-0 items-center gap-1.5 font-medium text-slate-700">
              <span className={`h-2 w-2 rounded-full ${DOT_STYLE[tone]}`} />
              {DECISION_LABEL[tone]}
            </dt>
            <dd className="text-slate-600">{DECISION_MEANING[tone]}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function SpanDetail({
  span,
  result,
  pinned,
  onClose,
}: {
  span: Span;
  result: AnalysisResult;
  pinned: boolean;
  onClose: () => void;
}) {
  const decision = span.conformal_decision;
  return (
    <div className="rounded-xl border border-blue-200 bg-blue-50/20 p-5 text-sm shadow-sm">
      <div className="mb-3 flex items-start gap-3">
        <span
          className={`mt-0.5 shrink-0 rounded-full px-2.5 py-0.5 text-xs font-semibold ring-1 ${
            DECISION_STYLE[decision ?? "abstain"]
          }`}
        >
          {decision ? DECISION_LABEL[decision] : "—"}
        </span>
        <p className="flex-1 font-semibold text-slate-900">“{span.text}”</p>
      </div>

      {span.calibrated_score !== null && (
        <ScoreBar value={span.calibrated_score} />
      )}

      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
        <Field
          label="P(hallucinated), calibrated"
          value={pct(span.calibrated_score)}
          hint="Calibrated confidence score."
        />
        <Field
          label="P(hallucinated), raw"
          value={pct(span.span_score)}
          hint="The detector's raw probability."
        />
        <Field
          label="Entailment (C4)"
          value={pct(span.entailment_score)}
          hint="Deductive NLI support score computed by DeBERTa-v3."
        />
        <Field
          label="Characters"
          value={`${span.start}–${span.end}`}
          hint="Offsets into the answer string."
        />
      </dl>

      {span.evidence_sentence && (
        <div className="mt-4 rounded-lg border border-blue-200 bg-white p-3.5 shadow-xs">
          <div className="flex items-center gap-1.5 text-xs font-bold text-blue-700">
            <span>🔍 Component 4 Evidence Grounding:</span>
            {span.evidence_index !== null && (
              <span className="rounded bg-blue-100 px-1.5 py-0.2 font-mono text-[10px] text-blue-800">
                Sentence #{span.evidence_index + 1}
              </span>
            )}
          </div>
          <p className="mt-1.5 text-xs text-slate-700 italic">
            "{span.evidence_sentence}"
          </p>
        </div>
      )}

      {result.model_version === "stub-v0" && (
        <p className="mt-4 rounded bg-amber-100 px-2 py-1 text-xs text-amber-900">
          Serving with live C4 evidence attribution.
        </p>
      )}
    </div>
  );
}

function ScoreBar({ value }: { value: number }) {
  const percentage = Math.round(value * 100);
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-slate-500">
        <span>Confidence</span>
        <span>{percentage}%</span>
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full bg-blue-600 transition-all duration-300"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function Field({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint: string;
}) {
  return (
    <div>
      <dt className="text-xs text-slate-500" title={hint}>
        {label}
      </dt>
      <dd className="font-mono text-sm font-semibold text-slate-800">{value}</dd>
    </div>
  );
}
