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

export default function HighlightedAnswer({ result }: { result: AnalysisResult }) {
  const pieces = useMemo(() => buildPieces(result), [result]);
  const [pinned, setPinned] = useState<number | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);

  // A new result invalidates the old indices.
  useEffect(() => {
    setPinned(null);
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
              className={`cursor-pointer rounded px-1 py-0.5 ring-1 transition ${
                DECISION_STYLE[decision]
              } ${isActive ? "ring-2 ring-offset-1" : ""}`}
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
          Hover a marked passage for its calibrated score, or click to keep it open.
        </p>
      )}

      <Legend />
    </div>
  );
}

function EmptyState({ alpha }: { alpha: number | null }) {
  return (
    <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-900">
      <span className="font-medium">Nothing marked.</span> Every passage of this
      answer was decided <span className="font-medium">Supported</span>
      {alpha !== null && <> at α = {alpha.toFixed(2)}</>}. That is a decision, not
      a failure to run — but recall is the detector's weaker side, so treat it as
      evidence, not proof.
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
      className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ring-1 ${DECISION_STYLE[tone]}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${DOT_STYLE[tone]}`} />
      {count} {label}
    </span>
  );
}

function Legend() {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <p className="mb-3 text-xs font-medium uppercase tracking-wide text-slate-500">
        How to read this
      </p>
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
      <p className="mt-3 border-t border-slate-200 pt-3 text-xs text-slate-500">
        Decisions come from split conformal prediction, so the guarantee is over
        the calibration draw and holds on average across spans — not for any one
        span. Raising α asks for a weaker promise; at large α the prediction set
        can come back empty, which is also reported as Uncertain.
      </p>
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
    <div className="rounded-xl border border-slate-200 bg-white p-5 text-sm shadow-sm">
      <div className="mb-3 flex items-start gap-3">
        <span
          className={`mt-0.5 shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ${
            DECISION_STYLE[decision ?? "abstain"]
          }`}
        >
          {decision ? DECISION_LABEL[decision] : "—"}
        </span>
        <p className="flex-1 text-slate-900">“{span.text}”</p>
        {pinned && (
          <button
            onClick={onClose}
            className="shrink-0 rounded px-2 py-0.5 text-xs text-slate-500 hover:bg-slate-100"
          >
            unpin
          </button>
        )}
      </div>

      {span.calibrated_score !== null && (
        <ScoreBar value={span.calibrated_score} />
      )}

      <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-3 sm:grid-cols-4">
        <Field
          label="P(hallucinated), calibrated"
          value={pct(span.calibrated_score)}
          hint="What C2 says the chance is. Calibrated means this number matches the observed rate."
        />
        <Field
          label="P(hallucinated), raw"
          value={pct(span.span_score)}
          hint="The detector's own mean token probability, before calibration."
        />
        <Field
          label="Non-conformity"
          value={span.nonconformity?.toFixed(3) ?? "—"}
          hint="How unusual this span would be if it really were hallucinated. Compared against the calibration set's quantile at this α."
        />
        <Field
          label="Characters"
          value={`${span.start}–${span.end}`}
          hint="Offsets into the answer. These are the numbers the offline evaluation scores."
        />
      </dl>

      {span.evidence_sentence && (
        <p className="mt-4 border-l-2 border-slate-300 pl-3 text-slate-600">
          <span className="font-medium">Evidence (C4):</span> {span.evidence_sentence}
        </p>
      )}
      {span.explanation && (
        <p className="mt-2 text-slate-600">
          <span className="font-medium">Why (C3):</span> {span.explanation}
        </p>
      )}
      {result.model_version === "stub-v0" && (
        <p className="mt-4 rounded bg-amber-100 px-2 py-1 text-xs text-amber-900">
          Placeholder detector — these numbers are not model output.
        </p>
      )}
    </div>
  );
}

function ScoreBar({ value }: { value: number }) {
  return (
    <div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-slate-100">
        <div
          className="h-full rounded-full bg-slate-800 transition-all"
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-slate-400">
        <span>supported</span>
        <span>hallucinated</span>
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
  hint?: string;
}) {
  return (
    <div title={hint}>
      <dt className="text-[11px] uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="font-mono text-slate-900">{value}</dd>
    </div>
  );
}
