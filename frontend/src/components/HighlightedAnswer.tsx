import { useMemo, useState } from "react";
import type { AnalysisResult, ConformalDecision, Span } from "../types";

const DECISION_STYLE: Record<ConformalDecision, string> = {
  flag: "bg-red-100 text-red-900 decoration-red-500 ring-red-300",
  abstain: "bg-amber-100 text-amber-900 decoration-amber-500 ring-amber-300",
  pass: "bg-emerald-50 text-emerald-900 decoration-emerald-500 ring-emerald-200",
};

const DECISION_LABEL: Record<ConformalDecision, string> = {
  flag: "Hallucinated",
  abstain: "Needs review",
  pass: "Supported",
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
  const [active, setActive] = useState<Span | null>(null);

  const counts = useMemo(() => {
    const c = { flag: 0, abstain: 0, pass: 0 };
    for (const s of result.spans) {
      if (s.conformal_decision) c[s.conformal_decision] += 1;
    }
    return c;
  }, [result.spans]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge tone="flag" count={counts.flag} label="Hallucinated" />
        <Badge tone="abstain" count={counts.abstain} label="Needs review" />
        <Badge tone="pass" count={counts.pass} label="Supported" />
        <span className="ml-auto text-slate-500">
          {result.model_version} · α = {result.alpha ?? "—"} ·{" "}
          {result.latency_ms !== null ? `${result.latency_ms} ms` : "—"}
        </span>
      </div>

      <p className="rounded-lg border border-slate-200 bg-white p-4 text-[15px] leading-8 text-slate-800">
        {pieces.map((piece, i) =>
          piece.kind === "text" ? (
            <span key={i}>{piece.text}</span>
          ) : (
            <mark
              key={i}
              onMouseEnter={() => setActive(piece.span)}
              onMouseLeave={() => setActive(null)}
              className={`cursor-help rounded px-1 py-0.5 underline decoration-wavy underline-offset-4 ring-1 ${
                DECISION_STYLE[piece.span.conformal_decision ?? "abstain"]
              }`}
            >
              {piece.text}
            </mark>
          ),
        )}
      </p>

      {active ? (
        <SpanDetail span={active} result={result} />
      ) : (
        <p className="text-sm text-slate-500">
          Hover a highlighted span to see its calibrated confidence.
        </p>
      )}
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
      className={`rounded-full px-2.5 py-1 font-medium ring-1 ${DECISION_STYLE[tone]}`}
    >
      {count} {label}
    </span>
  );
}

function SpanDetail({ span, result }: { span: Span; result: AnalysisResult }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 text-sm">
      <div className="mb-2 font-medium text-slate-900">"{span.text}"</div>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-1 sm:grid-cols-4">
        <Field
          label="Decision"
          value={
            span.conformal_decision ? DECISION_LABEL[span.conformal_decision] : "—"
          }
        />
        <Field label="Calibrated" value={pct(span.calibrated_score)} />
        <Field label="Raw (uncalibrated)" value={pct(span.span_score)} />
        <Field
          label="Non-conformity"
          value={span.nonconformity?.toFixed(3) ?? "—"}
        />
      </dl>

      {span.evidence_sentence && (
        <p className="mt-3 border-l-2 border-slate-300 pl-3 text-slate-600">
          <span className="font-medium">Evidence (C4):</span> {span.evidence_sentence}
        </p>
      )}
      {span.explanation && (
        <p className="mt-2 text-slate-600">
          <span className="font-medium">Why (C3):</span> {span.explanation}
        </p>
      )}
      {result.model_version === "stub-v0" && (
        <p className="mt-3 rounded bg-amber-100 px-2 py-1 text-xs text-amber-900">
          Placeholder detector — these numbers are not model output.
        </p>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="font-mono text-slate-900">{value}</dd>
    </div>
  );
}
