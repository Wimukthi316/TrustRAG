import { useMemo } from "react";
import { VERDICT } from "../copy";
import type { AnalysisResult, ConformalDecision, Span } from "../types";
import { Dot } from "./ui";

/**
 * The answer, with the marks on it.
 *
 * This is the one screen the whole project exists to produce, so it gets the
 * largest type in the app and the most leading. A mark may not change the line
 * box it sits in -- padding and a ring, never a border and never a different
 * font size -- or the paragraph starts to look like a ransom note and stops
 * being readable as prose.
 */

const MARK_STYLE: Record<ConformalDecision, string> = {
  flag: "bg-flag-soft [--mark-ring:var(--color-flag-ring)] text-flag-ink",
  abstain: "bg-abstain-soft [--mark-ring:var(--color-abstain-ring)] text-abstain-ink",
  pass: "bg-pass-soft [--mark-ring:var(--color-pass-ring)] text-pass-ink",
};

const MARK_ACTIVE: Record<ConformalDecision, string> = {
  flag: "outline outline-2 outline-offset-2 outline-flag/70",
  abstain: "outline outline-2 outline-offset-2 outline-abstain/70",
  pass: "outline outline-2 outline-offset-2 outline-pass/70",
};

export type Piece =
  | { kind: "text"; text: string }
  | { kind: "span"; text: string; span: Span; index: number };

/** Split the answer into plain-text and marked pieces, in order. */
export function buildPieces(result: AnalysisResult): Piece[] {
  const spans = [...result.spans].sort((a, b) => a.start - b.start);
  const pieces: Piece[] = [];
  let cursor = 0;
  let index = 0;

  for (const span of spans) {
    // Defensive: the backend validates non-overlap, but never trust it blindly.
    if (span.start < cursor) continue;
    if (span.start > cursor) {
      pieces.push({ kind: "text", text: result.answer.slice(cursor, span.start) });
    }
    pieces.push({
      kind: "span",
      text: result.answer.slice(span.start, span.end),
      span,
      index: index++,
    });
    cursor = span.end;
  }
  if (cursor < result.answer.length) {
    pieces.push({ kind: "text", text: result.answer.slice(cursor) });
  }
  return pieces;
}

export function sortedSpans(result: AnalysisResult): Span[] {
  return [...result.spans].sort((a, b) => a.start - b.start);
}

export function countByVerdict(result: AnalysisResult): Record<ConformalDecision, number> {
  const counts: Record<ConformalDecision, number> = { flag: 0, abstain: 0, pass: 0 };
  for (const span of result.spans) {
    if (span.conformal_decision) counts[span.conformal_decision] += 1;
  }
  return counts;
}

export function MarkedAnswer({
  result,
  active = null,
  onSelect,
  interactive = true,
}: {
  result: AnalysisResult;
  active?: number | null;
  onSelect?: (index: number) => void;
  interactive?: boolean;
}) {
  const pieces = useMemo(() => buildPieces(result), [result]);

  return (
    <p className="t-answer text-slate-800">
      {pieces.map((piece, i) =>
        piece.kind === "text" ? (
          <span key={i}>{piece.text}</span>
        ) : (
          <MarkedSpan
            key={i}
            piece={piece}
            isActive={active === piece.index}
            interactive={interactive}
            onSelect={() => onSelect?.(piece.index)}
          />
        ),
      )}
    </p>
  );
}

/**
 * One marked passage.
 *
 * A `<span role="button">` rather than a `<button>`, deliberately. A real
 * button is a single inline block and cannot break across lines, so a marked
 * passage longer than the remaining line jumps to its own line and the
 * paragraph stops being a paragraph.
 */
function MarkedSpan({
  piece,
  isActive,
  interactive,
  onSelect,
}: {
  piece: Extract<Piece, { kind: "span" }>;
  isActive: boolean;
  interactive: boolean;
  onSelect: () => void;
}) {
  const decision = piece.span.conformal_decision ?? "abstain";
  const verdict = VERDICT[decision];

  if (!interactive) {
    return (
      <span className={`mark ${MARK_STYLE[decision]}`} title={verdict.label}>
        {piece.text}
      </span>
    );
  }

  return (
    <span
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onSelect();
        }
      }}
      aria-pressed={isActive}
      aria-label={`${verdict.label}: ${piece.text}`}
      className={`mark cursor-pointer ${MARK_STYLE[decision]} ${
        isActive ? MARK_ACTIVE[decision] : ""
      }`}
    >
      {piece.text}
    </span>
  );
}

/**
 * The key to the colours.
 *
 * It is always present, and every entry is a coloured dot beside a word, never
 * a colour on its own -- which is both the accessibility rule and the only way
 * a person seeing this for the first time knows what amber means.
 */
export function VerdictLegend({ className = "" }: { className?: string }) {
  return (
    <ul className={`flex flex-wrap items-center gap-x-7 gap-y-3 ${className}`}>
      <li className="inline-flex items-center gap-2.5">
        <span className="mark bg-flag-soft [--mark-ring:var(--color-flag-ring)] text-flag-ink">
          <Dot tone="flag" />
        </span>
        <span className="t-small font-medium text-slate-700">{VERDICT.flag.label}</span>
      </li>
      <li className="inline-flex items-center gap-2.5">
        <span className="mark bg-abstain-soft [--mark-ring:var(--color-abstain-ring)]">
          <Dot tone="abstain" />
        </span>
        <span className="t-small font-medium text-slate-700">{VERDICT.abstain.label}</span>
      </li>
      <li className="inline-flex items-center gap-2.5">
        <span className="inline-flex items-center rounded-md px-1.5 py-1">
          <span className="h-2.5 w-2.5 rounded-full bg-slate-300" />
        </span>
        <span className="t-small font-medium text-slate-700">
          Left plain — {VERDICT.pass.short.toLowerCase()}
        </span>
      </li>
    </ul>
  );
}
