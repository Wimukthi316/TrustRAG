import { useState, type ReactNode } from "react";
import type { CoverageLevel, EvidenceChart, ReliabilityChart } from "../proof";

/**
 * Three pictures, and no more.
 *
 * The rule this file was written under: a person who has never heard of this
 * project should be able to read each one in about ten seconds and say out loud
 * what it shows. Everything that failed that test -- the per-task breakdowns,
 * the risk-control sweeps, the shift table -- is in the paper instead.
 *
 * The two series colours are `--series-1` and `--series-2` from index.css. They
 * are the pair that passed the palette validator on a white surface; they are
 * not to be swapped for something that looks nicer without re-running it.
 */

const S1 = "var(--series-1)"; // after correction / the system's own result
const S2 = "var(--series-2)"; // before correction / the raw model

type Tip = { x: number; y: number; title: string; body: string } | null;

/**
 * A plot may not shrink below the width its own labels need.
 *
 * Letting an SVG scale to a phone turns 16px axis text into 5px text, which is
 * worse than not showing the chart at all. So the plot keeps a floor width and
 * the narrow screen scrolls it sideways instead.
 */
function Plot({ children, min = 660 }: { children: ReactNode; min?: number }) {
  return (
    <div className="-mx-2 overflow-x-auto px-2 pb-3">
      <div style={{ minWidth: min }}>{children}</div>
    </div>
  );
}

function Tooltip({ tip }: { tip: Tip }) {
  if (!tip) return null;
  return (
    <div
      className="pointer-events-none absolute z-20 w-56 -translate-x-1/2 -translate-y-full rounded-xl bg-slate-900 px-4 py-3 text-left shadow-xl"
      style={{ left: `${tip.x}%`, top: `${tip.y}%` }}
    >
      <p className="text-[0.875rem] font-semibold text-white">{tip.title}</p>
      <p className="tnum mt-1 text-[0.8125rem] leading-5 text-slate-300">{tip.body}</p>
    </div>
  );
}

/* ---------------------------------------------------------------------------
   1. Does the percentage mean anything?
--------------------------------------------------------------------------- */

const R_W = 880;
const R_H = 520;
const R_PAD = { top: 30, right: 130, bottom: 78, left: 96 };
const R_MIN = 0.2;
const R_MAX = 1.0;

const rx = (v: number) =>
  R_PAD.left + ((v - R_MIN) / (R_MAX - R_MIN)) * (R_W - R_PAD.left - R_PAD.right);
const ry = (v: number) =>
  R_H - R_PAD.bottom - ((v - R_MIN) / (R_MAX - R_MIN)) * (R_H - R_PAD.top - R_PAD.bottom);

export function ReliabilityFigure({ data }: { data: ReliabilityChart }) {
  const [tip, setTip] = useState<Tip>(null);
  const ticks = [0.2, 0.4, 0.6, 0.8, 1.0];

  const series = [
    { key: "after", label: "After our correction", colour: S1, rows: data.after },
    { key: "before", label: "Before, straight out of the model", colour: S2, rows: data.before },
  ];

  return (
    <figure className="relative">
      <Plot>
      <svg
        viewBox={`0 0 ${R_W} ${R_H}`}
        className="w-full"
        role="img"
        aria-label="Scatter plot comparing what the system said against what actually happened, before and after correction. After correction the points sit close to the perfect diagonal; before correction they sit well below it."
      >
        {/* grid */}
        {ticks.map((t) => (
          <g key={`g${t}`}>
            <line
              x1={R_PAD.left}
              x2={R_W - R_PAD.right}
              y1={ry(t)}
              y2={ry(t)}
              stroke="var(--chart-grid)"
              strokeWidth={1}
            />
            <line
              x1={rx(t)}
              x2={rx(t)}
              y1={R_PAD.top}
              y2={R_H - R_PAD.bottom}
              stroke="var(--chart-grid)"
              strokeWidth={1}
            />
            <text x={R_PAD.left - 14} y={ry(t) + 6} textAnchor="end" fontSize={13} fill="#64748b">
              {Math.round(t * 100)}%
            </text>
            <text
              x={rx(t)}
              y={R_H - R_PAD.bottom + 28}
              textAnchor="middle"
              fontSize={13}
              fill="#64748b"
            >
              {Math.round(t * 100)}%
            </text>
          </g>
        ))}

        {/* the diagonal every honest system should sit on */}
        <line
          x1={rx(R_MIN)}
          y1={ry(R_MIN)}
          x2={rx(R_MAX)}
          y2={ry(R_MAX)}
          stroke="#94a3b8"
          strokeWidth={2}
          strokeDasharray="7 6"
        />
        <text
          x={rx(0.86)}
          y={ry(0.86) - 20}
          fontSize={13}
          fill="#64748b"
          textAnchor="end"
        >
          perfectly honest
        </text>

        {series.map((s) => (
          <g key={s.key}>
            <polyline
              points={s.rows.map((r) => `${rx(r.confidence)},${ry(r.actual)}`).join(" ")}
              fill="none"
              stroke={s.colour}
              strokeWidth={2.5}
              strokeLinejoin="round"
            />
            {s.rows.map((r, i) => (
              <circle
                key={i}
                cx={rx(r.confidence)}
                cy={ry(r.actual)}
                r={6}
                fill={s.colour}
                stroke="var(--chart-surface)"
                strokeWidth={2}
                onMouseEnter={() =>
                  setTip({
                    x: (rx(r.confidence) / R_W) * 100,
                    y: (ry(r.actual) / R_H) * 100 - 2,
                    title: s.label,
                    body: `Said ${Math.round(r.confidence * 100)}% · really was ${Math.round(
                      r.actual * 100,
                    )}% · ${r.n} passages`,
                  })
                }
                onMouseLeave={() => setTip(null)}
              />
            ))}
            {/* direct label at the end of the run */}
            <text
              x={rx(s.rows[s.rows.length - 1].confidence) + 14}
              y={ry(s.rows[s.rows.length - 1].actual) + 6}
              fontSize={13}
              fontWeight={600}
              fill={s.colour}
            >
              {s.key === "after" ? "after" : "before"}
            </text>
          </g>
        ))}

        <text
          x={R_PAD.left + (R_W - R_PAD.left - R_PAD.right) / 2}
          y={R_H - 22}
          textAnchor="middle"
          fontSize={14}
          fontWeight={600}
          fill="#334155"
        >
          What the system said the chance was
        </text>
        <text
          x={24}
          y={R_PAD.top + (R_H - R_PAD.top - R_PAD.bottom) / 2}
          textAnchor="middle"
          fontSize={14}
          fontWeight={600}
          fill="#334155"
          transform={`rotate(-90 24 ${R_PAD.top + (R_H - R_PAD.top - R_PAD.bottom) / 2})`}
        >
          How often it turned out to be true
        </text>
      </svg>
      </Plot>

      <Tooltip tip={tip} />

      <figcaption className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2.5">
        {series.map((s) => (
          <span key={s.key} className="inline-flex items-center gap-2.5">
            <span
              className="h-3 w-3 rounded-full"
              style={{ backgroundColor: s.colour }}
              aria-hidden
            />
            <span className="t-small font-medium text-slate-700">{s.label}</span>
          </span>
        ))}
        <span className="t-small tnum ml-auto text-slate-500">
          {data.n.toLocaleString()} passages held back from training
        </span>
      </figcaption>
    </figure>
  );
}

/* ---------------------------------------------------------------------------
   2. Did it keep the promise it made?
--------------------------------------------------------------------------- */

const C_W = 880;
const C_ROW = 62;
const C_LEFT = 252;
const C_RIGHT = 706;
const C_MIN = 0.5;
const C_MAX = 1.0;

const cx = (v: number) => C_LEFT + ((v - C_MIN) / (C_MAX - C_MIN)) * (C_RIGHT - C_LEFT);

export function CoverageFigure({ data }: { data: CoverageLevel[] }) {
  const [tip, setTip] = useState<Tip>(null);
  const rows = [...data].sort((a, b) => b.promised - a.promised);
  const height = rows.length * C_ROW + 116;
  const ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

  return (
    <figure className="relative">
      <Plot>
      <svg
        viewBox={`0 0 ${C_W} ${height}`}
        className="w-full"
        role="img"
        aria-label="One row per promise level. At every level the number it actually caught lands inside the allowance below what it promised."
      >
        {ticks.map((t) => (
          <g key={t}>
            <line
              x1={cx(t)}
              x2={cx(t)}
              y1={34}
              y2={rows.length * C_ROW + 40}
              stroke="var(--chart-grid)"
              strokeWidth={1}
            />
            <text x={cx(t)} y={rows.length * C_ROW + 70} textAnchor="middle" fontSize={13} fill="#64748b">
              {Math.round(t * 100)}
            </text>
          </g>
        ))}

        {rows.map((row, i) => {
          const y = 40 + i * C_ROW + C_ROW / 2 - 6;
          const bandLeft = cx(row.promised - row.band);
          const bandRight = cx(row.promised);
          return (
            <g key={row.promised}>
              <text x={C_LEFT - 24} y={y + 6} textAnchor="end" fontSize={14} fill="#334155">
                Promised {Math.round(row.promised * 100)} in 100
              </text>

              {/* the allowance a single honest run may fall short by */}
              <rect
                x={bandLeft}
                y={y - 15}
                width={Math.max(2, bandRight - bandLeft)}
                height={30}
                rx={6}
                fill="var(--series-1)"
                opacity={0.14}
              />
              {/* the promise itself */}
              <line
                x1={cx(row.promised)}
                x2={cx(row.promised)}
                y1={y - 19}
                y2={y + 19}
                stroke="#475569"
                strokeWidth={2.5}
              />
              {/* what it actually did */}
              <circle
                cx={cx(row.delivered)}
                cy={y}
                r={7.5}
                fill="var(--series-1)"
                stroke="var(--chart-surface)"
                strokeWidth={2.5}
                onMouseEnter={() =>
                  setTip({
                    x: (cx(row.delivered) / C_W) * 100,
                    y: ((y - 14) / height) * 100,
                    title: `Promised ${Math.round(row.promised * 100)} in 100`,
                    body: `Caught ${(row.delivered * 100).toFixed(1)} · allowed to fall ${(
                      row.band * 100
                    ).toFixed(1)} short · ${row.held ? "kept" : "missed"}`,
                  })
                }
                onMouseLeave={() => setTip(null)}
              />
              <text
                x={C_RIGHT + 22}
                y={y + 6}
                fontSize={14}
                fontWeight={600}
                fill="#334155"
                className="tnum"
              >
                {(row.delivered * 100).toFixed(1)}
              </text>
            </g>
          );
        })}

        <text
          x={C_LEFT + (C_RIGHT - C_LEFT) / 2}
          y={height - 14}
          textAnchor="middle"
          fontSize={14}
          fontWeight={600}
          fill="#334155"
        >
          Out of every 100 made-up passages, how many it caught
        </text>
      </svg>
      </Plot>

      <Tooltip tip={tip} />

      <figcaption className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2.5">
        <span className="inline-flex items-center gap-2.5">
          <span className="h-4 w-1 rounded bg-slate-600" aria-hidden />
          <span className="t-small font-medium text-slate-700">What it promised</span>
        </span>
        <span className="inline-flex items-center gap-2.5">
          <span
            className="h-3 w-3 rounded-full"
            style={{ backgroundColor: S1 }}
            aria-hidden
          />
          <span className="t-small font-medium text-slate-700">What it caught</span>
        </span>
        <span className="inline-flex items-center gap-2.5">
          <span
            className="h-4 w-7 rounded"
            style={{ backgroundColor: S1, opacity: 0.14 }}
            aria-hidden
          />
          <span className="t-small font-medium text-slate-700">
            How far short one fair run is allowed to land
          </span>
        </span>
      </figcaption>
    </figure>
  );
}

/* ---------------------------------------------------------------------------
   3. How much reading it saves.

   Not a chart. Two numbers whose whole point is "how many things do I have to
   read", and the honest form for that is one square per thing.
--------------------------------------------------------------------------- */

export function EvidenceFigure({ data }: { data: EvidenceChart }) {
  const pool = Math.round(data.pool);
  const returned = Math.round(data.returned);
  const squares = Array.from({ length: pool }, (_, i) => i < returned);

  return (
    <figure>
      <div
        className="grid max-w-lg grid-cols-10 gap-2 sm:gap-2.5"
        role="img"
        aria-label={data.sentence}
      >
        {squares.map((filled, i) => (
          <span
            key={i}
            title={filled ? "you read this one" : "you can skip this one"}
            className={`aspect-square rounded-lg ${filled ? "" : "bg-slate-200"}`}
            style={filled ? { backgroundColor: S1 } : undefined}
          />
        ))}
      </div>

      <figcaption className="mt-5 flex flex-wrap items-center gap-x-6 gap-y-2.5">
        <span className="inline-flex items-center gap-2.5">
          <span className="h-3.5 w-3.5 rounded" style={{ backgroundColor: S1 }} aria-hidden />
          <span className="t-small font-medium text-slate-700">
            Sentences it hands you — about {data.returned.toFixed(1)}
          </span>
        </span>
        <span className="inline-flex items-center gap-2.5">
          <span className="h-3.5 w-3.5 rounded bg-slate-200" aria-hidden />
          <span className="t-small font-medium text-slate-700">
            Sentences you can skip — about {(data.pool - data.returned).toFixed(1)}
          </span>
        </span>
      </figcaption>
    </figure>
  );
}
