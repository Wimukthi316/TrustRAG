import { useEffect, useState } from "react";
import { figureUrl, getMetrics } from "../api";
import type { MetricsResponse } from "../types";

// The evidence behind the number on the Analyse tab. Everything here is read
// from JSON the offline runs wrote; nothing is computed in the browser, so the
// tab and the report cannot drift apart.
//
// Two presentation rules are enforced in this file rather than left to whoever
// reads it. Coverage is never shown without the band it is judged against, and
// an ECE is never shown without the constant-base-rate row beneath it.

const pct = (value: number) => `${(100 * value).toFixed(1)}%`;
const num = (value: number | null | undefined, places = 4) =>
  value === null || value === undefined ? "—" : value.toFixed(places);

export default function MetricsPanel() {
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
        {error}
      </div>
    );
  }
  if (!metrics) {
    return <p className="text-sm text-slate-500">Loading measurements…</p>;
  }
  if (!metrics.available) {
    return (
      <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950">
        {metrics.notes[0] ?? "No C2 results on disk."}
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <Headline metrics={metrics} />
      <Calibration metrics={metrics} />
      <Coverage metrics={metrics} />
      <PerTask metrics={metrics} />
      {metrics.shift_available && <Shift metrics={metrics} />}
      {metrics.risk_control.length > 0 && <RiskControl metrics={metrics} />}
      <Figures names={metrics.figures} />
      <Notes notes={metrics.notes} />
    </div>
  );
}

function Headline({ metrics }: { metrics: MetricsResponse }) {
  const floor = metrics.calibration.find((row) => row.is_floor);
  return (
    <section className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <Stat
        label="ECE, before → after"
        value={`${num(metrics.ece_before)} → ${num(metrics.ece_after)}`}
        hint={
          floor
            ? `an uninformative predictor scores ${num(floor.ece)}`
            : undefined
        }
      />
      <Stat
        label="AUROC"
        value={num(metrics.auroc)}
        hint="0.5 is chance. This is what calibration cannot change."
      />
      <Stat
        label="Calibration / test spans"
        value={`${metrics.n_calibration.toLocaleString()} / ${metrics.n_test.toLocaleString()}`}
        hint={`${pct(metrics.positive_rate_test)} of test spans are hallucinated`}
      />
      <Stat
        label="Selected calibrator"
        value={metrics.selected_calibrator ?? "—"}
        hint="chosen by lowest test ECE across four candidates — a declared peek"
      />
    </section>
  );
}

function Calibration({ metrics }: { metrics: MetricsResponse }) {
  return (
    <Block
      title="Calibration"
      subtitle="Fitted on the held-out calibration split, every figure measured on test."
    >
      <Table head={["method", "ECE", "MCE", "Brier"]}>
        {metrics.calibration.map((row) => (
          <tr
            key={row.method}
            className={row.is_floor ? "bg-slate-50 text-slate-500" : undefined}
          >
            <Cell>
              {row.selected ? (
                <span className="font-medium text-slate-900">
                  {row.method} <Tag tone="ok">selected</Tag>
                </span>
              ) : row.is_floor ? (
                <span className="italic">
                  {row.method} <Tag tone="warn">ignores the input</Tag>
                </span>
              ) : (
                row.method
              )}
            </Cell>
            <Cell mono>{num(row.ece)}</Cell>
            <Cell mono>{num(row.mce)}</Cell>
            <Cell mono>{num(row.brier)}</Cell>
          </tr>
        ))}
      </Table>
    </Block>
  );
}

function Coverage({ metrics }: { metrics: MetricsResponse }) {
  return (
    <Block
      title="Coverage vs α"
      subtitle="Below target is only a shortfall outside the band — the guarantee holds in expectation over calibration draws."
    >
      <Table
        head={["α", "target", "empirical", "±band", "abstain", "empty", "flag", ""]}
      >
        {metrics.coverage.map((row) => (
          <tr key={row.alpha}>
            <Cell mono>{row.alpha.toFixed(2)}</Cell>
            <Cell mono>{row.target_coverage.toFixed(3)}</Cell>
            <Cell mono>{num(row.empirical_coverage)}</Cell>
            <Cell mono>±{num(row.band)}</Cell>
            <Cell mono>{num(row.abstention_rate)}</Cell>
            <Cell mono>{num(row.empty_set_rate)}</Cell>
            <Cell mono>{num(row.flag_rate)}</Cell>
            <Cell>
              {row.inside_band ? (
                <Tag tone="ok">in band</Tag>
              ) : (
                <Tag tone="bad">outside band</Tag>
              )}
            </Cell>
          </tr>
        ))}
      </Table>
      <p className="mt-2 text-xs text-slate-500">
        The abstain column is not monotone in α, and the empty column beside it
        is why: past a point the threshold is tight enough that some spans have
        neither label clearing it. An empty set means the same thing to a
        reviewer as an undecided one — send it to a human.
      </p>
    </Block>
  );
}

function PerTask({ metrics }: { metrics: MetricsResponse }) {
  if (metrics.per_task.length === 0) return null;
  return (
    <Block
      title="Coverage within each task, α = 0.10"
      subtitle="A separate threshold per task, so the promise holds inside each rather than only on average across them."
    >
      <Table head={["task", "test spans", "calib spans", "coverage", "±band", "abstain", ""]}>
        {metrics.per_task.map((row) => (
          <tr key={row.group}>
            <Cell>{row.group}</Cell>
            <Cell mono>{row.n_test.toLocaleString()}</Cell>
            <Cell mono>{row.n_calibration.toLocaleString()}</Cell>
            <Cell mono>{num(row.empirical_coverage)}</Cell>
            <Cell mono>±{num(row.band)}</Cell>
            <Cell mono>{num(row.abstention_rate)}</Cell>
            <Cell>
              {row.inside_band ? <Tag tone="ok">in band</Tag> : <Tag tone="bad">outside</Tag>}
            </Cell>
          </tr>
        ))}
      </Table>
    </Block>
  );
}

function Shift({ metrics }: { metrics: MetricsResponse }) {
  return (
    <Block
      title="Under domain shift — RAGBench"
      subtitle="Calibrated on RAGTruth, applied unchanged to another corpus."
    >
      <div className="mb-3 rounded-md border border-rose-200 bg-rose-50 px-3 py-2 text-xs text-rose-900">
        <span className="font-semibold">VOID.</span> Calibration and test data
        from different corpora are not exchangeable, so the coverage guarantee
        does not apply to the shifted column. These numbers measure the break;
        they are not a working method.
      </div>
      <Table head={["α", "target", "in-domain", "shifted (VOID)", "best repair", "method"]}>
        {metrics.shift.map((row) => (
          <tr key={row.alpha}>
            <Cell mono>{row.alpha.toFixed(2)}</Cell>
            <Cell mono>{row.target_coverage.toFixed(3)}</Cell>
            <Cell mono>{num(row.in_domain)}</Cell>
            <Cell mono>
              <span className="text-rose-700">{num(row.shifted)}</span>
            </Cell>
            <Cell mono>{num(row.repaired)}</Cell>
            <Cell>
              <span className="text-slate-500">{row.repaired_method ?? "—"}</span>
            </Cell>
          </tr>
        ))}
      </Table>
      <p className="mt-2 text-xs text-slate-500">
        Neither reweighting restores coverage, so no third was attempted. The
        honest deployment rule is to recalibrate on your own data before
        trusting the dial.
      </p>
    </Block>
  );
}

function RiskControl({ metrics }: { metrics: MetricsResponse }) {
  return (
    <Block
      title="Conformal risk control"
      subtitle="At most α of hallucinated tokens go unflagged — the promise in the product's own language."
    >
      <Table head={["α", "threshold", "measured risk", "tokens flagged", ""]}>
        {metrics.risk_control.map((row) => (
          <tr key={row.alpha}>
            <Cell mono>{row.alpha.toFixed(2)}</Cell>
            <Cell mono>
              {num(row.threshold)}
              {row.on_grid_edge && <Tag tone="warn">grid edge</Tag>}
            </Cell>
            <Cell mono>{num(row.test_risk)}</Cell>
            <Cell mono>{row.token_flag_rate === null ? "—" : pct(row.token_flag_rate)}</Cell>
            <Cell>
              {row.bound_held === null ? (
                <Tag tone="warn">unreachable</Tag>
              ) : row.bound_held ? (
                <Tag tone="ok">held</Tag>
              ) : (
                <Tag tone="bad">missed</Tag>
              )}
            </Cell>
          </tr>
        ))}
      </Table>
      <p className="mt-2 text-xs text-slate-500">
        Read the flag rate beside every risk. Any false-negative bound can be
        satisfied by highlighting the whole answer, and a threshold at the edge
        of the search grid has done exactly that.
      </p>
    </Block>
  );
}

function Figures({ names }: { names: string[] }) {
  if (names.length === 0) return null;
  return (
    <Block
      title="Figures"
      subtitle="Regenerated from the same JSON by `python -m src.c2_calibration.figures`."
    >
      <div className="grid gap-4 sm:grid-cols-2">
        {names.map((name) => (
          <figure key={name} className="rounded-lg border border-slate-200 p-2">
            <img
              src={figureUrl(name)}
              alt={name.replace(/_/g, " ")}
              className="w-full rounded"
              loading="lazy"
            />
            <figcaption className="mt-1 text-center font-mono text-[11px] text-slate-400">
              {name}
            </figcaption>
          </figure>
        ))}
      </div>
    </Block>
  );
}

function Notes({ notes }: { notes: string[] }) {
  if (notes.length === 0) return null;
  return (
    <section className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-600">
        How to read this
      </h3>
      <ul className="space-y-1.5 text-xs leading-5 text-slate-600">
        {notes.map((note) => (
          <li key={note} className="flex gap-2">
            <span className="text-slate-400">—</span>
            <span>{note}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

// --- small presentational pieces --------------------------------------------

function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
        {label}
      </p>
      <p className="mt-1 font-mono text-sm text-slate-900">{value}</p>
      {hint && <p className="mt-1 text-[11px] leading-4 text-slate-400">{hint}</p>}
    </div>
  );
}

function Block({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <h3 className="text-sm font-semibold text-slate-900">{title}</h3>
      {subtitle && <p className="mb-3 mt-0.5 text-xs text-slate-500">{subtitle}</p>}
      {children}
    </section>
  );
}

function Table({
  head,
  children,
}: {
  head: string[];
  children: React.ReactNode;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border border-slate-200">
      <table className="w-full text-xs">
        <thead className="bg-slate-50 text-slate-500">
          <tr>
            {head.map((cell, index) => (
              <th
                key={`${cell}-${index}`}
                className="px-3 py-2 text-left font-medium"
              >
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100">{children}</tbody>
      </table>
    </div>
  );
}

function Cell({ children, mono }: { children: React.ReactNode; mono?: boolean }) {
  return (
    <td className={`px-3 py-1.5 ${mono ? "font-mono text-slate-700" : "text-slate-700"}`}>
      {children}
    </td>
  );
}

function Tag({ tone, children }: { tone: "ok" | "warn" | "bad"; children: React.ReactNode }) {
  const tones = {
    ok: "bg-emerald-50 text-emerald-800 ring-emerald-200",
    warn: "bg-amber-50 text-amber-900 ring-amber-200",
    bad: "bg-rose-50 text-rose-800 ring-rose-200",
  } as const;
  return (
    <span
      className={`ml-1 rounded-full px-1.5 py-0.5 text-[10px] ring-1 ${tones[tone]}`}
    >
      {children}
    </span>
  );
}
