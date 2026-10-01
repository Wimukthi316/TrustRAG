import { useEffect, useMemo, useRef, useState } from "react";
import { useReducedMotion } from "framer-motion";
import { ArrowRight, Loader2, Pause, Play, RotateCcw, UserCheck } from "lucide-react";
import { getAdaptive, type AdaptiveTimeline, type SeriesPoint } from "../adaptive";
import { Band, BandHead, Button, Card, Pill, Reveal, Stat } from "../components/ui";

/**
 * The hard case, and the answer to it.
 *
 * Split conformal keeps its promise only while the new documents look like the
 * ones it was calibrated on. The shift study measured what happens when they do
 * not -- on RAGBench the promised 90% coverage collapses to about 74% -- and
 * both static repairs failed. This page is the fix that did not: a reviewer
 * confirms each span the system shows, and that one bit of feedback re-tunes the
 * dial until the long-run promise holds again.
 *
 * Every number here is measured by `src/c2_calibration/online_conformal.py` and
 * read from `public/adaptive-conformal.json`. The reviewer is simulated by the
 * gold label, which the page says out loud rather than implies.
 */

const PASS = "var(--color-pass)"; // the adaptive loop -- the result
const FLAG = "var(--color-flag)"; // the static method -- what breaks

export default function Adaptive({ onProof }: { onProof: () => void }) {
  const [data, setData] = useState<AdaptiveTimeline | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAdaptive()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) {
    return (
      <Band tone="white">
        <BandHead
          title="The adaptive run did not load."
          lead={`Nothing was found to draw from: ${error}`}
        />
      </Band>
    );
  }

  if (!data) {
    return (
      <Band tone="white">
        <div className="flex items-center gap-4 text-slate-500">
          <Loader2 className="h-5 w-5 animate-spin" aria-hidden />
          <p className="t-lead">Loading the adaptive run…</p>
        </div>
      </Band>
    );
  }

  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const staticShort = data.target_coverage - data.static.coverage;

  return (
    <>
      {/* ---- hero ------------------------------------------------------- */}
      <Band tone="white">
        <div className="mx-auto max-w-2xl text-center">
          <p className="t-eyebrow text-brand-600">The hard case</p>
          <h1 className="t-title mt-4 text-slate-900">
            When the documents change, the promise breaks — then heals itself.
          </h1>
          <p className="t-lead mt-4 text-slate-600">
            The guarantee only holds while new answers look like the ones the
            system was tuned on. Point it at a different kind of document and the
            promise quietly fails. Here it is failing — and here is a review loop
            that brings it back, one correction at a time, with no assumption
            about what changed.
          </p>
          <div className="mt-6 flex flex-wrap justify-center gap-2">
            <Pill tone="flag">Static: promised {pct(data.target_coverage)}, kept {pct(data.static.coverage)}</Pill>
            <Pill tone="pass">Adaptive: back to {pct(data.adaptive.coverage)}</Pill>
          </div>
        </div>
      </Band>

      {/* ---- the live chart --------------------------------------------- */}
      <Band tone="plain">
        <div className="mx-auto max-w-4xl">
          <BandHead
            step={1}
            eyebrow="Watch it recover"
            title="One promise, two methods, the same stream of answers."
            lead="Play the stream. The red line is today's method with a fixed setting — it settles below the promised line and stays there. The green line is the same system with the review loop switched on: it climbs back up to the promise and holds."
          />
          <Reveal>
            <Card className="p-5 sm:p-8">
              <SelfHealingChart data={data} />
            </Card>
          </Reveal>
        </div>
      </Band>

      {/* ---- the scoreboard --------------------------------------------- */}
      <Band tone="white">
        <div className="mx-auto max-w-5xl">
          <BandHead
            step={2}
            eyebrow="What it cost"
            title="The promise is back. Nothing was swept under the rug to get it."
            lead="Recovering a guarantee off-distribution is not free, and the honest price is shown next to the win: to keep the promise on documents this different, the system has to hand far more of them to a human instead of deciding itself."
          />
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            <Card className="p-6">
              <Stat
                label="Promised coverage"
                value={pct(data.target_coverage)}
                tone="neutral"
                hint="The target set before any document is read: catch the true label this often."
              />
            </Card>
            <Card className="p-6">
              <Stat
                label="Static method, off-distribution"
                value={pct(data.static.coverage)}
                tone="bad"
                floor={`${(staticShort * 100).toFixed(1)} points under the promise`}
                hint="Fixed setting calibrated on the original corpus, applied to the new one. This is what breaks."
              />
            </Card>
            <Card className="p-6">
              <Stat
                label="With the review loop"
                value={pct(data.adaptive.coverage)}
                tone="good"
                floor={`target ${pct(data.target_coverage)} — met`}
                hint="Adaptive conformal: the miscoverage level is nudged by each review until the long-run promise holds."
              />
            </Card>
            <Card className="p-6">
              <Stat
                label="Answers sent to a human"
                value={pct(data.adaptive.abstain_rate)}
                unit="abstain"
                tone="warn"
                floor={`was ${pct(data.static.abstain_rate)} before`}
                hint="The cost of the repair. Under a shift this large the system keeps its promise mainly by deferring, not by deciding."
              />
            </Card>
          </div>

          <Reveal delay={0.1}>
            <div className="mt-6 rounded-2xl border-l-4 border-brand-500 bg-white px-6 py-6 shadow-[0_1px_2px_rgba(15,23,42,0.04)] sm:px-8">
              <p className="t-eyebrow text-brand-600">Read this honestly</p>
              <p className="t-lead mt-3 max-w-3xl text-slate-700">
                The static method does not just under-deliver — it under-delivers{" "}
                <Num>{(staticShort * 100).toFixed(0)} points</Num> below what it
                promised, which for a guarantee is not a weak result, it is a
                wrong one. The loop earns the promise back, and the meter above
                shows what that takes: abstention rises from{" "}
                <Num>{pct(data.static.abstain_rate)}</Num> to{" "}
                <Num>{pct(data.adaptive.abstain_rate)}</Num>. The method does not
                hide the price; it moves the dial to wherever the promise requires
                and tells you where that landed.
              </p>
            </div>
          </Reveal>
        </div>
      </Band>

      {/* ---- how it works ----------------------------------------------- */}
      <Band tone="plain">
        <div className="mx-auto max-w-4xl">
          <BandHead
            step={3}
            eyebrow="How the dial learns"
            title="One reviewer, one bit of feedback, one update."
            lead="No model of the shift, no guess at what the new documents look like. After each answer the system shows someone, they say whether the marked span was really unsupported, and that single yes-or-no moves the dial."
          />
          <div className="grid gap-5 md:grid-cols-3">
            <Step
              n="a"
              title="Show and decide"
              body="For the next span the system builds its answer at the dial's current setting — flag, pass, or send to a human."
            />
            <Step
              n="b"
              title="A person confirms"
              body="The reviewer says whether the true label was inside the answer. That is the only signal the loop needs."
              icon
            />
            <Step
              n="c"
              title="Nudge the dial"
              body="Missed it → promise a little more next time. Covered it → relax a little. Over the stream the misses settle at exactly the rate promised."
            />
          </div>

          <Reveal delay={0.1}>
            <Card className="mt-6 p-6 sm:p-8">
              <p className="t-eyebrow text-slate-500">The whole update</p>
              <p className="tnum mt-3 text-center text-[1.15rem] text-slate-900 sm:text-[1.35rem]">
                α<sub>next</sub> = α<sub>now</sub> + γ · (target − missed?)
              </p>
              <p className="t-small mt-4 text-slate-600">
                That is the entire method (Adaptive Conformal Inference, Gibbs &amp;
                Candès, 2021). It carries a <b>long-run</b> guarantee: averaged over
                the stream, the miss rate lands on the target for any sequence at
                all — which is exactly the promise plain conformal gives up the
                moment the documents change. It says nothing about any single
                answer, and it pays in wider answers and more deferral.
              </p>
            </Card>
          </Reveal>

          <Reveal delay={0.15}>
            <div className="mt-5 flex items-start gap-3 rounded-xl bg-abstain-soft px-5 py-4 text-abstain-ink ring-1 ring-inset ring-abstain-ring">
              <UserCheck className="mt-0.5 h-5 w-5 shrink-0" aria-hidden />
              <p className="t-small">
                <b>The reviewer here is simulated by the gold label</b>, which is
                the standard way this method is tested. These are results from a
                simulated review loop, not from a study with real reviewers — and
                the same run shows that under a shift this large almost every span
                is sent for review anyway ({pct(data.review_only.review_fraction)}),
                so the review-only loop reaches nearly the same place (
                {pct(data.review_only.coverage)}).
              </p>
            </div>
          </Reveal>
        </div>
      </Band>

      {/* ---- robustness ------------------------------------------------- */}
      <Band tone="white">
        <div className="mx-auto max-w-4xl">
          <BandHead
            step={4}
            eyebrow="Not one lucky setting"
            title="It lands on the promise across learning rates and stream orders."
            lead="A single hand-picked setting would not be evidence of much. The loop was run at three learning rates and on a stream reordered so the domain changes in blocks. The promised level is 90%."
          />
          <Reveal>
            <Card className="overflow-hidden">
              <table className="w-full text-left">
                <thead>
                  <tr className="border-b border-slate-200 bg-slate-50/60 text-slate-600">
                    <th className="px-5 py-3 t-small font-semibold">Setting</th>
                    <th className="px-5 py-3 t-small font-semibold">Learning rate γ</th>
                    <th className="px-5 py-3 t-small font-semibold">Coverage reached</th>
                    <th className="px-5 py-3 t-small font-semibold">Sent to a human</th>
                  </tr>
                </thead>
                <tbody className="tnum">
                  {data.gamma_sweep.map((g) => (
                    <tr key={g.gamma} className="border-b border-slate-100 last:border-0">
                      <td className="px-5 py-3 t-body text-slate-500">shuffled stream</td>
                      <td className="px-5 py-3 t-body text-slate-800">{g.gamma.toFixed(2)}</td>
                      <td className="px-5 py-3 t-body font-semibold text-pass-ink">
                        {pct(g.coverage)}
                      </td>
                      <td className="px-5 py-3 t-body text-slate-700">{pct(g.abstain_rate)}</td>
                    </tr>
                  ))}
                  <tr className="border-b border-slate-100 last:border-0">
                    <td className="px-5 py-3 t-body text-slate-500">blocks by domain</td>
                    <td className="px-5 py-3 t-body text-slate-800">{data.gamma.toFixed(2)}</td>
                    <td className="px-5 py-3 t-body font-semibold text-pass-ink">
                      {pct(data.natural_order.coverage)}
                    </td>
                    <td className="px-5 py-3 t-body text-slate-700">
                      {pct(data.natural_order.abstain_rate)}
                    </td>
                  </tr>
                </tbody>
              </table>
            </Card>
          </Reveal>
          <p className="t-small mt-4 text-slate-500">
            Every row clears the promised 90%. Measured over{" "}
            {data.n_stream.toLocaleString()} predicted spans on RAGBench, streamed
            past a threshold calibrated only on the original corpus.
          </p>
        </div>
      </Band>

      {/* ---- CTA -------------------------------------------------------- */}
      <Band tone="dark">
        <div className="max-w-3xl">
          <h2 className="t-h2 text-white">This is the promise the static page makes — repaired.</h2>
          <p className="t-lead mt-4 text-brand-100/85">
            The proof page shows the guarantee holding on the corpus it was built
            for. This one shows what happens when the world moves, and how a human
            in the loop puts the guarantee back.
          </p>
          <div className="mt-7">
            <Button variant="onDark" onClick={onProof}>
              See the in-domain proof
              <ArrowRight className="h-4 w-4" aria-hidden />
            </Button>
          </div>
        </div>
      </Band>
    </>
  );
}

/* ------------------------------------------------------------------------- */

function Num({ children }: { children: React.ReactNode }) {
  return <span className="tnum font-semibold text-slate-900">{children}</span>;
}

function Step({
  n,
  title,
  body,
  icon = false,
}: {
  n: string;
  title: string;
  body: string;
  icon?: boolean;
}) {
  return (
    <Card className="h-full p-6">
      <span className="tnum inline-flex h-8 w-8 items-center justify-center rounded-full bg-brand-600 text-[0.8125rem] font-semibold uppercase text-white">
        {icon ? <UserCheck className="h-4 w-4" aria-hidden /> : n}
      </span>
      <h3 className="t-h3 mt-4 text-slate-900">{title}</h3>
      <p className="t-body mt-2 text-slate-600">{body}</p>
    </Card>
  );
}

/* ---------------------------------------------------------------------------
   The self-healing chart.

   Two rolling-coverage curves over one stream, with a playhead a reader drives.
   The point it has to make in ten seconds: the red line settles below the
   promised line, the green line climbs back to it. Everything else -- the axes,
   the live readout -- serves that one comparison.
--------------------------------------------------------------------------- */

const W = 880;
const H = 470;
const PAD = { top: 28, right: 116, bottom: 64, left: 64 };
const Y_MIN = 0.5;
const Y_MAX = 1.0;

function SelfHealingChart({ data }: { data: AdaptiveTimeline }) {
  const still = useReducedMotion();
  const series = data.series;
  const last = series.length - 1;
  const [cursor, setCursor] = useState(last);
  const [playing, setPlaying] = useState(false);
  const raf = useRef<number | null>(null);

  // Reduced motion: never auto-animate, just show the finished curves.
  const canAnimate = !still && series.length > 4;

  useEffect(() => {
    if (!playing) return;
    let stop = false;
    let acc = 0;
    let prev = performance.now();
    const tick = (nowMs: number) => {
      if (stop) return;
      acc += nowMs - prev;
      prev = nowMs;
      // About 7 seconds end to end regardless of point count.
      const perStep = 7000 / series.length;
      if (acc >= perStep) {
        const steps = Math.floor(acc / perStep);
        acc -= steps * perStep;
        setCursor((c) => {
          const next = c + steps;
          if (next >= last) {
            setPlaying(false);
            return last;
          }
          return next;
        });
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      stop = true;
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [playing, series.length, last]);

  const nStream = data.n_stream;
  const xMax = series[last]?.t ?? nStream;
  const sx = (t: number) => PAD.left + (t / xMax) * (W - PAD.left - PAD.right);
  const clampY = (v: number) => Math.max(Y_MIN, Math.min(Y_MAX, v));
  const sy = (v: number) =>
    H - PAD.bottom - ((clampY(v) - Y_MIN) / (Y_MAX - Y_MIN)) * (H - PAD.top - PAD.bottom);

  const shown = useMemo(() => series.slice(0, cursor + 1), [series, cursor]);
  const line = (key: "static" | "adaptive") =>
    shown.map((p) => `${sx(p.t)},${sy(p[key])}`).join(" ");

  const cur: SeriesPoint = series[cursor] ?? series[last];
  const yTicks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const targetY = sy(data.target_coverage);

  function play() {
    if (cursor >= last) setCursor(0);
    setPlaying(true);
  }
  function restart() {
    setPlaying(false);
    setCursor(0);
  }

  return (
    <figure className="relative">
      <div className="-mx-2 overflow-x-auto px-2 pb-2">
        <div style={{ minWidth: 640 }}>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full"
            role="img"
            aria-label={`Two rolling-coverage lines over a stream of ${nStream} answers. The static method settles around ${(
              data.static.coverage * 100
            ).toFixed(0)} percent, below the promised 90; the adaptive loop climbs back to about ${(
              data.adaptive.coverage * 100
            ).toFixed(0)} percent.`}
          >
            {/* y grid + labels */}
            {yTicks.map((t) => (
              <g key={t}>
                <line
                  x1={PAD.left}
                  x2={W - PAD.right}
                  y1={sy(t)}
                  y2={sy(t)}
                  stroke="var(--chart-grid)"
                  strokeWidth={1}
                />
                <text x={PAD.left - 12} y={sy(t) + 5} textAnchor="end" fontSize={13} fill="#64748b">
                  {Math.round(t * 100)}%
                </text>
              </g>
            ))}

            {/* the promised line */}
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={targetY}
              y2={targetY}
              stroke="#475569"
              strokeWidth={2}
              strokeDasharray="7 6"
            />
            <text x={W - PAD.right + 8} y={targetY + 5} fontSize={13} fontWeight={600} fill="#475569">
              promised
            </text>

            {/* static (breaks) */}
            <polyline
              points={line("static")}
              fill="none"
              stroke={FLAG}
              strokeWidth={2.75}
              strokeLinejoin="round"
              strokeLinecap="round"
            />
            {/* adaptive (recovers) */}
            <polyline
              points={line("adaptive")}
              fill="none"
              stroke={PASS}
              strokeWidth={2.75}
              strokeLinejoin="round"
              strokeLinecap="round"
            />

            {/* playhead + live dots */}
            <line
              x1={sx(cur.t)}
              x2={sx(cur.t)}
              y1={PAD.top}
              y2={H - PAD.bottom}
              stroke="#cbd5e1"
              strokeWidth={1.5}
            />
            <circle cx={sx(cur.t)} cy={sy(cur.static)} r={6} fill={FLAG} stroke="#fff" strokeWidth={2} />
            <circle cx={sx(cur.t)} cy={sy(cur.adaptive)} r={6} fill={PASS} stroke="#fff" strokeWidth={2} />

            {/* end labels */}
            <text x={W - PAD.right + 8} y={sy(cur.adaptive) + 5} fontSize={13} fontWeight={700} fill={PASS}>
              {(cur.adaptive * 100).toFixed(0)}
            </text>
            <text x={W - PAD.right + 8} y={sy(cur.static) + 5} fontSize={13} fontWeight={700} fill={FLAG}>
              {(cur.static * 100).toFixed(0)}
            </text>

            <text
              x={PAD.left + (W - PAD.left - PAD.right) / 2}
              y={H - 22}
              textAnchor="middle"
              fontSize={14}
              fontWeight={600}
              fill="#334155"
            >
              Answers seen so far →
            </text>
          </svg>
        </div>
      </div>

      {/* controls + live readout */}
      <div className="mt-4 flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <Button
            size="md"
            variant="primary"
            onClick={() => (playing ? setPlaying(false) : play())}
            disabled={!canAnimate}
          >
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
            {playing ? "Pause" : cursor >= last ? "Replay" : "Play"}
          </Button>
          <Button size="md" variant="secondary" onClick={restart} disabled={!canAnimate}>
            <RotateCcw className="h-4 w-4" />
            Reset
          </Button>
        </div>

        <input
          type="range"
          min={0}
          max={last}
          value={cursor}
          onChange={(e) => {
            setPlaying(false);
            setCursor(Number(e.target.value));
          }}
          className="h-1.5 flex-1 cursor-pointer accent-brand-600"
          aria-label="Scrub through the stream"
        />

        <div className="tnum flex items-center gap-5 text-[0.8125rem]">
          <span className="text-slate-500">
            span{" "}
            <span className="font-semibold text-slate-800">
              {cur.t.toLocaleString()}
            </span>
            /{nStream.toLocaleString()}
          </span>
          <span className="text-slate-500">
            dial α{" "}
            <span className="font-semibold text-slate-800">{cur.alpha.toFixed(3)}</span>
          </span>
        </div>
      </div>

      <figcaption className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2.5">
        <span className="inline-flex items-center gap-2.5">
          <span className="h-3 w-3 rounded-full" style={{ backgroundColor: PASS }} aria-hidden />
          <span className="t-small font-medium text-slate-700">With the review loop</span>
        </span>
        <span className="inline-flex items-center gap-2.5">
          <span className="h-3 w-3 rounded-full" style={{ backgroundColor: FLAG }} aria-hidden />
          <span className="t-small font-medium text-slate-700">Fixed setting (today)</span>
        </span>
        <span className="inline-flex items-center gap-2.5">
          <span className="h-3.5 w-4 rounded" style={{ borderTop: "2px dashed #475569" }} aria-hidden />
          <span className="t-small font-medium text-slate-700">What it promised</span>
        </span>
      </figcaption>
    </figure>
  );
}
