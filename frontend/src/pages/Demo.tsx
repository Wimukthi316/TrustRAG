import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, ReactNode } from "react";
import {
  ArrowRight,
  CircleHelp,
  Clock,
  FileText,
  Loader2,
  MessageSquare,
  PenLine,
  Paperclip,
  Quote,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import { analyze, extractFile, getMetrics } from "../api";
import type { ExtractResult } from "../api";
import { saveCheck } from "../history";
import {
  CORRECTION,
  DIAL,
  ERROR_TYPE_DETAIL,
  PLAIN,
  EVIDENCE,
  FIELD,
  FIT,
  PROMISE,
  RULED,
  TASK_LABEL,
  TASK_PROMISE,
  VERDICT,
  careLabel,
} from "../copy";
import { resultAt, type RecordedCase } from "../recorded";
import type {
  AnalysisResult,
  AnalyzeRequest,
  ConformalDecision,
  CoverageRow,
  DistributionCheck,
  ErrorType,
  FeatureCheck,
  MetricsResponse,
  Span,
  TaskType,
} from "../types";
import {
  MarkedAnswer,
  VerdictLegend,
  countByVerdict,
  sortedSpans,
} from "../components/Marked";
import { Band, BandHead, Button, Card, Hint, Reveal } from "../components/ui";

/**
 * The working page, read by scrolling.
 *
 * Six bands in the order the process happens: pick a text, see what the system
 * was given, see what it marked, read each mark, find out whether the promise
 * covers this input, and decide what to do. Every band says in plain words what
 * is happening in it, because the person reading this is not the person who
 * built it and will not infer the step from the layout.
 *
 * Nothing is behind a tab. A tab would fit more on the screen, which is exactly
 * the problem it would be solving in the wrong direction.
 */

const VERDICT_ICON = {
  flag: ShieldAlert,
  abstain: CircleHelp,
  pass: ShieldCheck,
} as const;

const SKIN: Record<
  ConformalDecision,
  { chip: string; mark: string; ring: string; bar: string }
> = {
  flag: {
    chip: "bg-flag-soft text-flag-ink",
    mark: "bg-flag-soft text-flag-ink [--mark-ring:var(--color-flag-ring)]",
    ring: "ring-flag-ring",
    bar: "bg-flag",
  },
  abstain: {
    chip: "bg-abstain-soft text-abstain-ink",
    mark: "bg-abstain-soft text-abstain-ink [--mark-ring:var(--color-abstain-ring)]",
    ring: "ring-abstain-ring",
    bar: "bg-abstain",
  },
  pass: {
    chip: "bg-pass-soft text-pass-ink",
    mark: "bg-pass-soft text-pass-ink [--mark-ring:var(--color-pass-ring)]",
    ring: "ring-pass-ring",
    bar: "bg-pass",
  },
};

const pct = (x: number | null) => (x === null ? "—" : `${Math.round(x * 100)}%`);

export default function Demo({
  recorded,
  alphas,
  opened = null,
  onProof,
}: {
  recorded: RecordedCase[];
  /** The settings the dial may offer. Empty means no dial. */
  alphas: number[];
  /** A check reopened from the history page, if the reader came that way. */
  opened?: AnalysisResult | null;
  onProof: () => void;
}) {
  const [slug, setSlug] = useState<string | null>(null);
  // The result of a live check or a reopened one. A recorded case does not go
  // through here: it is looked up from the file at the dial's current setting,
  // so that turning the dial cannot leave the two out of step.
  const [ownResult, setOwnResult] = useState<AnalysisResult | null>(null);
  const [active, setActive] = useState<number>(0);

  // Own-text mode. Kept collapsed: it needs the backend running, and the point
  // of the recorded runs is that the demo works when it is not.
  const [own, setOwn] = useState(false);
  const [question, setQuestion] = useState("");
  const [context, setContext] = useState("");
  const [answer, setAnswer] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("qa");
  const [alpha, setAlpha] = useState(0.1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<"saved" | "failed" | null>(null);
  const [reopened, setReopened] = useState(false);
  // What the last live check was run with, so the dial can run it again at a
  // different setting. A recorded case needs none of this; a reopened one
  // cannot be re-run, and the dial says so rather than going quiet.
  const [lastRequest, setLastRequest] = useState<AnalyzeRequest | null>(null);

  // C2's offline evidence, for the per-task figure at the bottom of the page.
  //
  // Fetched once, and allowed to fail without consequence: what it carries is
  // an addition to the result on screen, never a precondition for it. The
  // recorded runs exist so this page works with no backend at all, and a
  // network error here must not take that away.
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  useEffect(() => {
    getMetrics()
      .then((next) => setMetrics(next.available ? next : null))
      .catch(() => setMetrics(null));
  }, []);

  // A check reopened from the history page wins over everything: the reader
  // clicked a specific row and is owed that row.
  useEffect(() => {
    if (!opened) return;
    setSlug(null);
    setOwn(false);
    setError(null);
    setSaved(null);
    setReopened(true);
    setLastRequest(null);
    setOwnResult(opened);
    if (opened.alpha !== null) setAlpha(opened.alpha);
  }, [opened]);

  // Otherwise open on the first recorded case, so the page is never a blank
  // form.
  useEffect(() => {
    if (ownResult || opened || recorded.length === 0) return;
    setSlug(recorded[0].slug);
  }, [recorded, ownResult, opened]);

  // What is on screen. A recorded case is read out of the file at the current
  // setting; anything else is whatever the last check returned.
  const chosen = slug ? (recorded.find((c) => c.slug === slug) ?? null) : null;
  const result = chosen ? resultAt(chosen, alpha) : ownResult;

  const spans = useMemo(() => (result ? sortedSpans(result) : []), [result]);

  useEffect(() => {
    setActive(0);
  }, [result]);

  function chooseRecorded(next: string) {
    const item = recorded.find((c) => c.slug === next);
    if (!item) return;
    setError(null);
    setOwn(false);
    setSaved(null);
    setReopened(false);
    setSlug(next);
    setOwnResult(null);
    setLastRequest(null);
  }

  async function runOwn() {
    setBusy(true);
    setError(null);
    setSaved(null);
    setReopened(false);
    const request: AnalyzeRequest = {
      question,
      context,
      answer,
      alpha,
      task_type: taskType,
    };
    try {
      const next = await analyze(request);
      setSlug(null);
      setLastRequest(request);
      setOwnResult(next);

      // Only a real check is kept. Clicking through the three recorded
      // examples is not something anyone wants a history of, and filling the
      // list with them would bury the reader's own work.
      //
      // The save is awaited but never allowed to throw: the check has already
      // happened and is already on screen, and losing that to a storage
      // failure would be the wrong trade.
      const entry = await saveCheck(next);
      setSaved(entry ? "saved" : "failed");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  /**
   * Move the dial.
   *
   * A recorded case has every setting already built, so this only changes
   * which one is read. A live check does not, and re-running it is the only
   * honest way to move it: recomputing the threshold in the browser would be a
   * second implementation of the guarantee, and the two would drift.
   *
   * A reopened check cannot be moved at all. It is a record of something that
   * already happened at a setting somebody chose, and quietly re-deciding it
   * here would turn a record into a live result without saying so.
   */
  async function moveDial(next: number) {
    setAlpha(next);
    if (chosen || reopened || !lastRequest) return;
    setBusy(true);
    setError(null);
    // The saved note referred to the run at the old setting. Leaving it up
    // would have the page claim that what is on screen is in the history when
    // what is in the history is a different result.
    setSaved(null);
    try {
      setOwnResult(await analyze({ ...lastRequest, alpha: next }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const counts = result ? countByVerdict(result) : { flag: 0, abstain: 0, pass: 0 };

  // "The document check read every sentence and would cite none of them" and
  // "no document check was attached" used to arrive as the same three empty
  // fields, so this had to guess from whether ANY span carried evidence -- and
  // a run that legitimately found nothing for every mark then read as "not
  // checked". The contract now carries the answer, so the guess is gone.
  //
  // The fallback survives for one reason: a result stored in history before the
  // field existed has no `attribution_checked`, and for those the old heuristic
  // is still the most honest thing available.
  const attributed =
    result?.attribution_checked ??
    result?.spans.some(
      (s) => s.evidence_sentence !== null || s.entailment_score !== null,
    ) ??
    false;

  // Share of the answer that carries no mark at all. A count of words would be
  // a different unit from the two counts beside it; a share is the same unit.
  const plainShare = result
    ? 1 -
      result.spans.reduce((n, s) => n + (s.end - s.start), 0) /
        Math.max(1, result.answer.length)
    : 0;

  return (
    <>
      {/* ---- 1. what to check ------------------------------------------ */}
      <Band id="band-pick" tone="white">
        <BandHead
          step={1}
          eyebrow="Start here"
          title="Pick something for it to check."
          lead="Each of these is a real answer an AI produced from a real document, together with the result the system gave when it read them. Nothing here is staged."
        />

        <div className="grid gap-5 lg:grid-cols-3">
          {recorded.map((item, i) => {
            const selected = slug === item.slug && !own;
            return (
              <Reveal key={item.slug} delay={i * 0.06}>
                <button
                  type="button"
                  onClick={() => chooseRecorded(item.slug)}
                  className={`h-full w-full rounded-2xl border p-5 text-left transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600 ${
                    selected
                      ? "border-brand-500 bg-brand-50/60 shadow-[0_0_0_1px_var(--color-brand-500)]"
                      : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50"
                  }`}
                >
                  <span className="t-eyebrow text-brand-600">
                    {TASK_LABEL[item.result.task_type]}
                  </span>
                  <span className="t-h3 mt-3 block text-slate-900">{item.title}</span>
                  <span className="t-small mt-3 block text-slate-600">
                    {item.result.spans.length} passages marked ·{" "}
                    {item.result.context_sentences.length} sentences in the document
                  </span>
                </button>
              </Reveal>
            );
          })}
        </div>

        <div className="mt-7">
          <Button variant="ghost" size="md" onClick={() => setOwn((v) => !v)}>
            <PenLine className="h-4 w-4" aria-hidden />
            {own ? "Hide the paste-your-own box" : "Or paste your own document"}
          </Button>
        </div>

        <AnimatePresence initial={false}>
          {own && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.25 }}
              className="overflow-hidden"
            >
              <Card className="mt-5 p-6 sm:p-8">
                <p className="t-body max-w-2xl text-slate-600">
                  This one calls the live system, so it needs the local server to
                  be running. If it is not, the three examples above still work —
                  they were recorded from real runs. It checks at{" "}
                  <span className="font-medium text-slate-800">
                    {careLabel(alpha).name.toLowerCase()}
                  </span>
                  ; you can move that on the dial once there is a result.
                </p>

                <div className="mt-6 space-y-5">
                  <Field
                    label={FIELD.question.label}
                    plain={FIELD.question.plain}
                    value={question}
                    onChange={setQuestion}
                    rows={2}
                  />
                  <Field
                    label={FIELD.context.label}
                    plain={FIELD.context.plain}
                    value={context}
                    onChange={setContext}
                    rows={7}
                  />
                  <ContextAttach onText={setContext} />
                  <Field
                    label={FIELD.answer.label}
                    plain={FIELD.answer.plain}
                    value={answer}
                    onChange={setAnswer}
                    rows={5}
                  />

                  {/* There is no second setting control here. It used to
                      offer three of the six settings from a dropdown while the
                      dial below offered all six, which is one value with two
                      controls -- the reader has no way to know which one won.
                      The dial is the control; this form takes the text. */}
                  <label className="block">
                    <span className="t-body font-medium text-slate-800">
                      {FIELD.task.label}
                    </span>
                    <select
                      value={taskType}
                      onChange={(e) => setTaskType(e.target.value as TaskType)}
                      className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-[0.9375rem] text-slate-800 sm:max-w-sm"
                    >
                      {(Object.keys(TASK_LABEL) as TaskType[]).map((t) => (
                        <option key={t} value={t}>
                          {TASK_LABEL[t]}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                <div className="mt-7 flex flex-wrap items-center gap-4">
                  <Button onClick={runOwn} disabled={busy || !answer.trim()}>
                    {busy ? (
                      <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
                    ) : (
                      <Sparkles className="h-4 w-4" aria-hidden />
                    )}
                    {busy ? "Reading it…" : "Check this answer"}
                  </Button>
                  {error && (
                    <p className="t-small max-w-md text-flag-ink">
                      The live system did not answer: {error}
                    </p>
                  )}
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      </Band>

      {result && (
        <>
          {/* ---- 2. what it was given --------------------------------- */}
          <Band id="band-input" tone="plain">
            <BandHead
              step={2}
              eyebrow="Before the check"
              title="This is everything it was allowed to use."
              lead="The document below is the only source. Anything in the answer that is not supported by it is, by definition, something the AI supplied itself — and that is the thing we are looking for."
            />

            <div className="grid gap-6 lg:grid-cols-5">
              <Reveal className="lg:col-span-2">
                <Card className="flex h-full flex-col p-6">
                  <p className="t-eyebrow flex items-center gap-2 text-slate-500">
                    <MessageSquare className="h-3.5 w-3.5" aria-hidden />
                    {FIELD.question.label}
                  </p>
                  <p className="t-lead mt-3 text-slate-800">
                    {result.question.trim() || "Nobody asked anything — it was told to summarise."}
                  </p>
                  <p className="t-small mt-auto pt-6 text-slate-500">
                    {TASK_LABEL[result.task_type]}
                    {chosen ? ` · ${chosen.source}` : " · your own text"}
                  </p>
                </Card>
              </Reveal>

              <Reveal delay={0.07} className="lg:col-span-3">
                <Card className="h-full p-6">
                  <p className="t-eyebrow flex items-center gap-2 text-slate-500">
                    <FileText className="h-3.5 w-3.5" aria-hidden />
                    {FIELD.context.label} · {result.context_sentences.length} sentences
                  </p>
                  <div className="mt-3 max-h-64 overflow-y-auto pr-3">
                    <p className="t-body whitespace-pre-wrap text-slate-700">
                      {result.context}
                    </p>
                  </div>
                </Card>
              </Reveal>
            </div>
          </Band>

          {/* ---- 3. what it marked ------------------------------------ */}
          <Band id="band-marks" tone="white">
            <BandHead
              step={3}
              eyebrow="The result"
              title={headline(counts)}
              lead="Below is the answer exactly as the AI wrote it, with the system's marks on top. Colour is the only thing that has been added — no words were changed. Click any marked passage to read what it found."
            />

            <Reveal>
              <PlainSummary
                result={result}
                counts={counts}
                plainShare={plainShare}
                nTest={metrics?.n_test ?? null}
              />
            </Reveal>

            {alphas.length > 1 && (
              <Reveal delay={0.05}>
                <CareDial
                  alphas={alphas}
                  alpha={alpha}
                  onChange={moveDial}
                  row={
                    metrics?.coverage.find(
                      (r) => Math.abs(r.alpha - alpha) < 1e-9,
                    ) ?? null
                  }
                  busy={busy}
                  frozen={reopened}
                  live={!chosen && !reopened && lastRequest !== null}
                />
              </Reveal>
            )}

            <Reveal delay={0.1}>
              <Card className="mt-5 overflow-hidden">
                <div className="border-b border-slate-100 px-6 py-4 sm:px-8">
                  <VerdictLegend />
                </div>
                <div className="px-6 py-7 sm:px-8">
                  <MarkedAnswer result={result} active={active} onSelect={setActive} />
                </div>
              </Card>
            </Reveal>

            {(saved || reopened) && (
              <Reveal delay={0.05}>
                <p className="t-small mt-4 flex items-center gap-2 text-slate-500">
                  <Clock className="h-4 w-4" aria-hidden />
                  {reopened
                    ? "Reopened from your history. This is exactly what came back when you ran it."
                    : saved === "saved"
                      ? "Kept in your history. You can open this again later."
                      : "This one could not be saved, so it will not be in your history."}
                </p>
              </Reveal>
            )}

          </Band>

          {/* ---- 4. each mark ----------------------------------------- */}
          {spans.length > 0 && (
            <Band id="band-detail" tone="plain">
              <BandHead
                step={4}
                eyebrow="One at a time"
                title="What it found, and where to check it."
                lead="For every mark: how likely the passage is to be made up, what kind of mistake it is, and the one sentence in the document you should read to settle it yourself. The percentages are not the model's raw opinion — they were corrected against thousands of answers whose truth we already knew, so they read as real rates."
              />

              <div className="space-y-4">
                {spans.map((span, i) => (
                  <Reveal key={`${span.start}-${span.end}`} delay={Math.min(i, 4) * 0.05}>
                    <SpanCard
                      span={span}
                      index={i}
                      total={spans.length}
                      attributed={attributed}
                      sentenceCount={result.context_sentences.length}
                      isActive={active === i}
                      onSelect={() => setActive(i)}
                    />
                  </Reveal>
                ))}
              </div>
            </Band>
          )}

          {/* ---- 5. does the promise cover this? ---------------------- */}
          <Band id="band-promise" tone="white">
            <BandHead
              step={5}
              eyebrow="The honest part"
              title="Does the promise cover this text?"
              lead="The percentages above are only worth something if this text resembles the material the system was tested against. So before it shows you anything, it checks — and when the answer is no, it says so instead of quietly carrying on."
            />
            <Reveal>
              <PromiseNotice result={result} metrics={metrics} />
            </Reveal>
          </Band>

          {/* ---- 6. what next ---------------------------------------- */}
          <Band id="band-next" tone="dark">
            <div className="max-w-3xl">
              <h2 className="t-h2 text-white">So why believe any of this?</h2>
              <p className="t-lead mt-4 text-brand-100/85">
                Fair question. The next page answers it with three pictures: whether
                the percentages mean what they say, whether the system kept the
                promise it made, and how much reading it saves you.
              </p>
              <div className="mt-7">
                <Button variant="onDark" onClick={onProof}>
                  Show me the evidence
                  <ArrowRight className="h-4 w-4" aria-hidden />
                </Button>
              </div>
            </div>
          </Band>
        </>
      )}
    </>
  );
}

/**
 * The four questions a person who did not build this actually asks.
 *
 * In this order, because it is the order they are asked in: what do I do now,
 * how much work did that save me, can I trust the number, and does any of it
 * apply to the text I just pasted in. Everything else on the page answers a
 * question an author would ask.
 *
 * Every quantity in it is already elsewhere on the page in a form only an
 * author can read: the counts are C2's conformal decisions, the trust line is
 * its calibration, the last line is its distribution check. Nothing is
 * computed here that is not computed there.
 *
 * It replaced three stat boxes that printed the same three counts with no
 * sentence attached to any of them. A count is not an answer to "what do I do
 * now", and this page has now been rebuilt twice to remove exactly that.
 */
function PlainSummary({
  result,
  counts,
  plainShare,
  nTest,
}: {
  result: AnalysisResult;
  counts: Record<ConformalDecision, number>;
  /** Share of the answer carrying no mark at all. */
  plainShare: number;
  /** How many held-out answers the correction was checked against. */
  nTest: number | null;
}) {
  const sentences = result.context_sentences.length;
  const marks = result.spans.length;
  const check = result.distribution_check;
  const good = result.guarantee_applies;
  const applies = good
    ? PLAIN.appliesYes
    : check?.checked
      ? PLAIN.appliesNo
      : PLAIN.appliesNone;

  const rows: {
    tone: ConformalDecision;
    n: string;
    title: string;
    body: string;
  }[] = [];
  if (counts.flag > 0) {
    rows.push({
      tone: "flag",
      n: String(counts.flag),
      title: VERDICT.flag.label,
      body: PLAIN.flagBody,
    });
  }
  if (counts.abstain > 0) {
    rows.push({
      tone: "abstain",
      n: String(counts.abstain),
      title: VERDICT.abstain.label,
      body: PLAIN.unsureBody,
    });
  }
  rows.push({
    tone: "pass",
    n: `${Math.round(plainShare * 100)}%`,
    title: PLAIN.cleanTitle,
    body: PLAIN.cleanBody,
  });

  return (
    <Card className="p-6 sm:p-8">
      <p className="t-h3 text-slate-900">{PLAIN.title}</p>

      {marks === 0 && (
        <p className="t-lead mt-3 max-w-2xl text-slate-700">{PLAIN.nothing}</p>
      )}

      <ul className="mt-6 space-y-5">
        {rows.map((row) => (
          <li key={row.title} className="flex gap-4 sm:gap-5">
            <span
              className={`tnum flex h-11 w-14 shrink-0 items-center justify-center rounded-xl text-[1.25rem] font-semibold ${SKIN[row.tone].chip}`}
            >
              {row.n}
            </span>
            <div className="min-w-0">
              <p className="t-body font-semibold text-slate-900">{row.title}</p>
              <p className="t-small mt-1 max-w-xl leading-relaxed text-slate-600">
                {row.body}
              </p>
            </div>
          </li>
        ))}
      </ul>

      <div className="mt-7 space-y-3 border-t border-slate-200/80 pt-6">
        <p className="t-small max-w-2xl text-slate-600">
          {marks === 0
            ? PLAIN.effortNothing(sentences)
            : PLAIN.effort(sentences, marks)}
        </p>
        <p className="t-small max-w-2xl text-slate-600">
          {nTest === null ? PLAIN.trustWithout : PLAIN.trustWith(nTest)}
        </p>
        <p
          className={`t-small flex max-w-2xl items-start gap-2 font-medium ${
            good ? "text-pass-ink" : "text-abstain-ink"
          }`}
        >
          {good ? (
            <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          ) : (
            <TriangleAlert className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
          )}
          {applies}
        </p>
      </div>
    </Card>
  );
}

function headline(counts: Record<ConformalDecision, number>): string {
  if (counts.flag === 0 && counts.abstain === 0) {
    return "It found nothing the document does not support.";
  }
  const parts: string[] = [];
  if (counts.flag) {
    parts.push(`${counts.flag} ${counts.flag === 1 ? "passage" : "passages"} the document does not back up`);
  }
  if (counts.abstain) {
    parts.push(`${counts.abstain} it will not decide on`);
  }
  return `It found ${parts.join(", and ")}.`;
}

/**
 * The one dial.
 *
 * Six stops, not a continuous slider, and that is the point rather than a
 * limitation: these are the settings C2's offline run actually measured, so
 * every position has a real coverage figure to sit beside it. A continuous
 * control would let a reader land somewhere nothing was ever measured and
 * still show them a number, which is the habit this whole component exists to
 * argue against.
 *
 * The figures under it are read from /api/metrics, which reads the same JSON
 * the report is built from. If the API is not answering they are simply
 * absent; the dial still works, because the results it selects between were
 * built offline and are already in the page.
 */
function CareDial({
  alphas,
  alpha,
  onChange,
  row,
  busy,
  frozen,
  live,
}: {
  alphas: number[];
  alpha: number;
  onChange: (next: number) => void;
  /** What was measured at this exact setting, when the API is answering. */
  row: CoverageRow | null;
  busy: boolean;
  /** A reopened check: a record of a run, not a run. */
  frozen: boolean;
  /** Moving it will call the live system again. */
  live: boolean;
}) {
  const stops = [...alphas].sort((a, b) => b - a); // careless on the left
  const index = Math.max(
    0,
    stops.findIndex((a) => Math.abs(a - alpha) < 1e-9),
  );
  const care = careLabel(alpha);

  return (
    <Card className="p-6 sm:p-8">
      <div className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-2">
        <p className="t-h3 text-slate-900">{DIAL.title}</p>
        <p className="t-small font-medium text-slate-700">{care.name}</p>
      </div>
      <p className="t-body mt-2.5 max-w-2xl text-slate-600">{DIAL.lead}</p>

      <div className="mt-6 max-w-2xl">
        <label className="sr-only" htmlFor="care-dial">
          {DIAL.title}
        </label>
        <input
          id="care-dial"
          type="range"
          min={0}
          max={stops.length - 1}
          step={1}
          value={index}
          disabled={frozen || busy}
          onChange={(e) => onChange(stops[Number(e.target.value)])}
          className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-200 accent-brand-600 disabled:cursor-not-allowed disabled:opacity-50"
        />
        <div className="mt-2 flex justify-between">
          {stops.map((stop) => (
            <button
              key={stop}
              type="button"
              disabled={frozen || busy}
              onClick={() => onChange(stop)}
              className={`tnum rounded px-1 py-0.5 text-[0.75rem] transition disabled:cursor-not-allowed ${
                Math.abs(stop - alpha) < 1e-9
                  ? "font-semibold text-slate-900"
                  : "text-slate-400 hover:text-slate-700"
              }`}
            >
              {Math.round((1 - stop) * 100)}
            </button>
          ))}
        </div>
        <p className="t-small mt-1 text-slate-500">{care.line}</p>
      </div>

      {row && (
        <>
          {/* The same four numbers as one sentence, above the four numbers.
              Somebody who reads only the sentence has still been told the
              trade the dial makes, which is the only thing they need from it. */}
          <p className="t-body mt-7 max-w-2xl border-t border-slate-200/80 pt-6 text-slate-700">
            {DIAL.summary(
              Math.round(row.target_coverage * 100),
              Math.round(row.empirical_coverage * 100),
              Math.round(row.abstention_rate * 100),
            )}
          </p>
          <div className="mt-6 grid gap-5 sm:grid-cols-4">
            <DialStat label={DIAL.asked} value={row.target_coverage} muted />
            <DialStat label={DIAL.caught} value={row.empirical_coverage} />
            <DialStat label={DIAL.unsure} value={row.abstention_rate} />
            <DialStat label={DIAL.stands} value={row.flag_rate} />
          </div>
        </>
      )}

      <p className="t-small mt-5 text-slate-500">
        {frozen
          ? DIAL.frozen
          : busy
            ? DIAL.rerunning
            : row
              ? `${row.inside_band ? DIAL.inside : DIAL.outside} ${DIAL.measured}`
              : live
                ? DIAL.live
                : ""}
      </p>
    </Card>
  );
}

/**
 * One of the dial's four numbers, as a count out of a hundred.
 *
 * Not a percentage. "48%" is a unit a reader has to convert before it means
 * anything, and the readers this page is for will not convert it; "48 in 100"
 * is already the thing they wanted to know.
 */
function DialStat({
  label,
  value,
  muted = false,
}: {
  label: string;
  /** A rate in [0, 1]. Printed as a count out of a hundred. */
  value: number;
  muted?: boolean;
}) {
  return (
    <div>
      <p className="t-small text-slate-500">{label}</p>
      <p className="mt-1 flex items-baseline gap-1.5">
        <span
          className={`tnum text-[1.375rem] font-semibold leading-none tracking-tight ${
            muted ? "text-slate-500" : "text-slate-900"
          }`}
        >
          {Math.round(value * 100)}
        </span>
        <span className="t-small text-slate-500">{DIAL.perHundred}</span>
      </p>
    </div>
  );
}

/**
 * One marked passage, and everything the system knows about it.
 *
 * The layout is answering a complaint, and the complaint was right: the first
 * version put the verdict, the type and the score in a narrow left column and
 * gave the evidence panel three fifths of the width with `h-full`, so whenever
 * the document check abstained -- which is half the time by design, measured --
 * the card was two thirds empty grey. Two columns only work when both sides
 * carry comparable weight, and these two do not: the evidence is one sentence
 * or none.
 *
 * The deeper fault was repetition. On a flagged, unsupported passage the old
 * card said the same thing six times: a chip reading "Not in the document", a
 * line reading "The document does not say this", a heading reading "The
 * document never mentions this", a paragraph reading "Nothing in the document
 * is about this at all", a panel titled "Nothing in the document says this" and
 * a body repeating it once more. Six sentences, one fact, and the reader has to
 * check each one in case it is different.
 *
 * So the card is now three bands that each answer a different question, in the
 * order a person asks them:
 *
 *   1. WHAT was marked            -- the passage, large, in its own colour
 *   2. HOW BAD is it              -- the verdict and the number, on one row
 *   3. WHY                        -- the kind of mistake, and what to go read
 *
 * The number moves to the top row beside the verdict because it is the single
 * most important quantity on the card and it was previously last. The two "why"
 * panels sit in an equal two-column grid with `items-start`, so a short one
 * stops at its own last line instead of stretching to match its neighbour.
 *
 * `verdict.lead` now prints only for ABSTAIN. On flag and pass it restates the
 * chip; on abstain it says something the chip cannot ("we could not tell either
 * way, a person should look at this one"), which is the whole point of the
 * class.
 */
function SpanCard({
  span,
  index,
  total,
  attributed,
  sentenceCount,
  isActive,
  onSelect,
}: {
  span: Span;
  index: number;
  total: number;
  /** Did the document check run on this result at all? See `attributed`. */
  attributed: boolean;
  /** How many sentences the document was split into, for the empty state. */
  sentenceCount: number;
  isActive: boolean;
  onSelect: () => void;
}) {
  const decision = span.conformal_decision ?? "abstain";
  const skin = SKIN[decision];
  const verdict = VERDICT[decision];
  const Icon = VERDICT_ICON[decision];
  const score = span.calibrated_score;
  const raw = span.span_score;
  // A point of difference is the smallest movement worth a sentence; below it
  // the two numbers print the same and the line would be claiming a change the
  // reader cannot see.
  const moved =
    raw !== null && score !== null && Math.abs(raw - score) >= 0.01;
  const checked = attributed;
  const typed = span.error_type && span.error_type !== "unknown";

  return (
    <Card
      className={`overflow-hidden transition ${
        isActive ? "ring-2 ring-brand-400" : "ring-1 ring-transparent"
      }`}
    >
      {/* 1. what was marked */}
      <button
        type="button"
        onClick={onSelect}
        className="block w-full px-6 pt-6 text-left sm:px-8"
      >
        <span className="t-eyebrow text-slate-400">
          Mark {index + 1} of {total}
        </span>
        <span className="t-answer mt-3 block">
          <span className={`mark ${skin.mark}`}>{span.text}</span>
        </span>
      </button>

      {/* 2. the verdict and the number, on one row. The number is the reason
          anyone is reading this card, so it sits where the eye lands rather
          than at the bottom of a column. */}
      <div className="mt-5 border-b border-slate-200/80 px-6 pb-5 sm:px-8">
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-4">
          <span
            className={`inline-flex shrink-0 items-center gap-2 rounded-full px-3 py-1.5 text-[0.8125rem] font-semibold ${skin.chip}`}
          >
            <Icon className="h-4 w-4" aria-hidden />
            {verdict.label}
          </span>

          {/* Capped, not `flex-1`. Stretched across a 1,180px card the label
              ended up a hand's width from its own number and the pairing broke.
              A measure this narrow keeps "Chance it is made up" and "39%" on
              one visual unit at every width the card is used at. */}
          {score !== null && (
            <div className="w-full sm:w-72">
              <div className="flex items-baseline justify-between gap-4">
                <p className="t-small font-medium text-slate-600">
                  Chance it is made up
                </p>
                <p className="tnum text-[1.625rem] font-semibold leading-none tracking-tight text-slate-900">
                  {pct(score)}
                </p>
              </div>
              {/* The bar carries two quantities: the fill is the corrected
                  score, and the hairline is where the detector's own score
                  sat before C2 touched it. The reliability diagram on the
                  proof page argues the correction over 2,390 passages, which
                  is the right way to argue it and the wrong way to show it.
                  Here it is one number moving, on the passage in front of the
                  reader, on the screen where the system is actually judged. */}
              <div className="relative mt-2">
                <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200/70">
                  <motion.div
                    initial={{ width: 0 }}
                    whileInView={{ width: `${Math.round(score * 100)}%` }}
                    viewport={{ once: true, amount: 0.6 }}
                    transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                    className={`h-full rounded-full ${skin.bar}`}
                  />
                </div>
                {raw !== null && (
                  <span
                    aria-hidden
                    style={{ left: `${Math.max(0, Math.min(100, raw * 100))}%` }}
                    className="absolute -top-1 h-4 w-[2px] -translate-x-1/2 rounded-full bg-slate-500"
                  />
                )}
              </div>
              {raw !== null && (
                <p className="t-small mt-2 flex items-center gap-1.5 text-slate-500">
                  <span className="h-3 w-[2px] rounded-full bg-slate-500" aria-hidden />
                  {CORRECTION.label}{" "}
                  <span className="tnum font-medium text-slate-700">{pct(raw)}</span>
                  <Hint text={moved ? CORRECTION.moved : CORRECTION.same} />
                </p>
              )}
            </div>
          )}
        </div>

        {/* Only ABSTAIN gets a line here. On flag and pass the sentence
            restated the chip directly above it; on abstain it says the one
            thing the chip cannot -- that nobody is guessing and this one is
            for a person. The "corrected against thousands of answers"
            explanation used to print on every card, four times a page; it is
            said once now, in this band's own introduction. */}
        <RuledOut set={span.prediction_set} />

        {/* The fallback, and only the fallback. RuledOut says this and says
            which of the two ABSTAINs it was; it can only do that when the
            result carries the set. A check saved before the contract did
            cannot, and for those this sentence is still the most honest thing
            available. */}
        {decision === "abstain" && span.prediction_set === null && (
          <p className="t-small mt-4 text-slate-600">{verdict.lead}</p>
        )}
      </div>

      {/* 3. why -- two panels of equal weight, neither stretched to the other.
          The band is tinted and the panels are white, rather than the other way
          round: three shades of the same grey stacked down a card gave the eye
          nothing to separate, and inverting it makes each panel read as an
          object you can look at one at a time. */}
      <div className="grid items-start gap-5 bg-slate-50/70 px-6 py-6 sm:px-8 md:grid-cols-2">
        {typed ? (
          <MistakeKind
            type={span.error_type as ErrorType}
            explanation={span.explanation}
          />
        ) : (
          <div />
        )}
        <EvidencePanel
          checked={checked}
          decision={decision}
          sentence={span.evidence_sentence}
          index={span.evidence_index}
          sentenceCount={sentenceCount}
        />
      </div>
    </Card>
  );
}

/**
 * What to go and read in the document.
 *
 * Three states, and the empty one is the one that used to look broken. It is
 * not an error and it is not rare: the shipped attributor abstains on 50.4% of
 * spans at its calibrated threshold, because most flagged passages are things
 * the document never mentions and there is genuinely no sentence to cite. So
 * the empty state has to read as a finding rather than as a hole -- which is
 * why it now names how many sentences were actually read. "We read all twelve
 * and none of them backs this up" is a result; a grey rectangle is not.
 */
function EvidencePanel({
  checked,
  decision,
  sentence,
  index,
  sentenceCount,
}: {
  checked: boolean;
  decision: ConformalDecision;
  sentence: string | null;
  index: number | null;
  sentenceCount: number;
}) {
  const body = (title: string, children: ReactNode) => (
    <div className="rounded-xl border border-slate-200 border-l-4 border-l-slate-300 bg-white p-4">
      <p className="t-eyebrow text-slate-400">What to check in the document</p>
      <p className="t-body mt-2.5 font-semibold text-slate-900">{title}</p>
      {children}
    </div>
  );

  if (!checked) {
    return body(
      EVIDENCE.notRunTitle,
      <p className="t-small mt-1.5 leading-relaxed text-slate-600">
        {EVIDENCE.notRunBody}
      </p>,
    );
  }

  if (sentence) {
    return (
      <div className="rounded-xl border border-slate-200 border-l-4 border-l-brand-400 bg-white p-4">
        <p className="t-eyebrow text-slate-400">What to check in the document</p>
        <p className="t-body mt-2.5 font-semibold text-slate-900">
          {decision === "flag"
            ? EVIDENCE.foundTitle
            : EVIDENCE.foundTitleNeutral}
        </p>
        <p className="t-body mt-3 leading-relaxed text-slate-700">
          <Quote
            className="mr-1.5 inline h-4 w-4 -translate-y-0.5 text-slate-400"
            aria-hidden
          />
          {sentence}
        </p>
        <p className="t-small mt-3.5 border-t border-slate-200/80 pt-3 text-slate-500">
          Sentence {(index ?? 0) + 1}
          {sentenceCount ? ` of ${sentenceCount}` : ""}. Read it and judge for
          yourself — that is the point of showing it.
        </p>
      </div>
    );
  }

  return body(
    EVIDENCE.noneTitle,
    <p className="t-small mt-1.5 leading-relaxed text-slate-600">
      {sentenceCount
        ? `We read all ${sentenceCount} sentences in the document and not one of them backs this up.`
        : EVIDENCE.noneBody}
    </p>,
  );
}

/**
 * C3, made legible.
 *
 * The type layer has always returned a category and a written reason on every
 * span, and for six days nothing on this page read either field -- so the demo
 * showed how likely a passage was made up and never what kind of mistake it
 * was, which is the first thing anyone asks.
 *
 * The first version of this fix printed the raw category name. That is worse
 * than it sounds: "Extra detail nobody gave it" is a compound of two separate
 * judgements and a reader has to unpack it in their head while someone is
 * talking to them. So the two axes are shown as two rows, which is also what
 * the model actually is -- it predicts mechanism and severity on separate
 * heads, and it is far better at the first than the second.
 *
 * The colour carries the mechanism and nothing else. Across four stacked cards
 * a reader can see "these two contradict the document, those two are absent
 * from it" without reading a word, and that distinction is the one the model
 * gets right 75% of the time. Severity is deliberately plain grey; it is the
 * weak axis and it should not look as confident as the row above it.
 *
 * `explanation` is printed verbatim underneath, because it is the only part of
 * this block that is about THIS passage rather than about the category, and
 * because it is where the system says it could not decide.
 */
function MistakeKind({
  type,
  explanation,
}: {
  type: ErrorType;
  explanation: string | null;
}) {
  const detail = ERROR_TYPE_DETAIL[type];
  const conflict = detail.family === "conflict";

  return (
    <div
      className={`rounded-xl border border-slate-200 border-l-4 bg-white p-4 ${
        conflict ? "border-l-amber-400" : "border-l-violet-400"
      }`}
    >
      <p className="t-eyebrow text-slate-400">What kind of mistake</p>

      <p className="t-body mt-2.5 font-semibold text-slate-900">
        {detail.mechanism}
      </p>
      <p className="t-small mt-1.5 leading-relaxed text-slate-600">
        {detail.meaning}
      </p>

      <div className="mt-3.5 border-t border-slate-200/80 pt-3">
        <p className="t-small text-slate-500">
          <span className="font-medium text-slate-600">How easy to spot:</span>{" "}
          {detail.severity}
        </p>
      </div>

      {explanation && (
        <p className="t-small mt-3.5 leading-relaxed text-slate-500">
          {explanation}
        </p>
      )}
    </div>
  );
}

/**
 * The decision, shown as the set it actually is.
 *
 * A split-conformal decision is not a label the model picked. It is the set of
 * answers the calibrated threshold could not rule out, and the three verdicts
 * on screen are only the shapes that set can take. Once a reader has seen the
 * set, "not sure" stops reading as a number that happened to land in the
 * middle and starts reading as what it is: a refusal the method arrived at on
 * purpose. That is the most common misreading of this system and two chips fix
 * it better than a paragraph does.
 *
 * It renders nothing when the set is absent. A result saved before the
 * contract carried it cannot say which of the two ABSTAINs it was, and
 * guessing the commoner one on screen would be exactly the class of mistake
 * this component exists to prevent.
 */
function RuledOut({ set }: { set: number[] | null }) {
  if (set === null) return null;

  // 1 is "not in the document": the detector predicts hallucination, so the
  // positive label is the bad news. The verdict is deliberately not a prop --
  // the set is what the verdict was read off, not the other way round, and
  // taking both would let them disagree on screen.
  const keptNo = set.includes(1);
  const keptYes = set.includes(0);
  const kept = set.length;

  const chip = (live: boolean, tone: "flag" | "pass", label: string) => (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[0.8125rem] font-medium ring-1 ring-inset ${
        live
          ? `${SKIN[tone].chip} ring-transparent`
          : "bg-transparent text-slate-400 line-through ring-slate-200"
      }`}
    >
      {label}
    </span>
  );

  return (
    <div className="mt-4">
      {/* The sentence prints only when it says something the verdict chip
          above has not already said. On a flag or a pass it would be the
          third restatement of one fact on one card; on either kind of
          ABSTAIN it is the only place the difference is stated at all. */}
      {kept !== 1 && (
        <p className="t-small font-medium text-slate-700">
          {kept === 2 ? RULED.both : RULED.none}
        </p>
      )}
      <div
        className={`flex flex-wrap items-center gap-x-3 gap-y-2 ${
          kept === 1 ? "" : "mt-2.5"
        }`}
      >
        <span className="t-small flex items-center gap-1.5 text-slate-500">
          {RULED.title}
          <Hint text={RULED.hint} />
        </span>
        {chip(keptNo, "flag", RULED.no)}
        {chip(keptYes, "pass", RULED.yes)}
      </div>
    </div>
  );
}

/**
 * The distribution check, opened up.
 *
 * It shipped as one green or amber card, which asks a reader to take the
 * verdict on trust -- from the component whose entire purpose is not asking
 * anyone to take a number on trust. Every quantity below was already being
 * measured and sent on every response, and nothing read them.
 *
 * The track is a percentile, not a value: where this text sits among the
 * material the promise was calibrated on. The middle is ordinary and both ends
 * are far from anything the system was tested against, so it is drawn as a
 * range with a marker rather than as a bar filling from the left, which would
 * read as "more is worse".
 */
function FitStrip({ check }: { check: DistributionCheck }) {
  if (check.features.length === 0) return null;

  return (
    <Card className="mt-5 p-6 sm:p-8">
      <p className="t-h3 text-slate-900">{FIT.title}</p>
      <p className="t-body mt-2.5 max-w-2xl text-slate-600">{FIT.lead}</p>

      <ul className="mt-6 space-y-4">
        {check.features.map((f: FeatureCheck) => {
          const at = Math.max(0, Math.min(1, f.percentile));
          const where = at <= 0.05 ? FIT.low : at >= 0.95 ? FIT.high : FIT.middle;
          return (
            <li key={f.name} className="grid gap-2 sm:grid-cols-[14rem_1fr] sm:gap-5">
              <p
                className={`t-small font-medium ${
                  f.unusual ? "text-abstain-ink" : "text-slate-700"
                }`}
              >
                {f.label}
              </p>
              <div>
                <div className="relative h-2 rounded-full bg-slate-200/70">
                  <span
                    aria-hidden
                    style={{ left: `${at * 100}%` }}
                    className={`absolute -top-1 h-4 w-[3px] -translate-x-1/2 rounded-full ${
                      f.unusual ? "bg-abstain" : "bg-slate-600"
                    }`}
                  />
                </div>
                <p className="t-small mt-1.5 text-slate-500">
                  {where}
                  {f.unusual ? ` — ${FIT.unusual.toLowerCase()}` : ""}
                </p>
              </div>
            </li>
          );
        })}
      </ul>

      <p className="t-small mt-6 border-t border-slate-200/80 pt-4 text-slate-500">
        {FIT.onesided}
      </p>
    </Card>
  );
}

/**
 * The promise for one kind of text, rather than the pooled figure.
 *
 * Coverage has always been reported per task in the results and only the
 * pooled number ever reached a screen. Pooled is the weaker claim -- a system
 * can hold 88% across everything while missing badly on one task of three --
 * and a reader looking at a summarisation result is owed the summarisation
 * number. `inside_band` is not a fudge factor: the band comes from the two
 * split sizes and says how far a fair run is allowed to fall short.
 */
function TaskPromise({
  result,
  metrics,
}: {
  result: AnalysisResult;
  metrics: MetricsResponse;
}) {
  const row = metrics.per_task.find((r) => r.group === result.task_type);

  return (
    <Card className="mt-5 p-6 sm:p-8">
      <p className="t-h3 text-slate-900">{TASK_PROMISE.title}</p>
      <p className="t-body mt-2.5 max-w-2xl text-slate-600">{TASK_PROMISE.lead}</p>

      {row ? (
        <>
          <div className="mt-6 flex flex-wrap items-end gap-x-10 gap-y-5">
            <div>
              <p className="t-small text-slate-500">{TASK_PROMISE.asked}</p>
              <p className="tnum mt-1 text-[1.625rem] font-semibold leading-none tracking-tight text-slate-500">
                90%
              </p>
            </div>
            <div>
              <p className="t-small text-slate-500">
                {TASK_PROMISE.caught} · {TASK_LABEL[result.task_type]}
              </p>
              <p className="tnum mt-1 text-[1.625rem] font-semibold leading-none tracking-tight text-slate-900">
                {pct(row.empirical_coverage)}
              </p>
            </div>
          </div>
          <p
            className={`t-small mt-5 ${
              row.inside_band ? "text-slate-600" : "text-abstain-ink"
            }`}
          >
            {row.inside_band ? TASK_PROMISE.inside : TASK_PROMISE.outside} Measured
            on {row.n_test.toLocaleString()} passages of this kind.
          </p>
        </>
      ) : (
        <p className="t-small mt-5 text-slate-600">{TASK_PROMISE.none}</p>
      )}
    </Card>
  );
}

function PromiseNotice({
  result,
  metrics,
}: {
  result: AnalysisResult;
  metrics: MetricsResponse | null;
}) {
  const check = result.distribution_check;
  const state = !result.guarantee_applies
    ? check?.checked
      ? PROMISE.withdrawn
      : PROMISE.absent
    : PROMISE.holds;
  const good = result.guarantee_applies;

  return (
    <>
      <Card
        className={`p-6 ring-1 ring-inset sm:p-8 ${
          good ? "ring-pass-ring" : "ring-abstain-ring"
        }`}
      >
        <div className="flex flex-col gap-5 sm:flex-row">
          <span
            className={`flex h-12 w-12 shrink-0 items-center justify-center rounded-xl ${
              good ? "bg-pass-soft text-pass-ink" : "bg-abstain-soft text-abstain-ink"
            }`}
          >
            {good ? (
              <ShieldCheck className="h-6 w-6" aria-hidden />
            ) : (
              <TriangleAlert className="h-6 w-6" aria-hidden />
            )}
          </span>
          <div className="min-w-0">
            <h3 className="t-h3 text-slate-900">{state.title}</h3>
            <p className="t-lead mt-3 max-w-2xl text-slate-600">{state.body}</p>
            {!good && (
              <p className="t-body mt-4 max-w-2xl text-slate-500">
                Most tools would have shown you the same percentages here
                without mentioning it. Saying so costs us a clean demo and is
                the more useful thing to do.
              </p>
            )}
          </div>
        </div>
      </Card>

      {check?.checked && <FitStrip check={check} />}
      {metrics && <TaskPromise result={result} metrics={metrics} />}
    </>
  );
}

/**
 * Attach a PDF, Word .docx or text file and drop its text into the document
 * box. The file goes to /api/extract, which returns plain text; the model only
 * ever sees text, and nothing is stored. A person can read and edit what came
 * out before running the check -- the whole point of the tool is showing what
 * it is working from.
 */
function ContextAttach({ onText }: { onText: (text: string) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<ExtractResult | null>(null);

  async function onPick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    // Reset so the same file can be picked again after an edit.
    e.target.value = "";
    if (!file) return;
    setBusy(true);
    setError(null);
    setInfo(null);
    try {
      const res = await extractFile(file);
      onText(res.text);
      setInfo(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,.docx,.txt,.md"
        onChange={onPick}
        className="sr-only"
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={busy}
        className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-[0.8125rem] font-medium text-slate-700 transition hover:bg-slate-50 disabled:cursor-not-allowed disabled:opacity-60"
      >
        {busy ? (
          <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
        ) : (
          <Paperclip className="h-3.5 w-3.5" aria-hidden />
        )}
        {busy ? "Reading the file…" : "Attach a PDF, Word or text file"}
      </button>
      {info && (
        <p className="t-small mt-1.5 text-slate-500">
          Loaded{" "}
          <span className="font-medium text-slate-700">{info.filename}</span>
          {typeof info.pages === "number"
            ? ` · ${info.pages} page${info.pages === 1 ? "" : "s"}`
            : ""}{" "}
          · {info.chars.toLocaleString()} characters into the document box.
          {info.note && (
            <span className="mt-1 block text-abstain-ink">{info.note}</span>
          )}
        </p>
      )}
      {error && (
        <p className="t-small mt-1.5 text-flag-ink">
          Could not read the file: {error}
        </p>
      )}
    </div>
  );
}

function Field({
  label,
  plain,
  value,
  onChange,
  rows,
}: {
  label: string;
  plain: string;
  value: string;
  onChange: (next: string) => void;
  rows: number;
}) {
  return (
    <label className="block">
      <span className="t-body font-medium text-slate-800">{label}</span>
      {plain && <span className="t-small mt-1 block text-slate-500">{plain}</span>}
      <textarea
        value={value}
        rows={rows}
        onChange={(e) => onChange(e.target.value)}
        className="mt-2 w-full resize-y rounded-lg border border-slate-300 bg-white px-3 py-2.5 text-[0.9375rem] leading-6 text-slate-800 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-200"
      />
    </label>
  );
}
