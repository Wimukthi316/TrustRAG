import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  CircleHelp,
  Clock,
  FileText,
  Loader2,
  MessageSquare,
  PenLine,
  Quote,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TriangleAlert,
} from "lucide-react";
import { analyze } from "../api";
import { saveCheck } from "../history";
import {
  EVIDENCE,
  FIELD,
  PROMISE,
  TASK_LABEL,
  VERDICT,
  careLabel,
} from "../copy";
import type { RecordedCase } from "../recorded";
import type { AnalysisResult, ConformalDecision, Span, TaskType } from "../types";
import {
  MarkedAnswer,
  VerdictLegend,
  countByVerdict,
  sortedSpans,
} from "../components/Marked";
import { Band, BandHead, Button, Card, Reveal } from "../components/ui";

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
  opened = null,
  onProof,
}: {
  recorded: RecordedCase[];
  /** A check reopened from the history page, if the reader came that way. */
  opened?: AnalysisResult | null;
  onProof: () => void;
}) {
  const [slug, setSlug] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
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

  // A check reopened from the history page wins over everything: the reader
  // clicked a specific row and is owed that row.
  useEffect(() => {
    if (!opened) return;
    setSlug(null);
    setOwn(false);
    setError(null);
    setSaved(null);
    setReopened(true);
    setResult(opened);
  }, [opened]);

  // Otherwise open on the first recorded case, so the page is never a blank
  // form.
  useEffect(() => {
    if (result || opened || recorded.length === 0) return;
    setSlug(recorded[0].slug);
    setResult(recorded[0].result);
  }, [recorded, result, opened]);

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
    setResult(item.result);
  }

  async function runOwn() {
    setBusy(true);
    setError(null);
    setSaved(null);
    setReopened(false);
    try {
      const next = await analyze({
        question,
        context,
        answer,
        alpha,
        task_type: taskType,
      });
      setSlug(null);
      setResult(next);

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

  const chosen = slug ? (recorded.find((c) => c.slug === slug) ?? null) : null;
  const counts = result ? countByVerdict(result) : { flag: 0, abstain: 0, pass: 0 };

  // "The document check read every sentence and would cite none of them" and
  // "no document check was attached" arrive as the same three empty fields, and
  // only one of them is a claim worth making. If nothing in the whole result
  // carries evidence, assume the layer is absent and say the weaker thing.
  //
  // The cost is a run where it legitimately found nothing for every single
  // mark, which then reads as "not checked". That errs towards claiming less
  // than happened, which is the right direction to err in.
  const attributed =
    result?.spans.some(
      (s) => s.evidence_sentence !== null || s.entailment_score !== null,
    ) ?? false;

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
                  they were recorded from real runs.
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
                  <Field
                    label={FIELD.answer.label}
                    plain={FIELD.answer.plain}
                    value={answer}
                    onChange={setAnswer}
                    rows={5}
                  />

                  <div className="grid gap-5 sm:grid-cols-2">
                    <label className="block">
                      <span className="t-body font-medium text-slate-800">
                        {FIELD.task.label}
                      </span>
                      <select
                        value={taskType}
                        onChange={(e) => setTaskType(e.target.value as TaskType)}
                        className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-[0.9375rem] text-slate-800"
                      >
                        {(Object.keys(TASK_LABEL) as TaskType[]).map((t) => (
                          <option key={t} value={t}>
                            {TASK_LABEL[t]}
                          </option>
                        ))}
                      </select>
                    </label>

                    <label className="block">
                      <span className="t-body font-medium text-slate-800">
                        How careful should it be
                      </span>
                      <select
                        value={alpha}
                        onChange={(e) => setAlpha(Number(e.target.value))}
                        className="mt-2 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-[0.9375rem] text-slate-800"
                      >
                        {[0.1, 0.2, 0.3].map((a) => (
                          <option key={a} value={a}>
                            {careLabel(a).name} — {careLabel(a).line}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
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
              <Card className="overflow-hidden">
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

            <Reveal delay={0.1}>
              <div className="mt-6 grid gap-4 sm:grid-cols-3">
                <Tally
                  tone="flag"
                  n={counts.flag}
                  label={VERDICT.flag.label}
                  note="The document does not back these up."
                />
                <Tally
                  tone="abstain"
                  n={counts.abstain}
                  label={VERDICT.abstain.label}
                  note="Handed to a person rather than guessed at."
                />
                <Tally
                  tone="pass"
                  n={`${Math.round(plainShare * 100)}%`}
                  label="Of the answer left plain"
                  note="Backed by the document. Read it and move on."
                />
              </div>
            </Reveal>
          </Band>

          {/* ---- 4. each mark ----------------------------------------- */}
          {spans.length > 0 && (
            <Band id="band-detail" tone="plain">
              <BandHead
                step={4}
                eyebrow="One at a time"
                title="What it found, and where to check it."
                lead="For every mark the system gives you two things: how likely it thinks the passage is made up, and the one sentence in the document you should read to settle it yourself."
              />

              <div className="space-y-4">
                {spans.map((span, i) => (
                  <Reveal key={`${span.start}-${span.end}`} delay={Math.min(i, 4) * 0.05}>
                    <SpanCard
                      span={span}
                      index={i}
                      total={spans.length}
                      attributed={attributed}
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
              <PromiseNotice result={result} />
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

function Tally({
  tone,
  n,
  label,
  note,
}: {
  tone: ConformalDecision;
  n: number | string;
  label: string;
  note: string;
}) {
  return (
    <div className={`rounded-xl bg-white p-5 ring-1 ring-inset ${SKIN[tone].ring}`}>
      <p className="tnum text-[1.75rem] font-semibold leading-none tracking-tight text-slate-900">
        {n}
      </p>
      <p className="t-body mt-2.5 font-medium text-slate-800">{label}</p>
      <p className="t-small mt-1 text-slate-500">{note}</p>
    </div>
  );
}

function SpanCard({
  span,
  index,
  total,
  attributed,
  isActive,
  onSelect,
}: {
  span: Span;
  index: number;
  total: number;
  /** Did the document check run on this result at all? See `attributed`. */
  attributed: boolean;
  isActive: boolean;
  onSelect: () => void;
}) {
  const decision = span.conformal_decision ?? "abstain";
  const skin = SKIN[decision];
  const verdict = VERDICT[decision];
  const Icon = VERDICT_ICON[decision];
  const score = span.calibrated_score;

  const checked = attributed;

  return (
    <Card
      className={`overflow-hidden transition ${
        isActive ? "ring-2 ring-brand-400" : "ring-1 ring-transparent"
      }`}
    >
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

      <div className="grid gap-6 px-6 py-6 sm:px-8 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <span
            className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[0.8125rem] font-semibold ${skin.chip}`}
          >
            <Icon className="h-4 w-4" aria-hidden />
            {verdict.label}
          </span>
          <p className="t-body mt-4 text-slate-600">{verdict.lead}</p>

          {score !== null && (
            <div className="mt-6">
              <div className="flex items-baseline justify-between gap-4">
                <p className="t-small font-medium text-slate-600">
                  Chance it is made up
                </p>
                <p className="tnum text-[1.5rem] font-semibold text-slate-900">
                  {pct(score)}
                </p>
              </div>
              <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100">
                <motion.div
                  initial={{ width: 0 }}
                  whileInView={{ width: `${Math.round(score * 100)}%` }}
                  viewport={{ once: true, amount: 0.6 }}
                  transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
                  className={`h-full rounded-full ${skin.bar}`}
                />
              </div>
              <p className="t-small mt-2.5 text-slate-500">
                Not the model's raw opinion. It was corrected against thousands of
                answers whose truth we already knew, so this reads as a real rate.
              </p>
            </div>
          )}
        </div>

        <div className="lg:col-span-3">
          <div className="h-full rounded-xl bg-slate-50 p-5">
            {!checked ? (
              <>
                <p className="t-body font-semibold text-slate-800">
                  {EVIDENCE.notRunTitle}
                </p>
                <p className="t-body mt-2 text-slate-600">{EVIDENCE.notRunBody}</p>
              </>
            ) : span.evidence_sentence ? (
              <>
                <p className="t-body font-semibold text-slate-800">
                  {decision === "flag"
                    ? EVIDENCE.foundTitle
                    : EVIDENCE.foundTitleNeutral}
                </p>
                <p className="t-lead mt-3 text-slate-700">
                  <Quote
                    className="mr-2 inline h-4 w-4 -translate-y-0.5 text-slate-400"
                    aria-hidden
                  />
                  {span.evidence_sentence}
                </p>
                <p className="t-small mt-4 text-slate-500">
                  Sentence {(span.evidence_index ?? 0) + 1} of the document. Read it
                  and judge for yourself — that is the point of showing it.
                </p>
              </>
            ) : (
              <>
                <p className="t-body font-semibold text-slate-800">
                  {EVIDENCE.noneTitle}
                </p>
                <p className="t-body mt-2 text-slate-600">{EVIDENCE.noneBody}</p>
              </>
            )}
          </div>
        </div>
      </div>
    </Card>
  );
}

function PromiseNotice({ result }: { result: AnalysisResult }) {
  const check = result.distribution_check;
  const state = !result.guarantee_applies
    ? check?.checked
      ? PROMISE.withdrawn
      : PROMISE.absent
    : PROMISE.holds;
  const good = result.guarantee_applies;

  return (
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
              Most tools would have shown you the same percentages here without
              mentioning it. Saying so costs us a clean demo and is the more
              useful thing to do.
            </p>
          )}
        </div>
      </div>
    </Card>
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
