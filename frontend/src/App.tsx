import { useEffect, useState } from "react";
import { analyze, getExample, getHealth } from "./api";
import HighlightedAnswer from "./components/HighlightedAnswer";
import MetricsPanel from "./components/MetricsPanel";
import type { AnalysisResult, HealthResponse, TaskType } from "./types";

// Two tabs, not two pages. Analyse is what the system does; Evidence is why
// anyone should believe the number it just produced. Keeping them one click
// apart is the point -- a calibrated score that cannot be traced back to a
// measurement is exactly the thing this project exists to complain about.
type Tab = "analyse" | "evidence";

// Labels are presentation, so they live here; the requests themselves come from
// the backend, which owns the corpus record and its offsets.
const EXAMPLES: { name: string; label: string; note: string }[] = [
  {
    name: "ragtruth_qa",
    label: "RAGTruth record",
    note: "A real corpus record, the distribution the detector was trained on.",
  },
  {
    name: "handwritten",
    label: "Hand-written",
    note: "Invented, and far shorter than anything in RAGTruth. The detector over-flags on it — a limitation worth seeing.",
  },
];

const TASK_TYPES: { value: TaskType; label: string }[] = [
  { value: "qa", label: "Question answering" },
  { value: "summarization", label: "Summarization" },
  { value: "data2text", label: "Data to text" },
  { value: "other", label: "Other" },
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [context, setContext] = useState("");
  const [answer, setAnswer] = useState("");
  const [alpha, setAlpha] = useState(0.1);
  const [taskType, setTaskType] = useState<TaskType>("qa");

  const [tab, setTab] = useState<Tab>("analyse");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  async function loadExample(name: string) {
    setError(null);
    try {
      const ex = await getExample(name);
      setQuestion(ex.question);
      setContext(ex.context);
      setAnswer(ex.answer);
      setAlpha(ex.alpha);
      setTaskType(ex.task_type);
      setResult(null);
      setLoaded(name);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await analyze({ question, context, answer, alpha, task_type: taskType }));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const ready = Boolean(context.trim() && answer.trim());

  return (
    <div className="min-h-full bg-slate-50">
      <header className="sticky top-0 z-10 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-4 gap-y-2 px-6 py-3.5">
          <div className="flex items-baseline gap-2.5">
            <h1 className="text-base font-semibold tracking-tight text-slate-900">
              TrustRAG
            </h1>
            <p className="text-xs text-slate-500">
              Span-level calibrated hallucination detection for RAG
            </p>
          </div>
          <nav className="ml-auto flex gap-1 rounded-lg bg-slate-100 p-0.5">
            {(
              [
                ["analyse", "Analyse"],
                ["evidence", "Evidence"],
              ] as [Tab, string][]
            ).map(([value, label]) => (
              <button
                key={value}
                onClick={() => setTab(value)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                  tab === value
                    ? "bg-white text-slate-900 shadow-sm"
                    : "text-slate-500 hover:text-slate-800"
                }`}
              >
                {label}
              </button>
            ))}
          </nav>
          <HealthBadge health={health} />
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {tab === "evidence" ? (
          <MetricsPanel />
        ) : (
        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:items-start">
          <section className="space-y-4 rounded-xl border border-slate-200 bg-white p-6">
            <div className="flex flex-wrap items-center gap-2">
              <h2 className="mr-auto text-sm font-semibold text-slate-900">Input</h2>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.name}
                  onClick={() => loadExample(ex.name)}
                  title={ex.note}
                  className={`rounded-md border px-2.5 py-1 text-xs transition ${
                    loaded === ex.name
                      ? "border-slate-900 bg-slate-900 text-white"
                      : "border-slate-300 text-slate-600 hover:bg-slate-50"
                  }`}
                >
                  {ex.label}
                </button>
              ))}
            </div>

            {loaded && (
              <p className="rounded-md bg-slate-50 px-3 py-2 text-xs text-slate-500">
                {EXAMPLES.find((e) => e.name === loaded)?.note}
              </p>
            )}

            <Field
              label="Question"
              hint="Blank for summarization and data-to-text, which have no question."
            >
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="What the user asked the RAG system"
                className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-900 focus:outline-none focus:ring-1 focus:ring-slate-900"
              />
            </Field>

            <Field
              label="Retrieved context"
              hint="Everything the answer is allowed to rely on."
            >
              <textarea
                value={context}
                onChange={(e) => setContext(e.target.value)}
                rows={9}
                placeholder="The passages the retriever returned"
                className="w-full resize-y rounded-md border border-slate-300 px-3 py-2 font-mono text-xs leading-5 focus:border-slate-900 focus:outline-none focus:ring-1 focus:ring-slate-900"
              />
            </Field>

            <Field
              label="Generated answer"
              hint="The only text that is scored. Context and question are read, never judged."
            >
              <textarea
                value={answer}
                onChange={(e) => setAnswer(e.target.value)}
                rows={5}
                placeholder="What the LLM produced — this is what gets checked"
                className="w-full resize-y rounded-md border border-slate-300 px-3 py-2 font-mono text-xs leading-5 focus:border-slate-900 focus:outline-none focus:ring-1 focus:ring-slate-900"
              />
            </Field>

            <Field label="Task type" hint="Selects the prompt template for the baseline detector.">
              <select
                value={taskType}
                onChange={(e) => setTaskType(e.target.value as TaskType)}
                className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm focus:border-slate-900 focus:outline-none focus:ring-1 focus:ring-slate-900"
              >
                {TASK_TYPES.map((t) => (
                  <option key={t.value} value={t.value}>
                    {t.label}
                  </option>
                ))}
              </select>
            </Field>

            <AlphaControl alpha={alpha} onChange={setAlpha} />

            <button
              onClick={run}
              disabled={!ready || busy}
              className="w-full rounded-md bg-slate-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-slate-700 disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {busy ? "Analysing…" : "Analyse"}
            </button>
          </section>

          <section className="rounded-xl border border-slate-200 bg-white p-6">
            <h2 className="mb-4 text-sm font-semibold text-slate-900">Result</h2>
            {error ? (
              <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
                {error}
              </div>
            ) : result ? (
              <HighlightedAnswer result={result} />
            ) : (
              <div className="rounded-lg border border-dashed border-slate-300 px-4 py-10 text-center text-sm text-slate-500">
                Load an example or paste your own, then press Analyse.
              </div>
            )}
          </section>
        </div>
        )}
      </main>
    </div>
  );
}

function HealthBadge({ health }: { health: HealthResponse | null }) {
  if (!health) {
    return (
      <span className="rounded-full bg-rose-100 px-2.5 py-1 text-xs text-rose-900 ring-1 ring-rose-200">
        backend offline
      </span>
    );
  }
  if (!health.detector_loaded) {
    return (
      <span
        className="rounded-full bg-amber-100 px-2.5 py-1 text-xs text-amber-950 ring-1 ring-amber-300"
        title="TRUSTRAG_DETECTOR is unset, so the placeholder is serving. Its scores are not model output."
      >
        stub detector — scores are not real
      </span>
    );
  }
  return (
    <span className="flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-xs text-emerald-900 ring-1 ring-emerald-200">
      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
      <span className="font-mono">{health.model_version}</span>
    </span>
  );
}

function AlphaControl({
  alpha,
  onChange,
}: {
  alpha: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
      <div className="flex items-baseline justify-between">
        {/* No `uppercase` here: CSS would fold the Greek alpha to a capital
            Alpha, which renders as a plain "A" and reads as a typo. */}
        <span className="text-xs font-medium tracking-wide text-slate-600">
          MISCOVERAGE α
        </span>
        <span className="font-mono text-sm text-slate-900">{alpha.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={0.05}
        max={0.4}
        step={0.05}
        value={alpha}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-2 w-full accent-slate-900"
      />
      <p className="mt-2 text-xs text-slate-500">
        Asks the conformal layer for {((1 - alpha) * 100).toFixed(0)}% coverage.
        Lower α is a stronger promise, so more passages come back{" "}
        <span className="font-medium">Uncertain</span> rather than decided. This
        is the honest lever — the detector's own cutoff is not adjustable here on
        purpose.
      </p>
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="flex items-baseline gap-2">
        <span className="text-xs font-medium uppercase tracking-wide text-slate-600">
          {label}
        </span>
        {hint && <span className="text-[11px] text-slate-400">{hint}</span>}
      </span>
      {children}
    </label>
  );
}
