import { useEffect, useState } from "react";
import { analyze, getExample, getHealth } from "./api";
import HighlightedAnswer from "./components/HighlightedAnswer";
import type { AnalysisResult, HealthResponse } from "./types";

export default function App() {
  const [question, setQuestion] = useState("");
  const [context, setContext] = useState("");
  const [answer, setAnswer] = useState("");
  const [alpha, setAlpha] = useState(0.1);

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  async function loadExample() {
    setError(null);
    try {
      const ex = await getExample();
      setQuestion(ex.question);
      setContext(ex.context);
      setAnswer(ex.answer);
      setAlpha(ex.alpha);
      setResult(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(
        await analyze({ question, context, answer, alpha, task_type: "other" }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const ready = question.trim() && context.trim() && answer.trim();

  return (
    <div className="min-h-full bg-slate-100">
      <header className="border-b border-slate-200 bg-white">
        <div className="mx-auto flex max-w-5xl items-center gap-3 px-6 py-4">
          <div>
            <h1 className="text-lg font-semibold text-slate-900">TrustRAG</h1>
            <p className="text-xs text-slate-500">
              Span-level calibrated hallucination detection for RAG
            </p>
          </div>
          <span className="ml-auto text-xs">
            {health ? (
              <span
                className={
                  health.detector_loaded
                    ? "rounded-full bg-emerald-100 px-2.5 py-1 text-emerald-800"
                    : "rounded-full bg-amber-100 px-2.5 py-1 text-amber-900"
                }
              >
                {health.detector_loaded
                  ? `model: ${health.model_version}`
                  : "stub detector — scores are fake"}
              </span>
            ) : (
              <span className="rounded-full bg-red-100 px-2.5 py-1 text-red-800">
                backend offline
              </span>
            )}
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-5xl space-y-6 px-6 py-8">
        <section className="space-y-4 rounded-lg border border-slate-200 bg-white p-6">
          <div className="flex items-center justify-between">
            <h2 className="font-medium text-slate-900">Input</h2>
            <button
              onClick={loadExample}
              className="rounded-md border border-slate-300 px-3 py-1.5 text-sm text-slate-700 hover:bg-slate-50"
            >
              Load example
            </button>
          </div>

          <Field label="Question">
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What the user asked the RAG system"
              className="w-full rounded-md border border-slate-300 px-3 py-2 text-sm focus:border-slate-500 focus:outline-none"
            />
          </Field>

          <Field label="Retrieved context">
            <textarea
              value={context}
              onChange={(e) => setContext(e.target.value)}
              rows={6}
              placeholder="The passages the retriever returned"
              className="w-full resize-y rounded-md border border-slate-300 px-3 py-2 font-mono text-sm focus:border-slate-500 focus:outline-none"
            />
          </Field>

          <Field label="Generated answer">
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              rows={4}
              placeholder="What the LLM produced — this is what gets checked"
              className="w-full resize-y rounded-md border border-slate-300 px-3 py-2 font-mono text-sm focus:border-slate-500 focus:outline-none"
            />
          </Field>

          <div className="flex flex-wrap items-center gap-4 pt-2">
            <label className="flex items-center gap-3 text-sm text-slate-700">
              <span className="whitespace-nowrap">
                α = <span className="font-mono">{alpha.toFixed(2)}</span>
              </span>
              <input
                type="range"
                min={0.05}
                max={0.4}
                step={0.05}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
                className="w-48"
              />
              <span className="whitespace-nowrap text-xs text-slate-500">
                target coverage {((1 - alpha) * 100).toFixed(0)}%
              </span>
            </label>

            <button
              onClick={run}
              disabled={!ready || busy}
              className="ml-auto rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-slate-300"
            >
              {busy ? "Analysing…" : "Analyse"}
            </button>
          </div>
        </section>

        {error && (
          <div className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            {error}
          </div>
        )}

        {result && (
          <section className="rounded-lg border border-slate-200 bg-white p-6">
            <h2 className="mb-4 font-medium text-slate-900">Result</h2>
            <HighlightedAnswer result={result} />
          </section>
        )}
      </main>
    </div>
  );
}

function Field({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="text-xs font-medium uppercase tracking-wide text-slate-600">
        {label}
      </span>
      {children}
    </label>
  );
}
