import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Check,
  CircleHelp,
  Cloud,
  Copy,
  Download,
  Loader2,
  Monitor,
  Search,
  ShieldAlert,
  ShieldCheck,
  Trash2,
} from "lucide-react";
import { TASK_LABEL } from "../copy";
import {
  BROWSER_LIMIT,
  clearHistory,
  deleteCheck,
  exportHistory,
  listHistory,
  openCheck,
  type HistoryBackend,
} from "../history";
import type { AnalysisResult, HistoryEntry, TaskType } from "../types";
import { Band, BandHead, Button, Card, EmptyState, Reveal } from "../components/ui";

/**
 * Every check this person has run.
 *
 * The page exists because a tool that shows you a result once and then loses it
 * is a demo. Someone checking a contract at four o'clock has to be able to find
 * it again at five.
 *
 * Two things it refuses to be vague about. It always says which store answered
 * -- a cluster on the other side of a network, or this browser -- because
 * "kept on your device" and "kept on a server" are different promises and the
 * reader is entitled to know which one they have. And deleting is real and
 * immediate: a person who pasted a document in here must be able to take it
 * back out without asking anyone.
 */

const PAGE_SIZE = 25;

const BACKEND_LABEL: Record<HistoryBackend, { icon: typeof Cloud; text: string }> = {
  mongodb: { icon: Cloud, text: "Saved to your database" },
  browser: { icon: Monitor, text: "Saved on this device only" },
  none: { icon: Monitor, text: "Nothing is being saved" },
};

const TASK_FILTERS: { value: TaskType | "all"; label: string }[] = [
  { value: "all", label: "Everything" },
  { value: "qa", label: TASK_LABEL.qa },
  { value: "summarization", label: TASK_LABEL.summarization },
  { value: "data2text", label: TASK_LABEL.data2text },
];

function when(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function History({
  onOpen,
  onDemo,
}: {
  onOpen: (result: AnalysisResult) => void;
  onDemo: () => void;
}) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [backend, setBackend] = useState<HistoryBackend>("browser");
  const [note, setNote] = useState("");
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [exportText, setExportText] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const [search, setSearch] = useState("");
  const [task, setTask] = useState<TaskType | "all">("all");
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [page, setPage] = useState(0);

  // Typing in a search box must not put a network round trip on every
  // keystroke, and the server-backed store is on the other side of one.
  const [term, setTerm] = useState("");
  useEffect(() => {
    const id = window.setTimeout(() => {
      setTerm(search.trim());
      setPage(0);
    }, 250);
    return () => window.clearTimeout(id);
  }, [search]);

  const refresh = useCallback(async () => {
    setLoading(true);
    const result = await listHistory({
      q: term || undefined,
      task,
      only_flagged: onlyFlagged,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    });
    setEntries(result.entries);
    setTotal(result.total);
    setBackend(result.backend === "mongodb" ? "mongodb" : "browser");
    setNote(result.note);
    setLoading(false);
  }, [term, task, onlyFlagged, page]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const filtered = term !== "" || task !== "all" || onlyFlagged;

  async function open(id: string) {
    setBusyId(id);
    const detail = await openCheck(id);
    setBusyId(null);
    if (detail) onOpen(detail.result);
  }

  async function remove(id: string) {
    setBusyId(id);
    await deleteCheck(id);
    setBusyId(null);
    await refresh();
  }

  async function clearEverything() {
    setConfirmClear(false);
    setLoading(true);
    await clearHistory();
    setPage(0);
    await refresh();
  }

  /**
   * Export, in two ways, because one of them does not always work.
   *
   * A browser-initiated download is inert inside a sandboxed viewer -- an
   * `<a download>` with a blob href silently does nothing there. Rather than
   * detect the situation, which cannot be done reliably, the panel offers both:
   * the file, which works when the page is served normally, and the text
   * itself, which works everywhere. Copying is the one that always succeeds, so
   * it is the one on the left.
   */
  async function prepareExport() {
    const payload = await exportHistory();
    setExportText(JSON.stringify(payload, null, 2));
    setCopied(false);
  }

  function downloadExport() {
    if (!exportText) return;
    const blob = new Blob([exportText], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `trustrag-history-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  async function copyExport() {
    if (!exportText) return;
    try {
      await navigator.clipboard.writeText(exportText);
      setCopied(true);
    } catch {
      // Clipboard permission refused, or an insecure origin. The text is on
      // screen and selectable, which is the fallback.
      setCopied(false);
    }
  }

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const Badge = BACKEND_LABEL[backend];

  const headline = useMemo(() => {
    if (loading) return "Looking…";
    if (total === 0 && filtered) return "Nothing matches that.";
    if (total === 0) return "Nothing saved yet.";
    return `${total} ${total === 1 ? "check" : "checks"}${filtered ? " match" : ""}.`;
  }, [loading, total, filtered]);

  return (
    <Band tone="white">
      <BandHead
        eyebrow="Your history"
        title="Everything you have checked."
        lead="Every answer you have run through this is here. Open one to see it exactly as it came back — the marks, the percentages and the sentence the document was checked against."
      />

      {/* Where the checks actually live. Never implied, always printed. */}
      <Reveal>
        <div className="mb-6 flex flex-wrap items-center gap-4 rounded-2xl bg-slate-50 px-5 py-4">
          <span className="inline-flex items-center gap-2.5 text-[0.875rem] font-medium text-slate-700">
            <Badge.icon className="h-4 w-4 text-brand-600" aria-hidden />
            {Badge.text}
          </span>
          <span className="t-small max-w-xl text-slate-500">
            {backend === "browser"
              ? `They stay in this browser and never leave your machine. Clearing your browser data removes them, and the most recent ${BROWSER_LIMIT} are kept.`
              : "They are stored in your database, so they survive a cleared browser and follow you between machines."}
            {note ? ` ${note}` : ""}
          </span>
        </div>
      </Reveal>

      {/* Filters, in one row above the list. */}
      <Reveal delay={0.05}>
        <div className="flex flex-wrap items-center gap-3">
          <label className="relative min-w-0 flex-1 sm:max-w-sm">
            <Search
              className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
              aria-hidden
            />
            <span className="sr-only">Search your checks</span>
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search what you asked or what it wrote"
              className="w-full rounded-full border border-slate-300 bg-white py-2.5 pl-10 pr-4 text-[0.9375rem] text-slate-800 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-200"
            />
          </label>

          <select
            value={task}
            onChange={(e) => {
              setTask(e.target.value as TaskType | "all");
              setPage(0);
            }}
            className="rounded-full border border-slate-300 bg-white px-4 py-2.5 text-[0.9375rem] text-slate-800"
          >
            {TASK_FILTERS.map((f) => (
              <option key={f.value} value={f.value}>
                {f.label}
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => {
              setOnlyFlagged((v) => !v);
              setPage(0);
            }}
            aria-pressed={onlyFlagged}
            className={`inline-flex items-center gap-2 rounded-full border px-4 py-2.5 text-[0.9375rem] font-medium transition ${
              onlyFlagged
                ? "border-flag-ring bg-flag-soft text-flag-ink"
                : "border-slate-300 bg-white text-slate-700 hover:bg-slate-50"
            }`}
          >
            <ShieldAlert className="h-4 w-4" aria-hidden />
            Only ones with something flagged
          </button>

          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="ghost"
              size="md"
              onClick={() => void prepareExport()}
              disabled={total === 0}
            >
              <Download className="h-4 w-4" aria-hidden />
              Export
            </Button>
            <Button
              variant="ghost"
              size="md"
              onClick={() => setConfirmClear(true)}
              disabled={total === 0}
            >
              <Trash2 className="h-4 w-4" aria-hidden />
              Clear all
            </Button>
          </div>
        </div>
      </Reveal>

      {exportText !== null && (
        <div className="mt-5 rounded-2xl border border-slate-200 bg-slate-50 p-5">
          <div className="flex flex-wrap items-center gap-3">
            <p className="t-body font-medium text-slate-800">
              Your history, as JSON
            </p>
            <div className="ml-auto flex flex-wrap gap-2">
              <Button size="md" onClick={() => void copyExport()}>
                {copied ? (
                  <Check className="h-4 w-4" aria-hidden />
                ) : (
                  <Copy className="h-4 w-4" aria-hidden />
                )}
                {copied ? "Copied" : "Copy"}
              </Button>
              <Button variant="secondary" size="md" onClick={downloadExport}>
                <Download className="h-4 w-4" aria-hidden />
                Download file
              </Button>
              <Button variant="ghost" size="md" onClick={() => setExportText(null)}>
                Close
              </Button>
            </div>
          </div>
          <p className="t-small mt-2 text-slate-500">
            Copying always works. Downloading a file does not when this page is
            embedded somewhere that blocks it, which is why both are here.
          </p>
          <textarea
            readOnly
            value={exportText}
            rows={8}
            onFocus={(e) => e.currentTarget.select()}
            className="tnum mt-3 w-full resize-y rounded-xl border border-slate-300 bg-white p-3 font-mono text-[0.8125rem] leading-5 text-slate-700"
          />
        </div>
      )}

      {confirmClear && (
        <div className="mt-5 flex flex-wrap items-center gap-4 rounded-2xl bg-flag-soft px-5 py-4 ring-1 ring-inset ring-flag-ring">
          <p className="t-body text-flag-ink">
            This deletes every saved check. It cannot be undone.
          </p>
          <div className="ml-auto flex gap-2">
            <Button variant="secondary" size="md" onClick={() => setConfirmClear(false)}>
              Keep them
            </Button>
            <Button size="md" onClick={() => void clearEverything()}>
              Delete everything
            </Button>
          </div>
        </div>
      )}

      <p className="t-small mt-6 text-slate-500">{headline}</p>

      <div className="mt-4 space-y-3">
        {loading && entries.length === 0 && (
          <div className="flex items-center gap-3 px-1 py-8 text-slate-500">
            <Loader2 className="h-5 w-5 animate-spin" aria-hidden />
            <p className="t-body">Loading your checks…</p>
          </div>
        )}

        {!loading && entries.length === 0 && (
          <EmptyState
            icon={<Search className="h-5 w-5" />}
            title={filtered ? "Nothing matches that" : "You have not checked anything yet"}
            body={
              filtered
                ? "Try a different word, or clear the filters."
                : "Run an answer through the checker and it will show up here."
            }
          />
        )}

        {entries.map((entry, i) => (
          <Reveal key={entry.id} delay={Math.min(i, 6) * 0.03}>
            <Row
              entry={entry}
              busy={busyId === entry.id}
              onOpen={() => void open(entry.id)}
              onDelete={() => void remove(entry.id)}
            />
          </Reveal>
        ))}
      </div>

      {pages > 1 && (
        <div className="mt-8 flex items-center gap-3">
          <Button
            variant="secondary"
            size="md"
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            disabled={page === 0}
          >
            Newer
          </Button>
          <p className="t-small tnum text-slate-500">
            Page {page + 1} of {pages}
          </p>
          <Button
            variant="secondary"
            size="md"
            onClick={() => setPage((p) => Math.min(pages - 1, p + 1))}
            disabled={page >= pages - 1}
          >
            Older
          </Button>
        </div>
      )}

      {!loading && total === 0 && !filtered && (
        <div className="mt-8">
          <Button onClick={onDemo}>Check something</Button>
        </div>
      )}
    </Band>
  );
}

function Row({
  entry,
  busy,
  onOpen,
  onDelete,
}: {
  entry: HistoryEntry;
  busy: boolean;
  onOpen: () => void;
  onDelete: () => void;
}) {
  const [confirm, setConfirm] = useState(false);

  return (
    <Card className="overflow-hidden">
      <div className="flex flex-col gap-4 p-5 sm:flex-row sm:items-center">
        <button
          type="button"
          onClick={onOpen}
          className="min-w-0 flex-1 text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-600"
        >
          <span className="t-small flex flex-wrap items-center gap-x-3 gap-y-1 text-slate-500">
            <span className="tnum">{when(entry.created_at)}</span>
            <span aria-hidden>·</span>
            <span>{TASK_LABEL[entry.task_type]}</span>
            {!entry.guarantee_applies && (
              <>
                <span aria-hidden>·</span>
                <span className="text-abstain-ink">promise did not apply</span>
              </>
            )}
          </span>
          <span className="t-body mt-1.5 block truncate font-medium text-slate-900">
            {entry.question || "No question — it was told to summarise"}
          </span>
          <span className="t-small mt-1 block truncate text-slate-600">
            {entry.answer_preview}
          </span>
        </button>

        <div className="flex shrink-0 items-center gap-2">
          <Count
            n={entry.n_flagged}
            tone="flag"
            icon={ShieldAlert}
            label="not in the document"
          />
          <Count n={entry.n_unsure} tone="abstain" icon={CircleHelp} label="not sure" />
          {entry.n_flagged === 0 && entry.n_unsure === 0 && (
            <Count n={0} tone="pass" icon={ShieldCheck} label="nothing found" />
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <Button variant="secondary" size="md" onClick={onOpen} disabled={busy}>
            {busy ? <Loader2 className="h-4 w-4 animate-spin" aria-hidden /> : null}
            Open
          </Button>
          <button
            type="button"
            onClick={() => (confirm ? onDelete() : setConfirm(true))}
            onBlur={() => setConfirm(false)}
            aria-label={confirm ? "Confirm delete" : "Delete this check"}
            className={`flex h-10 items-center gap-2 rounded-full px-3 text-[0.875rem] font-medium transition ${
              confirm
                ? "bg-flag-soft text-flag-ink ring-1 ring-inset ring-flag-ring"
                : "text-slate-500 hover:bg-slate-100 hover:text-slate-800"
            }`}
          >
            <Trash2 className="h-4 w-4" aria-hidden />
            {confirm ? "Sure?" : ""}
          </button>
        </div>
      </div>
    </Card>
  );
}

function Count({
  n,
  tone,
  icon: Icon,
  label,
}: {
  n: number;
  tone: "flag" | "abstain" | "pass";
  icon: typeof ShieldAlert;
  label: string;
}) {
  if (n === 0 && tone !== "pass") return null;
  const tones = {
    flag: "bg-flag-soft text-flag-ink ring-flag-ring",
    abstain: "bg-abstain-soft text-abstain-ink ring-abstain-ring",
    pass: "bg-pass-soft text-pass-ink ring-pass-ring",
  } as const;
  return (
    <span
      title={label}
      className={`tnum inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-[0.875rem] font-semibold ring-1 ring-inset ${tones[tone]}`}
    >
      <Icon className="h-3.5 w-3.5" aria-hidden />
      {tone === "pass" ? "clean" : n}
    </span>
  );
}
