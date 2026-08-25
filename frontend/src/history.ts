import { API_ROOT } from "./api";
import type {
  AnalysisResult,
  HistoryDetail,
  HistoryEntry,
  HistoryPage,
  HistoryQuery,
  TaskType,
} from "./types";

/**
 * Where a check goes after it has been run.
 *
 * Two stores, one interface. The API answers when a MongoDB connection is
 * configured behind it; when it is not -- no cluster, no wifi, or the packaged
 * single-file copy, which has no server at all -- this file keeps the checks in
 * the browser instead.
 *
 * The fallback is not a stub. `localStorage` is real persistence: the checks
 * survive a reload, a restart and a closed laptop. What it is not is shared
 * between devices, and the interface says which store answered on every screen
 * rather than letting a reader assume.
 *
 * Every `localStorage` call is wrapped. It throws outright in a private window
 * with site data blocked, and returning an empty history there is correct
 * behaviour; taking the page down with it is not.
 */

const KEY = "trustrag.history.v1";

// A cap, because localStorage is a few megabytes and one check carries a whole
// document. Oldest out first, and the interface says so when it happens.
const BROWSER_MAX = 100;

export type HistoryBackend = "mongodb" | "browser" | "none";

interface StoredCheck {
  entry: HistoryEntry;
  result: AnalysisResult;
}

// --- the browser store ------------------------------------------------------

function readAll(): StoredCheck[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as StoredCheck[]) : [];
  } catch {
    return [];
  }
}

function writeAll(checks: StoredCheck[]): boolean {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(checks));
    return true;
  } catch {
    // Almost always the quota. Halving is cheaper than working out which check
    // is the large one, and losing the oldest half beats losing the write.
    try {
      window.localStorage.setItem(KEY, JSON.stringify(checks.slice(0, Math.floor(BROWSER_MAX / 2))));
      return true;
    } catch {
      return false;
    }
  }
}

function preview(text: string, limit = 240): string {
  const flat = text.split(/\s+/).filter(Boolean).join(" ");
  return flat.length <= limit ? flat : `${flat.slice(0, limit - 1).trimEnd()}…`;
}

/** The list row for a result. Mirrors `entry_from_result` in the Python. */
function entryFor(result: AnalysisResult, id: string, createdAt: string): HistoryEntry {
  return {
    id,
    created_at: createdAt,
    task_type: result.task_type,
    question: result.question.split(/\s+/).filter(Boolean).join(" "),
    answer_preview: preview(result.answer),
    n_spans: result.spans.length,
    n_flagged: result.spans.filter((s) => s.conformal_decision === "flag").length,
    n_unsure: result.spans.filter((s) => s.conformal_decision === "abstain").length,
    guarantee_applies: result.guarantee_applies,
    alpha: result.alpha,
    model_version: result.model_version,
  };
}

function matches(entry: HistoryEntry, query: HistoryQuery): boolean {
  if (query.q) {
    const needle = query.q.toLowerCase();
    const hay = `${entry.question} ${entry.answer_preview}`.toLowerCase();
    if (!hay.includes(needle)) return false;
  }
  if (query.task && query.task !== "all" && entry.task_type !== query.task) return false;
  if (query.only_flagged && entry.n_flagged === 0) return false;
  return true;
}

const browserStore = {
  backend: "browser" as const,

  list(query: HistoryQuery): HistoryPage {
    const all = readAll();
    const filtered = all.map((c) => c.entry).filter((e) => matches(e, query));
    const offset = query.offset ?? 0;
    const limit = query.limit ?? 25;
    return {
      schema_version: "1.0.0",
      available: true,
      backend: "browser",
      note: "",
      total: filtered.length,
      entries: filtered.slice(offset, offset + limit),
    };
  },

  save(result: AnalysisResult): HistoryEntry {
    const id = `b${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`;
    const entry = entryFor(result, id, new Date().toISOString().replace(/\.\d+Z$/, "+00:00"));
    const next = [{ entry, result }, ...readAll()].slice(0, BROWSER_MAX);
    writeAll(next);
    return entry;
  },

  get(id: string): HistoryDetail | null {
    const found = readAll().find((c) => c.entry.id === id);
    return found ? { entry: found.entry, result: found.result } : null;
  },

  remove(id: string): boolean {
    const all = readAll();
    const next = all.filter((c) => c.entry.id !== id);
    if (next.length === all.length) return false;
    writeAll(next);
    return true;
  },

  clear(): number {
    const n = readAll().length;
    writeAll([]);
    return n;
  },

  exportAll() {
    const checks = readAll();
    return {
      schema_version: "1.0.0",
      exported_at: new Date().toISOString(),
      backend: "browser",
      count: checks.length,
      checks,
    };
  },
};

// --- the API store ----------------------------------------------------------
//
// Whether the API can answer is decided once, by asking it, and remembered for
// the life of the page. Asking on every keystroke in the search box would put a
// network round trip in front of a filter.

let serverBacked: boolean | null = null;

async function askTheServer(): Promise<HistoryPage | null> {
  try {
    const res = await fetch(`${API_ROOT}/history?limit=1`, {
      signal: AbortSignal.timeout(4000),
    });
    if (!res.ok) return null;
    return (await res.json()) as HistoryPage;
  } catch {
    return null;
  }
}

/** Which store this page is talking to. Decided once, then cached. */
export async function resolveBackend(): Promise<HistoryBackend> {
  if (serverBacked !== null) return serverBacked ? "mongodb" : "browser";
  const page = await askTheServer();
  serverBacked = page !== null && page.available;
  return serverBacked ? "mongodb" : "browser";
}

function params(query: HistoryQuery): string {
  const search = new URLSearchParams();
  if (query.q) search.set("q", query.q);
  if (query.task && query.task !== "all") search.set("task", query.task as TaskType);
  if (query.only_flagged) search.set("only_flagged", "true");
  if (query.limit !== undefined) search.set("limit", String(query.limit));
  if (query.offset) search.set("offset", String(query.offset));
  const qs = search.toString();
  return qs ? `?${qs}` : "";
}

// --- the interface the page uses -------------------------------------------

export async function listHistory(query: HistoryQuery = {}): Promise<HistoryPage> {
  if ((await resolveBackend()) === "mongodb") {
    const res = await fetch(`${API_ROOT}/history${params(query)}`);
    if (res.ok) return (await res.json()) as HistoryPage;
    // A store that answered a moment ago and does not now: fall back rather
    // than show an error, and let the page say where it ended up.
    serverBacked = false;
  }
  return browserStore.list(query);
}

/**
 * Keep a result.
 *
 * Never throws. A failed save must not take down the screen showing the result
 * that was just produced -- the check still happened, and the person can still
 * read it. It returns null so the caller can say the saving failed.
 */
export async function saveCheck(result: AnalysisResult): Promise<HistoryEntry | null> {
  try {
    if ((await resolveBackend()) === "mongodb") {
      const res = await fetch(`${API_ROOT}/history`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(result),
      });
      if (res.ok) return (await res.json()) as HistoryEntry;
      serverBacked = false;
    }
    return browserStore.save(result);
  } catch {
    try {
      return browserStore.save(result);
    } catch {
      return null;
    }
  }
}

export async function openCheck(id: string): Promise<HistoryDetail | null> {
  if ((await resolveBackend()) === "mongodb") {
    const res = await fetch(`${API_ROOT}/history/${encodeURIComponent(id)}`);
    if (res.ok) return (await res.json()) as HistoryDetail;
    if (res.status === 404) return null;
    serverBacked = false;
  }
  return browserStore.get(id);
}

export async function deleteCheck(id: string): Promise<boolean> {
  if ((await resolveBackend()) === "mongodb") {
    const res = await fetch(`${API_ROOT}/history/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (res.ok || res.status === 404) return res.ok;
    serverBacked = false;
  }
  return browserStore.remove(id);
}

export async function clearHistory(): Promise<number> {
  if ((await resolveBackend()) === "mongodb") {
    const res = await fetch(`${API_ROOT}/history`, { method: "DELETE" });
    if (res.ok) return ((await res.json()) as { deleted: number }).deleted;
    serverBacked = false;
  }
  return browserStore.clear();
}

export async function exportHistory(): Promise<unknown> {
  if ((await resolveBackend()) === "mongodb") {
    const res = await fetch(`${API_ROOT}/history/export`);
    if (res.ok) return await res.json();
    serverBacked = false;
  }
  return browserStore.exportAll();
}

export const BROWSER_LIMIT = BROWSER_MAX;
