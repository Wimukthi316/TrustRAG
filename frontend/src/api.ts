import type {
  AnalysisResult,
  AnalyzeRequest,
  HealthResponse,
  MetricsResponse,
} from "./types";

// Vite proxies /api -> http://127.0.0.1:8000 in dev (see vite.config.ts).
//
// Exported, because history.ts talks to the same server and a second copy of
// this string is a second thing to change when the deployment moves.
export const API_ROOT = "/api";
const BASE = API_ROOT;

/**
 * A copy of the offline artefacts, inlined into the page at package time.
 *
 * The same build serves two situations. Normally it talks to the backend and
 * this is absent. Packaged as a single file — for someone who wants to look at
 * the work without installing anything, or for a demo whose backend has died —
 * the measurements and the figures travel with the page.
 *
 * The API is still tried first wherever it could answer, so a live install
 * never quietly shows a stale copy.
 */
export interface OfflineBundle {
  metrics?: MetricsResponse;
  figures?: Record<string, string>;
  generated?: string;
}

let bundleCache: OfflineBundle | null | undefined;

export function offlineBundle(): OfflineBundle | null {
  if (bundleCache !== undefined) return bundleCache;
  const element = document.getElementById("trustrag-offline");
  if (!element || !element.textContent) {
    bundleCache = null;
    return null;
  }
  try {
    bundleCache = JSON.parse(element.textContent) as OfflineBundle;
  } catch {
    bundleCache = null;
  }
  return bundleCache;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response had no JSON body; keep statusText
    }
    throw new Error(`${res.status} — ${detail}`);
  }
  return res.json() as Promise<T>;
}

export async function getHealth(): Promise<HealthResponse> {
  return json<HealthResponse>(await fetch(`${BASE}/health`));
}

export async function getExample(name = "ragtruth_qa"): Promise<AnalyzeRequest> {
  return json<AnalyzeRequest>(
    await fetch(`${BASE}/example?name=${encodeURIComponent(name)}`),
  );
}

export async function getMetrics(): Promise<MetricsResponse> {
  try {
    return await json<MetricsResponse>(await fetch(`${BASE}/metrics`));
  } catch (error) {
    const bundle = offlineBundle();
    if (bundle?.metrics) return bundle.metrics;
    throw error;
  }
}

// Figures are served by the backend rather than bundled, so a regenerated
// figure appears on a reload instead of needing a rebuild. In a packaged
// single-file copy there is no backend to serve them, so they travel inlined.
export function figureUrl(name: string): string {
  const inlined = offlineBundle()?.figures?.[name];
  return inlined ?? `${BASE}/figures/${encodeURIComponent(name)}.png`;
}

export async function analyze(req: AnalyzeRequest): Promise<AnalysisResult> {
  return json<AnalysisResult>(
    await fetch(`${BASE}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
    }),
  );
}

// --- file upload -----------------------------------------------------------
// A PDF, Word .docx or text file, turned into plain text on the server so it
// can drop into the document box. The model is only ever sent that text, never
// the file, and nothing is stored -- the bytes live for one request.

export interface ExtractResult {
  filename: string;
  chars: number;
  pages?: number | null;
  text: string;
  note?: string | null;
}

export async function extractFile(file: File): Promise<ExtractResult> {
  const form = new FormData();
  form.append("file", file);
  // No Content-Type header -- the browser sets it with the multipart boundary.
  return json<ExtractResult>(
    await fetch(`${BASE}/extract`, { method: "POST", body: form }),
  );
}
