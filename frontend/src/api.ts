import type {
  AnalysisResult,
  AnalyzeRequest,
  HealthResponse,
  MetricsResponse,
} from "./types";

// Vite proxies /api -> http://127.0.0.1:8000 in dev (see vite.config.ts).
const BASE = "/api";

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
