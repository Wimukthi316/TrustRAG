import type {
  AnalysisResult,
  AnalyzeRequest,
  HealthResponse,
  MetricsResponse,
} from "./types";

// Vite proxies /api -> http://127.0.0.1:8000 in dev (see vite.config.ts).
const BASE = "/api";

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
  return json<MetricsResponse>(await fetch(`${BASE}/metrics`));
}

// Figures are served by the backend rather than bundled, so a regenerated
// figure appears on a reload instead of needing a rebuild.
export function figureUrl(name: string): string {
  return `${BASE}/figures/${encodeURIComponent(name)}.png`;
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
