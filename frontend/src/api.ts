import type { AnalysisResult, AnalyzeRequest, HealthResponse } from "./types";

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

export async function getExample(): Promise<AnalyzeRequest> {
  return json<AnalyzeRequest>(await fetch(`${BASE}/example`));
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
