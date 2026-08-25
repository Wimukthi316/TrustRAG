import type { AnalysisResult } from "./types";

/**
 * Recorded runs, loaded from a static file rather than from the API.
 *
 * Two reasons they exist. A demo whose backend dies has nothing to show, and a
 * person who wants to look at this on a phone should not have to install
 * anything. Both are solved by shipping a handful of real results.
 *
 * They are **recorded, not simulated**. `src/common/recorded_analysis.py`
 * rebuilds each one from artefacts on disk -- C1's recorded token
 * probabilities, C2's own calibrator and conformal class, C2's exchangeability
 * reference, C4's attributor -- so every field is what the live endpoint would
 * have returned for that response. The interface still labels them as recorded,
 * because a number that is real and a number that is live are different claims.
 */

export interface RecordedCase {
  slug: string;
  title: string;
  why: string;
  response_id: string;
  source: string;
  result: AnalysisResult;
}

export interface RecordedRuns {
  note: string;
  alpha: number;
  cases: RecordedCase[];
}

export async function getRecordedRuns(): Promise<RecordedRuns> {
  // A packaged single-file copy carries them inline; a normal install fetches
  // the static file Vite serves out of public/.
  const inlined = document.getElementById("trustrag-recorded");
  if (inlined?.textContent) {
    return JSON.parse(inlined.textContent) as RecordedRuns;
  }
  const res = await fetch("recorded-runs.json");
  if (!res.ok) throw new Error(`recorded runs unavailable (${res.status})`);
  return (await res.json()) as RecordedRuns;
}
