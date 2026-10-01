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
  /** The default setting's build. The home page and the packager read it. */
  result: AnalysisResult;
  /**
   * The same case at the dial's other settings, keyed the way
   * `alpha_key` in recorded_analysis.py writes them: "0.05", "0.15", "0.2".
   * The default setting is not repeated here.
   *
   * They are built offline, by C2's own SplitConformal, because the browser
   * must not do this arithmetic. A threshold recomputed in JavaScript would be
   * a second implementation of the guarantee, and two implementations drift.
   *
   * Optional: a file generated before the dial existed has none, and the
   * interface then offers one setting instead of six.
   */
  variants?: Record<string, AnalysisResult>;
}

export interface RecordedRuns {
  note: string;
  alpha: number;
  /** The settings the dial may offer, in order, as alpha keys. */
  alphas?: string[];
  cases: RecordedCase[];
}

/**
 * How an alpha is spelled as a key. Mirrors `alpha_key` in
 * recorded_analysis.py, which uses %g -- so "0.1", never "0.10".
 */
export function alphaKey(alpha: number): string {
  return String(Number(alpha));
}

/**
 * The build of this case at one setting.
 *
 * Falls back to the default build rather than throwing, because a missing
 * variant means an older recorded file and the page should still work. The
 * dial only ever offers settings that came out of the same file, so the
 * fallback is not a path a reader can reach by turning it.
 */
export function resultAt(item: RecordedCase, alpha: number): AnalysisResult {
  return item.variants?.[alphaKey(alpha)] ?? item.result;
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
