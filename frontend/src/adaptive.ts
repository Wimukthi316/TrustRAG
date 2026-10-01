/**
 * The adaptive-conformal timeline the "Self-healing" page animates.
 *
 * `src/c2_calibration/online_conformal.py` writes it, out of the served C2
 * artifact and the RAGBench probability dump. Nothing on that page is computed
 * in the browser: the coverage curves, the alpha path and the cost figures are
 * all measured by C2's own code and read from here.
 *
 * The reviewer in this run is simulated by the gold RAGBench label -- the
 * standard way adaptive conformal is evaluated. `note` carries that caveat in
 * the words the page prints, so the claim stays "a simulated review loop", never
 * "a measured one".
 */

export interface SeriesPoint {
  /** Position in the stream (which span, 1-based). */
  t: number;
  /** Rolling coverage of the static method up to here. */
  static: number;
  /** Rolling coverage of the adaptive loop up to here. */
  adaptive: number;
  /** The running miscoverage level at this step. */
  alpha: number;
}

export interface Rates {
  coverage: number;
  flag_rate: number;
  abstain_rate: number;
  mean_set_size: number;
}

export interface GammaRow {
  gamma: number;
  coverage: number;
  abstain_rate: number;
  final_alpha_t: number;
}

export interface AdaptiveTimeline {
  note: string;
  detector: string | null;
  n_stream: number;
  alpha_target: number;
  target_coverage: number;
  gamma: number;
  final_alpha_t: number;
  review_fraction: number;
  static: Rates;
  adaptive: Rates;
  series: SeriesPoint[];
  gamma_sweep: GammaRow[];
  review_only: { coverage: number; abstain_rate: number; review_fraction: number };
  natural_order: { coverage: number; abstain_rate: number; final_alpha_t: number };
}

export async function getAdaptive(): Promise<AdaptiveTimeline> {
  // A packaged single-file copy carries it inline; a normal install fetches the
  // static file Vite serves out of public/.
  const inlined = document.getElementById("trustrag-adaptive");
  if (inlined?.textContent) {
    return JSON.parse(inlined.textContent) as AdaptiveTimeline;
  }
  const res = await fetch("adaptive-conformal.json");
  if (!res.ok) throw new Error(`adaptive timeline unavailable (${res.status})`);
  return (await res.json()) as AdaptiveTimeline;
}
