/**
 * The three measurements the proof page draws, loaded from a static file.
 *
 * `src/common/proof_charts.py` writes it. Nothing here is computed in the
 * browser: the reliability bins come from C2's own binning function, the
 * tolerance band from C2's own `coverage_tolerance`, and the evidence figures
 * from C4's conformal sweep. If a number on that page looks wrong, the fix is
 * in the Python, not here.
 */

export interface ReliabilityBin {
  confidence: number;
  actual: number;
  n: number;
}

export interface ReliabilityChart {
  before: ReliabilityBin[];
  after: ReliabilityBin[];
  ece_before: number;
  ece_after: number;
  n: number;
  n_bins: number;
}

export interface CoverageLevel {
  promised: number;
  delivered: number;
  /** How far below target a single honest run is allowed to land. */
  band: number;
  held: boolean;
  sent_to_person: number;
}

export interface EvidenceChart {
  alpha: number;
  coverage: number;
  pool: number;
  returned: number;
  singleton_rate: number;
  share_returned: number;
  n: number;
  sentence: string;
}

export interface ProofCharts {
  generated: string;
  source: string;
  reliability: ReliabilityChart;
  coverage: CoverageLevel[];
  evidence: EvidenceChart;
}

export async function getProofCharts(): Promise<ProofCharts> {
  // A packaged single-file copy carries them inline; a normal install fetches
  // the static file Vite serves out of public/.
  const inlined = document.getElementById("trustrag-proof");
  if (inlined?.textContent) {
    return JSON.parse(inlined.textContent) as ProofCharts;
  }
  const res = await fetch("proof-charts.json");
  if (!res.ok) throw new Error(`proof charts unavailable (${res.status})`);
  return (await res.json()) as ProofCharts;
}
