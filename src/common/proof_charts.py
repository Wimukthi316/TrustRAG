"""The three figures the proof page draws, computed here and not in the browser.

The measurements tab already follows one rule: the browser does no arithmetic,
so what a viewer sees and what the report says cannot drift apart. These charts
follow it too. Everything below is derived from artefacts on disk -- C2's own
results file and C1's recorded probabilities -- using C2's own binning code.

**Why three and not twelve.** The offline runs produce a dozen figures and every
one of them is in the paper. A person who has never met a reliability diagram
cannot be shown twelve of them and be expected to leave believing anything. So
the page carries three claims, each with one picture and one sentence:

    1. when it says 70%, about 70 in 100 really are wrong
    2. ask for 90% and you get at least 90%
    3. it narrows nineteen sentences down to about five

Everything else stays in the paper, where a reader has time.

    python -m src.common.proof_charts
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

C2_RESULTS = Path("results/c2/c1/c2_results.json")
TEST_DUMP = Path("results/c1/test/probabilities.jsonl")
C4_CONFORMAL = Path("results/c4/conformal/c4_conformal.json")
C4_COST = Path("results/c4/analysis/review_cost.json")
OUT = Path("frontend/public/proof-charts.json")

# The same bin count the report uses, so the curve on the page and the ECE in
# the paper are the same measurement. Empty bins cost nothing: the binning code
# drops bins with no members, so each curve starts where its data does -- which
# is why the two curves here have different point counts.
N_BINS = 15
ALPHA = 0.10


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def reliability(
    results_path: Path = C2_RESULTS, dump_path: Path = TEST_DUMP
) -> Dict[str, Any]:
    """Stated confidence against what actually happened, before and after.

    The "after" bins are read from C2's results, which is what the report
    quotes. The "before" bins are not stored there, so they are recomputed from
    C1's raw span scores with **C2's own `reliability_bins`** rather than a
    second implementation -- the two curves have to be binned identically or
    comparing them means nothing.
    """
    from src.c2_calibration.calibration import reliability_bins
    from src.c2_calibration.run_c2 import span_units

    results = json.loads(results_path.read_text(encoding="utf-8"))["span"]

    raw_scores, labels, _ = span_units(_read_jsonl(dump_path))
    before = [
        {
            "confidence": round(b.mean_score, 4),
            "actual": round(b.positive_rate, 4),
            "n": b.count,
        }
        for b in reliability_bins(raw_scores, labels, n_bins=N_BINS)
    ]
    after = [
        {
            "confidence": round(b["mean_score"], 4),
            "actual": round(b["positive_rate"], 4),
            "n": b["count"],
        }
        for b in results["reliability_bins_after"]
    ]
    return {
        "before": before,
        "after": after,
        "ece_before": round(results["calibration"]["before"]["ece"], 4),
        "ece_after": round(results["calibration"]["after"]["ece"], 4),
        "n": results["calibration"]["after"]["n"],
        "n_bins": N_BINS,
    }


def coverage(results_path: Path = C2_RESULTS) -> List[Dict[str, Any]]:
    """What was promised against what was delivered, with the honest tolerance."""
    from src.c2_calibration.conformal import coverage_tolerance

    results = json.loads(results_path.read_text(encoding="utf-8"))["span"]
    rows: List[Dict[str, Any]] = []
    for row in results["conformal"]["coverage"]:
        # The stored rows predate the tolerance function, so the band is
        # recomputed here with C2's own `coverage_tolerance` rather than
        # hard-coded. A number below target but inside this band is not a miss,
        # and a chart that omitted it would be asking a reader to judge a
        # shortfall with no idea how big a shortfall means anything.
        band = coverage_tolerance(
            row["alpha"], row["n_calibration"], row["n_test"]
        )
        delivered = row["empirical_coverage"]
        rows.append(
            {
                "promised": round(row["target_coverage"], 4),
                "delivered": round(delivered, 4),
                "band": round(band, 4),
                "held": bool(delivered >= row["target_coverage"] - band),
                "sent_to_person": round(row["abstention_rate"], 4),
            }
        )
    return rows


def evidence(
    conformal_path: Path = C4_CONFORMAL, cost_path: Path = C4_COST
) -> Optional[Dict[str, Any]]:
    """How much of the document a reviewer still has to read."""
    if not conformal_path.exists():
        return None
    conformal = json.loads(conformal_path.read_text(encoding="utf-8"))
    row = next(
        (r for r in conformal["coverage_table"] if abs(r["alpha"] - ALPHA) < 1e-9),
        None,
    )
    if row is None:
        return None
    headline = None
    if cost_path.exists():
        headline = json.loads(cost_path.read_text(encoding="utf-8")).get("headline")
    return {
        "alpha": ALPHA,
        "coverage": round(row["empirical_coverage"], 4),
        "pool": round(row["mean_pool_size"], 2),
        "returned": round(row["mean_set_size"], 2),
        "singleton_rate": round(row["singleton_rate"], 4),
        # Named for exactly what it is. The cost artefact reports a different
        # reduction -- the mean of the per-span ratios rather than the ratio of
        # the means -- and two numbers both called "reduction" on one page is
        # how a reader stops trusting either.
        "share_returned": round(row["mean_set_size"] / row["mean_pool_size"], 4),
        "n": row["n_test"],
        "sentence": (headline or {}).get("sentence"),
    }


def build() -> Dict[str, Any]:
    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": "results/c2/c1/c2_results.json, results/c1/test/probabilities.jsonl, results/c4",
        "reliability": reliability(),
        "coverage": coverage(),
        "evidence": evidence(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    payload = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    r = payload["reliability"]
    print(f"reliability : {len(r['before'])} bins before, {len(r['after'])} after")
    print(f"              ECE {r['ece_before']} -> {r['ece_after']} on {r['n']} passages")
    print(f"coverage    : {len(payload['coverage'])} levels, all held = "
          f"{all(row['held'] for row in payload['coverage'])}")
    if payload["evidence"]:
        e = payload["evidence"]
        print(f"evidence    : {e['returned']} of {e['pool']} sentences at "
              f"{e['coverage']:.4f} coverage")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
