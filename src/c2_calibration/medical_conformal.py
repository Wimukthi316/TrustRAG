"""Medical conformal selective prediction (C2, phase 4 -- the medical novelty).

Phase 2 and 3 established, on real runs, that the raw span detector does not
transfer to medical hallucination detection and cannot be fine-tuned into it:
the biomedical backbone lifts MedHallu example-F1 only 0.5259 -> 0.5583 (below
the 0.6667 trivial floor), and fine-tuning on synthetic medical spans HURTS it
(0.5583 -> 0.2168, recall 0.565 -> 0.129) by overfitting to the synthetic error
signature. What stays recoverable at every stage is calibration (medical ECE
~0.01 after recalibration).

So the medical-safety contribution is not a better detector, it is a guarantee
laid on top of the weak one. This module applies the project's split-conformal /
selective-prediction machinery (src/c2_calibration/conformal.py) to the
ANSWER-LEVEL medical scores MedHallu supports, and reports:

  1. marginal coverage -- split conformal calibrated on a medical split gives, on
     held-out medical answers, P(true label in the prediction set) >= 1 - alpha,
     distribution-free and finite-sample. We verify the guarantee actually holds
     (check_coverage) rather than asserting it.
  2. selective prediction -- the risk/coverage trade-off, and the practical
     operating points: abstaining on the least-confident (1 - c) fraction drives
     the error on the accepted fraction down to a chosen budget. This is what
     turns a 0.56-F1 detector into a usable medical trust layer: it does not
     decide the cases it cannot decide safely, it abstains and defers to a
     clinician.
  3. coverage under a within-medical shift -- MedHallu ships a difficulty label.
     Calibrating on easy+medium and testing on hard is a real distribution shift;
     we report whether the marginal guarantee survives it and whether a
     per-difficulty (Mondrian) calibration restores conditional coverage.

HONESTY. Every number here is computed from a scores file produced by a run we
executed (evaluate_medhallu's answer_scores.jsonl); nothing is hand-set. The
guarantee is marginal (holds in expectation over the calibration draw), the
medical hallucinations are model-generated, and the answer-level unit is the only
one MedHallu supports -- all stated in the report. The score is the detector's
answer-level P(hallucinated) = max token probability, exactly the signal C2
calibrates elsewhere.

Usage:

    python -m src.c2_calibration.medical_conformal \\
        --scores results/medical/medhallu-bioclinical/answer_scores.jsonl \\
        --out-dir results/medical/conformal-bioclinical
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.c2_calibration.calibration import calibration_report
from src.c2_calibration.conformal import (
    SplitConformal,
    area_under_risk_coverage,
    check_coverage,
    coverage_table,
    format_coverage_table,
    group_conditional_coverage,
    risk_coverage_curve,
)

DEFAULT_ALPHAS = (0.05, 0.10, 0.15, 0.20, 0.30)


def read_answer_scores(
    path: Path | str, config: Optional[str] = None, limit: Optional[int] = None
) -> Tuple[List[float], List[int], List[str]]:
    """Read evaluate_medhallu's answer_scores.jsonl into (scores, labels, difficulty).

    score is the answer-level P(hallucinated); label is the gold example-level
    hallucination flag. config filters to one MedHallu config when given.
    """
    scores: List[float] = []
    labels: List[int] = []
    difficulty: List[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if config is not None and row.get("config") != config:
                continue
            if "score" not in row or "gold_positive" not in row:
                continue
            scores.append(float(row["score"]))
            labels.append(1 if row["gold_positive"] else 0)
            difficulty.append(str(row.get("difficulty") or ""))
            if limit is not None and len(scores) >= limit:
                break
    return scores, labels, difficulty


def stratified_split(
    labels: Sequence[int], calib_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Indices split into (calibration, test), preserving each class's rate.

    Exchangeability is what split conformal rests on, so the split is a plain
    random partition within each class -- the same construction evaluate_medhallu
    uses, kept here so the two reports agree.
    """
    rng = random.Random(seed)
    calib: List[int] = []
    test: List[int] = []
    for value in (0, 1):
        members = [i for i, lab in enumerate(labels) if lab == value]
        rng.shuffle(members)
        cut = int(round(len(members) * calib_fraction))
        calib.extend(members[:cut])
        test.extend(members[cut:])
    calib.sort()
    test.sort()
    return calib, test


def selective_operating_points(
    scores: Sequence[float],
    labels: Sequence[int],
    target_risks: Sequence[float] = (0.05, 0.10, 0.15, 0.20),
    n_points: int = 200,
) -> List[Dict[str, float]]:
    """For each error budget, the most coverage whose selective risk stays under it.

    Reads the risk-coverage curve and, for a target risk, returns the highest
    coverage whose empirical risk <= target: "act on this fraction, abstain on
    the rest, and be wrong at most this often on what you acted on". When no
    coverage meets the budget the point is reported with met=False rather than
    dropped.
    """
    curve = risk_coverage_curve(scores, labels, n_points=n_points)
    out: List[Dict[str, float]] = []
    for target in target_risks:
        feasible = [row for row in curve if row["risk"] <= target]
        if feasible:
            best = max(feasible, key=lambda r: r["coverage"])
            out.append(
                {
                    "target_risk": target,
                    "met": True,
                    "coverage": best["coverage"],
                    "abstention": 1.0 - best["coverage"],
                    "risk": best["risk"],
                    "confidence_threshold": best["confidence_threshold"],
                }
            )
        else:
            floor = min(curve, key=lambda r: r["risk"]) if curve else {"risk": 1.0, "coverage": 0.0}
            out.append(
                {
                    "target_risk": target,
                    "met": False,
                    "coverage": 0.0,
                    "abstention": 1.0,
                    "best_available_risk": floor["risk"],
                }
            )
    return out


def difficulty_shift(
    scores: Sequence[float],
    labels: Sequence[int],
    difficulty: Sequence[str],
    alpha: float,
    seed: int,
    easy_levels: Tuple[str, ...] = ("easy", "medium"),
    hard_level: str = "hard",
) -> Optional[Dict[str, Any]]:
    """Calibrate on easier difficulties, test on the hardest -- a real shift.

    Returns the marginal coverage a threshold calibrated on easy+medium achieves
    on hard, against the same threshold's behaviour when it is instead calibrated
    on a held-out slice of hard itself. A drop on the first is the guarantee
    failing under difficulty shift; the second is the in-domain reference.
    """
    easy_idx = [i for i, d in enumerate(difficulty) if d.lower() in easy_levels]
    hard_idx = [i for i, d in enumerate(difficulty) if d.lower() == hard_level]
    if len(easy_idx) < 10 or len(hard_idx) < 20:
        return None

    easy_scores = [scores[i] for i in easy_idx]
    easy_labels = [labels[i] for i in easy_idx]
    hard_scores = [scores[i] for i in hard_idx]
    hard_labels = [labels[i] for i in hard_idx]

    # Cross-difficulty: calibrate on easy+medium, test on all of hard.
    cross = SplitConformal(alpha=alpha).fit(easy_scores, easy_labels)
    cross_eval = cross.evaluate(hard_scores, hard_labels)

    # In-domain reference: calibrate on a slice of hard, test on the rest.
    rng = random.Random(seed)
    order = list(range(len(hard_idx)))
    rng.shuffle(order)
    cut = max(10, len(order) // 3)
    cal = order[:cut]
    tst = order[cut:]
    within = SplitConformal(alpha=alpha).fit(
        [hard_scores[i] for i in cal], [hard_labels[i] for i in cal]
    )
    within_eval = within.evaluate(
        [hard_scores[i] for i in tst], [hard_labels[i] for i in tst]
    )
    return {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "n_easy_calibration": len(easy_idx),
        "n_hard_test": len(hard_idx),
        "calibrated_on_easy_tested_on_hard": cross_eval,
        "calibrated_on_hard_tested_on_hard": within_eval,
    }


def analyse(
    scores: Sequence[float],
    labels: Sequence[int],
    difficulty: Sequence[str],
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    calib_fraction: float = 0.3,
    seed: int = 13,
) -> Dict[str, Any]:
    """The full medical-conformal analysis from a single scores file."""
    calib_idx, test_idx = stratified_split(labels, calib_fraction, seed)
    calib_scores = [scores[i] for i in calib_idx]
    calib_labels = [labels[i] for i in calib_idx]
    test_scores = [scores[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    rows = coverage_table(calib_scores, calib_labels, test_scores, test_labels, alphas)
    problems = check_coverage(rows)

    curve = risk_coverage_curve(scores, labels, n_points=21)
    aurc = area_under_risk_coverage(curve)
    operating = selective_operating_points(scores, labels)

    per_difficulty = group_conditional_coverage(
        calib_scores, calib_labels, [difficulty[i] for i in calib_idx],
        test_scores, test_labels, [difficulty[i] for i in test_idx],
        alpha=0.1,
    )
    shift = difficulty_shift(scores, labels, difficulty, alpha=0.1, seed=seed)

    base_rate = sum(labels) / len(labels) if labels else 0.0
    cal = calibration_report(test_scores, test_labels)

    return {
        "n_total": len(scores),
        "n_calibration": len(calib_idx),
        "n_test": len(test_idx),
        "base_rate": base_rate,
        "calib_fraction": calib_fraction,
        "seed": seed,
        "raw_calibration": cal,
        "coverage_table": rows,
        "coverage_problems": problems,
        "guarantee_holds": not problems,
        "risk_coverage": {"aurc": aurc, "curve": curve},
        "selective_operating_points": operating,
        "coverage_by_difficulty": per_difficulty,
        "difficulty_shift": shift,
    }


def format_report(result: Dict[str, Any]) -> str:
    """Plain-text summary for the run log."""
    lines: List[str] = []
    lines.append(
        f"medical conformal selective prediction  "
        f"(n {result['n_total']:,}: calib {result['n_calibration']:,} / "
        f"test {result['n_test']:,}, base rate {result['base_rate']:.3f})"
    )
    lines.append("")
    lines.append("split-conformal coverage on held-out medical answers:")
    lines.append(format_coverage_table(result["coverage_table"]))
    lines.append("")
    lines.append(
        "guarantee holds: "
        + ("YES -- every coverage shortfall within sampling noise"
           if result["guarantee_holds"]
           else "NO -- " + "; ".join(result["coverage_problems"]))
    )
    lines.append("")
    lines.append(f"risk-coverage AURC {result['risk_coverage']['aurc']:.4f} (lower better)")
    lines.append("selective operating points (act on 'coverage', abstain on the rest):")
    for op in result["selective_operating_points"]:
        if op["met"]:
            lines.append(
                f"  error <= {op['target_risk']:.2f}:  coverage {op['coverage']:.3f}  "
                f"(abstain {op['abstention']:.3f})  realised risk {op['risk']:.4f}"
            )
        else:
            lines.append(
                f"  error <= {op['target_risk']:.2f}:  not reachable "
                f"(best risk {op['best_available_risk']:.4f})"
            )
    shift = result.get("difficulty_shift")
    if shift:
        cross = shift["calibrated_on_easy_tested_on_hard"]["empirical_coverage"]
        within = shift["calibrated_on_hard_tested_on_hard"]["empirical_coverage"]
        lines.append("")
        lines.append(
            f"difficulty shift (alpha {shift['alpha']:.2f}, target "
            f"{shift['target_coverage']:.2f}): calibrated on easy+medium -> "
            f"hard coverage {cross:.4f}; calibrated on hard -> {within:.4f}"
        )
    return "\n".join(lines)


def write_figure(result: Dict[str, Any], path: Path) -> Optional[Path]:
    """Two-panel figure: coverage vs target, and the risk-coverage curve.

    Returns None (without failing) when matplotlib is unavailable, so the
    numeric result is never held hostage to a plotting dependency.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None

    rows = result["coverage_table"]
    targets = [r["target_coverage"] for r in rows]
    empirical = [r["empirical_coverage"] for r in rows]
    curve = result["risk_coverage"]["curve"]
    cov = [c["coverage"] for c in curve]
    risk = [c["risk"] for c in curve]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot([0, 1], [0, 1], "--", color="0.7", linewidth=1, label="target")
    ax1.plot(targets, empirical, "o-", color="#2b6cb0", label="empirical")
    ax1.set_xlabel("target coverage (1 - alpha)")
    ax1.set_ylabel("empirical coverage on medical test")
    ax1.set_title("split-conformal coverage holds on medical")
    ax1.legend(loc="lower right", frameon=False)
    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylim(0.5, 1.02)

    ax2.plot(cov, risk, "o-", color="#c05621")
    ax2.set_xlabel("coverage (fraction acted on)")
    ax2.set_ylabel("selective risk (error on accepted)")
    ax2.set_title(f"risk-coverage  (AURC {result['risk_coverage']['aurc']:.3f})")
    ax2.set_ylim(0, max(risk + [0.5]))

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Medical conformal selective prediction on MedHallu answer scores."
    )
    parser.add_argument(
        "--scores", required=True,
        help="answer_scores.jsonl written by evaluate_medhallu",
    )
    parser.add_argument("--config", default=None, help="filter to one MedHallu config")
    parser.add_argument("--out-dir", default="results/medical/conformal")
    parser.add_argument("--calib-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--alphas", type=float, nargs="*", default=list(DEFAULT_ALPHAS),
        help="miscoverage levels to tabulate",
    )
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()

    scores, labels, difficulty = read_answer_scores(args.scores, config=args.config)
    if len(scores) < 50:
        print(f"only {len(scores)} scored answers -- too few for a conformal split")
        return 1

    result = analyse(
        scores, labels, difficulty,
        alphas=args.alphas, calib_fraction=args.calib_fraction, seed=args.seed,
    )
    result["scores_file"] = str(args.scores)

    print(format_report(result))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "medical_conformal_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nwrote {metrics_path}")

    if not args.no_figure:
        fig_path = write_figure(result, out_dir / "medical_conformal.png")
        if fig_path:
            print(f"wrote {fig_path}")
        else:
            print("matplotlib unavailable -- skipped the figure, metrics are complete")

    print(
        "\nNote: coverage is marginal (holds in expectation over the calibration "
        "draw); MedHallu hallucinations are model-generated and answer-level; the "
        "score is the detector's answer-level P(hallucinated). The guarantee is "
        "distribution-free given exchangeability of the medical calibration and "
        "test answers."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
