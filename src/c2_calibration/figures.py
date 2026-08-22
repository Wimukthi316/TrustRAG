"""Every C2 figure, regenerated from the JSON on disk.

Same rule as `tables.py`: no figure is drawn from a number that is not in an
artefact, and a missing artefact produces a skipped figure with a printed reason
rather than an empty axis that looks like a measurement of nothing.

Figures land in `paper/figures/` rather than under `results/`, because
`results/**/*.png` is gitignored -- those are the throwaway diagnostics a run
prints on its way past. These are the ones that go in the report and on the
metrics tab, so they are versioned.

Four of them, and each answers a question a reader will actually ask:

    coverage_vs_alpha        does the promise hold? The diagonal is the target
                             and the shaded band is what sampling noise allows,
                             so a point slightly under the diagonal reads as
                             "fine" instead of as "missed".
    abstention_vs_alpha      what does the promise cost? Abstention is the price
                             of coverage, and this is the curve a compliance
                             officer chooses an operating point from.
    reliability              is the number honest? Before and after calibration
                             on the same axes, against the diagonal.
    shift                    does it survive a change of corpus? Three lines --
                             in-domain, shifted and VOID, and repaired -- which
                             is the single figure Block C exists to produce.

Run:  python -m src.c2_calibration.figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.c2_calibration.conformal import coverage_tolerance

# One place for the palette, so the four figures look like a set rather than
# four separate accidents.
INK = "#1b1b1b"
TARGET = "#8a8a8a"
IN_DOMAIN = "#1f5fa9"
SHIFTED = "#b3202c"
REPAIRED = "#2f7d32"
BEFORE = "#c06a00"
BAND = "#cfd8e3"


def load(root: Path, relative: str) -> Optional[Dict[str, Any]]:
    path = root / relative
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    return plt


def _save(figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    # A vector copy as well: IEEE wants figures as PDF, and regenerating them
    # later from a raster would be a worse day than saving both now.
    figure.savefig(out_dir / f"{name}.pdf")
    return path


def coverage_vs_alpha(root: Path, out_dir: Path) -> Optional[Path]:
    """Empirical coverage against target, with the band sampling noise allows."""
    results = load(root, "c2/c1/c2_results.json")
    if not results or "span" not in results:
        print("skipped coverage_vs_alpha: results/c2/c1/c2_results.json is missing")
        return None
    plt = _pyplot()

    rows = results["span"]["conformal"]["coverage"]
    alphas = [row["alpha"] for row in rows]
    target = [row["target_coverage"] for row in rows]
    empirical = [row["empirical_coverage"] for row in rows]
    bands = [
        coverage_tolerance(
            row["alpha"], int(row["n_calibration"]), int(row["n_test"])
        )
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(4.6, 3.6))
    axis.fill_between(
        alphas,
        [t - b for t, b in zip(target, bands)],
        [t + b for t, b in zip(target, bands)],
        color=BAND,
        label="3-sigma sampling band",
        zorder=1,
    )
    axis.plot(alphas, target, "--", color=TARGET, linewidth=1.2, label="target 1 - alpha", zorder=2)
    axis.plot(
        alphas, empirical, "o-", color=IN_DOMAIN, linewidth=1.6,
        markersize=4, label="empirical, RAGTruth test", zorder=3,
    )
    axis.set_xlabel("alpha (miscoverage level)")
    axis.set_ylabel("coverage")
    axis.set_title("Split-conformal coverage, span level")
    axis.legend(frameon=False, loc="upper right")
    return _save(figure, out_dir, "c2_coverage_vs_alpha")


def abstention_vs_alpha(root: Path, out_dir: Path) -> Optional[Path]:
    """What coverage costs: abstention and flag rate, with the empty set split out."""
    results = load(root, "c2/c1/c2_results.json")
    if not results or "span" not in results:
        print("skipped abstention_vs_alpha: results/c2/c1/c2_results.json is missing")
        return None
    plt = _pyplot()

    rows = results["span"]["conformal"]["coverage"]
    alphas = [row["alpha"] for row in rows]

    figure, axis = plt.subplots(figsize=(4.6, 3.6))
    axis.plot(
        alphas, [r["abstention_rate"] for r in rows], "o-", color=IN_DOMAIN,
        linewidth=1.6, markersize=4, label="abstain (sent to a human)",
    )
    axis.plot(
        alphas, [r["flag_rate"] for r in rows], "s-", color=SHIFTED,
        linewidth=1.4, markersize=3.5, label="flag",
    )
    # The empty-set line is what makes the abstention curve's turn explicable:
    # past a certain alpha every abstention is an empty set, not a {0,1} set.
    axis.plot(
        alphas, [r["empty_set_rate"] for r in rows], "^--", color=TARGET,
        linewidth=1.2, markersize=3.5, label="of which empty set",
    )
    axis.set_xlabel("alpha (miscoverage level)")
    axis.set_ylabel("share of predicted spans")
    axis.set_title("What the guarantee costs, span level")
    axis.legend(frameon=False)
    return _save(figure, out_dir, "c2_abstention_vs_alpha")


def reliability(root: Path, out_dir: Path) -> Optional[Path]:
    """Before and after calibration on one pair of axes."""
    results = load(root, "c2/c1/c2_results.json")
    uncertainty = load(root, "c2/c1/c2_uncertainty.json")
    if not results or "span" not in results:
        print("skipped reliability: results/c2/c1/c2_results.json is missing")
        return None
    plt = _pyplot()

    after = results["span"]["reliability_bins_after"]
    figure, axis = plt.subplots(figsize=(4.2, 4.0))
    axis.plot([0, 1], [0, 1], "--", color=TARGET, linewidth=1.2, label="perfect")
    axis.plot(
        [b["mean_score"] for b in after],
        [b["positive_rate"] for b in after],
        "o-", color=IN_DOMAIN, linewidth=1.6, markersize=4,
        label=f"after {results['span']['calibration']['selected']}",
    )

    before_ece = results["span"]["calibration"]["before"]["ece"]
    after_ece = results["span"]["calibration"]["after"]["ece"]
    floor = (uncertainty or {}).get("span", {}).get("floor")
    caption = f"ECE {before_ece:.4f} -> {after_ece:.4f}"
    if floor:
        caption += (
            f"\nconstant-base-rate floor {floor['constant_base_rate']['ece']:.4f} "
            f"at AUROC {floor['constant_base_rate']['auroc']:.2f}"
        )
    axis.set_xlabel("mean predicted P(hallucinated)")
    axis.set_ylabel("observed hallucination rate")
    axis.set_title("Reliability, span level")
    axis.legend(frameon=False, loc="upper left")
    axis.text(
        0.98, 0.02, caption, transform=axis.transAxes,
        ha="right", va="bottom", fontsize=7.5, color=INK,
    )
    return _save(figure, out_dir, "c2_reliability")


def shift(root: Path, out_dir: Path) -> Optional[Path]:
    """The Block C figure: three lines, and the one that matters is the red one.

    In-domain sits on the target inside its band. Shifted falls far below it and
    is labelled VOID, because the guarantee does not apply to those points at
    all -- they are a measurement of the break. Repaired is the best either
    reweighting manages, and it does not reach the target either, which is the
    finding.
    """
    results = load(root, "c2/c1/c2_results.json")
    repairs = load(root, "c2/ragbench/repair.json")
    if not results or not repairs:
        print("skipped shift: results/c2/c1/c2_results.json or c2/ragbench/repair.json is missing")
        return None
    plt = _pyplot()

    in_rows = results["span"]["conformal"]["coverage"]
    alphas = [row["alpha"] for row in in_rows]
    target = [row["target_coverage"] for row in in_rows]
    in_domain = [row["empirical_coverage"] for row in in_rows]
    bands = [
        coverage_tolerance(row["alpha"], int(row["n_calibration"]), int(row["n_test"]))
        for row in in_rows
    ]

    verdict = {entry["alpha"]: entry for entry in repairs["verdict"]}
    shifted = [verdict[a]["unrepaired"]["coverage"] if a in verdict else None for a in alphas]

    # Plot whichever repair did best at each alpha, and name it in the legend,
    # rather than quietly picking one and letting the reader assume it was the
    # only one tried.
    def best_repair(alpha: float) -> Optional[float]:
        entry = verdict.get(alpha)
        if not entry:
            return None
        candidates = [
            entry[name]["coverage"]
            for name in ("label_shift_estimated", "covariate_shift")
            if entry.get(name)
        ]
        return max(candidates) if candidates else None

    repaired = [best_repair(a) for a in alphas]
    oracle = [
        verdict[a]["label_shift_oracle"]["coverage"]
        if a in verdict and verdict[a].get("label_shift_oracle")
        else None
        for a in alphas
    ]

    figure, axis = plt.subplots(figsize=(5.0, 3.8))
    axis.fill_between(
        alphas,
        [t - b for t, b in zip(target, bands)],
        [t + b for t, b in zip(target, bands)],
        color=BAND, zorder=1,
    )
    axis.plot(alphas, target, "--", color=TARGET, linewidth=1.2, label="target 1 - alpha", zorder=2)
    axis.plot(
        alphas, in_domain, "o-", color=IN_DOMAIN, linewidth=1.7, markersize=4,
        label="in-domain (RAGTruth)", zorder=4,
    )
    axis.plot(
        alphas, shifted, "s-", color=SHIFTED, linewidth=1.7, markersize=4,
        label="shifted (RAGBench) — VOID", zorder=4,
    )
    axis.plot(
        alphas, repaired, "^-", color=REPAIRED, linewidth=1.5, markersize=4,
        label="best deployable reweighting", zorder=3,
    )
    axis.plot(
        alphas, oracle, "v:", color=REPAIRED, linewidth=1.2, markersize=3.5,
        alpha=0.75, label="label shift, oracle prior (not deployable)", zorder=3,
    )
    axis.set_xlabel("alpha (miscoverage level)")
    axis.set_ylabel("coverage")
    axis.set_title("The guarantee does not survive the shift")
    axis.legend(frameon=False, fontsize=7.5, loc="lower left")
    axis.text(
        0.98, 0.97,
        "VOID: calibration and test data are not\nexchangeable, so the guarantee\ndoes not apply to the red line",
        transform=axis.transAxes, ha="right", va="top", fontsize=7, color=SHIFTED,
    )
    return _save(figure, out_dir, "c2_shift")


def risk_control(root: Path, out_dir: Path) -> Optional[Path]:
    """The product promise and its price, on one pair of axes."""
    risk = load(root, "c2/c1/c2_risk_control.json")
    if not risk:
        print("skipped risk_control: results/c2/c1/c2_risk_control.json is missing")
        return None
    plt = _pyplot()

    usable = [row for row in risk["rows"] if row["chosen"]["feasible"]]
    if not usable:
        print("skipped risk_control: no alpha was reachable")
        return None
    alphas = [row["chosen"]["alpha"] for row in usable]

    figure, axis = plt.subplots(figsize=(4.8, 3.6))
    axis.plot(alphas, alphas, "--", color=TARGET, linewidth=1.2, label="bound (alpha)")
    axis.plot(
        alphas, [row["test_risk"] for row in usable], "o-",
        color=IN_DOMAIN, linewidth=1.6, markersize=4,
        label="measured missed-token rate",
    )
    axis.plot(
        alphas, [row["test"]["token_flag_rate"] for row in usable], "s-",
        color=BEFORE, linewidth=1.6, markersize=4,
        label="share of tokens flagged (the cost)",
    )
    edge = [row for row in usable if row["chosen"]["on_grid_edge"]]
    for row in edge:
        axis.annotate(
            "grid edge:\nflags everything",
            xy=(row["chosen"]["alpha"], row["test"]["token_flag_rate"]),
            xytext=(row["chosen"]["alpha"] + 0.06, 0.82),
            fontsize=7, color=SHIFTED,
            arrowprops={"arrowstyle": "->", "color": SHIFTED, "linewidth": 0.8},
        )
    axis.set_xlabel("alpha (bound on missed hallucinated tokens)")
    axis.set_ylabel("rate")
    axis.set_title("Conformal risk control, and what it costs")
    axis.legend(frameon=False, fontsize=7.5)
    return _save(figure, out_dir, "c2_risk_control")


FIGURES = (
    coverage_vs_alpha,
    abstention_vs_alpha,
    reliability,
    shift,
    risk_control,
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the C2 report figures.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args(argv)

    root, out_dir = Path(args.results), Path(args.out_dir)
    written: List[Path] = []
    for builder in FIGURES:
        path = builder(root, out_dir)
        if path is not None:
            written.append(path)
            print(f"wrote {path}")
    print(f"\n{len(written)} of {len(FIGURES)} figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
