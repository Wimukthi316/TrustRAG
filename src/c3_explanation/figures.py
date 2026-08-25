"""C3's figures, regenerated from the same JSON the tables come from.

Five figures, PNG and PDF, written to paper/figures/. Each one exists because a
table row is hard to read and a picture of it is not:

    c3_severity_confusion   the evident/subtle confusion matrix, captioned with
                            the label's own agreement rate so the classifier's
                            errors are read against the label's, not against zero
    c3_set_size_vs_alpha    average set size and singleton rate against alpha --
                            the C3 analogue of C2's risk-coverage curve, and the
                            figure that makes the useful-decision rate explicit
    c3_coverage_vs_alpha    empirical coverage against target, with the 3-sigma
                            band drawn rather than described
    c3_type_conditional_ece calibration inside each error type against the pooled
                            number, which is what pooling hides
    c3_escalation_budget    the share of spans a bounded agent would be asked
                            about, per alpha

A missing input skips its figure and says so. Nothing is drawn from a number that
is not on disk.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGSIZE = (6.4, 4.0)
GRID = {"alpha": 0.25, "linewidth": 0.6}


def load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save(fig: plt.Figure, out_dir: Path, name: str) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    for suffix in ("png", "pdf"):
        path = out_dir / f"{name}.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        written.append(str(path))
    plt.close(fig)
    return written


def severity_confusion(
    baseline: Dict[str, Any], audit: Dict[str, Any], evaluation: str = "predicted_test"
) -> Optional[plt.Figure]:
    block = (baseline.get("evaluations") or {}).get(evaluation)
    if not block:
        return None
    confusion = block["severity"]["model"]["confusion"]
    labels = block["severity"]["labels"]
    matrix = [[confusion.get(g, {}).get(p, 0) for p in labels] for g in labels]
    totals = [sum(row) or 1 for row in matrix]
    normalised = [[cell / total for cell in row] for row, total in zip(matrix, totals)]

    word = audit["l1_2_meta_agreement"]["severity_by_annotator_word"]["independent"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    image = ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("predicted severity")
    ax.set_ylabel("annotated severity")
    for i, row in enumerate(matrix):
        for j, cell in enumerate(row):
            ax.text(
                j,
                i,
                f"{cell:,}\n{normalised[i][j]:.2f}",
                ha="center",
                va="center",
                color="white" if normalised[i][j] > 0.5 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, label="share of the true class")
    ax.set_title(
        "Severity confusion, "
        + evaluation.replace("_", " ")
        + f"\nlabel's own agreement on this axis: {word['agreement']:.4f}"
        + f" (n={word['n_comparable']:,}, independent vocabulary)",
        fontsize=10,
    )
    return fig


def set_size_vs_alpha(conformal: Dict[str, Any]) -> plt.Figure:
    rows = conformal["coverage"]
    alphas = [row["alpha"] for row in rows]
    sizes = [row["average_set_size"] for row in rows]
    singletons = [row["singleton_rate"] for row in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(alphas, sizes, marker="o", color="#1f4e79", label="average set size")
    ax.axhline(1.0, color="#888888", linestyle=":", linewidth=1.0)
    ax.set_xlabel("alpha (miscoverage level)")
    ax.set_ylabel("average number of types in the set")
    ax.set_ylim(0, len(conformal["classes"]) + 0.2)
    ax.grid(True, **GRID)

    twin = ax.twinx()
    twin.plot(
        alphas, singletons, marker="s", color="#b45309", label="singleton rate"
    )
    twin.set_ylabel("fraction of spans committed to one type")
    twin.set_ylim(0, 1)

    handles = ax.get_lines()[:1] + twin.get_lines()[:1]
    ax.legend(handles, [h.get_label() for h in handles], loc="center right", fontsize=9)
    ax.set_title(
        "What a coverage guarantee costs in decisiveness\n"
        "a set of one is a decision; a set of two is a stated limit",
        fontsize=10,
    )
    return fig


def coverage_vs_alpha(conformal: Dict[str, Any]) -> plt.Figure:
    rows = conformal["coverage"]
    alphas = [row["alpha"] for row in rows]
    targets = [row["target_coverage"] for row in rows]
    empirical = [row["empirical_coverage"] for row in rows]
    lower = [row["target_coverage"] - row["band"] for row in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(alphas, targets, color="#444444", linestyle="--", label="target (1 - alpha)")
    ax.fill_between(
        alphas,
        lower,
        [1.0] * len(alphas),
        color="#9ecae1",
        alpha=0.25,
        label="inside the 3-sigma band",
    )
    ax.plot(alphas, empirical, marker="o", color="#1f4e79", label="empirical coverage")

    intervals = [row.get("coverage_interval") or {} for row in rows]
    if all(interval.get("ci_low") is not None for interval in intervals):
        low = [interval["ci_low"] for interval in intervals]
        high = [interval["ci_high"] for interval in intervals]
        if all(not math.isnan(v) for v in low + high):
            ax.fill_between(alphas, low, high, color="#1f4e79", alpha=0.15)
    ax.set_xlabel("alpha (miscoverage level)")
    ax.set_ylabel("coverage of the true error type")
    ax.grid(True, **GRID)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title(
        "Conformal coverage over the error type\n"
        "shaded interval is a response-level cluster bootstrap",
        fontsize=10,
    )
    return fig


def type_conditional_ece(addons: Dict[str, Any]) -> Optional[plt.Figure]:
    block = addons.get("addon_1_type_conditional_calibration") or {}
    if not block:
        return None
    names = list(block["by_error_type"])
    values = [block["by_error_type"][n]["ece"] for n in names]
    counts = [block["by_error_type"][n]["n"] for n in names]
    overall = block["overall"]["ece"]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    positions = range(len(names))
    ax.bar(positions, values, color="#1f4e79")
    ax.axhline(
        overall,
        color="#b45309",
        linestyle="--",
        label=f"pooled ECE {overall:.4f}",
    )
    ax.set_xticks(list(positions), [n.replace("_info", "") for n in names], rotation=20, ha="right")
    ax.set_ylabel("expected calibration error")
    for position, (value, n) in enumerate(zip(values, counts)):
        if not math.isnan(value):
            ax.text(position, value, f"n={n:,}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", **GRID)
    ax.legend(fontsize=9)
    ax.set_title(
        "Calibration inside each error type\n"
        "the pooled number is not the number any single class gets",
        fontsize=10,
    )
    return fig


def escalation_budget(addons: Dict[str, Any]) -> Optional[plt.Figure]:
    budget = addons.get("layer_6_escalation_budget") or {}
    if not budget:
        return None
    alphas = [float(a) for a in budget["per_alpha"]]
    shares = [budget["per_alpha"][a]["share_of_all_spans"] for a in budget["per_alpha"]]
    of_abstained = [
        budget["per_alpha"][a]["share_of_abstained"] for a in budget["per_alpha"]
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(alphas, shares, marker="o", color="#1f4e79", label="share of all predicted spans")
    ax.plot(alphas, of_abstained, marker="s", color="#b45309", label="share of abstained spans")
    ax.axhline(
        budget["abstention_rate"],
        color="#888888",
        linestyle=":",
        label=f"C2 abstention rate {budget['abstention_rate']:.4f}",
    )
    ax.set_xlabel("alpha (miscoverage level)")
    ax.set_ylabel("fraction of spans escalated")
    ax.set_ylim(0, 1)
    ax.grid(True, **GRID)
    ax.legend(fontsize=9)
    ax.set_title(
        "The escalation budget\n"
        "escalate only where C2 abstains and the type set has more than one member",
        fontsize=10,
    )
    return fig


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the C3 figures.")
    parser.add_argument("--results-dir", default="results/c3")
    parser.add_argument("--out-dir", default="paper/figures")
    args = parser.parse_args(argv)

    root = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    audit = load(root / "analysis" / "label_audit.json")
    addons = load(root / "analysis" / "addons.json")
    baseline = load(root / "type" / "baseline_metrics.json")
    conformal = load(root / "type" / "c3_conformal.json")

    written: List[str] = []
    skipped: List[str] = []

    if baseline and audit:
        figure = severity_confusion(baseline, audit)
        if figure is not None:
            written += save(figure, out_dir, "c3_severity_confusion")
        else:
            skipped.append("c3_severity_confusion (no predicted_test evaluation)")
    else:
        skipped.append("c3_severity_confusion (needs baseline_metrics.json and label_audit.json)")

    if conformal:
        written += save(set_size_vs_alpha(conformal), out_dir, "c3_set_size_vs_alpha")
        written += save(coverage_vs_alpha(conformal), out_dir, "c3_coverage_vs_alpha")
    else:
        skipped.append("c3_set_size_vs_alpha and c3_coverage_vs_alpha (needs c3_conformal.json)")

    if addons:
        figure = type_conditional_ece(addons)
        if figure is not None:
            written += save(figure, out_dir, "c3_type_conditional_ece")
        else:
            skipped.append("c3_type_conditional_ece (addons.json has no calibration block)")
        figure = escalation_budget(addons)
        if figure is not None:
            written += save(figure, out_dir, "c3_escalation_budget")
        else:
            skipped.append("c3_escalation_budget (addons.json has no escalation block)")
    else:
        skipped.append("c3_type_conditional_ece and c3_escalation_budget (needs addons.json)")

    for path in written:
        print(f"written: {path}")
    for note in skipped:
        print(f"skipped: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
