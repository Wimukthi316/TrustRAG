"""C4's figures. PNG for the web app, PDF for the paper, both from the same artefacts.

A figure whose artefact is missing is skipped with a printed reason rather than
drawn from nothing. The names are added to the backend's `FIGURE_ORDER`
whitelist so `/api/figures/{name}.png` can serve them; that route checks the
name against the whitelist and never path-joins user input, and it stays that
way.

    python -m src.c4_attribution.figures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results/c4")
FIGURE_DIR = Path("paper/figures")

FIGURE_NAMES: Tuple[str, ...] = (
    "c4_theta_sweep",
    "c4_evidence_by_task",
    "c4_prefilter_recall",
    "c4_coverage_vs_alpha",
    "c4_set_size_vs_alpha",
    "c4_review_cost",
    "c4_error_distance",
)


def _load(relative: str) -> Optional[Dict[str, Any]]:
    path = RESULTS / relative
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save(figure: plt.Figure, name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / f"{name}.png", dpi=160)
    figure.savefig(FIGURE_DIR / f"{name}.pdf")
    plt.close(figure)
    print(f"  wrote {FIGURE_DIR / name}.png and .pdf")


def theta_sweep() -> bool:
    report = _load("analysis/attribution_matching_test.json")
    if not report:
        return False
    sweep = report["theta_sweep"]
    thetas = [r["theta"] for r in sweep]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(thetas, [r["convention_a_rate"] for r in sweep], marker="o", label="Convention A")
    axis.plot(thetas, [r["convention_b_rate"] for r in sweep], marker="s", label="Convention B")
    axis.axvline(report["theta"], color="grey", linestyle="--", linewidth=1)
    axis.annotate(
        f"operating point {report['theta']}\nfixed before any model result",
        xy=(report["theta"], 0.05),
        xytext=(0.52, 0.05),
        fontsize=8,
        color="grey",
    )
    axis.set_xlabel("theta -- content-token containment required to call a sentence gold")
    axis.set_ylabel("share of annotated spans with a gold sentence")
    axis.set_ylim(0, 1)
    axis.set_title("How much the matching convention decides")
    axis.legend()
    axis.grid(alpha=0.3)
    _save(figure, "c4_theta_sweep")
    return True


def evidence_by_task() -> bool:
    report = _load("analysis/evidence_audit.json")
    if not report:
        return False
    tasks = list(report["by_task"])
    verbatim = [report["by_task"][t].get("verbatim", 0.0) for t in tasks]
    contained = [report["by_task"][t].get("containment_ge_0.9", 0.0) for t in tasks]
    no_evidence = [report["by_task"][t].get("share_no_evidence", 0.0) for t in tasks]

    positions = range(len(tasks))
    width = 0.27
    figure, axis = plt.subplots(figsize=(6.5, 4))
    axis.bar([p - width for p in positions], verbatim, width, label="quote is verbatim")
    axis.bar(list(positions), contained, width, label="containment >= 0.9")
    axis.bar([p + width for p in positions], no_evidence, width, label="'no evidence exists'")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(tasks)
    axis.set_ylabel("share")
    axis.set_ylim(0, 1)
    axis.set_title("The three tasks are three different problems")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3, axis="y")
    _save(figure, "c4_evidence_by_task")
    return True


def prefilter_recall() -> bool:
    report = _load("scores/scorer_metrics.json")
    if not report:
        return False
    block = report["prefilter"]
    candidates = block.get("candidates") or {"embedding_used": block["rows"]}
    figure, axis = plt.subplots(figsize=(6, 4))
    styles = {"embedding_used": ("o", "-"), "idf_lexical_not_used": ("s", "--")}
    for name, series in candidates.items():
        marker, line = styles.get(name, ("o", "-"))
        axis.plot(
            [r["k"] for r in series],
            [r["recall"] for r in series],
            marker=marker,
            linestyle=line,
            label=name.replace("_", " "),
        )
    used = block["top_k_used"]
    axis.axvline(used, color="grey", linestyle=":", linewidth=1)
    axis.legend(fontsize=8, loc="lower right")
    axis.set_xlabel("k -- sentences kept by the embedding pre-filter")
    axis.set_ylabel("recall of the gold sentence")
    axis.set_ylim(0, 1.02)
    axis.set_title("The pre-filter's ceiling: what it drops, nothing downstream recovers")
    axis.grid(alpha=0.3)
    _save(figure, "c4_prefilter_recall")
    return True


def coverage_vs_alpha() -> bool:
    report = _load("conformal/c4_conformal.json")
    if not report:
        return False
    rows = report["coverage_table"]
    alphas = [r["alpha"] for r in rows]
    targets = [r["target_coverage"] for r in rows]
    empirical = [r["empirical_coverage"] for r in rows]
    lower = [t - r["band"] for t, r in zip(targets, rows)]

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(alphas, targets, linestyle="--", color="grey", label="target 1 - alpha")
    # Shade only the shortfall a single honest calibration draw may show. The
    # guarantee is marginal, so about half of all correct runs land below target;
    # what would be a real failure is landing below this band.
    axis.fill_between(alphas, lower, targets, alpha=0.18, color="grey",
                      label="3-sigma tolerance")
    axis.plot(alphas, empirical, marker="o", label="empirical coverage")
    axis.set_xlabel("alpha")
    axis.set_ylabel("coverage")
    axis.set_title("Conformal evidence sets: coverage against its own noise band")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    _save(figure, "c4_coverage_vs_alpha")
    return True


def set_size_vs_alpha() -> bool:
    report = _load("conformal/c4_conformal.json")
    if not report:
        return False
    rows = report["coverage_table"]
    alphas = [r["alpha"] for r in rows]
    sizes = [r["mean_set_size"] for r in rows]
    pools = [r["mean_pool_size"] for r in rows]
    figure, axis = plt.subplots(figsize=(6.5, 4.4))
    axis.plot(alphas, sizes, marker="o", label="mean set size")
    axis.plot(alphas, pools, linestyle=":", color="grey", label="sentences available")
    axis.set_xlabel("alpha")
    axis.set_ylabel("sentences")
    axis.set_title("Coverage is free without this column")
    # Headroom so the legend never sits on top of a line.
    axis.set_ylim(0, max(pools + sizes) * 1.45)
    twin = axis.twinx()
    twin.plot(alphas, [r["singleton_rate"] for r in rows], marker="s", color="tab:orange",
              label="singleton rate")
    twin.set_ylabel("singleton rate")
    twin.set_ylim(0, 1.45)
    handles, labels = axis.get_legend_handles_labels()
    extra_handles, extra_labels = twin.get_legend_handles_labels()
    axis.legend(
        handles + extra_handles,
        labels + extra_labels,
        fontsize=8,
        loc="upper right",
        framealpha=0.95,
    )
    axis.grid(alpha=0.3)
    _save(figure, "c4_set_size_vs_alpha")
    return True


def review_cost() -> bool:
    report = _load("analysis/review_cost.json")
    if not report:
        return False
    rows = report["curve"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(
        [r["empirical_coverage"] for r in rows],
        [r["mean_sentences_per_span"] for r in rows],
        marker="o",
        label="conformal set",
    )
    axis.axhline(rows[0]["mean_pool_per_span"], color="grey", linestyle="--",
                 label="return everything")
    axis.axhline(1.0, color="tab:green", linestyle=":", label="return one guess")
    axis.set_xlabel("measured coverage")
    axis.set_ylabel("context sentences a reviewer opens, per flagged span")
    axis.set_title("What the guarantee costs a human reader")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.3)
    _save(figure, "c4_review_cost")
    return True


def error_distance() -> bool:
    report = _load("analysis/faithfulness.json")
    if not report:
        return False
    block = next((v for v in report.values() if isinstance(v, dict) and v.get("index_distance_histogram")), None)
    if not block:
        return False
    histogram = block["index_distance_histogram"]
    keys = sorted(histogram, key=lambda k: (k == "6+", k))
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.bar(keys, [histogram[k] for k in keys])
    axis.set_xlabel("|returned sentence index - gold sentence index|")
    axis.set_ylabel("errors")
    axis.set_title("Are the mistakes near misses, or are they random?")
    axis.grid(alpha=0.3, axis="y")
    _save(figure, "c4_error_distance")
    return True


BUILDERS: Sequence[Tuple[str, Callable[[], bool]]] = (
    ("c4_theta_sweep", theta_sweep),
    ("c4_evidence_by_task", evidence_by_task),
    ("c4_prefilter_recall", prefilter_recall),
    ("c4_coverage_vs_alpha", coverage_vs_alpha),
    ("c4_set_size_vs_alpha", set_size_vs_alpha),
    ("c4_review_cost", review_cost),
    ("c4_error_distance", error_distance),
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    made: List[str] = []
    for name, builder in BUILDERS:
        if builder():
            made.append(name)
        else:
            print(f"  skipped {name}: its artefact is not on disk")
    print(f"{len(made)} of {len(BUILDERS)} figures written")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
