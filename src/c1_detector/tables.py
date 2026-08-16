"""Regenerate every reported table from the JSON on disk.

The freeze rule for this project is that no number reaches the report by hand.
This module is the only thing that formats a table, it reads only artefacts
written by a run, and where an artefact is missing it prints TODO rather than a
plausible-looking number. A table with a TODO in it is a table telling the truth
about what has been measured.

Run it with:  python -m src.c1_detector.tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TODO = "TODO"
LEVELS = ("token", "span_exact", "span_overlap", "example")


def load(root: Path, relative: str) -> Optional[Dict[str, Any]]:
    """An artefact, or None. Missing is a legitimate state, not an error."""
    path = root / relative
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def f1(block: Optional[Dict[str, Any]], level: str) -> str:
    if not block or level not in block:
        return TODO
    return f"{block[level]['f1']:.4f}"


def trivial_f1(positive_rate: float) -> float:
    """What "flag everything" scores at this positive rate: 2p / (1 + p)."""
    return (2 * positive_rate) / (1 + positive_rate) if positive_rate else 0.0


def table_1(root: Path) -> List[str]:
    """Baselines, all four metric levels, response-level models marked n/a."""
    c1 = load(root, "c1/test/metrics.json")
    cw3 = load(root, "c1/test-cw3/metrics.json")
    lettuce = load(root, "lettucedetect/test/metrics.json")
    hhem = load(root, "hhem/hhem_metrics.json")
    judge = load(root, "llm_judge/llm_judge_metrics.json")

    rows: List[Sequence[str]] = []

    def span_model(name: str, report: Optional[Dict[str, Any]]) -> None:
        block = report.get("overall") if report else None
        rows.append([name] + [f1(block, level) for level in LEVELS])

    span_model("C1 ModernBERT-base (ours)", c1)
    span_model("C1 + class weights", cw3)
    span_model("LettuceDetect-large, re-scored by us", lettuce)

    if hhem:
        chosen = hhem["test"]["adapted"]["f1"]
        at_half = hhem["test"]["at_half"]["f1"]
        rows.append(
            [
                f"HHEM-2.1-Open (t={hhem['chosen_threshold']})",
                "n/a",
                "n/a",
                "n/a",
                f"{chosen:.4f}",
            ]
        )
        rows.append(["HHEM-2.1-Open (t=0.5)", "n/a", "n/a", "n/a", f"{at_half:.4f}"])
    else:
        rows.append(["HHEM-2.1-Open", "n/a", "n/a", "n/a", TODO])

    if judge:
        rows.append(
            [
                f"LLM judge ({judge['model']}, n={judge['n_judged']})",
                "n/a",
                f1(judge, "span_exact"),
                f1(judge, "span_overlap"),
                f1(judge, "example"),
            ]
        )
    else:
        rows.append(["LLM judge", "n/a", TODO, TODO, TODO])

    if c1:
        # The floor every positive-class F1 must be printed beside. It depends
        # only on how many responses really contain a hallucination, so it is
        # derived from the confusion counts rather than stored anywhere.
        counts = c1["overall"]["example"]
        total = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
        rate = (counts["tp"] + counts["fn"]) / total if total else 0.0
        rows.append(
            [
                f"trivial: flag every response (p={rate:.4f})",
                "n/a",
                "n/a",
                "n/a",
                f"{trivial_f1(rate):.4f}",
            ]
        )

    return render(
        "Table 1 - baselines on the RAGTruth test split",
        ["system", "token F1", "span-exact F1", "span-overlap F1", "example F1"],
        rows,
        note=(
            "n/a means the system cannot produce spans at all. That column of n/a "
            "is the gap this component addresses, stated without argument."
        ),
    )


def table_2(root: Path) -> List[str]:
    """Ablations against the same test split, with the paired verdict."""
    paired = load(root, "c1/analysis/cw3_vs_baseline.json")
    bounds = load(root, "c1/analysis/decode_bounds.json")

    rows: List[Sequence[str]] = []
    if paired:
        for level, block in paired.items():
            rows.append(
                [
                    f"+ class weights, {level}",
                    f"{block['delta_points']:+.2f}p",
                    f"[{block['ci_low_points']:+.2f}, {block['ci_high_points']:+.2f}]",
                    block["verdict"],
                ]
            )
    else:
        rows.append(["+ class weights", TODO, TODO, TODO])

    if bounds:
        rows.append(
            [
                "+ fragment gluing, span_exact (bound, not run)",
                f"{bounds['best_gain_points']:+.2f}p",
                "upper bound",
                f"costs {bounds['exact_matches_lost_at_best']} exact matches",
            ]
        )
    else:
        rows.append(["+ fragment gluing (bound)", TODO, TODO, TODO])

    return render(
        "Table 2 - ablations",
        ["variant", "delta", "95% paired interval", "verdict"],
        rows,
        note=(
            "Intervals are paired bootstraps over the same resampled responses. "
            "Comparing one model's point estimate to the other's interval is not a "
            "test of the difference between them."
        ),
    )


def table_3(root: Path) -> List[str]:
    """The localisation decomposition: where the strict metric actually goes."""
    loc = load(root, "c1/analysis/localisation_report.json")
    if not loc:
        return render("Table 3 - localisation decomposition", ["bucket"], [[TODO]])

    buckets = loc["overall"]["buckets"]
    n_gold, n_pred = loc["n_gold_spans"], loc["n_pred_spans"]

    def side(count: int, total: int, possible: bool) -> Sequence[str]:
        # "missed" has no predictions and "spurious" has no gold by definition.
        # Printing 0 there invites a reader to treat it as a measurement.
        if not possible:
            return ["-", "-"]
        return [f"{count:,}", f"{100 * count / total:.1f}%" if total else TODO]

    rows = [
        [name]
        + list(side(block["gold"], n_gold, name != "spurious"))
        + list(side(block["pred"], n_pred, name != "missed"))
        for name, block in buckets.items()
    ]
    ceiling = loc["tokenisation_ceiling"]["span_exact"]["f1"]
    lengths = loc["span_lengths"]["ALL"]
    return render(
        "Table 3 - localisation decomposition of the strict span metric",
        ["bucket", "gold spans", "% gold", "predicted spans", "% pred"],
        rows,
        note=(
            f"Buckets are connected components of the overlap graph, so they are "
            f"mutually exclusive and exhaustive by construction. Tokenisation "
            f"ceiling for this split: span-exact F1 {ceiling:.4f} - read the "
            f"measured score against that, never against 1.0. Median gold span "
            f"{lengths['gold_chars']['p50']:.0f} characters against "
            f"{lengths['pred_chars']['p50']:.0f} predicted."
        ),
    )


def table_4(root: Path) -> List[str]:
    """Cross-domain, including the fairly chosen operating point."""
    ood = load(root, "ood/ragbench-scores/operating_point.json")
    if not ood:
        return render("Table 4 - cross-domain", ["row"], [[TODO]])

    rows: List[Sequence[str]] = []
    whole = ood["whole_corpus_argmax"]
    rows.append(
        [
            "whole corpus, argmax",
            f"{whole['f1']:.4f}",
            f"{whole['trivial_f1']:.4f}",
            f"{whole['margin_over_trivial_points']:+.2f}p",
            "-",
        ]
    )
    for name in ("argmax", "adapted"):
        row = ood["test"][name]
        rows.append(
            [
                f"held-out half, {name}",
                f"{row['f1']:.4f}",
                f"{row['trivial_f1']:.4f}",
                f"{row['margin_over_trivial_points']:+.2f}p",
                f"{100 * row['flagged_rate']:.1f}%",
            ]
        )
    return render(
        "Table 4 - RAGBench, out of domain",
        ["rule", "F1", "trivial F1", "margin", "responses flagged"],
        rows,
        note=(
            f"Threshold {ood['chosen_threshold']} was chosen on a stratified "
            "calibration half and applied once to the held-out half. A rule that "
            "flags nearly every response has become the trivial classifier: report "
            "reaching the floor, never clearing it."
        ),
    )


def table_5(root: Path) -> List[str]:
    """Reproducibility: repeat runs, and honesty about how many there are."""
    variance = load(root, "c1/analysis/variance.json")
    if not variance:
        return render(
            "Table 5 - reproducibility",
            ["level", "mean", "sd", "range"],
            [[TODO, TODO, TODO, TODO]],
            note=(
                "No variance.json on disk. Run repeat trainings, then "
                "python -m src.c1_detector.variance <run dirs>."
            ),
        )

    rows = []
    for level, block in variance["levels"].items():
        sd = TODO if block["sd"] is None else f"{block['sd']:.4f}"
        rows.append(
            [level, f"{block['mean']:.4f}", sd, f"{block['range_points']:.2f}p"]
        )
    note = f"n = {variance['n_runs']} runs, seeds {variance['distinct_seeds']}."
    if not variance["sd_reported"]:
        note += " Too few runs for a standard deviation; the range is a spread."
    if variance["mixes_seeds"]:
        note += " Mixed seeds, so seed variation and GPU non-determinism are pooled."
    return render(
        "Table 5 - reproducibility across repeat runs",
        ["level", "mean F1", "sd", "range"],
        rows,
        note=note,
    )


def render(
    title: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    note: str = "",
) -> List[str]:
    lines = [f"## {title}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    if note:
        lines += ["", note]
    lines.append("")
    return lines


def build(root: Path) -> str:
    lines: List[str] = [
        "# C1 reported tables",
        "",
        "Generated by `python -m src.c1_detector.tables`. Every value is read from "
        "an artefact under `results/`; `TODO` means the artefact does not exist yet "
        "and no number has been invented in its place.",
        "",
    ]
    for builder in (table_1, table_2, table_3, table_4, table_5):
        lines += builder(root)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the C1 report tables.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/c1/analysis/tables.md")
    args = parser.parse_args(argv)

    text = build(Path(args.results))
    print(text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"written: {out}")

    missing = text.count(TODO)
    if missing:
        print(f"\n{missing} TODO cell(s) remain. They are honest; fill them by running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
