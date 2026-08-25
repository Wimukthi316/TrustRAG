"""Regenerate every C3 table from the JSON on disk.

The tables are generated, never typed. A number that appears in the report has a
file behind it, and if that file is missing the row says TODO rather than being
quietly dropped -- a table with a row missing reads as a table that was complete.

    C3-1  label audit                     results/c3/analysis/label_audit.json
    C3-2  typed span matching             results/c3/analysis/typed_matching_test.json
    C3-3  type prediction                 results/c3/type/baseline_metrics.json
    C3-4  type calibration                results/c3/type/c3_conformal.json
    C3-5  conformal type sets             results/c3/type/c3_conformal.json
    C3-6  severity-weighted risk control  results/c3/analysis/addons.json
    C3-7  explanations                    results/c3/explanation/explanation_metrics.json
    C3-8  generator against error type    results/c3/analysis/addons.json
    C3-9  escalation budget               results/c3/analysis/addons.json

Output goes to results/c3/tables.md, which is gitignored on purpose: the
generator is the artefact, the markdown is a rendering of it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

TODO = "TODO"


def load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def number(value: Any, places: int = 4) -> str:
    if value is None:
        return TODO
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(as_float):
        return "n/a"
    if math.isinf(as_float):
        return "inf"
    return f"{as_float:.{places}f}"


def count(value: Any) -> str:
    if value is None:
        return TODO
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def missing(name: str, path: Path) -> List[str]:
    return [
        f"### {name}",
        "",
        f"{TODO} - `{path}` not on disk. Run the layer that writes it.",
        "",
    ]


# --------------------------------------------------------------------------
# The tables
# --------------------------------------------------------------------------


def c3_1(report: Dict[str, Any]) -> List[str]:
    cov = report["l1_1_label_coverage"]
    meta = report["l1_2_meta_agreement"]
    census = report["l1_3_census"]
    floors = report["l1_4_floors"]

    lines = ["### C3-1 Label audit", ""]
    lines.append(
        f"`assert_label_coverage` over {count(report['n_spans'])} annotated spans: "
        f"**{'PASS' if cov['assert_label_coverage_passed'] else 'FAIL'}**. "
        f"{cov['n_distinct']} distinct `label_type` strings, no unmapped value."
    )
    lines.append("")
    lines += table(
        ["label_type", "spans", "share of corpus"],
        [
            [raw, count(n), number(n / report["n_spans"])]
            for raw, n in cov["distinct_raw_label_types"].items()
        ],
    )
    lines.append("")
    lines.append(
        "Annotator comment against published label. The `evident`/`subtle` rows "
        "share the label's own vocabulary and are therefore not independent "
        "evidence; the `high`/`low` rows are."
    )
    lines.append("")
    word = meta["severity_by_annotator_word"]
    lines += table(
        ["annotator's word", "labelled evident", "labelled subtle", "independent"],
        [
            [
                w,
                count(cells.get("evident", 0)),
                count(cells.get("subtle", 0)),
                "yes" if w in word["independent_words"] else "no",
            ]
            for w, cells in word["table"].items()
        ],
    )
    lines.append("")
    lines.append(
        f"Mechanism agreement {number(meta['mechanism']['agreement'])} "
        f"({count(meta['mechanism']['n_agree'])}/{count(meta['mechanism']['n_comparable'])}); "
        f"independent-vocabulary severity agreement "
        f"{number(word['independent']['agreement'])} "
        f"({count(word['independent']['n_agree'])}/{count(word['independent']['n_comparable'])}). "
        f"{number(meta['prefix_mapped_rate'])} of spans carry a comment whose prefix "
        f"names a mechanism at all."
    )
    lines.append("")
    lines += table(
        ["field", "spans", "share"],
        [
            [
                "implicit_true (annotated hallucinated, factually correct)",
                count(census["implicit_true"]["n"]),
                number(census["implicit_true"]["rate"]),
            ],
            [
                "due_to_null (data-to-text null artefact)",
                count(census["due_to_null"]["n"]),
                number(census["due_to_null"]["rate"]),
            ],
            [
                "`Original:` quote present in the comment",
                count(meta["structure"]["counts"]["original"]),
                number(meta["structure"]["rates"]["original"]),
            ],
        ],
    )
    lines.append("")
    lines.append("Per-type base rates and the floor every C3 number is read against.")
    lines.append("")
    rows: List[List[str]] = []
    for split in ("train", "calib", "test"):
        block = floors["base_rates"].get(split)
        if not block:
            continue
        floor = floors["majority_class_floor"][split]
        rows.append(
            [
                split,
                count(block["error_type"]["n"]),
                number(floor["error_type"]["accuracy"]),
                number(floor["error_type"]["macro_f1"]),
                number(floor["severity"]["accuracy"]),
                number(floor["severity"]["macro_f1"]),
            ]
        )
    lines += table(
        [
            "split",
            "spans",
            "4-way floor accuracy",
            "4-way floor macro-F1",
            "severity floor accuracy",
            "severity floor macro-F1",
        ],
        rows,
    )
    lines.append("")
    return lines


def c3_2(report: Dict[str, Any]) -> List[str]:
    f6 = report["footnote_6"]
    typeable = report["typeable"]
    lines = ["### C3-2 Typed span matching", ""]
    lines.append(
        "How a predicted span inherits a type, and how often the rule that "
        "arXiv:2504.04335 footnote 6 calls non-trivial actually bites."
    )
    lines.append("")
    lines += table(
        ["localisation bucket", "predicted spans"],
        [[name, count(n)] for name, n in report["predicted_by_bucket"].items()],
    )
    lines.append("")
    lines += table(
        ["quantity", "value"],
        [
            ["gold spans", count(report["n_gold_spans"])],
            ["predicted spans", count(report["n_predicted_spans"])],
            [
                "typeable (touch at least one gold span)",
                f"{count(typeable['n'])} ({number(typeable['rate_of_predicted'])})",
            ],
            [
                "excluded as spurious (no gold type exists)",
                f"{count(typeable['n_excluded_spurious'])} "
                f"({number(typeable['excluded_rate'])})",
            ],
            [
                "components covering more than one gold span",
                count(f6["n_multi_gold_components"]),
            ],
            [
                "of those, crossing a type boundary",
                f"{count(f6['n_multi_gold_crossing'])} "
                f"({number(f6['crossing_rate_of_multi_gold'])})",
            ],
            [
                "typeable predictions affected",
                f"{count(f6['n_typeable_predictions_affected'])} "
                f"({number(f6['affected_rate_of_typeable'])} of typeable)",
            ],
            [
                "gold spans already holding two types (a preprocessing merge)",
                f"{count(report['gold_multi_type']['n'])} "
                f"({number(report['gold_multi_type']['rate_of_gold'])})",
            ],
        ],
    )
    lines.append("")
    return lines


def c3_3(report: Dict[str, Any]) -> List[str]:
    lines = ["### C3-3 Type prediction", ""]
    lines.append(
        "Every model row has its majority-class floor directly beneath it. "
        "Intervals are response-level cluster bootstraps."
    )
    for name, block in report["evaluations"].items():
        lines.append("")
        lines.append(f"**{name}** - {count(block['n'])} spans in {count(block['n_responses'])} responses")
        lines.append("")
        rows: List[List[str]] = []
        for axis in ("mechanism", "severity", "error_type_flat", "error_type_axis_product"):
            cell = block.get(axis)
            if not cell:
                continue
            intervals = cell.get("intervals", {})
            macro = intervals.get("macro_f1", {})
            rows.append(
                [
                    axis,
                    "model",
                    number(cell["model"]["accuracy"]),
                    number(cell["model"]["macro_f1"]),
                    f"[{number(macro.get('ci_low'))}, {number(macro.get('ci_high'))}]"
                    if macro
                    else TODO,
                ]
            )
            rows.append(
                [
                    axis,
                    f"floor (always '{cell['floor']['predicted_class']}')",
                    number(cell["floor"]["accuracy"]),
                    number(cell["floor"]["macro_f1"]),
                    "ignores the input",
                ]
            )
        lines += table(
            ["axis", "row", "accuracy", "macro-F1", "macro-F1 95% CI"], rows
        )
    lines.append("")
    return lines


def c3_4(report: Dict[str, Any]) -> List[str]:
    lines = ["### C3-4 Type calibration", ""]
    lines.append(
        "Top-label calibration of the type head. The floor row ignores its input "
        "entirely; an ECE that does not beat it is not a calibration result."
    )
    lines.append("")
    lines += table(
        ["method", "ECE", "MCE", "Brier", "NLL", "accuracy", ""],
        [
            [
                row["method"],
                number(row["ece"]),
                number(row["mce"]),
                number(row["brier"]),
                number(row["nll"]),
                number(row["accuracy"]),
                "**floor, ignores the input**"
                if row.get("is_floor")
                else ("selected" if row.get("selected") else ""),
            ]
            for row in report["calibration"]
        ],
    )
    lines.append("")
    return lines


def c3_4b(addons: Dict[str, Any]) -> List[str]:
    block = addons.get("addon_1_type_conditional_calibration") or {}
    if not block:
        return []
    lines = ["", "Type-conditional calibration (add-on 1).", ""]
    lines += table(
        ["true type", "spans", "ECE", "mean confidence", "accuracy"],
        [[name, count(cell["n"]), number(cell["ece"]), number(cell["mean_confidence"]),
          number(cell["accuracy"])]
         for name, cell in block["by_error_type"].items()]
        + [["overall", count(block["overall"]["n"]), number(block["overall"]["ece"]),
            number(block["overall"]["mean_confidence"]),
            number(block["overall"]["accuracy"])]],
    )
    lines.append("")
    return lines


def c3_5(report: Dict[str, Any]) -> List[str]:
    lines = ["### C3-5 Conformal type sets", ""]
    lines.append(
        "Coverage against the 3-sigma band a single honest calibration draw may "
        "fall short by, the same rule C2's coverage rows are judged by."
    )
    lines.append("")
    lines += table(
        ["alpha", "target", "coverage", "95% CI", "band", "inside band",
         "avg set size", "singleton rate", "empty rate"],
        [
            [
                number(row["alpha"], 2),
                number(row["target_coverage"]),
                number(row["empirical_coverage"]),
                f"[{number((row.get('coverage_interval') or {}).get('ci_low'))}, "
                f"{number((row.get('coverage_interval') or {}).get('ci_high'))}]",
                number(row["band"]),
                "yes" if row["inside_band"] else "**no**",
                number(row["average_set_size"]),
                number(row["singleton_rate"]),
                number(row["empty_set_rate"]),
            ]
            for row in report["coverage"]
        ],
    )
    headline = report["headline_alpha"]
    row = next(r for r in report["coverage"] if r["alpha"] == headline)
    lines.append("")
    lines.append(f"What the sets say at alpha = {headline}.")
    lines.append("")
    lines += table(
        ["shape", "sets", "share"],
        [
            [shape, count(row["set_shapes"][shape]), number(rate)]
            for shape, rate in sorted(
                row["set_shape_rates"].items(), key=lambda kv: -kv[1]
            )
        ],
    )
    lines.append("")
    lines.append(
        "Coverage inside each true type. Split conformal promises marginal "
        "coverage, not per-class coverage, so these rows are diagnostic - but a "
        "marginal guarantee carried entirely by the majority classes is worth "
        "knowing before anyone deploys it."
    )
    lines.append("")
    lines += table(
        ["true type", "test n", "calibration n", "coverage", "avg set size"],
        [
            [
                name,
                count(cell["n_test"]),
                count(cell["n_calibration"]),
                number(cell["empirical_coverage"])
                if cell["measurable"]
                else "n/a - absent from calibration",
                number(cell["average_set_size"]),
            ]
            for name, cell in report["per_class_coverage"].items()
        ],
    )
    lines.append("")
    lines.append(f"*{report['ood_note']}*")
    lines.append("")
    return lines


def c3_6(addons: Dict[str, Any]) -> List[str]:
    risk = addons.get("addon_2_severity_weighted_risk_control") or {}
    lines = ["### C3-6 Severity-weighted conformal risk control", ""]
    if not risk:
        lines += [f"{TODO} - addons.json has no risk-control block.", ""]
        return lines
    lines.append(f"Loss: {risk['loss']}.")
    lines.append(
        f"Weights: evident {risk['weights']['evident']}, subtle "
        f"{risk['weights']['subtle']}, implicit_true "
        f"{risk['weights']['implicit_true']} - {risk['weights_note']}"
    )
    lines.append("")
    lines += table(
        ["alpha", "threshold", "test risk", "bound held", "span flag rate",
         "spans flagged", "note"],
        [
            [
                number(row["alpha"], 2),
                number(row.get("threshold")),
                number(row.get("test_risk")),
                ("yes" if row.get("bound_held") else "**no**")
                if row.get("feasible")
                else "-",
                number(row.get("span_flag_rate")),
                count(row.get("n_spans_flagged")) if row.get("feasible") else TODO,
                "" if row.get("feasible") else "not reachable at this alpha",
            ]
            for row in risk["rows"]
        ],
    )
    lines.append("")
    lines.append(
        "The flag-rate column is the cost. A risk bound reported without one is "
        "marketing."
    )
    lines.append("")
    return lines


def c3_7(report: Dict[str, Any]) -> List[str]:
    lines = ["### C3-7 Explanations", ""]
    lines.append(
        "Snippet match is the primary metric: does C3 quote the same source text "
        "the human annotator quoted. ROUGE-L is reported for completeness and is "
        "the weaker number."
    )
    lines.append("")
    rows: List[List[str]] = []
    for name, block in report["systems"].items():
        rows.append(
            [
                name,
                count(block["n_spans"]),
                number(block["snippet"]["mean_containment"]),
                number(block["snippet"]["rate_at_threshold"].get("1.0")),
                number(block["self_faithfulness"]["verbatim_rate"]),
                number(block["rouge_l_vs_meta"]["mean_f1"]),
            ]
        )
    lines += table(
        ["system", "spans", "mean snippet containment", "exact snippet rate",
         "self-faithfulness", "ROUGE-L vs meta"],
        rows,
    )
    lines.append("")
    lines.append(f"*{report['systems']['template_gold_type']['rouge_l_vs_meta']['caption']}*")
    lines.append("")
    lines.append(f"*Snippet metric: {report.get('snippet_metric_note', '')}*")
    lines.append("")
    lines.append(f"*LLM comparison: n = {report['llm_comparison']['n']}. {report['llm_comparison']['status']}*")
    lines.append("")
    template = report["systems"].get("template_gold_type", {})
    if template.get("by_task"):
        lines.append("Snippet containment by task.")
        lines.append("")
        lines += table(
            ["task", "spans", "spans with a reference", "mean containment"],
            [
                [task, count(cell["n"]), count(cell["n_with_reference"]),
                 number(cell["mean_containment"])]
                for task, cell in template["by_task"].items()
            ],
        )
        lines.append("")
    return lines


def c3_8(addons: Dict[str, Any]) -> List[str]:
    gen = addons.get("addon_3_generator_by_error_type") or {}
    lines = ["### C3-8 Generator model against error type", ""]
    if not gen:
        lines += [f"{TODO} - addons.json has no generator block.", ""]
        return lines
    types = sorted(
        {t for cell in gen["by_model"].values() for t in cell["rates"]}
    )
    lines += table(
        ["generator", "spans", "spans per response"] + types,
        [
            [model, count(cell["n_spans"]), number(cell["spans_per_response"], 3)]
            + [number(cell["rates"].get(t), 3) for t in types]
            for model, cell in gen["by_model"].items()
        ],
    )
    lines.append("")
    return lines


def c3_9(addons: Dict[str, Any]) -> List[str]:
    budget = addons.get("layer_6_escalation_budget") or {}
    lines = ["### C3-9 Escalation budget", ""]
    if not budget:
        lines += [f"{TODO} - addons.json has no escalation block.", ""]
        return lines
    lines.append(f"Rule: {budget['rule']}.")
    lines.append(
        f"{count(budget['n_abstained'])} of {count(budget['n_spans'])} predicted "
        f"spans abstain ({number(budget['abstention_rate'])})."
    )
    lines.append("")
    lines += table(
        ["alpha", "spans escalated", "share of all spans", "share of abstained"],
        [
            [alpha, count(cell["n_escalated"]), number(cell["share_of_all_spans"]),
             number(cell["share_of_abstained"])]
            for alpha, cell in budget["per_alpha"].items()
        ],
    )
    lines.append("")
    lines.append(f"*{budget['loop_status']}*")
    lines.append("")
    return lines


def c3_extra(addons: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    det = addons.get("addon_4_type_conditional_detectability")
    if det:
        lines += ["### C3-10 Type-conditional detectability (add-on 4)", ""]
        lines += table(
            ["error type", "gold spans", "span-overlap recall", "mean conformal set size"],
            [
                [name, count(cell["n_gold"]), number(cell["overlap_recall"]),
                 number(cell["mean_conformal_set_size"])]
                for name, cell in det["by_error_type"].items()
            ],
        )
        lines.append("")
    rows: List[List[str]] = []
    for key, label in (
        ("addon_5_implicit_true_ablation", "implicit_true excluded"),
        ("addon_9_due_to_null_ablation", "due_to_null excluded"),
    ):
        block = addons.get(key)
        if not block:
            continue
        rows.append(
            [
                label,
                count(block["n_spans_flagged"]),
                count(block["n_responses_that_change_label"]),
                number(block["example_before"]["f1"]),
                number(block["example_after"]["f1"]),
                f"{block['delta_f1']:+.4f}",
            ]
        )
    if rows:
        lines += ["### C3-11 Confound ablations on C1's headline (add-ons 5 and 9)", ""]
        lines += table(
            ["ablation", "spans removed", "responses that change label",
             "example F1 before", "example F1 after", "delta"],
            rows,
        )
        lines.append("")
    return lines


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def build(paths: Dict[str, Path]) -> str:
    lines: List[str] = ["# C3 tables", ""]
    lines.append(
        "Generated by `python -m src.c3_explanation.tables`. Every number here "
        "comes from a JSON file written by a run; a missing file leaves a TODO "
        "rather than a gap."
    )
    lines.append("")

    sections: List[tuple] = [
        ("C3-1 Label audit", paths["label_audit"], c3_1),
        ("C3-2 Typed span matching", paths["typed_matching"], c3_2),
        ("C3-3 Type prediction", paths["baseline_metrics"], c3_3),
        ("C3-4 Type calibration", paths["conformal"], c3_4),
        ("C3-5 Conformal type sets", paths["conformal"], c3_5),
        ("C3-6 Severity-weighted risk control", paths["addons"], c3_6),
        ("C3-7 Explanations", paths["explanation"], c3_7),
        ("C3-8 Generator against error type", paths["addons"], c3_8),
        ("C3-9 Escalation budget", paths["addons"], c3_9),
    ]
    addons = load(paths["addons"])
    for name, path, builder in sections:
        report = load(path)
        if report is None:
            lines += missing(name, path)
            continue
        lines += builder(report)
        if builder is c3_4 and addons:
            lines += c3_4b(addons)
    if addons:
        lines += c3_extra(addons)
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the C3 tables from JSON.")
    parser.add_argument("--results-dir", default="results/c3")
    parser.add_argument("--out", default="results/c3/tables.md")
    args = parser.parse_args(argv)

    root = Path(args.results_dir)
    paths = {
        "label_audit": root / "analysis" / "label_audit.json",
        "typed_matching": root / "analysis" / "typed_matching_test.json",
        "addons": root / "analysis" / "addons.json",
        "baseline_metrics": root / "type" / "baseline_metrics.json",
        "conformal": root / "type" / "c3_conformal.json",
        "explanation": root / "explanation" / "explanation_metrics.json",
    }
    text = build(paths)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    todo = text.count(TODO)
    print(f"written: {out_path}  ({todo} TODO marker{'s' if todo != 1 else ''})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
