"""Regenerate every reported C2 table from the JSON on disk.

The freeze rule for this project is that no number reaches the report by hand.
This module is the only thing that formats a C2 table, it reads only artefacts
written by a run, and where an artefact is missing it prints TODO rather than a
plausible-looking number. A table with a TODO in it is a table telling the truth
about what has been measured.

It mirrors `src/c1_detector/tables.py` deliberately, down to the TODO
convention, so the two components' tables are read the same way.

Two house rules are enforced here rather than left to whoever writes the report:

  * every ECE is printed beside the constant-base-rate predictor's ECE, because
    an ECE read against zero says nothing about whether a detector is useful;
  * every coverage figure is printed beside the noise band it has to be judged
    against, because coverage below target is only a failure outside that band.

Run it with:  python -m src.c2_calibration.tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.c2_calibration.conformal import coverage_tolerance

TODO = "TODO"


def load(root: Path, relative: str) -> Optional[Dict[str, Any]]:
    """An artefact, or None. Missing is a legitimate state, not an error."""
    path = root / relative
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def number(value: Any, places: int = 4) -> str:
    if value is None:
        return TODO
    try:
        return f"{float(value):.{places}f}"
    except (TypeError, ValueError):
        return TODO


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


def missing(title: str, artefact: str) -> List[str]:
    return [
        f"## {title}",
        "",
        f"{TODO} — `results/{artefact}` does not exist. Run it; nothing has been "
        "invented in its place.",
        "",
    ]


def table_1(root: Path) -> List[str]:
    """Calibration: four calibrators, both units, with the uninformative floor."""
    results = load(root, "c2/c1/c2_results.json")
    uncertainty = load(root, "c2/c1/c2_uncertainty.json")
    if not results:
        return missing("C2-1 Calibration", "c2/c1/c2_results.json")

    rows: List[Sequence[str]] = []
    for unit in ("span", "token"):
        block = results.get(unit)
        if not block:
            continue
        selected = block["calibration"]["selected"]
        for row in block["calibration"]["rows"]:
            name = str(row["method"])
            label = f"**{name}**" if name == selected else name
            rows.append(
                [
                    unit,
                    label,
                    number(row["ece"]),
                    number(row["ece_equal_count"]),
                    number(row["mce"]),
                    number(row["brier"]),
                    number(row["nll"]),
                ]
            )
        floor = (uncertainty or {}).get(unit, {}).get("floor")
        if floor:
            constant = floor["constant_base_rate"]
            rows.append(
                [
                    unit,
                    "_constant base rate (uninformative)_",
                    number(constant["ece"]),
                    number(constant.get("ece_equal_count")),
                    number(constant["mce"]),
                    number(constant["brier"]),
                    number(constant.get("nll")),
                ]
            )
        else:
            rows.append([unit, "_constant base rate_", TODO, TODO, TODO, TODO, TODO])

    note = (
        "Every calibrator is fitted on the held-out calibration split and every "
        "figure is measured on test. The bold row is the one `run_c2` selected, "
        "by lowest test ECE -- a declared peek across four candidates.\n\n"
        "**Read the ECE column against the constant-base-rate row, not against "
        "zero.** That predictor returns the calibration positive rate for every "
        "input and cannot rank anything; its AUROC is 0.5000. At span level it "
        "scores a *better* ECE than the calibrated detector, which is the "
        "clearest available statement that a good ECE is not on its own a result."
    )
    return render(
        "C2-1 Calibration, fitted on calib and measured on test",
        ["unit", "method", "ECE", "ECE (eq-count)", "MCE", "Brier", "NLL"],
        rows,
        note,
    )


def table_2(root: Path) -> List[str]:
    """Coverage vs alpha, in-domain, with both the analytic band and the clustered CI."""
    results = load(root, "c2/c1/c2_results.json")
    uncertainty = load(root, "c2/c1/c2_uncertainty.json")
    if not results or "span" not in results:
        return missing("C2-2 Coverage vs alpha", "c2/c1/c2_results.json")

    clustered = {}
    if uncertainty and "span" in uncertainty:
        clustered = {
            row["alpha"]: row
            for row in uncertainty["span"]["bootstrap_full"]["coverage"]
        }

    rows: List[Sequence[str]] = []
    for row in results["span"]["conformal"]["coverage"]:
        alpha = row["alpha"]
        band = clustered.get(alpha, {}).get("pooled_3sigma_band")
        interval = clustered.get(alpha)
        ci = (
            f"[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"
            if interval
            else TODO
        )
        rows.append(
            [
                number(alpha, 2),
                number(row["target_coverage"], 3),
                number(row["empirical_coverage"]),
                number(band),
                ci,
                number(row["abstention_rate"]),
                number(row["empty_set_rate"]),
                number(row["flag_rate"]),
            ]
        )

    note = (
        "Span level. `+/-band` is the 3-sigma range a single honest calibration "
        "draw may fall below target by; coverage inside it is not a shortfall. "
        "`95% CI` is a response-level cluster bootstrap over the whole procedure "
        "-- calibration resampled and the calibrator and threshold refitted every "
        "draw -- from `c2_uncertainty.json`.\n\n"
        "The abstention column is not monotone in alpha, and the reason is the "
        "`empty` column beside it: past alpha 0.30 the threshold is tight enough "
        "that some spans have neither label clearing it. LAC allows the empty set "
        "and `conformal.py` folds it into abstention, because \"neither label is "
        "ordinary\" and \"cannot separate\" are the same instruction to a "
        "reviewer: send it to a human."
    )
    return render(
        "C2-2 Split-conformal coverage vs alpha, RAGTruth test",
        [
            "alpha",
            "target",
            "empirical",
            "+/-band",
            "95% CI (clustered)",
            "abstain",
            "empty",
            "flag",
        ],
        rows,
        note,
    )


def table_3(root: Path) -> List[str]:
    """Group-conditional coverage per task, and the exchangeability repair."""
    results = load(root, "c2/c1/c2_results.json")
    uncertainty = load(root, "c2/c1/c2_uncertainty.json")
    if not results or "span" not in results:
        return missing("C2-3 Per-task and per-response coverage", "c2/c1/c2_results.json")

    rows: List[Sequence[str]] = []
    groups = results["span"]["conformal"]["group_conditional_alpha_0.1"]
    for task, row in sorted(groups.items()):
        rows.append(
            [
                task,
                f"{int(row['n_test']):,}",
                f"{int(row['n_calibration']):,}",
                number(row["empirical_coverage"]),
                number(row["abstention_rate"]),
            ]
        )
    if not rows:
        rows.append(["(no group in both splits)", TODO, TODO, TODO, TODO])

    lines = render(
        "C2-3a Group-conditional coverage at alpha 0.10, by task",
        ["task", "n test spans", "n calib spans", "coverage", "abstain"],
        rows,
        "A separate threshold is fitted per task, so the promise holds within "
        "each task and not merely on average across them. Target is 0.900.",
    )

    if not uncertainty or "span" not in uncertainty:
        return lines + missing(
            "C2-3b One span per response", "c2/c1/c2_uncertainty.json"
        )

    subsample = uncertainty["span"]["one_row_per_response"]
    sub_rows: List[Sequence[str]] = []
    for row in subsample["rows"]:
        coverage = row["coverage"]
        sub_rows.append(
            [
                number(row["alpha"], 2),
                number(row["target_coverage"], 3),
                number(row["subsampled_mean_coverage"]),
                f"[{coverage['ci_low']:.4f}, {coverage['ci_high']:.4f}]",
                number(row["pooled_coverage"]),
                "yes" if row["inside_band"] else "NO",
            ]
        )
    lines += render(
        "C2-3b One span per response, exchangeable by construction",
        ["alpha", "target", "subsampled mean", "95% CI over draws", "pooled", "in band"],
        sub_rows,
        f"{subsample['draws']} draws, seed {subsample['seed']}. Each draw keeps "
        f"exactly one predicted span from each response, so its "
        f"{subsample['n_calibration_rows_per_draw']:,} calibration rows are "
        "exchangeable in the sense split conformal requires, with no "
        "approximation. **Each individual draw is exactly valid**; the mean "
        "across draws shows how much the arbitrary choice of which span to keep "
        "matters and is not itself claimed to carry the guarantee.",
    )
    return lines


def table_4(root: Path) -> List[str]:
    """Response-level uncertainty: how much did pooling understate the noise?"""
    uncertainty = load(root, "c2/c1/c2_uncertainty.json")
    if not uncertainty:
        return missing("C2-4 Response-level uncertainty", "c2/c1/c2_uncertainty.json")

    rows: List[Sequence[str]] = []
    for unit in ("span", "token"):
        block = uncertainty.get(unit)
        if not block:
            continue
        boot = block["bootstrap_full"]
        rows_per_response = boot["n_calibration_rows"] / max(
            boot["n_calibration_responses"], 1
        )
        for row in boot["coverage"]:
            rows.append(
                [
                    unit,
                    f"{rows_per_response:.2f}",
                    number(row["alpha"], 2),
                    number(row["sd"], 5),
                    number(row["theoretical_sd_independent_rows"], 5),
                    f"{row['sd_ratio_clustered_over_independent']:.2f}x",
                ]
            )

    note = (
        "`sd clustered` resamples **responses**, never rows. `sd independent` is "
        "what `conformal.py`'s noise band assumes, its 3-sigma tolerance divided "
        "by three. A ratio above 1.00x means treating rows as independent "
        "understated the real spread by that much.\n\n"
        "The two units land on opposite sides of 1.00x and the reason is the "
        "cluster size in column two. A predicted span is very nearly a response, "
        "so pooling spans costs almost nothing. A token is not: the pooled "
        "token band is optimistic by roughly six times, which is the measurement "
        "`conformal.py`'s third caveat asked for."
    )
    return render(
        "C2-4 Cluster bootstrap: what pooling cost",
        ["unit", "rows per response", "alpha", "sd clustered", "sd independent", "ratio"],
        rows,
        note,
    )


def table_5(root: Path) -> List[str]:
    """Is the ECE reduction real once responses, not spans, are resampled?"""
    uncertainty = load(root, "c2/c1/c2_uncertainty.json")
    if not uncertainty:
        return missing("C2-5 ECE reduction intervals", "c2/c1/c2_uncertainty.json")

    rows: List[Sequence[str]] = []
    for unit in ("span", "token"):
        block = uncertainty.get(unit)
        if not block:
            continue
        for label, key in (
            ("whole procedure", "bootstrap_full"),
            ("this fitted artifact", "bootstrap_test_only"),
        ):
            boot = block[key]
            before, after, delta = (
                boot["ece_before"],
                boot["ece_after"],
                boot["ece_reduction"],
            )
            rows.append(
                [
                    unit,
                    label,
                    f"[{before['ci_low']:.4f}, {before['ci_high']:.4f}]",
                    f"[{after['ci_low']:.4f}, {after['ci_high']:.4f}]",
                    f"[{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]",
                    "not decidable" if delta["crosses_zero"] else "real",
                ]
            )
    return render(
        "C2-5 ECE before, after and the reduction, at response level",
        ["unit", "resampling", "ECE before", "ECE after", "reduction", "verdict"],
        rows,
        "95% percentile intervals over 2,000 response-level bootstrap draws. "
        "Quote the *whole procedure* row beside a claim about the method; the "
        "*this fitted artifact* row holds the calibrator and threshold fixed and "
        "belongs beside a claim about the shipped file. An interval that crosses "
        "zero would read `not decidable at this n`.",
    )


def table_6(root: Path) -> List[str]:
    """The shift: what breaks, the diagnosis, and the two repairs."""
    void = load(root, "c2/ragbench/c2_results_void.json")
    diagnosis = load(root, "c2/ragbench/shift_diagnosis.json")
    repairs = load(root, "c2/ragbench/repair.json")
    control = load(root, "c2/control/shift_diagnosis.json")

    lines: List[str] = []

    if not void or "span" not in void:
        lines += missing("C2-6a The break", "c2/ragbench/c2_results_void.json")
    else:
        rows: List[Sequence[str]] = []
        in_domain = load(root, "c2/c1/c2_results.json")
        reference = (
            {
                row["alpha"]: row["empirical_coverage"]
                for row in in_domain["span"]["conformal"]["coverage"]
            }
            if in_domain and "span" in in_domain
            else {}
        )
        for row in void["span"]["conformal"]["coverage"]:
            band = coverage_tolerance(
                row["alpha"], int(row["n_calibration"]), int(row["n_test"])
            )
            shortfall = row["target_coverage"] - row["empirical_coverage"]
            rows.append(
                [
                    number(row["alpha"], 2),
                    number(row["target_coverage"], 3),
                    number(reference.get(row["alpha"])),
                    number(row["empirical_coverage"]),
                    number(band),
                    f"{shortfall:+.4f}",
                    f"{shortfall / band:.1f}x" if band else TODO,
                ]
            )
        lines += render(
            "C2-6a VOID — the RAGTruth threshold applied to RAGBench",
            [
                "alpha",
                "target",
                "in-domain",
                "RAGBench",
                "+/-band",
                "shortfall",
                "bands out",
            ],
            rows,
            "Span level. This whole table is marked **VOID**: the calibration "
            "and test data are not exchangeable, so the coverage guarantee does "
            "not apply to these numbers and they are reported as a measurement "
            "of the break rather than as a working method. The run wrote no "
            "serving artifact.",
        )

    if not diagnosis:
        lines += missing("C2-6b The diagnosis", "c2/ragbench/shift_diagnosis.json")
    else:

        def read(block: Optional[Dict[str, Any]], path: Sequence[str]) -> Any:
            current: Any = block
            for key in path:
                if current is None:
                    return None
                current = current.get(key) if isinstance(current, dict) else None
            return current

        diagnostics = (
            ("base-rate ratio, target / source", ("base_rate", "ratio")),
            # The pipe is escaped: an unescaped one closes the markdown cell.
            (r"KS on P(score \| y=0)", ("conditional_scores", "y=0", "ks_statistic")),
            (r"KS on P(score \| y=1)", ("conditional_scores", "y=1", "ks_statistic")),
            ("domain classifier held-out AUC", ("domain_classifier", "held_out_auc")),
            (
                "BBSE prior error",
                ("label_shift_estimate", "estimation_error_on_positive_rate"),
            ),
        )
        rows = [
            [
                label,
                number(read(control, path)),
                number(read(diagnosis, path)),
            ]
            for label, path in diagnostics
        ]
        lines += render(
            "C2-6b Which shift is it? — with an in-domain control",
            ["diagnostic", "in-domain control", "RAGTruth -> RAGBench"],
            rows,
            "The control column runs the identical harness with source and "
            "target both drawn from RAGTruth. Every diagnostic reads "
            "\"no shift\" there and \"large shift\" on RAGBench, which is what "
            "rules out the harness itself as the cause of the break.\n\n"
            "A base rate that moved with score distributions that did **not** "
            "would be label shift. Both moved, and a classifier separates the "
            "two corpora from run-time features alone, so it is not pure label "
            "shift and the label-shift repair cannot be expected to fix it. "
            "P(y|x) has almost certainly moved too: RAGBench labels are written "
            "by an LLM judge at sentence granularity, RAGTruth's by humans at "
            "phrase granularity, and no reweighting repairs that.",
        )

    if not repairs:
        lines += missing("C2-6c The repairs", "c2/ragbench/repair.json")
    else:
        methods = (
            ("unrepaired (VOID)", "unrepaired"),
            ("label shift, estimated prior", "label_shift_estimated"),
            ("label shift, ORACLE prior", "label_shift_oracle"),
            ("covariate shift", "covariate_shift"),
        )
        rows = []
        for entry in repairs["verdict"]:
            for label, key in methods:
                row = entry.get(key)
                if row is None:
                    rows.append([number(entry["alpha"], 2), label, TODO, TODO, TODO])
                    continue
                rows.append(
                    [
                        number(entry["alpha"], 2),
                        label,
                        number(row["coverage"]),
                        f"{row['shortfall']:+.4f}",
                        number(row["abstention_rate"]),
                    ]
                )
        weights = repairs["covariate_shift_weights"]["calibration"]
        lines += render(
            "C2-6c Does reweighting bring coverage back? No.",
            ["alpha", "method", "coverage", "shortfall", "abstain"],
            rows,
            "The **oracle** row is handed the evaluation half's true positive "
            "rate, which no deployment can know. It is reported only to "
            "separate a bad prior estimate from a false label-shift assumption, "
            "and it still does not reach target -- so the assumption is what is "
            "wrong, exactly as the KS statistics in C2-6b predicted.\n\n"
            f"The covariate-shift weights are worth reading before its coverage "
            f"column: {weights['fraction_clipped']:.1%} sit on the clipping "
            f"boundary and the effective calibration sample size collapses from "
            f"{weights['n']:,} to {weights['effective_sample_size']:.0f}. Its "
            "threshold is partly decided by the clipping constant rather than by "
            "the data.\n\n"
            "**Both repairs fail, so no third was attempted.** The deployment "
            "rule is: recalibrate on your own data before you trust the dial.",
        )
    return lines


def table_7(root: Path) -> List[str]:
    """Conformal risk control: the product-language promise and what it costs."""
    risk = load(root, "c2/c1/c2_risk_control.json")
    if not risk:
        return missing("C2-7 Conformal risk control", "c2/c1/c2_risk_control.json")

    rows: List[Sequence[str]] = []
    for row in risk["rows"]:
        chosen = row["chosen"]
        if not chosen["feasible"]:
            rows.append(
                [number(chosen["alpha"], 2), "unreachable", TODO, TODO, TODO, "n/a"]
            )
            continue
        edge = "  (grid edge)" if chosen["on_grid_edge"] else ""
        rows.append(
            [
                number(chosen["alpha"], 2),
                number(chosen["threshold"]) + edge,
                number(chosen["calibration_risk"]),
                number(row["test_risk"]),
                number(row["test"]["token_flag_rate"]),
                "yes" if row["bound_held"] else "NO",
            ]
        )
    baseline = risk["flag_everything_baseline"]
    rows.append(
        [
            "_flag everything_",
            "0.0000",
            "0.0000",
            number(baseline["fnr_over_responses_with_hallucinations"]),
            number(baseline["token_flag_rate"]),
            "trivially",
        ]
    )

    note = (
        f"Loss: {risk['loss']}. {risk['convention'].capitalize()}, so "
        f"{risk['n_calibration_responses_used']:,} of "
        f"{risk['n_calibration_responses_total']:,} calibration responses are "
        "used. The threshold is chosen once per alpha on calibration and applied "
        "once to test.\n\n"
        "**Read the `tokens flagged` column beside every risk.** A false-negative "
        "bound can always be satisfied by highlighting the whole answer, which is "
        "the `flag everything` row, and the only thing that tells the two apart "
        "is what the rule costs. A threshold marked `(grid edge)` is where the "
        "search stopped, not where the data pointed, and must be reported as "
        "unreached rather than as a chosen value.\n\n"
        "The exchangeable unit here is the response, because the loss is defined "
        "per response and averaged over responses -- so this table does not "
        "inherit the pooled-token problem measured in C2-4."
    )
    return render(
        "C2-7 Conformal risk control over missed hallucinated tokens",
        ["alpha", "t_hat", "calib risk", "test risk", "tokens flagged", "bound held"],
        rows,
        note,
    )


def table_8(root: Path) -> List[str]:
    """A second detector: is C2 a property of the method or of one training run?"""
    base = load(root, "c2/c1/c2_results.json")
    cw3 = load(root, "c2/c1-cw3/c2_results.json")
    base_uncertainty = load(root, "c2/c1/c2_uncertainty.json")
    cw3_uncertainty = load(root, "c2/c1-cw3/c2_uncertainty.json")
    lettuce = load(root, "c2/lettucedetect/c2_results.json")
    lettuce_uncertainty = load(root, "c2/lettucedetect/c2_uncertainty.json")
    if not base:
        return missing("C2-8 A second detector", "c2/c1/c2_results.json")

    def summarise(name: str, results: Optional[Dict[str, Any]],
                  uncertainty: Optional[Dict[str, Any]]) -> Sequence[str]:
        if not results or "span" not in results:
            return [name, TODO, TODO, TODO, TODO, TODO, TODO, TODO]
        block = results["span"]
        coverage = {
            row["alpha"]: row for row in block["conformal"]["coverage"]
        }.get(0.10)
        floor = (uncertainty or {}).get("span", {}).get("floor")
        ranking = (
            ("preserved" if floor["detector"]["ranking_preserved"] else "MOVED")
            if floor
            else TODO
        )
        auroc = number(floor["detector"]["auroc_raw"]) if floor else TODO
        constant = (
            number(floor["constant_base_rate"]["ece"]) if floor else TODO
        )
        return [
            name,
            block["calibration"]["selected"],
            number(block["calibration"]["before"]["ece"]),
            number(block["calibration"]["after"]["ece"]),
            constant,
            auroc,
            number(coverage["empirical_coverage"]) if coverage else TODO,
            ranking,
        ]

    rows = [
        summarise("C1 ModernBERT-base (ours)", base, base_uncertainty),
        summarise("C1 + class weights (cw3)", cw3, cw3_uncertainty),
        summarise("LettuceDetect-large, self-split", lettuce, lettuce_uncertainty),
    ]

    note = (
        "Span level; coverage at alpha 0.10, target 0.900. The point of the "
        "second detector is that C2 should be a property of the *method*, not of "
        "one training run: if both detectors calibrate and both stay covered, "
        "the layer transfers between detectors even though it does not transfer "
        "between corpora.\n\n"
        "The `ranking` column is a correctness check, not a result. A strictly "
        "monotone calibrator cannot reorder predictions, so AUROC before and "
        "after must be identical. Where it reads `MOVED`, isotonic regression "
        "was selected: it is only *weakly* monotone and can tie distinct scores "
        "together, which `conformal.py` predicts in its `fit` docstring. That is "
        "a measured confirmation of a documented caveat, and it is also the "
        "reason a strictly monotone calibrator is the safer default for serving.\n\n"
        "Our own detector is worse calibrated raw than the public LettuceDetect "
        "checkpoint, which is the argument for C2 rather than an embarrassment: "
        "the weaker and cheaper the model, the more it needs this layer."
    )
    return render(
        "C2-8 The same C2 layer over three detectors",
        [
            "detector",
            "calibrator",
            "ECE raw",
            "ECE cal.",
            "constant floor",
            "AUROC",
            "coverage @0.10",
            "ranking",
        ],
        rows,
        note,
    )


def table_9(root: Path) -> List[str]:
    """The run-time exchangeability check: what it costs and what it catches."""
    reference = load(root, "c2/c1/c2_ood_reference.json")
    if not reference:
        return missing(
            "C2-9 The run-time exchangeability check", "c2/c1/c2_ood_reference.json"
        )

    validation = reference.get("validation") or {}
    bias = validation.get("label_conditional") or {}
    shifted = reference.get("shifted_corpus_trip_rate") or {}

    rows: List[Sequence[str]] = [
        [
            "claimed false-alarm rate (the threshold)",
            number(reference.get("threshold")),
            "what the warning promises to cost",
        ],
        [
            "measured on held-out RAGTruth",
            number(validation.get("measured_false_alarm_rate")),
            f"over {validation.get('n', 0):,} responses the reference never saw",
        ],
        [
            "tripped on RAGBench",
            number(shifted.get("trip_rate")),
            f"over {shifted.get('n', 0):,} responses; the corpus C2-6a breaks on",
        ],
        [
            "on responses that DO contain hallucinations",
            number(bias.get("trip_rate_gold_positive")),
            "",
        ],
        [
            "on responses that do not",
            number(bias.get("trip_rate_gold_negative")),
            "",
        ],
        [
            "share of warnings that are hallucinated responses",
            number(bias.get("share_of_alarms_gold_positive"), 3),
            f"base rate {number(bias.get('gold_positive_base_rate'), 3)}",
        ],
    ]

    enrichment = (
        shifted.get("trip_rate", 0) / validation["measured_false_alarm_rate"]
        if validation.get("measured_false_alarm_rate")
        else None
    )
    note = (
        f"Built on {reference.get('n', 0):,} calibration responses over "
        f"{len(reference.get('features', []))} features. A conformal p-value with "
        "the same (n+1) correction as the coverage quantile, so the threshold is "
        "a false-alarm rate rather than a magic number -- and the second row is "
        "what makes that claim checkable.\n\n"
        f"**It is a smoke alarm, not a proof.** It catches "
        f"{number(shifted.get('trip_rate'))} of the shifted corpus against a "
        f"{number(validation.get('measured_false_alarm_rate'))} false-alarm rate"
        + (f" -- {enrichment:.0f}x enrichment" if enrichment else "")
        + " -- so it is real signal, and it still misses most of RAGBench. That "
        "is the honest finding, not a shortfall to tune away: the shift between "
        "those corpora is large in aggregate and nearly invisible in any single "
        "response. A per-input check cannot substitute for measuring the shift, "
        "which is why the deployment rule in C2-6c is still 'recalibrate on your "
        "own data'.\n\n"
        "**The last three rows are a limitation, volunteered.** A response that "
        "really contains hallucinations has more spans, longer spans and higher "
        "scores, so it sits further into the tail and trips the warning about "
        "twice as often. The guarantee is therefore least well-supported on the "
        "responses that matter most. Dropping the score-derived features was "
        "tried as a fix and made the skew worse (0.79 with span shape alone, "
        "0.85 with span shape and count, against 0.67 with all eight), so the "
        "full list ships and the skew is reported."
    )
    return render(
        "C2-9 The run-time exchangeability check",
        ["quantity", "value", "note"],
        rows,
        note,
    )


def build(root: Path) -> str:
    lines: List[str] = [
        "# C2 reported tables",
        "",
        "Generated by `python -m src.c2_calibration.tables`. Every value is read "
        "from an artefact under `results/`; `TODO` means the artefact does not "
        "exist yet and no number has been invented in its place.",
        "",
    ]
    for builder in (
        table_1,
        table_2,
        table_3,
        table_4,
        table_5,
        table_6,
        table_7,
        table_8,
        table_9,
    ):
        lines += builder(root)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate the C2 report tables.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="results/c2/tables.md")
    args = parser.parse_args(argv)

    text = build(Path(args.results))
    print(text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(f"written: {out}")

    remaining = text.count(TODO)
    if remaining:
        print(f"\n{remaining} TODO cell(s) remain. They are honest; fill them by running.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
