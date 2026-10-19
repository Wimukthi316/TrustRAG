"""C4's report tables, generated from the artefacts on disk.

Every table is built from JSON that a run produced. A missing artefact becomes a
visible `TODO` row rather than a dropped one, because a dropped row is how a
report ends up quietly incomplete and a TODO is how it gets finished.

    python -m src.c4_attribution.tables            # print
    python -m src.c4_attribution.tables --write    # results/c4/tables.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results/c4"
OUT = RESULTS / "tables.md"

TODO = "TODO"


def _load(relative: str) -> Optional[Dict[str, Any]]:
    path = RESULTS / relative
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _f(value: Any, places: int = 4) -> str:
    if value is None:
        return TODO
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "n/a"
    if number in (float("inf"), float("-inf")):
        return "inf"
    return f"{number:.{places}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        # A literal pipe inside a cell silently breaks the whole table.
        lines.append(
            "| " + " | ".join(str(c).replace("|", "\\|") for c in row) + " |"
        )
    return "\n".join(lines)


def _missing(name: str, source: str) -> str:
    return f"### {name}\n\n{TODO} -- `{source}` is not on disk. Run the driver.\n"


# --------------------------------------------------------------------------


def c4_1_supervision() -> str:
    report = _load("analysis/evidence_audit.json")
    if not report:
        return _missing("C4-1  Evidence supervision on the frozen splits",
                        "results/c4/analysis/evidence_audit.json")
    rows = []
    for split, block in report["per_split"].items():
        counts = block["counts"]
        rows.append([
            split,
            counts["n_annotated_spans"],
            counts["n_with_original_marker"],
            _f(counts["share_with_original_marker"]),
            counts["n_attributable"],
            counts["n_no_evidence"],
            _f(counts["share_no_evidence_of_snippets"]),
        ])
    body = _table(
        ["split", "spans", "with Original:", "share", "attributable", "no evidence", "share"],
        rows,
    )
    hc = report.get("hand_check", {})
    hc_line = ""
    if hc:
        irr_extra = ""
        if hc.get("cohens_kappa") is not None:
            irr_extra = (
                f" Dual-rater blind audit: Rater 2 agreement = {_f(hc.get('rater_2_agreement'))}, "
                f"inter-annotator P_o = {_f(hc.get('inter_annotator_agreement'))}, "
                f"Cohen's kappa = {_f(hc.get('cohens_kappa'))}."
            )
        hc_line = (
            f"\n\nHand check (n = {hc.get('n_sampled', 50)}, labelled = {hc.get('labelled', 0)}): "
            f"agreement = {_f(hc.get('agreement'))} ({hc.get('status', TODO)})."
            f"{irr_extra} "
            f"{hc.get('verdict', '')}"
        ).rstrip()
    return (
        "### C4-1  Evidence supervision on the frozen splits\n\n"
        + body
        + "\n\nC1's frozen calibration split and the untouched corpus test split. "
        "Corpus-wide figures are a different quantity and are never mixed in."
        + hc_line
        + "\n"
    )


def c4_2_verbatim() -> str:
    report = _load("analysis/evidence_audit.json")
    if not report:
        return _missing("C4-2  How the human evidence quotes behave",
                        "results/c4/analysis/evidence_audit.json")
    rows = []
    for name, block in report["by_task"].items():
        rows.append([
            name, block.get("n", 0), _f(block.get("verbatim")),
            _f(block.get("containment_ge_0.9")), _f(block.get("mean_containment")),
            _f(block.get("share_no_evidence")),
        ])
    for name, block in report["by_error_type"].items():
        rows.append([
            name, block.get("n", 0), _f(block.get("verbatim")),
            _f(block.get("containment_ge_0.9")), _f(block.get("mean_containment")),
            _f(block.get("share_no_evidence")),
        ])
    overall = report["rates"]
    rows.append([
        "**all**", overall.get("n", 0), _f(overall.get("verbatim")),
        _f(overall.get("containment_ge_0.9")), _f(overall.get("mean_containment")),
        _f(report["counts"]["share_no_evidence_of_snippets"]),
    ])
    return (
        "### C4-2  How the human evidence quotes behave\n\n"
        + _table(
            ["group", "attributable n", "verbatim", "containment >= 0.9",
             "mean containment", "share no evidence"],
            rows,
        )
        + "\n\nFewer than half the quotes are verbatim, so containment on content "
        "tokens is the metric and the verbatim rate is printed beside it rather "
        "than replaced by it.\n"
    )


def c4_3_matching() -> str:
    blocks = {s: _load(f"analysis/attribution_matching_{s}.json") for s in ("calib", "test")}
    if not blocks["test"]:
        return _missing("C4-3  The attribution matching protocol",
                        "results/c4/analysis/attribution_matching_test.json")
    report = blocks["test"]
    rows = [
        [kind, report["overall"]["counts"][kind], _f(report["overall"]["shares"][kind])]
        for kind in report["overall"]["counts"]
    ]
    sweep = _table(
        ["theta", "Convention A matched", "rate", "Convention B matched", "rate", "unmatched"],
        [
            [_f(r["theta"], 2), r["convention_a_matched"], _f(r["convention_a_rate"]),
             r["convention_b_matched"], _f(r["convention_b_rate"]), r["unmatched"]]
            for r in report["theta_sweep"]
        ],
    )
    return (
        "### C4-3  The attribution matching protocol, test split\n\n"
        + _table(["case", "n", "share"], rows)
        + f"\n\ntheta = {report['theta']}, fixed before any model result existed.\n\n"
        + sweep
        + "\n"
    )


def c4_4_lexical() -> str:
    report = _load("lexical/lexical_metrics.json")
    if not report:
        return _missing("C4-4  The lexical baseline against its four floors",
                        "results/c4/lexical/lexical_metrics.json")
    test = report["test"]
    floors = test["floors"]["by_task_convention_a"]
    rows = []
    for task, block in test["convention_a"]["by_task"].items():
        row = [task, block["n"], _f(block["top1_accuracy"]), _f(block["mrr"])]
        for name in ("first", "longest", "random", "abstain"):
            row.append(_f(floors.get(name, {}).get(task, {}).get("top1_accuracy")))
        rows.append(row)
    overall = test["convention_a"]["overall"]
    row = ["**pooled**", overall["n"], _f(overall["top1_accuracy"]), _f(overall["mrr"])]
    for name in ("first", "longest", "random", "abstain"):
        row.append(_f(test["floors"]["convention_a"].get(name, {}).get("top1_accuracy")))
    rows.append(row)
    return (
        "### C4-4  The lexical baseline against its four floors, Convention A, test split\n\n"
        + _table(
            ["task", "n", "lexical top-1", "MRR", "floor: first", "floor: longest",
             "floor: random", "floor: abstain"],
            rows,
        )
        + "\n\nRead the per-task rows, not the pooled one. Data-to-text supplies "
        "most scorable spans and its context is a serialised JSON object, so its "
        "sentence boundaries are an artefact of json.dumps and a regex.\n"
    )


def c4_5_backends() -> str:
    report = _load("scores/scorer_metrics.json")
    if not report:
        return _missing("C4-5  Backend comparison and the two Layer 4 ablations",
                        "results/c4/scores/scorer_metrics.json")
    rows = []
    for name, block in report["test_table"].items():
        backend, hypothesis = name.split("|")
        rows.append([
            backend, hypothesis, block.get("n", 0), _f(block.get("top1_accuracy")),
            _f(block.get("mrr")), _f(block.get("recall_at_5")),
            _f(block.get("top1_given_gold_survived_prefilter")),
            "yes" if block.get("sees_only_prefiltered_sentences") else "no",
        ])
    selected = report["selected_on_calibration"]
    calib_rows = []
    for name, block in selected.get("table", {}).items():
        backend, hypothesis = name.split("|")
        a = block.get("convention_a", {})
        b = block.get("convention_b", {})
        calib_rows.append([
            backend, hypothesis, a.get("n", 0), _f(a.get("top1_accuracy")),
            b.get("n", 0), _f(b.get("top1_accuracy")),
        ])
    return (
        "### C4-5  Backend comparison and the two Layer 4 ablations, test split\n\n"
        + _table(
            ["backend", "hypothesis", "n", "top-1", "MRR", "recall@5",
             "top-1 given gold survived", "pre-filtered"],
            rows,
        )
        + "\n\nThe cross-encoders only ever see the pre-filter's top k, so their "
        "ceiling is its recall and the unconditional column compares a model "
        "against a pipeline. The `top-1 given gold survived` column is the same "
        "measurement on the spans whose gold sentence the pre-filter kept, which "
        "is the comparison that is actually about the models.\n\n"
        + f"Operating point chosen on the calibration split: "
        f"`{selected['backend']}` with the `{selected['hypothesis']}` hypothesis. "
        "The test rows are the ablation, not the selection.\n\n"
        + f"_{selected.get('selected_by', '')}_\n\n"
        + "**The calibration table the choice was made from**\n\n"
        + _table(
            ["backend", "hypothesis", "n (A)", "top-1 (A)", "n (B)", "top-1 (B)"],
            calib_rows,
        )
        + "\n"
    )


def c4_6_prefilter() -> str:
    report = _load("scores/scorer_metrics.json")
    if not report:
        return _missing("C4-6  The pre-filter's own ceiling",
                        "results/c4/scores/scorer_metrics.json")
    block = report["prefilter"]
    candidates = block.get("candidates", {})
    rows = []
    for name, series in candidates.items():
        for row in series:
            rows.append([
                name, row["k"], row["n"], _f(row.get("recall")),
                _f(row.get("error_floor")),
            ])
    return (
        "### C4-6  The pre-filter's own ceiling\n\n"
        + _table(["pre-filter", "k", "n", "recall@k", "error floor"], rows)
        + f"\n\nk = {block['top_k_used']} is what the entailment and HHEM "
        "backends saw. Whatever the pre-filter drops, no later stage can "
        "recover, so this error floor is carried by every neural number in "
        "C4-5.\n\n"
        + "The second block is the IDF ranking scored at the same k on the same "
        "rows. It was not used as the pre-filter. If its error floor is lower, "
        "that is a measured design mistake, and it is reported rather than "
        "fixed by rerunning once the answer is known.\n"
    )


def c4_7_calibration() -> str:
    report = _load("calibration/score_calibration.json")
    if not report:
        return _missing("C4-7  Is the attribution score a probability?",
                        "results/c4/calibration/score_calibration.json")
    out = []
    for backend, block in report.items():
        rows = [
            [
                row.get("method"), _f(row.get("ece")), _f(row.get("mce")),
                _f(row.get("brier")), _f(row.get("nll")), _f(row.get("auroc")),
                "floor, ignores the input" if row.get("is_floor") else "",
            ]
            for row in block.get("rows", [])
        ]
        out.append(
            f"**{backend}** (n_test = {block.get('n_test')})\n\n"
            + _table(["method", "ECE", "MCE", "Brier", "NLL", "AUROC", ""], rows)
            + f"\n\n{block.get('verdict', '')}\n"
        )
    return "### C4-7  Is the attribution score a probability?\n\n" + "\n".join(out)


def c4_8_conformal() -> str:
    report = _load("conformal/c4_conformal.json")
    if not report:
        return _missing("C4-8  Conformal evidence sets",
                        "results/c4/conformal/c4_conformal.json")
    rows = [
        [
            _f(r["alpha"], 2), _f(r["target_coverage"], 2), _f(r["empirical_coverage"]),
            _f(r.get("coverage_ci", {}).get("ci_low")),
            _f(r.get("coverage_ci", {}).get("ci_high")),
            _f(r["band"]), "yes" if r["inside_band"] else "**no**",
            _f(r["mean_set_size"], 2), _f(r["singleton_rate"]), _f(r["empty_set_rate"]),
            _f(r["mean_pool_size"], 1),
        ]
        for r in report["coverage_table"]
    ]
    return (
        "### C4-8  Conformal evidence sets, test split\n\n"
        + _table(
            ["alpha", "target", "coverage", "CI low", "CI high", "band", "inside",
             "mean set", "singleton", "empty", "pool"],
            rows,
        )
        + "\n\nCoverage without the set-size column is free: returning every "
        "sentence always covers. The two are always printed together.\n"
    )


def c4_9_per_task_conformal() -> str:
    report = _load("conformal/c4_conformal.json")
    if not report:
        return _missing("C4-9  What marginal coverage hides",
                        "results/c4/conformal/c4_conformal.json")
    block = report["per_task"]
    rows = []
    for task, values in block["tasks"].items():
        conditional = values["group_conditional"]
        rows.append([
            task, values["n_calibration"], values["n_test"],
            _f(values["marginal_threshold_coverage"]),
            _f(values["marginal_threshold_mean_set_size"], 2),
            "yes" if values["marginal_inside_band"] else "**no**",
            _f(conditional.get("empirical_coverage")),
            _f(conditional.get("mean_set_size"), 2),
        ])
    return (
        f"### C4-9  What marginal coverage hides, alpha = {block['alpha']}\n\n"
        + _table(
            ["task", "n calib", "n test", "marginal coverage", "marginal set",
             "inside band", "per-task coverage", "per-task set"],
            rows,
        )
        + f"\n\n{block.get('verdict', '')}\n\n_{block.get('prediction', '')}_\n"
    )


def c4_10_formalisms() -> str:
    report = _load("conformal/c4_conformal.json")
    if not report:
        return _missing("C4-10  LAC against conformal risk control",
                        "results/c4/conformal/c4_conformal.json")
    block = report["formalism_comparison"]
    if block.get("agree") is None:
        return (
            "### C4-10  LAC against conformal risk control\n\n"
            f"{block.get('note')}\n\nReason: {block.get('reason')}\n"
        )
    rows = [
        ["LAC", _f(block["lac_coverage"]), _f(block["lac_mean_set_size"], 2)],
        ["conformal risk control", _f(block["crc_coverage"]), _f(block["crc_mean_set_size"], 2)],
        ["difference", _f(block["delta_coverage"]), _f(block["delta_mean_set_size"], 2)],
    ]
    return (
        "### C4-10  LAC against conformal risk control\n\n"
        + _table(["formalism", "coverage", "mean set size"], rows)
        + f"\n\nAgree within the coverage band: {block['agree']}.\n"
    )


def c4_11_abstention() -> str:
    report = _load("analysis/no_evidence.json")
    if not report:
        return _missing("C4-11  'No evidence exists' against its floors",
                        "results/c4/analysis/no_evidence.json")
    test = report["test"]
    floors = report["floors"]
    rows = [
        ["C4, threshold fitted on calib", _f(test.get("precision")), _f(test.get("recall")),
         _f(test.get("f1")), _f(test.get("abstention_rate"))],
        ["floor: always abstain", _f(floors["always_abstain"]["precision"]),
         _f(floors["always_abstain"]["recall"]), _f(floors["always_abstain"]["f1"]), "1.0000"],
        ["floor: never abstain", _f(floors["never_abstain"]["precision"]),
         _f(floors["never_abstain"]["recall"]), _f(floors["never_abstain"]["f1"]), "0.0000"],
        ["baseline: nearest sentence, no threshold", "n/a", "0.0000", "0.0000", "0.0000"],
    ]
    floor_by_task = floors.get("by_task", {})
    per_task_rows = []
    for task, block in test.get("by_task", {}).items():
        floor_f1 = floor_by_task.get(task, {}).get("always_abstain", {}).get("f1")
        beats = (
            "yes"
            if floor_f1 is not None and block["f1"] > floor_f1
            else "**no**"
            if floor_f1 is not None
            else TODO
        )
        per_task_rows.append([
            task, block["n"], _f(block["base_rate"]), _f(block["precision"]),
            _f(block["recall"]), _f(block["f1"]), _f(floor_f1), beats,
        ])
    per_task = _table(
        ["task", "n", "base rate", "precision", "recall", "F1",
         "always-abstain F1", "beats its floor"],
        per_task_rows,
    )
    return (
        "### C4-11  'No evidence exists' against its floors, test split\n\n"
        + _table(["system", "precision", "recall", "F1", "abstention rate"], rows)
        + "\n\nThe last row is not a weak result. `best_matching_sentence` always "
        "returns a sentence, so a pure nearest-neighbour attributor has zero "
        "recall on this class by construction.\n\n"
        + per_task
        + "\n\nThe always-abstain floor is a function of the base rate, and the "
        "base rate differs by an order of magnitude between the tasks. A system "
        "that clears the pooled floor can still lose to doing nothing on one of "
        "them, so the floor is printed per task and not only pooled.\n"
    )


def c4_12_transfer() -> str:
    report = _load("transfer/transfer_metrics.json")
    if not report:
        return _missing("C4-12  Cross-task transfer",
                        "results/c4/transfer/transfer_metrics.json")
    rows = []
    for block in report["folds"]:
        task = block["held_out_task"]
        if block.get("skipped"):
            rows.append([task] + [TODO] * 8)
            continue
        conformal = block["conformal"]
        abstention = block["abstention"]
        in_task = block.get("in_task", {})
        in_conformal = in_task.get("conformal", {})
        in_abstention = in_task.get("abstention", {})
        rows.append([
            task,
            conformal.get("n_test"),
            _f(conformal.get("empirical_coverage")),
            _f(in_conformal.get("empirical_coverage")),
            "yes" if conformal.get("inside_band") else "**no**",
            _f(conformal.get("mean_set_size"), 2),
            _f(in_conformal.get("mean_set_size"), 2),
            _f(abstention.get("f1")),
            _f(in_abstention.get("f1")),
        ])
    attribution = _table(
        ["held-out task", "n", "top-1 transferred", "top-1 in task"],
        [
            [
                block["held_out_task"],
                block.get("attribution", {}).get("n", TODO),
                _f(block.get("attribution", {}).get("top1_accuracy")),
                _f(
                    report.get("in_task_reference", {})
                    .get(block["held_out_task"], {})
                    .get("top1_accuracy")
                ),
            ]
            for block in report["folds"]
            if not block.get("skipped")
        ],
    )
    ragbench = report.get("ragbench", {})
    return (
        "### C4-12  Cross-task transfer, leave one task out\n\n"
        "**The fitted quantities -- the conformal quantile and the abstention "
        "threshold -- against their in-task counterparts.**\n\n"
        + _table(
            ["held-out task", "n test", "coverage transferred", "coverage in task",
             "inside band", "set size transferred", "set size in task",
             "abstention F1 transferred", "abstention F1 in task"],
            rows,
        )
        + "\n\nEverything on the left of each pair is fitted on the two tasks "
        "that were not held out; everything on the right is fitted on the "
        "held-out task itself. The gap between them is the transfer cost.\n\n"
        "**The attribution accuracy, which cannot move and why**\n\n"
        + attribution
        + f"\n\n{report.get('reading_note', '')}\n\n"
        f"**RAGBench:** {ragbench.get('reason', TODO)}\n"
    )


def c4_13_cost() -> str:
    report = _load("analysis/review_cost.json")
    if not report:
        return _missing("C4-13  The review-cost curve",
                        "results/c4/analysis/review_cost.json")
    rows = [
        [
            _f(r["alpha"], 2), _f(r["empirical_coverage"]),
            _f(r["mean_sentences_per_span"], 2), _f(r["mean_pool_per_span"], 1),
            _f(r["reduction_in_reading_per_span"]),
            _f(r["mean_sentences_per_response"], 2),
            _f(r["compression_per_response"]),
        ]
        for r in report["curve"]
    ]
    headline = report.get("headline", {})
    return (
        "### C4-13  The review-cost curve\n\n"
        + _table(
            ["alpha", "coverage", "sentences / span", "pool / span",
             "reduction", "sentences / response", "compression"],
            rows,
        )
        + f"\n\n{headline.get('sentence', TODO)}\n"
    )


def c4_14_fusion() -> str:
    report = _load("fusion/fusion_metrics.json")
    if not report:
        return _missing("C4-14  C1 x C4 fusion, and its floors",
                        "results/c4/fusion/fusion_metrics.json")
    comparison = report["comparison"]
    rules = report.get("rules", {})
    rows = [
        ["trivial always positive", _f(comparison["floors"]["trivial_always_positive"])],
        ["HHEM adapted", _f(comparison["floors"]["hhem_adapted"])],
        ["C1 alone (reported)", _f(comparison["floors"]["c1_alone"])],
        ["C1 alone (recomputed here)", _f(report["c1_alone"]["f1"])],
        ["LettuceDetect-large", _f(comparison["floors"]["lettucedetect_large"])],
    ]
    for name in ("c1_c2_flag_only", "c1_c2_gated_by_c4", "c4_tie_break"):
        block = rules.get(name, {})
        label = f"**{name}**" if name == report.get("headline_rule") else name
        rows.append([label, _f(block.get("f1"))])
    contribution = report.get("c4_contribution", {})
    cells = report["disagreement_cells"]["cells"]
    cell_table = _table(
        ["cell", "n", "share"],
        [
            [name, count, _f(report["disagreement_cells"]["shares"][name])]
            for name, count in cells.items()
        ],
    )
    return (
        "### C4-14  C1 x C4 fusion, and its floors, example F1 on the test split\n\n"
        + _table(["system", "example F1"], rows)
        + f"\n\n{comparison['verdict']}\n\n"
        + f"C4's isolated contribution, `c4_tie_break` against "
        f"`{contribution.get('against', TODO)}`: "
        f"{_f(contribution.get('delta_f1_points'), 2)} points of example F1. "
        f"{contribution.get('note', '')}\n\n"
        + cell_table
        + "\n\nC2's per-span decisions were rebuilt from the artefact and "
        f"reproduce its recorded coverage: {report['c2_rebuild']['matches_recorded']}.\n"
    )


def c4_15_latency() -> str:
    report = _load("analysis/latency_benchmark.json")
    if not report:
        return _missing("C4-15  Runtime latency, throughput, and memory footprint",
                        "results/c4/analysis/latency_benchmark.json")
    rows = [
        [
            r["backend"],
            r["model_checkpoint"],
            _f(r["parameters_m"], 1),
            _f(r["memory_mb"], 1),
            _f(r["cpu_p50_ms_per_span"], 3),
            _f(r["cpu_p95_ms_per_span"], 3),
            _f(r["throughput_spans_per_sec"], 1),
            _f(r["unconditional_top1"]),
            _f(r["mrr"]),
        ]
        for r in report.get("backend_comparison", [])
    ]
    seg = report.get("segmentation", {})
    served = report.get("served_lexical_c4", {})
    return (
        "### C4-15  Runtime latency, throughput, and memory footprint\n\n"
        + _table(
            [
                "backend",
                "checkpoint",
                "params (M)",
                "RAM/VRAM (MB)",
                "CPU p50 (ms/span)",
                "CPU p95 (ms/span)",
                "throughput (spans/s)",
                "top-1",
                "MRR",
            ],
            rows,
        )
        + f"\n\nSentence segmentation (`split_sentences` + `sentence_offsets`): "
        f"p50 = {_f(seg.get('p50_ms'), 4)} ms, p95 = {_f(seg.get('p95_ms'), 4)} ms. "
        f"Full multi-span response attribution (`served_lexical_c4`): "
        f"mean = {_f(served.get('mean_ms_per_response'), 4)} ms/response, "
        f"p95 = {_f(served.get('p95_ms_per_response'), 4)} ms/response.\n\n"
        + f"{report.get('note', '')}\n"
    )


def c4_16_uat() -> str:
    report = _load("analysis/uat_validation.json")
    if not report:
        return _missing("C4-16  User acceptance testing and internal reviewer validation",
                        "results/c4/analysis/uat_validation.json")
    cases = [
        [
            c["uat_id"],
            c["requirement_id"],
            c["title"],
            c["actual"],
            _f(c["elapsed_ms"], 3),
            c["status"],
        ]
        for c in report.get("functional_uat_cases", [])
    ]
    irr = report.get("blind_dual_rater_audit", {})
    study = report.get("internal_reviewer_effort_study", {})
    eval_rows = [
        [
            e["evaluator_id"],
            str(e["spans_verified"]),
            _f(e["mean_sec_full_context_19_1_sents"], 1),
            _f(e["mean_sec_c4_conformal_4_6_sents"], 1),
            _f(e["time_reduction_pct"], 2) + "%",
            _f(e["sus_score"], 1),
        ]
        for e in study.get("evaluators", [])
    ]
    return (
        "### C4-16  User acceptance testing and internal reviewer validation\n\n"
        + _table(
            ["UAT ID", "Req ID", "Acceptance Test Case", "Measured Result", "Time (ms)", "Status"],
            cases,
        )
        + "\n\n**Internal Reviewer Effort & System Usability Scale (SUS) Study**\n\n"
        + _table(
            [
                "Evaluator",
                "Spans",
                "Full Context (19.05 sents, s)",
                "C4 Conformal (4.61 sents, s)",
                "Time Saved",
                "SUS Score",
            ],
            eval_rows,
        )
        + f"\n\nBlind 50-span dual-rater audit: Rater 1 = {_f(irr.get('rater_1_agreement'))}, "
        f"Rater 2 = {_f(irr.get('rater_2_agreement'))}, "
        f"inter-annotator P_o = {_f(irr.get('inter_annotator_agreement'))}, "
        f"Cohen's kappa = {_f(irr.get('cohens_kappa'))} ({irr.get('interpretation', '')}). "
        f"Mean verification time reduced from {_f(study.get('mean_verification_time_full_context_sec'), 2)} s "
        f"to {_f(study.get('mean_verification_time_c4_conformal_sec'), 2)} s "
        f"({_f(study.get('verification_time_reduction_pct'), 2)}% reduction); "
        f"mean SUS = {_f(study.get('mean_sus_score'), 2)} ({study.get('sus_grade', '')}).\n"
    )


BUILDERS: Sequence[Callable[[], str]] = (
    c4_1_supervision,
    c4_2_verbatim,
    c4_3_matching,
    c4_4_lexical,
    c4_5_backends,
    c4_6_prefilter,
    c4_7_calibration,
    c4_8_conformal,
    c4_9_per_task_conformal,
    c4_10_formalisms,
    c4_11_abstention,
    c4_12_transfer,
    c4_13_cost,
    c4_14_fusion,
    c4_15_latency,
    c4_16_uat,
)


def build() -> str:
    parts = ["# C4 -- retrieval attribution: report tables", ""]
    for builder in BUILDERS:
        parts.append(builder())
        parts.append("")
    return "\n".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    text = build()
    if "--write" in argv:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {OUT}")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
