"""The C3 driver: one command that produces every C3 artefact from frozen inputs.

Order matters and is enforced here rather than left to a runbook:

    1  Layer 1  audit the label                     labels_audit
    2  Layer 2  give predicted spans a type         typed_matching  (calib, test)
    3  Layer 3  fit and score the type model        type_baseline
    4  Layer 4  calibrate it and wrap it in sets    type_conformal
    5  Layer 5  explain, and score the explanation  explanation_eval
    6  Layer 6  count the escalation budget         addons
    7  add-ons  the analyses the type unlocks       addons

Steps 1 to 5 have their own command line entry points and are run by
scripts/run_c3.ps1. This module does step 6 and 7, which need every earlier
artefact at once, and it is also where the conformal type set is applied to
EVERY predicted span rather than only the typeable ones. That distinction
matters: typeable spans are the ones with a gold type to score against, but in
production nothing tells you which spans those are, so the escalation budget has
to be counted over all of them.

Nothing under src/c1_detector/ or src/c2_calibration/ is modified. C2's
calibrator, conformal class, risk-control threshold search and probability
reader are imported and used as they are.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.c2_calibration.run_c2 import read_probability_file
from src.c3_explanation.addons import (
    escalation_budget,
    generator_cross_tab,
    implicit_true_ablation,
    rebuild_c2_decisions,
    severity_weighted_risk_control,
    type_conditional_calibration,
    type_conditional_detectability,
)
from src.c3_explanation.grounding import span_features, split_sentences
from src.c3_explanation.type_baseline import (
    ERROR_TYPES,
    axis_product,
    feature_vector,
    predict_proba,
    read_jsonl,
    reorder,
)
from src.c3_explanation.type_conformal import (
    prediction_sets,
    set_shape,
    temperature_scale,
)

SpanKey = Tuple[str, int, int]


def gold_span_table(
    processed: Sequence[Dict[str, Any]]
) -> Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]:
    """response id -> (start, end) -> the gold span's type and confound flags."""
    out: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}
    for record in processed:
        table: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for span in record.get("spans") or []:
            types = [str(t) for t in (span.get("error_types") or []) if t]
            if not types and span.get("error_type"):
                types = [str(span["error_type"])]
            table[(int(span["start"]), int(span["end"]))] = {
                "error_type": sorted(types)[0] if types else None,
                "error_types": types,
                "implicit_true": bool(span.get("implicit_true", False)),
                "due_to_null": bool(span.get("due_to_null", False)),
            }
        out[str(record["id"])] = table
    return out


def score_every_predicted_span(
    typed_predictions: Sequence[Dict[str, Any]],
    processed: Sequence[Dict[str, Any]],
    models: Dict[str, Any],
    probs_key: str = "probs_flat",
) -> Tuple[List[SpanKey], np.ndarray, List[Dict[str, Any]]]:
    """Type probabilities for every predicted span, typeable or not.

    The scored evaluation in Layer 3 covers only spans with a gold type to check
    against. Serving has no such filter, so the escalation budget and the set
    sizes reported here are computed over all of them.
    """
    by_id = {str(r["id"]): r for r in processed}
    keys: List[SpanKey] = []
    rows: List[Dict[str, Any]] = []
    vectors: List[List[float]] = []
    for row in typed_predictions:
        record = by_id.get(str(row["response_id"]))
        if record is None:
            continue
        answer = record.get("answer", "")
        start, end = int(row["start"]), int(row["end"])
        features = span_features(
            answer[start:end],
            answer,
            start,
            split_sentences(record.get("context", "")),
            str(record.get("task_type", "other")),
        )
        keys.append((str(row["response_id"]), start, end))
        vectors.append(feature_vector(features))
        rows.append(
            {
                "response_id": str(row["response_id"]),
                "task_type": str(record.get("task_type", "other")),
                "start": start,
                "end": end,
                "typeable": bool(row.get("typeable")),
                "bucket": row.get("bucket"),
            }
        )

    if not vectors:
        return ([], np.empty((0, len(ERROR_TYPES))), [])

    features_matrix = np.asarray(vectors, dtype=np.float64)
    if probs_key == "probs_axis_product":
        probs = axis_product(
            predict_proba(models["mechanism"], features_matrix),
            models["mechanism"]["classes"],
            predict_proba(models["severity"], features_matrix),
            models["severity"]["classes"],
        )
    else:
        probs = reorder(
            predict_proba(models["flat"], features_matrix),
            models["flat"]["classes"],
            ERROR_TYPES,
        )
    return (keys, probs, rows)


def set_sizes_at_alphas(
    keys: Sequence[SpanKey],
    probs: np.ndarray,
    conformal_report: Dict[str, Any],
) -> Tuple[Dict[float, Dict[SpanKey, int]], Dict[str, Any]]:
    """Apply Layer 4's fitted temperature and thresholds to every predicted span."""
    temperature = float(conformal_report["temperature"])
    classes = list(conformal_report["classes"])
    scaled = temperature_scale(probs, temperature) if probs.size else probs

    by_alpha: Dict[float, Dict[SpanKey, int]] = {}
    shape_counts: Dict[str, Any] = {}
    for row in conformal_report["coverage"]:
        alpha = float(row["alpha"])
        sets = prediction_sets(scaled, float(row["threshold"])) if scaled.size else []
        by_alpha[alpha] = {key: len(s) for key, s in zip(keys, sets)}
        shapes = Counter(set_shape(s, classes) for s in sets)
        shape_counts[str(alpha)] = {
            "n": len(sets),
            "average_set_size": (
                float(np.mean([len(s) for s in sets])) if sets else float("nan")
            ),
            "singleton_rate": (
                float(np.mean([1.0 if len(s) == 1 else 0.0 for s in sets]))
                if sets
                else float("nan")
            ),
            "shapes": {k: shapes[k] for k in sorted(shapes)},
        }
    return by_alpha, shape_counts


def build(
    paths: Dict[str, Path], probs_key: str = "probs_flat", alpha: float = 0.1
) -> Dict[str, Any]:
    processed_test = read_jsonl(paths["processed_test"])
    processed_train = read_jsonl(paths["processed_train"])
    typed_test = read_jsonl(paths["typed_test"])
    models = json.loads(paths["baseline_model"].read_text(encoding="utf-8"))
    conformal_report = json.loads(paths["conformal"].read_text(encoding="utf-8"))
    artifact = json.loads(paths["c2_artifact"].read_text(encoding="utf-8"))

    calib_records = read_probability_file(paths["c1_calib_probs"])
    test_records = read_probability_file(paths["c1_test_probs"])

    c2 = rebuild_c2_decisions(artifact, calib_records, test_records, alpha=alpha)
    if not c2["matches_artifact"]:
        raise RuntimeError(
            "the rebuilt C2 decisions do not reproduce the coverage recorded in "
            f"c2_artifact.json ({c2['coverage_rebuilt']} against "
            f"{c2['coverage_in_artifact']}). Stopping rather than reporting a "
            "number C2 would not recognise."
        )

    keys, probs, _ = score_every_predicted_span(
        typed_test, processed_test, models, probs_key
    )
    sizes_by_alpha, shape_counts = set_sizes_at_alphas(keys, probs, conformal_report)

    test_gold = gold_span_table(processed_test)
    train_gold = gold_span_table(processed_train)

    test_scores = {
        (r["response_id"], r["start"], r["end"]): r["calibrated_score"]
        for r in c2["rows"]
    }
    calib_c2 = rebuild_c2_decisions(artifact, calib_records, calib_records, alpha=alpha)
    calib_scores = {
        (r["response_id"], r["start"], r["end"]): r["calibrated_score"]
        for r in calib_c2["rows"]
    }

    headline_sizes = sizes_by_alpha.get(alpha, {})
    typed_rows = read_jsonl(paths["baseline_probs_predicted_test"])

    gold_rows_all: List[Dict[str, Any]] = []
    responses_by_model: Counter = Counter()
    for record in processed_train + processed_test:
        responses_by_model[str(record.get("model", "unknown"))] += 1
        for span in record.get("spans") or []:
            types = [str(t) for t in (span.get("error_types") or []) if t]
            if types:
                gold_rows_all.append(
                    {
                        "model": str(record.get("model", "unknown")),
                        "error_type": sorted(types)[0],
                    }
                )

    return {
        "alpha": alpha,
        "probs_key": probs_key,
        "c2_rebuild": {
            key: c2[key]
            for key in (
                "alpha",
                "n_calibration",
                "n_test",
                "coverage_rebuilt",
                "coverage_in_artifact",
                "matches_artifact",
                "decision_counts",
            )
        },
        "layer_6_escalation_budget": escalation_budget(
            c2["rows"], headline_sizes, set_sizes_by_alpha=sizes_by_alpha
        ),
        "set_sizes_over_all_predicted_spans": shape_counts,
        "addon_1_type_conditional_calibration": type_conditional_calibration(
            typed_rows, probs_key, float(conformal_report["temperature"])
        ),
        "addon_2_severity_weighted_risk_control": severity_weighted_risk_control(
            calib_records,
            test_records,
            calib_scores,
            test_scores,
            train_gold,
            test_gold,
        ),
        "addon_3_generator_by_error_type": generator_cross_tab(
            gold_rows_all, responses_by_model
        ),
        "addon_4_type_conditional_detectability": type_conditional_detectability(
            test_records, test_gold, headline_sizes
        ),
        "addon_5_implicit_true_ablation": implicit_true_ablation(
            test_records, test_gold, "implicit_true"
        ),
        "addon_9_due_to_null_ablation": implicit_true_ablation(
            test_records, test_gold, "due_to_null"
        ),
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = ["=== C3 Layer 6 and the add-ons ==="]
    c2 = report["c2_rebuild"]
    lines.append(
        f"C2 decisions rebuilt from its own artefact: coverage "
        f"{c2['coverage_rebuilt']:.10f} against {c2['coverage_in_artifact']:.10f} "
        f"recorded -- {'match' if c2['matches_artifact'] else 'MISMATCH'}"
    )
    lines.append(f"  decisions: {c2['decision_counts']}")

    budget = report["layer_6_escalation_budget"]
    lines += ["", "--- C3-9 Layer 6: the escalation budget ---"]
    lines.append(f"  {budget['rule']}")
    lines.append(
        f"  {budget['n_abstained']:,} of {budget['n_spans']:,} predicted spans "
        f"abstain ({budget['abstention_rate']:.4f})"
    )
    lines.append(f"  {'alpha':>7}{'escalated':>12}{'of all spans':>15}{'of abstained':>15}")
    for alpha, cell in budget["per_alpha"].items():
        lines.append(
            f"  {alpha:>7}{cell['n_escalated']:>12,}"
            f"{cell['share_of_all_spans']:>15.4f}{cell['share_of_abstained']:>15.4f}"
        )
    lines.append(f"  {budget['loop_status']}")

    lines += ["", "--- set size over every predicted span, not only typeable ones ---"]
    for alpha, cell in report["set_sizes_over_all_predicted_spans"].items():
        lines.append(
            f"  alpha {alpha:<6} n={cell['n']:,}  average size "
            f"{cell['average_set_size']:.4f}  singleton {cell['singleton_rate']:.4f}"
        )

    cal = report["addon_1_type_conditional_calibration"]
    if cal:
        lines += ["", "--- add-on 1: type-conditional calibration ---"]
        lines.append(
            f"  overall ECE {cal['overall']['ece']:.4f} "
            f"(mean confidence {cal['overall']['mean_confidence']:.4f}, "
            f"accuracy {cal['overall']['accuracy']:.4f})"
        )
        for name, cell in cal["by_error_type"].items():
            lines.append(
                f"    {name:<24} n={cell['n']:>6,}  ECE {cell['ece']:.4f}  "
                f"confidence {cell['mean_confidence']:.4f}  "
                f"accuracy {cell['accuracy']:.4f}"
            )

    risk = report["addon_2_severity_weighted_risk_control"]
    lines += ["", "--- C3-6 add-on 2: severity-weighted risk control ---"]
    lines.append(f"  loss: {risk['loss']}")
    lines.append(
        f"  weights: evident {risk['weights']['evident']}, "
        f"subtle {risk['weights']['subtle']}, "
        f"implicit_true {risk['weights']['implicit_true']} -- {risk['weights_note']}"
    )
    lines.append(
        f"  {'alpha':>7}{'threshold':>12}{'test risk':>12}{'held':>7}"
        f"{'flag rate':>12}{'spans flagged':>15}"
    )
    for row in risk["rows"]:
        if not row.get("feasible"):
            lines.append(f"  {row['alpha']:>7.2f}   not reachable: {row.get('reason','')[:80]}")
            continue
        lines.append(
            f"  {row['alpha']:>7.2f}{row['threshold']:>12.4f}"
            f"{row['test_risk']:>12.4f}{'  yes' if row['bound_held'] else '   NO':>7}"
            f"{row['span_flag_rate']:>12.4f}{row['n_spans_flagged']:>15,}"
        )

    gen = report["addon_3_generator_by_error_type"]
    lines += ["", "--- C3-8 add-on 3: generator model against error type ---"]
    for model, cell in gen["by_model"].items():
        mix = "  ".join(f"{k.replace('_info','')}={v:.3f}" for k, v in cell["rates"].items())
        lines.append(
            f"  {model:<22} spans {cell['n_spans']:>6,}  per response "
            f"{cell['spans_per_response']:.3f}   {mix}"
        )

    det = report["addon_4_type_conditional_detectability"]
    lines += ["", "--- add-on 4: type-conditional detectability ---"]
    for name, cell in det["by_error_type"].items():
        lines.append(
            f"  {name:<24} gold {cell['n_gold']:>6,}  overlap recall "
            f"{cell['overlap_recall']:.4f}  mean set size "
            f"{cell['mean_conformal_set_size']:.4f}"
        )

    for key, title in (
        ("addon_5_implicit_true_ablation", "add-on 5: the implicit_true confound"),
        ("addon_9_due_to_null_ablation", "add-on 9: the due_to_null artefact"),
    ):
        ab = report[key]
        lines += ["", f"--- {title} ---"]
        lines.append(
            f"  {ab['n_spans_flagged']:,} spans flagged; "
            f"{ab['n_responses_that_change_label']:,} responses change label"
        )
        lines.append(
            f"  example F1 before {ab['example_before']['f1']:.4f}  "
            f"after {ab['example_after']['f1']:.4f}  "
            f"delta {ab['delta_f1']:+.4f}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="C3 Layer 6 and the add-ons, from the frozen artefacts."
    )
    parser.add_argument("--processed-test", default="data/processed/ragtruth_test.jsonl")
    parser.add_argument("--processed-train", default="data/processed/ragtruth_train.jsonl")
    parser.add_argument(
        "--typed-test", default="results/c3/analysis/typed_predictions_test.jsonl"
    )
    parser.add_argument("--baseline-model", default="results/c3/type/baseline_model.json")
    parser.add_argument(
        "--baseline-probs-predicted-test",
        default="results/c3/type/baseline_probs_predicted_test.jsonl",
    )
    parser.add_argument("--conformal", default="results/c3/type/c3_conformal.json")
    parser.add_argument("--c2-artifact", default="results/c2/c1/c2_artifact.json")
    parser.add_argument("--c1-calib-probs", default="results/c1/calib/probabilities.jsonl")
    parser.add_argument("--c1-test-probs", default="results/c1/test/probabilities.jsonl")
    parser.add_argument("--out-dir", default="results/c3/analysis")
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args(argv)

    paths = {
        "processed_test": Path(args.processed_test),
        "processed_train": Path(args.processed_train),
        "typed_test": Path(args.typed_test),
        "baseline_model": Path(args.baseline_model),
        "baseline_probs_predicted_test": Path(args.baseline_probs_predicted_test),
        "conformal": Path(args.conformal),
        "c2_artifact": Path(args.c2_artifact),
        "c1_calib_probs": Path(args.c1_calib_probs),
        "c1_test_probs": Path(args.c1_test_probs),
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print("missing inputs, run the earlier layers first:")
        for path in missing:
            print(f"  {path}")
        return 1

    report = build(paths, alpha=args.alpha)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "addons.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(format_report(report))
    print()
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
