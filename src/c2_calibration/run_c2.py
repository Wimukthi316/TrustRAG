"""C2 end to end: read detector probabilities, calibrate, then conformalise.

Input is the `probabilities.jsonl` written by either `evaluate_c1.py` (our own
detector) or `lettucedetect_adapter.py` (the public baseline). They emit the same
schema on purpose, so swapping the detector is a change of path and nothing
else. That is what lets C2 be built and validated while C1 is still training.

Two calibration splits are required and they must be disjoint from each other
and from anything the detector trained on:

    --calib   fits the calibrator and the conformal threshold
    --test    every number that gets reported

Run:
    python -m src.c2_calibration.run_c2 \\
        --calib results/lettucedetect/calib/probabilities.jsonl \\
        --test  results/lettucedetect/test/probabilities.jsonl \\
        --out-dir results/c2/lettucedetect

The unit of analysis matters and is a flag:

    --unit span    one row per span the detector proposed, labelled by whether
                   it overlaps a gold span. This is the level the novelty claim
                   is made at. Its honest limitation: it can only calibrate
                   spans the detector actually proposed, so it says nothing
                   about hallucinations the detector missed entirely. Precision
                   is calibrated; recall is C1's problem.
    --unit token   one row per answer token, labelled from the gold spans. Far
                   more rows and no selection effect, at the cost of no longer
                   being a statement about spans.

Both are reported. They answer different questions and a reviewer will ask about
whichever one is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.c2_calibration.calibration import (
    calibration_report,
    compare_calibrators,
    format_calibration_table,
    reliability_bins,
)
from src.common.schema import SCHEMA_VERSION
from src.c2_calibration.conformal import (
    area_under_risk_coverage,
    check_coverage,
    coverage_table,
    coverage_tolerance,
    format_coverage_table,
    group_conditional_coverage,
    minimum_calibration_size,
    risk_coverage_curve,
)


def halve_by_response(
    records: Sequence[Dict[str, Any]], seed: int = 42, fraction: float = 0.5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split one probability file into calibration and evaluation halves.

    Needed whenever the detector was trained on everything except this file --
    which is the situation for the public LettuceDetect checkpoint, because it
    trained on the whole RAGTruth train split. Carving the calibration set out
    of train there puts the detector in-sample on calibration and out-of-sample
    on test, the two sets are no longer exchangeable, and split conformal
    correctly refuses to cover. Measured on this repository, 2026-08-11:

        LettuceDetect on the train-derived calibration split   example F1 0.9267
        LettuceDetect on the official RAGTruth test split      example F1 0.7918

    A 13-point gap is not sampling noise, it is memorisation, and calibrating
    on it produces a threshold that is far too tight for test data.

    Splitting is by response, never by span, so every span of a response lands
    on the same side. Splitting by span would let two spans from one answer
    straddle the boundary, and they are plainly not independent.

    Our own C1 does not need this: train_c1.py already holds its calibration
    split out of training, so that file is legitimately out-of-sample.
    """
    import random

    ordered = sorted(records, key=lambda r: str(r.get("id")))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    cut = int(round(len(ordered) * fraction))
    return ordered[:cut], ordered[cut:]


def read_probability_file(path: Path | str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def span_units(
    records: Sequence[Dict[str, Any]],
    score_key: str = "mean_prob",
    group_key: str = "task_type",
) -> Tuple[List[float], List[int], List[str]]:
    """One row per predicted span: (score, is_hallucinated, group).

    `group_key` names the field group-conditional coverage is broken down by.
    On RAGTruth that is `task_type`, the three tasks the corpus is built from.
    On RAGBench every record carries the same task_type but a different
    `subset`, and the twelve subsets differ by an order of magnitude in
    hallucination rate, so a coverage average across them hides exactly what the
    shift study is asking about. A missing field reads as "unknown" rather than
    raising, so a dump written before the field existed still analyses.
    """
    scores: List[float] = []
    labels: List[int] = []
    groups: List[str] = []
    for record in records:
        for span in record.get("pred_spans", []):
            scores.append(float(span[score_key]))
            labels.append(int(bool(span["is_hallucinated"])))
            groups.append(str(record.get(group_key, "unknown")))
    return scores, labels, groups


def token_units(
    records: Sequence[Dict[str, Any]],
    group_key: str = "task_type",
) -> Tuple[List[float], List[int], List[str]]:
    """One row per answer token, labelled by overlap with a gold span.

    The gold token label is recomputed here from `gold_spans` and
    `answer_offsets` rather than stored, so it always agrees with the offsets
    the model actually saw.
    """
    scores: List[float] = []
    labels: List[int] = []
    groups: List[str] = []
    for record in records:
        gold = [(g["start"], g["end"]) for g in record.get("gold_spans", [])]
        task = str(record.get(group_key, "unknown"))
        for prob, offset in zip(record["token_probs"], record["answer_offsets"]):
            start, end = offset
            scores.append(float(prob))
            labels.append(
                int(any(gs < end and start < ge for gs, ge in gold))
            )
            groups.append(task)
    return scores, labels, groups


def analyse(
    calib: Sequence[Dict[str, Any]],
    test: Sequence[Dict[str, Any]],
    unit: str,
    score_key: str,
    alphas: Sequence[float],
    group_key: str = "task_type",
) -> Dict[str, Any]:
    extract = (
        (lambda rows: span_units(rows, score_key, group_key))
        if unit == "span"
        else (lambda rows: token_units(rows, group_key))
    )
    calib_scores, calib_labels, calib_groups = extract(calib)
    test_scores, test_labels, test_groups = extract(test)

    if not calib_scores or not test_scores:
        raise SystemExit(
            f"unit={unit} produced no rows (calib {len(calib_scores)}, "
            f"test {len(test_scores)}). With --unit span this means the detector "
            "predicted no spans at all, so there is nothing to calibrate."
        )

    calibration_rows = compare_calibrators(
        calib_scores, calib_labels, test_scores, test_labels
    )
    winner = min(calibration_rows, key=lambda r: r["ece"])

    # Refit the winning calibrator and pass calibrated scores to the conformal
    # layer. Note what this does and does not buy: temperature and Platt scaling
    # are strictly monotone, and LAC thresholds the score, so the conformal
    # result is bit-identical either way. Calibration is what makes the number
    # shown to a human mean what it says; conformal is what carries the
    # guarantee. The two are complementary, not sequential, and the report
    # should not claim otherwise. Only isotonic regression, which is weakly
    # monotone and can create ties, moves the conformal numbers at all.
    from src.c2_calibration.calibration import all_calibrators

    calibrator = next(c for c in all_calibrators() if c.name == winner["method"])
    calibrator.fit(calib_scores, calib_labels)
    calib_cal = list(calibrator.transform(calib_scores))
    test_cal = list(calibrator.transform(test_scores))

    coverage_rows = coverage_table(
        calib_cal, calib_labels, test_cal, test_labels, alphas=alphas
    )
    curve_raw = risk_coverage_curve(test_scores, test_labels)
    curve_cal = risk_coverage_curve(test_cal, test_labels)

    return {
        "unit": unit,
        "score_key": score_key if unit == "span" else "token_prob",
        "group_key": group_key,
        "n_calibration": len(calib_scores),
        "n_test": len(test_scores),
        "positive_rate_calibration": sum(calib_labels) / len(calib_labels),
        "positive_rate_test": sum(test_labels) / len(test_labels),
        "calibration": {
            "rows": calibration_rows,
            "selected": winner["method"],
            "selected_params": winner["params"],
            "before": calibration_report(test_scores, test_labels),
            "after": calibration_report(test_cal, test_labels),
        },
        "conformal": {
            "coverage": coverage_rows,
            "violations": check_coverage(coverage_rows),
            "group_conditional_alpha_0.1": group_conditional_coverage(
                calib_cal,
                calib_labels,
                calib_groups,
                test_cal,
                test_labels,
                test_groups,
                alpha=0.1,
            ),
            "min_calibration_size": {
                str(a): minimum_calibration_size(a) for a in alphas
            },
        },
        "risk_coverage": {
            "uncalibrated": curve_raw,
            "calibrated": curve_cal,
            "aurc_uncalibrated": area_under_risk_coverage(curve_raw),
            "aurc_calibrated": area_under_risk_coverage(curve_cal),
        },
        "reliability_bins_after": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_score": b.mean_score,
                "positive_rate": b.positive_rate,
            }
            for b in reliability_bins(test_cal, test_labels)
        ],
    }


def build_serving_artifact(
    calib: Sequence[Dict[str, Any]],
    result: Dict[str, Any],
    score_key: str,
    detector: str,
) -> Dict[str, Any]:
    """Everything the FastAPI backend needs to reproduce this C2 layer at request time.

    Two pieces:

    * the fitted calibrator, as JSON rather than a pickle;
    * the sorted non-conformity scores from the calibration split.

    Storing the raw scores rather than a table of thresholds means the server can
    answer *any* alpha the user moves the slider to, exactly, by recomputing the
    ceil((n+1)(1-alpha)) quantile. A precomputed table would either restrict the
    slider to the alphas that happened to be in it, or invite interpolation
    between thresholds -- and interpolating a conformal quantile silently voids
    the finite-sample guarantee it exists to provide.

    Span unit only. The demo highlights spans, and the token-level score array
    runs to a couple of hundred thousand floats, which is a lot of JSON for
    something nothing serves.
    """
    calib_scores, calib_labels, _ = span_units(calib, score_key)
    calibrator_name = result["calibration"]["selected"]

    from src.c2_calibration.calibration import all_calibrators

    calibrator = next(c for c in all_calibrators() if c.name == calibrator_name)
    calibrator.fit(calib_scores, calib_labels)
    calibrated = calibrator.transform(calib_scores)

    nonconformity = [
        float(1.0 - p) if y == 1 else float(p)
        for p, y in zip(calibrated, calib_labels)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "detector": detector,
        "unit": "span",
        "score_key": score_key,
        "calibrator": calibrator.to_dict(),
        "n_calibration": len(nonconformity),
        "nonconformity_sorted": sorted(nonconformity),
        # Recorded so a stale artifact is obvious rather than silently wrong.
        "test_ece_before": result["calibration"]["before"]["ece"],
        "test_ece_after": result["calibration"]["after"]["ece"],
        "coverage_at_alpha_0.1": next(
            (
                row["empirical_coverage"]
                for row in result["conformal"]["coverage"]
                if abs(row["alpha"] - 0.1) < 1e-9
            ),
            None,
        ),
    }


def format_analysis(result: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"\n{'=' * 74}")
    lines.append(f"UNIT: {result['unit']}  (score = {result['score_key']})")
    lines.append(
        f"  calibration rows {result['n_calibration']:,} "
        f"(positive {result['positive_rate_calibration']:.4f})  |  "
        f"test rows {result['n_test']:,} "
        f"(positive {result['positive_rate_test']:.4f})"
    )
    lines.append(f"{'=' * 74}")

    lines.append("\nCALIBRATION -- fitted on calib, all numbers measured on test")
    lines.append(format_calibration_table(result["calibration"]["rows"]))
    before = result["calibration"]["before"]
    after = result["calibration"]["after"]
    lines.append(
        f"\n  selected: {result['calibration']['selected']} "
        f"{result['calibration']['selected_params']}"
    )
    reduction = (
        (before["ece"] - after["ece"]) / before["ece"] * 100 if before["ece"] else 0.0
    )
    lines.append(
        f"  ECE {before['ece']:.4f} -> {after['ece']:.4f}  "
        f"({reduction:+.1f}%)   Brier {before['brier']:.4f} -> {after['brier']:.4f}"
    )

    lines.append("\nSPLIT CONFORMAL -- coverage vs alpha")
    lines.append(format_coverage_table(result["conformal"]["coverage"]))
    violations = result["conformal"]["violations"]
    if violations:
        lines.append("\n  COVERAGE VIOLATIONS -- stop and debug the maths:")
        lines.extend(f"    {v}" for v in violations)
    else:
        lines.append("\n  empirical coverage meets target at every alpha")

    group_key = result.get("group_key", "task_type")
    groups = result["conformal"]["group_conditional_alpha_0.1"]
    lines.append(f"\nGROUP-CONDITIONAL COVERAGE at alpha = 0.10, by {group_key}")
    if not groups:
        # Empty is the correct answer when no group appears on both sides, and
        # saying so beats printing a blank table that reads as a crash. It
        # happens on the cross-corpus run: group-conditional coverage fits a
        # separate threshold per group, which needs calibration data from that
        # group, and RAGTruth has no RAGBench subsets in it. The per-subset
        # numbers for that run come from shift.py, which evaluates one shared
        # threshold on each subset instead of fitting twelve.
        lines.append(
            f"  no group appears in both the calibration and the test file under\n"
            f"  `{group_key}`, so no per-group threshold can be fitted. On a\n"
            "  cross-corpus run this is expected; see src/c2_calibration/shift.py\n"
            "  for per-subset coverage under one shared threshold."
        )
    else:
        lines.append(
            f"  {group_key:<17} empirical   +/-noise   abstain   n_test   n_calib"
        )
        lines.append("  " + "-" * 68)
        for task, row in groups.items():
            band = coverage_tolerance(
                0.1, int(row["n_calibration"]), int(row["n_test"])
            )
            short = 0.9 - row["empirical_coverage"]
            mark = "  <-- below noise band" if short > max(band, 0.005) else ""
            lines.append(
                f"  {task:<17} {row['empirical_coverage']:<11.4f} {band:<10.4f} "
                f"{row['abstention_rate']:<9.4f} {row['n_test']:<8,} "
                f"{row['n_calibration']:,}{mark}"
            )
    if groups and result["unit"] == "token":
        lines.append(
            "  note: tokens within a response are correlated, so the effective\n"
            "  sample size is nearer the response count than the token count and\n"
            "  this noise band is optimistic. See conformal.py, caveat 3."
        )

    rc = result["risk_coverage"]
    lines.append(
        f"\nRISK-COVERAGE  AURC uncalibrated {rc['aurc_uncalibrated']:.4f}  "
        f"calibrated {rc['aurc_calibrated']:.4f}"
    )
    lines.append("  coverage   risk")
    for row in rc["calibrated"][::4]:
        lines.append(f"  {row['coverage']:<10.3f} {row['risk']:.4f}")
    return "\n".join(lines)


def write_plots(result: Dict[str, Any], out_dir: Path) -> List[Path]:
    """Reliability diagram and risk-coverage curve. Never fatal."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return []

    written: List[Path] = []
    unit = result["unit"]

    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    bins = result["reliability_bins_after"]
    axis.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect")
    axis.plot(
        [b["mean_score"] for b in bins],
        [b["positive_rate"] for b in bins],
        marker="o",
        label=f"calibrated ({result['calibration']['selected']})",
    )
    axis.set_xlabel("mean predicted P(hallucinated)")
    axis.set_ylabel("observed hallucination rate")
    axis.set_title(f"Reliability, {unit} level")
    axis.legend()
    figure.tight_layout()
    path = out_dir / f"reliability_{unit}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)

    figure, axis = plt.subplots(figsize=(4.5, 4.5))
    for name in ("uncalibrated", "calibrated"):
        curve = result["risk_coverage"][name]
        axis.plot(
            [r["coverage"] for r in curve], [r["risk"] for r in curve], marker=".", label=name
        )
    axis.set_xlabel("coverage (fraction answered)")
    axis.set_ylabel("risk (error rate on answered)")
    axis.set_title(f"Risk-coverage, {unit} level")
    axis.legend()
    figure.tight_layout()
    path = out_dir / f"risk_coverage_{unit}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Run C2 calibration and conformal.")
    parser.add_argument(
        "--calib",
        default=None,
        help="calibration probabilities.jsonl; omit when using --self-split",
    )
    parser.add_argument("--test", required=True, help="test probabilities.jsonl")
    parser.add_argument(
        "--self-split",
        action="store_true",
        help=(
            "split --test in half by response into calibration and evaluation. "
            "Required when the detector trained on everything outside this file, "
            "as the public LettuceDetect checkpoint did"
        ),
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/c2/run")
    parser.add_argument(
        "--score-key",
        default="mean_prob",
        choices=["mean_prob", "max_prob", "min_prob"],
        help="how a span's token probabilities aggregate into one score",
    )
    parser.add_argument(
        "--alphas",
        default="0.05,0.10,0.15,0.20,0.30,0.40",
        help="comma-separated miscoverage levels",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--group-key",
        default="task_type",
        help=(
            "record field to break group-conditional coverage down by. "
            "task_type on RAGTruth, subset on RAGBench"
        ),
    )
    parser.add_argument(
        "--allow-violations",
        action="store_true",
        help=(
            "record coverage violations and mark the whole result VOID instead of "
            "exiting non-zero. For the shift study ONLY, where the violation IS "
            "the finding and the numbers still have to be written down. Never "
            "pass this on an in-distribution run: there a violation means the "
            "maths is wrong and the process should fail"
        ),
    )
    parser.add_argument(
        "--out-name",
        default="c2_results.json",
        help="results filename inside --out-dir",
    )
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    if args.self_split:
        if args.calib:
            raise SystemExit("--self-split and --calib are mutually exclusive")
        calib, test = halve_by_response(
            read_probability_file(args.test), seed=args.split_seed
        )
        print(
            f"self-split of {args.test}: {len(calib):,} calibration / "
            f"{len(test):,} evaluation responses"
        )
    else:
        if not args.calib:
            raise SystemExit("pass --calib, or --self-split to halve --test")
        calib = read_probability_file(args.calib)
        test = read_probability_file(args.test)
        print(
            f"calibration file {len(calib):,} responses | "
            f"test file {len(test):,} responses"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {
        "calib_file": str(args.calib),
        "test_file": str(args.test),
        "alphas": alphas,
        "group_key": args.group_key,
    }
    for unit in ("span", "token"):
        result = analyse(calib, test, unit, args.score_key, alphas, args.group_key)
        print(format_analysis(result))
        results[unit] = result
        if not args.no_plots:
            for path in write_plots(result, out_dir):
                print(f"  wrote {path}")

    all_violations = [
        v for unit in ("span", "token") for v in results[unit]["conformal"]["violations"]
    ]
    # Stamped into the file itself, not only into the console. A results file
    # that under-covers must announce it to anyone who opens it later, including
    # a table generator that has forgotten how the run was invoked.
    results["void"] = bool(all_violations) and args.allow_violations
    results["violations"] = all_violations
    if results["void"]:
        results["void_reason"] = (
            "empirical coverage falls below target by more than sampling noise "
            "explains. The exchangeability precondition of split conformal does "
            "not hold between the calibration file and the test file, so the "
            "coverage guarantee does not apply to these numbers. They are "
            "reported as a measurement of the break, not as a working method."
        )

    path = out_dir / args.out_name
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nwrote {path}")

    if results["void"]:
        # Deliberately no serving artifact. The artifact is the file the backend
        # loads to make live promises to a user, and this run has just measured
        # that the promise does not hold on this data. Writing one anyway would
        # leave a loaded gun in results/ for a later session to pick up.
        print(
            "\nno serving artifact written: this run is VOID and its threshold "
            "must never reach the backend"
        )
    else:
        artifact = build_serving_artifact(
            calib, results["span"], args.score_key, detector=str(args.test)
        )
        artifact_path = out_dir / "c2_artifact.json"
        with artifact_path.open("w", encoding="utf-8") as handle:
            json.dump(artifact, handle)
        print(
            f"wrote {artifact_path} "
            f"({artifact['calibrator']['name']}, "
            f"{artifact['n_calibration']:,} calibration scores)"
        )

    # Aug 17 trigger point: on an in-distribution run a coverage violation
    # invalidates the central claim, so it must fail the process rather than sit
    # in a log. --allow-violations turns that off for the shift study, where the
    # violation is the measurement being taken; the result is stamped VOID
    # instead, which is a louder signal than an exit code nobody reads.
    if not all_violations:
        return 0
    if args.allow_violations:
        print("\nCOVERAGE VIOLATIONS FOUND -- result marked VOID and recorded:")
        for violation in all_violations:
            print(f"  {violation}")
        print(
            "\n  This is the expected outcome of a run whose calibration and test "
            "data\n  come from different corpora. Report it as the break, never as "
            "a method."
        )
        return 0
    print("\nCOVERAGE VIOLATIONS FOUND -- do not report these numbers:")
    for violation in all_violations:
        print(f"  {violation}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
