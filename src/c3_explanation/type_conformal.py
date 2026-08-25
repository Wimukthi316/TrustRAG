"""Layer 4 of C3: a calibrated conformal SET over the error type.

This is the layer that makes C3 TrustRAG's C3 rather than a classifier somebody
bolted on, and it exists only because C2 exists. C2 built a distribution-free
coverage guarantee over the binary "is this span hallucinated" decision. The
same machinery, pointed at the four-way type label, turns Layer 3's finding into
a guarantee instead of an apology.

Layer 3 measured that the mechanism axis is learnable and the severity axis very
nearly is not. A flat 4-way classifier has to answer anyway, so it answers
severity by guessing. A conformal set does not: at a chosen alpha it returns

    {evident_conflict, subtle_conflict}

which reads as "at 90% coverage this is a conflict; which severity is not
decidable from this label". An average set size near two, concentrated on the
severity axis, is not a weak result. It is a precise statement of what is
knowable, backed by a bound that holds without assuming anything about the
distribution -- except exchangeability, which C2 already showed how to check and
when to withdraw.

**What is reused from C2, unchanged.** `conformal.conformal_quantile` computes
the (n+1) corrected quantile, and `conformal.coverage_tolerance` computes the
3-sigma band a single honest calibration draw may fall short by, so a C3 coverage
row is judged by exactly the rule a C2 coverage row is.
`calibration.expected_calibration_error` and its siblings score the top-label
confidence. Nothing under src/c2_calibration/ is modified; C2 is frozen.

**What is new here.** C2's SplitConformal is binary by construction -- its
nonconformity is `1 - p` or `p`. The multi-class version is the same idea with
the true class named explicitly, and it lives here rather than in C2 so that the
frozen component stays frozen.

**The gate.** When AnalysisResult.guarantee_applies is False, the type and the
explanation are still shown and no coverage number is printed beside them. That
rule belongs to the serving path; this module records it in its artefact so the
two cannot drift.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.c2_calibration.calibration import (
    expected_calibration_error,
    maximum_calibration_error,
)
from src.c2_calibration.conformal import conformal_quantile, coverage_tolerance
from src.c3_explanation.labels_audit import axes_of_error_type
from src.c3_explanation.type_eval import bootstrap_by_response
from src.common.schema import ErrorType

ALPHAS: Tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)
EPSILON = 1e-12


# --------------------------------------------------------------------------
# Multi-class calibration
# --------------------------------------------------------------------------


def temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Divide the log-probabilities by T and renormalise.

    T > 1 softens, T < 1 sharpens, T = 1 changes nothing. Because it is a
    monotone transform of every logit by the same factor it can change how well
    a confidence is calibrated without changing which class wins -- which is
    exactly the separation C2 documents between calibration and ranking.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    logits = np.log(np.clip(probs, EPSILON, None)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def multiclass_nll(probs: np.ndarray, truth: np.ndarray) -> float:
    if probs.size == 0:
        return float("nan")
    picked = probs[np.arange(probs.shape[0]), truth]
    return float(-np.mean(np.log(np.clip(picked, EPSILON, None))))


def multiclass_brier(probs: np.ndarray, truth: np.ndarray) -> float:
    """Mean squared error against the one-hot truth, summed over classes."""
    if probs.size == 0:
        return float("nan")
    onehot = np.zeros_like(probs)
    onehot[np.arange(probs.shape[0]), truth] = 1.0
    return float(np.mean(((probs - onehot) ** 2).sum(axis=1)))


def fit_temperature(
    probs: np.ndarray,
    truth: np.ndarray,
    low: float = 0.05,
    high: float = 20.0,
    iterations: int = 200,
) -> float:
    """Temperature minimising calibration NLL, by golden-section search.

    A search rather than a gradient step so that no optimiser dependency and no
    random seed enters the result: the same inputs give the same T on any
    machine.
    """
    if probs.size == 0:
        return 1.0
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc = multiclass_nll(temperature_scale(probs, c), truth)
    fd = multiclass_nll(temperature_scale(probs, d), truth)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = multiclass_nll(temperature_scale(probs, c), truth)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = multiclass_nll(temperature_scale(probs, d), truth)
        if abs(b - a) < 1e-6:
            break
    return float((a + b) / 2.0)


def top_label_report(probs: np.ndarray, truth: np.ndarray, n_bins: int = 15) -> Dict[str, float]:
    """ECE / MCE on the top-label confidence, plus Brier and NLL.

    Multi-class ECE here is the standard top-label version: bin by the winning
    class's probability and compare that to how often the winner was right. That
    reduces to a binary calibration question -- confidence against correctness --
    so C2's own ECE and MCE functions score it unchanged.
    """
    if probs.size == 0:
        return {
            "ece": float("nan"),
            "mce": float("nan"),
            "brier": float("nan"),
            "nll": float("nan"),
            "accuracy": float("nan"),
            "n": 0,
        }
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == truth).astype(int)
    return {
        "ece": float(expected_calibration_error(confidence, correct, n_bins=n_bins)),
        "mce": float(maximum_calibration_error(confidence, correct, n_bins=n_bins)),
        "brier": multiclass_brier(probs, truth),
        "nll": multiclass_nll(probs, truth),
        "accuracy": float(correct.mean()),
        "n": int(probs.shape[0]),
    }


def base_rate_floor(
    calib_truth: np.ndarray, test_probs_shape: Tuple[int, int], test_truth: np.ndarray
) -> Dict[str, float]:
    """The predictor that answers the calibration class frequencies, every time.

    It ignores its input completely. Every ECE above must be read against this
    row rather than against zero -- the same rule C2's constant-base-rate row
    enforces for the binary case.
    """
    n_rows, n_classes = test_probs_shape
    counts = np.bincount(calib_truth, minlength=n_classes).astype(np.float64)
    total = counts.sum()
    rates = counts / total if total else np.full(n_classes, 1.0 / n_classes)
    probs = np.tile(rates, (n_rows, 1))
    report = top_label_report(probs, test_truth)
    report["ignores_the_input"] = True
    report["base_rates"] = [float(v) for v in rates]
    return report


# --------------------------------------------------------------------------
# Multi-class split conformal (LAC)
# --------------------------------------------------------------------------


def nonconformity(probs: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """1 - p(true class). Small means the model already backed the truth."""
    return 1.0 - probs[np.arange(probs.shape[0]), truth]


def prediction_sets(probs: np.ndarray, threshold: float) -> List[Tuple[int, ...]]:
    """Every class whose 1 - p clears the calibrated bar."""
    keep = (1.0 - probs) <= threshold
    return [tuple(int(i) for i in np.flatnonzero(row)) for row in keep]


def set_shape(members: Sequence[int], classes: Sequence[str]) -> str:
    """Name what a set of types is actually saying.

    A pair sharing a mechanism is the interesting one: the system has committed
    to conflict-or-baseless and is declining to commit on severity. That is the
    Layer 3 finding stated as a guarantee, so it gets its own row rather than
    being averaged into "size 2".
    """
    if not members:
        return "empty"
    if len(members) == 1:
        return "singleton"
    if len(members) == len(classes):
        return "full"
    axes = [axes_of_error_type(ErrorType(classes[i])) for i in members]
    mechanisms = {a for a, _ in axes}
    severities = {b for _, b in axes}
    if len(members) == 2 and len(mechanisms) == 1:
        return "pair_severity_undecided"
    if len(members) == 2 and len(severities) == 1:
        return "pair_mechanism_undecided"
    if len(mechanisms) == 1:
        return "mechanism_committed"
    return "other"


def coverage_row(
    calib_probs: np.ndarray,
    calib_truth: np.ndarray,
    test_probs: np.ndarray,
    test_truth: np.ndarray,
    alpha: float,
    classes: Sequence[str],
) -> Dict[str, Any]:
    """One alpha: threshold, coverage against its band, and what the sets look like."""
    threshold = float(conformal_quantile(nonconformity(calib_probs, calib_truth), alpha))
    sets = prediction_sets(test_probs, threshold)
    covered = [1 if int(t) in s else 0 for s, t in zip(sets, test_truth)]
    sizes = np.asarray([len(s) for s in sets], dtype=np.float64)
    shapes = Counter(set_shape(s, classes) for s in sets)
    empirical = float(np.mean(covered)) if covered else float("nan")
    band = float(
        coverage_tolerance(alpha, int(calib_probs.shape[0]), int(test_probs.shape[0]))
    )
    target = 1.0 - alpha
    return {
        "alpha": alpha,
        "target_coverage": target,
        "threshold": threshold,
        "empirical_coverage": empirical,
        "band": band,
        "inside_band": bool(empirical >= target - band),
        "average_set_size": float(sizes.mean()) if sizes.size else float("nan"),
        "singleton_rate": float(np.mean(sizes == 1)) if sizes.size else float("nan"),
        "empty_set_rate": float(np.mean(sizes == 0)) if sizes.size else float("nan"),
        "full_set_rate": (
            float(np.mean(sizes == len(classes))) if sizes.size else float("nan")
        ),
        "set_shapes": {k: shapes[k] for k in sorted(shapes)},
        "set_shape_rates": {
            k: shapes[k] / len(sets) for k in sorted(shapes)
        }
        if sets
        else {},
        "n_calibration": int(calib_probs.shape[0]),
        "n_test": int(test_probs.shape[0]),
    }


def per_class_coverage(
    calib_probs: np.ndarray,
    calib_truth: np.ndarray,
    test_probs: np.ndarray,
    test_truth: np.ndarray,
    alpha: float,
    classes: Sequence[str],
) -> Dict[str, Any]:
    """Coverage inside each true class.

    Split conformal promises MARGINAL coverage, not coverage inside every class,
    so these rows are expected to differ and a low one is not a bug. They are
    reported because a marginal guarantee that is carried entirely by the
    majority class is worth knowing about before anybody deploys it.
    """
    threshold = float(conformal_quantile(nonconformity(calib_probs, calib_truth), alpha))
    sets = prediction_sets(test_probs, threshold)
    out: Dict[str, Any] = {}
    for index, name in enumerate(classes):
        rows = [i for i, t in enumerate(test_truth) if int(t) == index]
        calib_n = int(np.sum(calib_truth == index))
        out[name] = {
            "n_test": len(rows),
            "n_calibration": calib_n,
            "empirical_coverage": (
                float(np.mean([1 if index in sets[i] else 0 for i in rows]))
                if rows
                else float("nan")
            ),
            "average_set_size": (
                float(np.mean([len(sets[i]) for i in rows])) if rows else float("nan")
            ),
            "measurable": bool(rows and calib_n),
        }
    return out


def group_coverage(
    calib_probs: np.ndarray,
    calib_truth: np.ndarray,
    test_probs: np.ndarray,
    test_truth: np.ndarray,
    test_groups: Sequence[str],
    alpha: float,
) -> Dict[str, Any]:
    threshold = float(conformal_quantile(nonconformity(calib_probs, calib_truth), alpha))
    sets = prediction_sets(test_probs, threshold)
    out: Dict[str, Any] = {}
    for group in sorted(set(test_groups)):
        rows = [i for i, g in enumerate(test_groups) if g == group]
        out[group] = {
            "n_test": len(rows),
            "empirical_coverage": (
                float(np.mean([1 if int(test_truth[i]) in sets[i] else 0 for i in rows]))
                if rows
                else float("nan")
            ),
            "average_set_size": (
                float(np.mean([len(sets[i]) for i in rows])) if rows else float("nan")
            ),
        }
    return out


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def unpack(rows: Sequence[Dict[str, Any]], key: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Probability matrix, true class index, response ids and tasks."""
    if not rows:
        return (np.empty((0, 0)), np.empty(0, dtype=int), [], [])
    classes = list(rows[0]["classes"])
    index = {name: i for i, name in enumerate(classes)}
    probs = np.asarray([r[key] for r in rows], dtype=np.float64)
    truth = np.asarray([index[r["error_type"]] for r in rows], dtype=np.int64)
    return (
        probs,
        truth,
        [str(r["response_id"]) for r in rows],
        [str(r.get("task_type", "unknown")) for r in rows],
    )


def analyse(
    calib_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    probs_key: str,
    alphas: Sequence[float] = ALPHAS,
    resamples: int = 2000,
    with_intervals: bool = True,
) -> Dict[str, Any]:
    classes = list(calib_rows[0]["classes"])
    calib_probs, calib_truth, _, _ = unpack(calib_rows, probs_key)
    test_probs, test_truth, test_ids, test_tasks = unpack(test_rows, probs_key)

    temperature = fit_temperature(calib_probs, calib_truth)
    calib_scaled = temperature_scale(calib_probs, temperature)
    test_scaled = temperature_scale(test_probs, temperature)

    calibration_rows = [
        {
            "method": "identity",
            "selected": False,
            "is_floor": False,
            **top_label_report(test_probs, test_truth),
        },
        {
            "method": f"temperature (T = {temperature:.4f})",
            "selected": True,
            "is_floor": False,
            **top_label_report(test_scaled, test_truth),
        },
        {
            "method": "constant base rate",
            "selected": False,
            "is_floor": True,
            **base_rate_floor(calib_truth, test_probs.shape, test_truth),
        },
    ]

    coverage: List[Dict[str, Any]] = []
    for alpha in alphas:
        row = coverage_row(
            calib_scaled, calib_truth, test_scaled, test_truth, alpha, classes
        )
        if with_intervals:
            threshold = row["threshold"]
            sets = prediction_sets(test_scaled, threshold)
            covered = np.asarray(
                [1.0 if int(t) in s else 0.0 for s, t in zip(sets, test_truth)]
            )
            sizes = np.asarray([float(len(s)) for s in sets])
            row["coverage_interval"] = bootstrap_by_response(
                test_ids, lambda rows: float(covered[rows].mean()), resamples=resamples
            )
            row["set_size_interval"] = bootstrap_by_response(
                test_ids, lambda rows: float(sizes[rows].mean()), resamples=resamples
            )
        coverage.append(row)

    headline = 0.1 if 0.1 in tuple(alphas) else alphas[0]
    return {
        "classes": classes,
        "probs_key": probs_key,
        "n_calibration": int(calib_probs.shape[0]),
        "n_test": int(test_probs.shape[0]),
        "n_test_responses": len(set(test_ids)),
        "temperature": temperature,
        "calibration": calibration_rows,
        "coverage": coverage,
        "per_class_coverage": per_class_coverage(
            calib_scaled, calib_truth, test_scaled, test_truth, headline, classes
        ),
        "per_task_coverage": group_coverage(
            calib_scaled, calib_truth, test_scaled, test_truth, test_tasks, headline
        ),
        "calibration_class_counts": {
            name: int(np.sum(calib_truth == i)) for i, name in enumerate(classes)
        },
        "test_class_counts": {
            name: int(np.sum(test_truth == i)) for i, name in enumerate(classes)
        },
        "headline_alpha": headline,
        "guarantee_note": (
            "These coverage numbers describe inputs exchangeable with the "
            "calibration split. When AnalysisResult.guarantee_applies is False "
            "the serving path still shows the type and the explanation and "
            "prints no coverage number beside them."
        ),
        "ood_note": (
            "There is no out-of-domain coverage number for the type label and "
            "that absence is a finding: RAGBench carries no error-type "
            "annotation, so a typed detector is structurally un-evaluable "
            "outside RAGTruth. What can be reported out of domain is the shift "
            "in the PREDICTED type mix and the exchangeability gate's trip rate."
        ),
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = [
        "=== C3 Layer 4: calibrated conformal type sets ===",
        f"probabilities: {report['probs_key']}   "
        f"calibration {report['n_calibration']:,} spans   "
        f"test {report['n_test']:,} spans in {report['n_test_responses']:,} responses",
        "",
        "--- C3-4 type calibration (top-label) ---",
        f"  {'method':<28}{'ECE':>9}{'MCE':>9}{'Brier':>9}{'NLL':>9}{'acc':>9}",
    ]
    for row in report["calibration"]:
        tag = "  <- ignores the input" if row.get("is_floor") else (
            "  <- selected" if row.get("selected") else ""
        )
        lines.append(
            f"  {row['method']:<28}{row['ece']:>9.4f}{row['mce']:>9.4f}"
            f"{row['brier']:>9.4f}{row['nll']:>9.4f}{row['accuracy']:>9.4f}{tag}"
        )

    lines += ["", "--- C3-5 conformal type sets ---"]
    lines.append(
        f"  {'alpha':>6}{'target':>9}{'coverage':>10}{'band':>8}  in band"
        f"{'avg size':>10}{'singleton':>11}{'empty':>8}"
    )
    for row in report["coverage"]:
        interval = row.get("coverage_interval") or {}
        span = (
            f"  [{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"
            if interval and np.isfinite(interval.get("ci_low", float("nan")))
            else ""
        )
        lines.append(
            f"  {row['alpha']:>6.2f}{row['target_coverage']:>9.4f}"
            f"{row['empirical_coverage']:>10.4f}{row['band']:>8.4f}"
            f"{'  yes  ' if row['inside_band'] else '  NO   '}"
            f"{row['average_set_size']:>10.4f}{row['singleton_rate']:>11.4f}"
            f"{row['empty_set_rate']:>8.4f}{span}"
        )

    headline = report["headline_alpha"]
    row = next(r for r in report["coverage"] if r["alpha"] == headline)
    lines += ["", f"--- what the sets say, at alpha = {headline} ---"]
    for shape, rate in sorted(row["set_shape_rates"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {shape:<28}{row['set_shapes'][shape]:>7,}  {rate:.4f}")
    lines.append(
        "  pair_severity_undecided is the system committing to conflict or "
        "baseless and declining to commit on severity."
    )

    lines += ["", f"--- coverage inside each true type, alpha = {headline} ---"]
    lines.append(
        "  split conformal promises marginal coverage, not per-class coverage; "
        "these rows are diagnostic"
    )
    for name, cell in report["per_class_coverage"].items():
        note = "" if cell["measurable"] else "   n/a - absent from calibration or test"
        coverage = (
            f"{cell['empirical_coverage']:.4f}"
            if not math.isnan(cell["empirical_coverage"])
            else "   n/a"
        )
        lines.append(
            f"  {name:<24} test n={cell['n_test']:>6,}  calib n={cell['n_calibration']:>6,}"
            f"  coverage {coverage}  avg size {cell['average_set_size']:.4f}{note}"
        )

    lines += ["", f"--- coverage per task, alpha = {headline} ---"]
    for task, cell in report["per_task_coverage"].items():
        lines.append(
            f"  {task:<16} n={cell['n_test']:>6,}  "
            f"coverage {cell['empirical_coverage']:.4f}  "
            f"avg size {cell['average_set_size']:.4f}"
        )

    lines += ["", report["guarantee_note"], "", report["ood_note"]]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="C3 Layer 4: calibrate the type head and wrap it in conformal sets."
    )
    parser.add_argument(
        "--calib", default="results/c3/type/baseline_probs_predicted_calib.jsonl"
    )
    parser.add_argument(
        "--test", default="results/c3/type/baseline_probs_predicted_test.jsonl"
    )
    parser.add_argument(
        "--probs-key",
        default="probs_flat",
        choices=("probs_flat", "probs_axis_product"),
    )
    parser.add_argument("--out-dir", default="results/c3/type")
    parser.add_argument("--out-name", default="c3_conformal.json")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    calib_rows = read_jsonl(Path(args.calib))
    test_rows = read_jsonl(Path(args.test))
    if not calib_rows or not test_rows:
        print("no probability rows found; run type_baseline (or the type head) first")
        return 1

    report = analyse(
        calib_rows, test_rows, args.probs_key, resamples=args.resamples
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.out_name
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(format_report(report))
    print()
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
