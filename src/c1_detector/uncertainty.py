"""How precise is a reported F1, and is a difference between two runs real?

Two different uncertainties get confused, so name them apart:

    run-to-run spread    train the same config again and the numbers differ.
                         Three runs at seeds 42, 7 and 13 give a standard
                         deviation of 0.38 points on example F1, 0.54 on
                         span-exact, 0.79 on span-overlap and 1.08 on token F1.
                         An earlier reading of 1.12 points came from two runs at
                         one seed; two runs cannot give a standard deviation, and
                         that figure was applied to every level when it had only
                         ever been measured for one. Measured by variance.py,
                         which needs a training run per point and is not here.

    bootstrap interval   hold the model fixed and resample the test responses.
                         Says how well 2,700 responses pin the number down.
                         Costs nothing, because the predictions already exist.

Both belong beside a reported number and neither substitutes for the other.

Comparing one model's point estimate against another's interval is not a test of
the difference. paired_bootstrap resamples the SAME responses for both models and
takes the difference on each draw, so the sampling noise the two share cancels
and what is left is the difference itself. An interval that straddles zero means
the data cannot tell the two apart.

Resampling is at the response level, which is the independent unit. Resampling
spans would treat two spans in one answer as independent, which they are not.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.c1_detector.evaluate_c1 import (
    example_metrics,
    span_char_metrics,
    span_overlap_metrics,
)
from src.c1_detector.localisation import load_probability_dump

Metric = Callable[[Sequence[Dict[str, Any]]], Dict[str, float]]

# token metrics are absent on purpose: the probability dump carries spans and
# per-token probabilities but not the gold BIO ids token_metrics needs.
LEVELS: Dict[str, Metric] = {
    "span_exact": span_char_metrics,
    "span_overlap": span_overlap_metrics,
    "example": example_metrics,
}

RESAMPLES = 2000
SEED = 42


def _percentile_interval(values: Sequence[float], b: int) -> Tuple[float, float]:
    ordered = sorted(values)
    return ordered[int(0.025 * b)], ordered[int(0.975 * b) - 1]


def bootstrap_ci(
    records: Sequence[Dict[str, Any]],
    resamples: int = RESAMPLES,
    seed: int = SEED,
    levels: Optional[Dict[str, Metric]] = None,
) -> Dict[str, Dict[str, float]]:
    """95% percentile interval for each metric, resampling responses."""
    levels = levels or LEVELS
    rng = random.Random(seed)
    n = len(records)
    draws: Dict[str, List[float]] = {name: [] for name in levels}

    for _ in range(resamples):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        for name, metric in levels.items():
            draws[name].append(metric(sample)["f1"])

    out: Dict[str, Dict[str, float]] = {}
    for name, metric in levels.items():
        low, high = _percentile_interval(draws[name], resamples)
        out[name] = {
            "point": metric(records)["f1"],
            "ci_low": low,
            "ci_high": high,
            "width_points": 100 * (high - low),
            "std_points": 100 * statistics.pstdev(draws[name]),
        }
    return out


def pair_by_id(
    baseline: Sequence[Dict[str, Any]], variant: Sequence[Dict[str, Any]]
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Line the two dumps up by response id, refusing to guess on a mismatch.

    Two evaluations of the same split should cover the same ids. If they do not,
    something upstream is wrong and quietly scoring the intersection would hide
    it behind a plausible number.
    """
    index = {row["id"]: row for row in variant}
    missing = [row["id"] for row in baseline if row["id"] not in index]
    if missing:
        raise ValueError(
            f"{len(missing)} response ids in the baseline are absent from the "
            f"variant, first few: {missing[:5]}"
        )
    return [(row, index[row["id"]]) for row in baseline]


def paired_bootstrap(
    baseline: Sequence[Dict[str, Any]],
    variant: Sequence[Dict[str, Any]],
    resamples: int = RESAMPLES,
    seed: int = SEED,
    levels: Optional[Dict[str, Metric]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Interval on (variant - baseline), in F1 points, over shared resamples."""
    levels = levels or LEVELS
    pairs = pair_by_id(baseline, variant)
    rng = random.Random(seed)
    n = len(pairs)
    draws: Dict[str, List[float]] = {name: [] for name in levels}

    for _ in range(resamples):
        index = [rng.randrange(n) for _ in range(n)]
        left = [pairs[i][0] for i in index]
        right = [pairs[i][1] for i in index]
        for name, metric in levels.items():
            draws[name].append(100 * (metric(right)["f1"] - metric(left)["f1"]))

    left_all = [pair[0] for pair in pairs]
    right_all = [pair[1] for pair in pairs]

    out: Dict[str, Dict[str, Any]] = {}
    for name, metric in levels.items():
        low, high = _percentile_interval(draws[name], resamples)
        crosses_zero = low <= 0.0 <= high
        out[name] = {
            "baseline": metric(left_all)["f1"],
            "variant": metric(right_all)["f1"],
            "delta_points": 100 * (metric(right_all)["f1"] - metric(left_all)["f1"]),
            "ci_low_points": low,
            "ci_high_points": high,
            "crosses_zero": crosses_zero,
            "std_points": statistics.pstdev(draws[name]),
            "verdict": (
                "not distinguishable from zero"
                if crosses_zero
                else ("higher" if low > 0 else "lower")
            ),
        }
    return out


def format_ci(result: Dict[str, Dict[str, float]]) -> str:
    lines = [f"{'level':<14}{'F1':>9}{'95% CI low':>12}{'high':>10}{'width':>9}{'sd':>8}"]
    for name, row in result.items():
        lines.append(
            f"{name:<14}{row['point']:>9.4f}{row['ci_low']:>12.4f}"
            f"{row['ci_high']:>10.4f}{row['width_points']:>8.2f}p"
            f"{row['std_points']:>7.2f}p"
        )
    return "\n".join(lines)


def format_paired(result: Dict[str, Dict[str, Any]]) -> str:
    lines = [
        f"{'level':<14}{'baseline':>10}{'variant':>10}{'delta':>9}"
        f"{'95% CI':>18}{'verdict':>32}"
    ]
    for name, row in result.items():
        interval = f"[{row['ci_low_points']:+.2f}, {row['ci_high_points']:+.2f}]"
        lines.append(
            f"{name:<14}{row['baseline']:>10.4f}{row['variant']:>10.4f}"
            f"{row['delta_points']:>+9.2f}{interval:>18}{row['verdict']:>32}"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap intervals for C1's reported F1 scores."
    )
    parser.add_argument("--probs", default="results/c1/test/probabilities.jsonl")
    parser.add_argument(
        "--compare",
        default=None,
        help="a second dump; reports the paired interval on the difference",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=None, help="write the result as JSON here")
    args = parser.parse_args(argv)

    baseline = load_probability_dump(args.probs)

    if args.compare:
        variant = load_probability_dump(args.compare)
        result = paired_bootstrap(baseline, variant, args.resamples, args.seed)
        print(
            f"paired on {len(baseline):,} responses, "
            f"{args.resamples:,} resamples, seed {args.seed}"
        )
        print(format_paired(result))
    else:
        result = bootstrap_ci(baseline, args.resamples, args.seed)
        print(
            f"{len(baseline):,} responses, {args.resamples:,} resamples, "
            f"seed {args.seed}"
        )
        print(format_ci(result))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nwritten: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
