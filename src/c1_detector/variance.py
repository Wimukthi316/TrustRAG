"""Turn repeated training runs into a variance estimate.

Two runs give a difference. They do not give a standard deviation you can put a
number of digits on: with n=2 the sample standard deviation is just the gap
divided by the square root of two, and it inherits the whole of whatever those
two runs happened to do. Every "1.1 points" quoted in this project came from
exactly that, and it is a spread, not a sigma.

This module reads whatever repeat runs exist on disk and reports mean, sample
standard deviation, min and max per metric level, refusing to print a standard
deviation at all when fewer than three runs are present. The point of Block G is
to move this from n=2 to n=5; the module is what reads the answer either way.

What the spread contains, stated so the report can state it: repeat runs at the
SAME seed still differ, because cuDNN kernel selection and the non-deterministic
reduction order on GPU are not seeded. Runs at DIFFERENT seeds add initialisation
of the classifier head, data order, and the train/val/calib split moving. A
pooled spread over mixed seeds therefore bounds both together and separates
neither, and the report should say which runs went in.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

LEVELS = ("token", "span_exact", "span_overlap", "example")

# Fewer than this and a standard deviation is theatre rather than a measurement.
MIN_RUNS_FOR_SD = 3


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def sample_sd(values: Sequence[float]) -> Optional[float]:
    """Sample standard deviation, or None when there is not enough to estimate."""
    if len(values) < MIN_RUNS_FOR_SD:
        return None
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def read_run(run_dir: Path, seed: Optional[int] = None) -> Dict[str, Any]:
    """One run's test metrics, plus the seed it actually used if recorded.

    The seed is read from summary.json rather than from the config file, because
    the CLI can override it and the override is what the run really used. Test
    metrics are written to their own directory, away from the training run's
    summary.json, so the caller may also state the seed outright.
    """
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"no metrics.json under {run_dir}")
    overall = json.loads(metrics_path.read_text(encoding="utf-8"))["overall"]

    if seed is None:
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            seed = summary.get("config", {}).get("seed")

    return {
        "dir": str(run_dir).replace("\\", "/"),
        "seed": seed,
        "f1": {level: overall[level]["f1"] for level in LEVELS if level in overall},
    }


def summarise(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean, sd, min and max per level over the runs given."""
    if not runs:
        raise ValueError("no runs to summarise")

    levels: Dict[str, Any] = {}
    for level in LEVELS:
        values = [r["f1"][level] for r in runs if level in r["f1"]]
        if not values:
            continue
        sd = sample_sd(values)
        levels[level] = {
            "n": len(values),
            "values": values,
            "mean": mean(values),
            "sd": sd,
            "sd_points": None if sd is None else 100 * sd,
            "min": min(values),
            "max": max(values),
            "range_points": 100 * (max(values) - min(values)),
        }

    seeds = [r["seed"] for r in runs]
    distinct = sorted({s for s in seeds if s is not None})
    return {
        "n_runs": len(runs),
        "runs": list(runs),
        "seeds": seeds,
        "distinct_seeds": distinct,
        "mixes_seeds": len(distinct) > 1,
        "sd_reported": len(runs) >= MIN_RUNS_FOR_SD,
        "levels": levels,
    }


def format_report(report: Dict[str, Any]) -> str:
    n = report["n_runs"]
    lines = [f"Repeat runs found: {n}", ""]
    lines.append(f"{'level':<14}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}{'range':>9}")
    for level, block in report["levels"].items():
        sd = "n/a" if block["sd"] is None else f"{block['sd']:.4f}"
        lines.append(
            f"{level:<14}{block['mean']:>9.4f}{sd:>9}"
            f"{block['min']:>9.4f}{block['max']:>9.4f}"
            f"{block['range_points']:>8.2f}p"
        )
    lines.append("")

    if not report["sd_reported"]:
        lines.append(
            f"WARNING: {n} run(s) is not enough for a standard deviation. Report the "
            "range as a spread and say n explicitly; do not write it as a sigma."
        )
    if report["mixes_seeds"]:
        lines.append(
            f"These runs use seeds {report['distinct_seeds']}, so the spread bounds "
            "seed variation and GPU non-determinism together and separates neither."
        )
    else:
        lines.append(
            "These runs share one seed, so the spread is GPU non-determinism alone; "
            "it says nothing about how much a different initialisation would move."
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise repeat C1 training runs into a variance estimate."
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "run directories, each containing metrics.json. Write DIR=SEED when "
            "the seed is not recoverable from a summary.json beside the metrics, "
            "which is the case for evaluation output directories"
        ),
    )
    parser.add_argument("--out", default="results/c1/analysis/variance.json")
    args = parser.parse_args(argv)

    runs: List[Dict[str, Any]] = []
    for item in args.runs:
        path, _, seed_text = item.partition("=")
        runs.append(read_run(Path(path), int(seed_text) if seed_text else None))
    report = summarise(runs)
    print(format_report(report))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
