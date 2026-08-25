"""Scoring for C3's type predictions, and the intervals that go beside them.

Two rules from C1 and C2 carry into every number this module produces.

**Print the floor beside the headline.** A classifier that always answers the
training majority class is scored exactly like the real one and reported in the
same table. On this corpus that matters more than usual: the severity axis is
88.52% evident on the test split, so an accuracy of 0.88 is what ignoring the
input already buys.

**Every interval is a response-level cluster bootstrap.** Spans inside one
response share a context, a generator and an annotator, so they are not
independent. C2 measured what forgetting that costs: span-level intervals came
out 1.1-1.3x too narrow and token-level ones 4.6-6.5x too narrow. The resampling
here draws whole responses with replacement, and it reuses C2's seed, resample
count and percentile helper so a C3 interval and a C2 interval mean the same
thing.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from src.c2_calibration.clustered import RESAMPLES, SEED, percentile_interval


def classification_report(
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Accuracy, macro F1 and per-class precision/recall/F1.

    `labels` fixes the class list so a class the model never predicts still gets
    a row of zeros rather than vanishing from the average. Leaving it out is how
    a macro F1 quietly improves by ignoring a rare class.
    """
    if len(gold) != len(predicted):
        raise ValueError(f"{len(gold)} gold labels but {len(predicted)} predictions")
    names = (
        sorted(set(labels))
        if labels is not None
        else sorted(set(gold) | set(predicted))
    )

    per_class: Dict[str, Dict[str, float]] = {}
    for name in names:
        tp = sum(1 for g, p in zip(gold, predicted) if g == name and p == name)
        fp = sum(1 for g, p in zip(gold, predicted) if g != name and p == name)
        fn = sum(1 for g, p in zip(gold, predicted) if g == name and p != name)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
            "predicted": tp + fp,
        }

    correct = sum(1 for g, p in zip(gold, predicted) if g == p)
    return {
        "n": len(gold),
        "accuracy": correct / len(gold) if gold else 0.0,
        "macro_f1": (
            sum(block["f1"] for block in per_class.values()) / len(per_class)
            if per_class
            else 0.0
        ),
        "weighted_f1": (
            sum(block["f1"] * block["support"] for block in per_class.values())
            / len(gold)
            if gold
            else 0.0
        ),
        "per_class": per_class,
        "confusion": {
            g: dict(Counter(p for gg, p in zip(gold, predicted) if gg == g))
            for g in names
        },
    }


def majority_label(gold: Sequence[str]) -> Optional[str]:
    """The most common label, ties broken on the sorted name so it is stable."""
    counts = Counter(gold)
    if not counts:
        return None
    best = max(counts.values())
    return sorted(name for name, n in counts.items() if n == best)[0]


def floor_report(
    gold: Sequence[str],
    constant: str,
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """The same report for a predictor that ignores its input entirely."""
    report = classification_report(gold, [constant] * len(gold), labels)
    report["predicted_class"] = constant
    report["ignores_the_input"] = True
    return report


# -------------------------------------------------------------------------
# Response-level cluster bootstrap
# -------------------------------------------------------------------------


def response_index(response_ids: Sequence[str]) -> List[np.ndarray]:
    """Row positions grouped by response, in first-seen order."""
    groups: Dict[str, List[int]] = defaultdict(list)
    for position, rid in enumerate(response_ids):
        groups[str(rid)].append(position)
    return [np.asarray(rows, dtype=np.int64) for rows in groups.values()]


def bootstrap_by_response(
    response_ids: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> Dict[str, float]:
    """Percentile interval for a statistic, resampling whole responses.

    `statistic` receives the row positions of one bootstrap draw and returns a
    single number. A draw that cannot produce it should return float('nan');
    those are dropped from the percentiles and counted, so they cannot hide.
    """
    index = response_index(response_ids)
    if not index:
        return percentile_interval([])
    rng = np.random.default_rng(seed)
    n = len(index)
    draws: List[float] = []
    for _ in range(resamples):
        picks = rng.integers(0, n, size=n)
        rows = np.concatenate([index[i] for i in picks])
        draws.append(float(statistic(rows)))
    return percentile_interval([v for v in draws if not math.isnan(v)])


def codes(values: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Label strings to column indices. A value outside `labels` becomes -1."""
    index = {name: i for i, name in enumerate(labels)}
    return np.asarray([index.get(str(v), -1) for v in values], dtype=np.int64)


def confusion_metrics(
    gold_codes: np.ndarray, pred_codes: np.ndarray, k: int
) -> Dict[str, Any]:
    """Accuracy, macro F1, per-class F1 and support from one confusion matrix.

    Vectorised because the cluster bootstrap calls it two thousand times per
    axis. classification_report above is the readable implementation of the same
    arithmetic, and tests/test_c3_type_eval.py asserts the two agree.
    """
    valid = (gold_codes >= 0) & (pred_codes >= 0)
    g = gold_codes[valid]
    p = pred_codes[valid]
    n = int(g.size)
    conf = np.bincount(g * k + p, minlength=k * k).reshape(k, k).astype(np.float64)
    tp = np.diag(conf)
    support = conf.sum(axis=1)
    predicted = conf.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(k), where=support > 0)
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros(k), where=denominator > 0
    )
    return {
        "accuracy": float(tp.sum() / n) if n else 0.0,
        "macro_f1": float(f1.mean()) if k else 0.0,
        "f1": f1,
        "support": support,
    }


def bootstrap_metrics(
    response_ids: Sequence[str],
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
    metrics: Sequence[str] = ("accuracy", "macro_f1"),
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Cluster-bootstrap intervals for the headline metrics and per-class F1.

    All metrics share one set of draws, so their intervals describe the same
    resampled corpora, exactly as C2's clustered.py does for ECE and coverage.
    A draw containing no instance of a class says nothing about that class's F1,
    so it is recorded as nan and counted rather than averaged in as a zero.
    """
    index = response_index(response_ids)
    if not index:
        return {}
    names = list(labels)
    k = len(names)
    gold_codes = codes(gold, names)
    pred_codes = codes(predicted, names)

    rng = np.random.default_rng(seed)
    n = len(index)
    collected: Dict[str, List[float]] = {name: [] for name in metrics}
    for label in names:
        collected[f"{label}.f1"] = []

    for _ in range(resamples):
        picks = rng.integers(0, n, size=n)
        rows = np.concatenate([index[i] for i in picks])
        block = confusion_metrics(gold_codes[rows], pred_codes[rows], k)
        for name in metrics:
            collected[name].append(float(block[name]))
        for column, label in enumerate(names):
            collected[f"{label}.f1"].append(
                float(block["f1"][column])
                if block["support"][column] > 0
                else float("nan")
            )

    return {
        name: {
            **percentile_interval([v for v in values if not math.isnan(v)]),
            "n_draws_without_the_class": int(
                sum(1 for v in values if math.isnan(v))
            ),
        }
        for name, values in collected.items()
    }


# -------------------------------------------------------------------------
# The full block one axis produces
# -------------------------------------------------------------------------


def evaluate_axis(
    response_ids: Sequence[str],
    gold: Sequence[str],
    predicted: Sequence[str],
    labels: Sequence[str],
    tasks: Optional[Sequence[str]] = None,
    floor_class: Optional[str] = None,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    with_intervals: bool = True,
) -> Dict[str, Any]:
    """Everything C3-3 needs for one axis: model, floor, per task, intervals."""
    constant = floor_class or majority_label(gold) or (labels[0] if labels else "")
    block: Dict[str, Any] = {
        "labels": list(labels),
        "model": classification_report(gold, predicted, labels),
        "floor": floor_report(gold, constant, labels),
    }

    if tasks is not None:
        per_task: Dict[str, Any] = {}
        for task in sorted(set(tasks)):
            rows = [i for i, t in enumerate(tasks) if t == task]
            task_gold = [gold[i] for i in rows]
            task_pred = [predicted[i] for i in rows]
            per_task[task] = {
                "model": classification_report(task_gold, task_pred, labels),
                "floor": floor_report(task_gold, constant, labels),
            }
        block["per_task"] = per_task

    if with_intervals:
        block["intervals"] = bootstrap_metrics(
            response_ids, gold, predicted, labels, resamples=resamples, seed=seed
        )
        block["interval_note"] = (
            "response-level cluster bootstrap, "
            f"{resamples} resamples, seed {seed}, 2.5th-97.5th percentile"
        )
    return block


def format_axis(name: str, block: Dict[str, Any]) -> str:
    """One axis as a table with the floor row directly beneath the model row."""
    model = block["model"]
    floor = block["floor"]
    intervals = block.get("intervals", {})

    def interval(key: str) -> str:
        row = intervals.get(key)
        if not row or not np.isfinite(row.get("ci_low", float("nan"))):
            return ""
        return f" [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"

    lines = [f"--- {name}  (n={model['n']:,}) ---"]
    lines.append(
        f"  model  accuracy {model['accuracy']:.4f}{interval('accuracy')}   "
        f"macro-F1 {model['macro_f1']:.4f}{interval('macro_f1')}"
    )
    lines.append(
        f"  floor  accuracy {floor['accuracy']:.4f}   "
        f"macro-F1 {floor['macro_f1']:.4f}   "
        f"(always '{floor['predicted_class']}', ignores the input)"
    )
    lines.append(f"  {'class':<24}{'P':>8}{'R':>8}{'F1':>8}{'support':>9}   floor F1")
    for label in block["labels"]:
        m = model["per_class"][label]
        f = floor["per_class"][label]
        lines.append(
            f"  {label:<24}{m['precision']:>8.4f}{m['recall']:>8.4f}"
            f"{m['f1']:>8.4f}{m['support']:>9,}   {f['f1']:.4f}"
            f"{interval(label + '.f1')}"
        )
    if "per_task" in block:
        lines.append("  per task (macro-F1, floor in brackets):")
        for task, cell in block["per_task"].items():
            lines.append(
                f"    {task:<16} n={cell['model']['n']:>6,}  "
                f"{cell['model']['macro_f1']:.4f}  "
                f"[{cell['floor']['macro_f1']:.4f}]"
            )
    return "\n".join(lines)
