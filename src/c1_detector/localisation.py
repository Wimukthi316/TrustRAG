"""Why span-exact match fails: a decomposition of C1's localisation errors.

Span-exact F1 on the RAGTruth test split is 0.1485 against an example-level 0.7623.
That single number says the detector localises badly but not how, and the two
plausible cures are opposite: a decoding fix helps when spans run together or
fragment, and a training fix helps when spans are missed or invented. This module
measures which it is.

Every predicted and every gold span is placed in exactly one bucket:

    exact       one gold, one prediction, identical offsets
    boundary    one gold, one prediction, offsets differ
    merge       one prediction spanning two or more gold spans
    split       two or more predictions over one gold span
    tangled     two or more on both sides, so neither name fits
    missed      a gold span no prediction touches
    spurious    a prediction touching no gold span

The buckets come from connected components of the gold-to-prediction overlap
graph rather than from an ordered chain of conditionals. That matters: with
components, every span belongs to exactly one group by construction, so the
counts reconcile whether or not the classification rules were written carefully.

The counts here will NOT equal the false positives and false negatives reported
by span_overlap_metrics, and the difference is informative rather than a bug.
That metric matches greedily, one prediction to one gold span, so a split leaves
surplus predictions counted as false positives and a merge leaves surplus gold
spans counted as false negatives. reconcile() derives one from the other and the
report records both, because the plan quoted the greedy figures as though they
meant "never touched" and "touching no gold", which they do not.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.c1_detector.bio import (
    ANSWER_SEQUENCE_ID,
    bio_to_char_spans,
    char_spans_to_bio,
)
from src.c1_detector.evaluate_c1 import span_char_metrics, span_overlap_metrics

Span = Tuple[int, int]
Component = Tuple[List[int], List[int]]

BUCKETS: Tuple[str, ...] = (
    "exact",
    "boundary",
    "merge",
    "split",
    "tangled",
    "missed",
    "spurious",
)

# A boundary error within this many characters on both edges is a nudge; beyond
# it the prediction is a different shape from the gold span, not a shifted copy.
# Reported explicitly in the output so a reader can see what the word "near"
# meant rather than having to guess.
NEAR_CHARS = 10


def overlaps(a: Span, b: Span) -> bool:
    """Half-open overlap, the same convention schema.py and bio.py use."""
    return a[0] < b[1] and b[0] < a[1]


def overlap_components(gold: Sequence[Span], pred: Sequence[Span]) -> List[Component]:
    """Connected components of the gold-to-prediction overlap graph.

    Returns (gold_indices, pred_indices) pairs. A gold span nothing touches comes
    back as ([i], []); a prediction touching no gold span as ([], [j]). Every
    index appears in exactly one component, which is what makes the bucket counts
    add up to the input totals without any further care.
    """
    gold_adjacent: Dict[int, set] = defaultdict(set)
    pred_adjacent: Dict[int, set] = defaultdict(set)
    for gi, gold_span in enumerate(gold):
        for pi, pred_span in enumerate(pred):
            if overlaps(gold_span, pred_span):
                gold_adjacent[gi].add(pi)
                pred_adjacent[pi].add(gi)

    seen_gold: set = set()
    seen_pred: set = set()
    components: List[Component] = []

    for start in range(len(gold)):
        if start in seen_gold:
            continue
        stack: List[Tuple[str, int]] = [("gold", start)]
        in_gold: set = set()
        in_pred: set = set()
        while stack:
            side, index = stack.pop()
            if side == "gold":
                if index in in_gold:
                    continue
                in_gold.add(index)
                seen_gold.add(index)
                stack.extend(("pred", pi) for pi in gold_adjacent[index])
            else:
                if index in in_pred:
                    continue
                in_pred.add(index)
                seen_pred.add(index)
                stack.extend(("gold", gi) for gi in pred_adjacent[index])
        components.append((sorted(in_gold), sorted(in_pred)))

    for pi in range(len(pred)):
        if pi not in seen_pred:
            components.append(([], [pi]))

    return components


def classify(n_gold: int, n_pred: int, identical: bool) -> str:
    """Name the bucket for one component. See the module docstring for meanings."""
    if n_gold == 0 and n_pred == 0:
        raise ValueError("a component must contain at least one span")
    if n_pred == 0:
        return "missed"
    if n_gold == 0:
        return "spurious"
    if n_gold == 1 and n_pred == 1:
        return "exact" if identical else "boundary"
    if n_pred == 1:
        return "merge"
    if n_gold == 1:
        return "split"
    return "tangled"


def edge_delta_bucket(delta: int) -> str:
    """Group an edge error by magnitude. One character and twenty are not the
    same disease: the first is usually punctuation, the second is a wrong span."""
    size = abs(delta)
    if size <= 2:
        return str(size)
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    return ">10"


def boundary_shape(dstart: int, dend: int, near: int = NEAR_CHARS) -> str:
    """Which way a one-to-one boundary error is wrong, beyond `near` characters.

    overrun     the prediction spills past the gold span
    undershoot  the prediction stops short of it
    shifted     both, so it is displaced rather than resized
    near        neither edge is off by more than `near`
    """
    over = (-dstart) > near or dend > near
    under = dstart > near or (-dend) > near
    if over and under:
        return "shifted"
    if over:
        return "overrun"
    if under:
        return "undershoot"
    return "near"


def decompose(
    records: Sequence[Dict[str, Any]],
    near: int = NEAR_CHARS,
    examples_per_bucket: int = 40,
) -> Dict[str, Any]:
    """Bucket every gold and predicted span, with a breakdown of boundary errors.

    Each record needs `gold_spans` and `pred_spans` as (start, end) pairs, plus
    `id` and `task_type` if the examples are wanted. Examples are kept so the
    Localisation Viewer can pick a clean case per category by measurement rather
    than by eye.
    """
    tally = {name: {"components": 0, "gold": 0, "pred": 0} for name in BUCKETS}
    start_delta: Counter = Counter()
    end_delta: Counter = Counter()
    shapes: Counter = Counter()
    which_edge: Counter = Counter()
    inside_gold = 0
    examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in records:
        gold = list(record["gold_spans"])
        pred = list(record["pred_spans"])
        for in_gold, in_pred in overlap_components(gold, pred):
            identical = (
                len(in_gold) == 1
                and len(in_pred) == 1
                and gold[in_gold[0]] == pred[in_pred[0]]
            )
            name = classify(len(in_gold), len(in_pred), identical)
            tally[name]["components"] += 1
            tally[name]["gold"] += len(in_gold)
            tally[name]["pred"] += len(in_pred)

            if len(examples[name]) < examples_per_bucket:
                examples[name].append(
                    {
                        "id": record.get("id"),
                        "task_type": record.get("task_type"),
                        "gold": [list(gold[i]) for i in in_gold],
                        "pred": [list(pred[i]) for i in in_pred],
                    }
                )

            if name != "boundary":
                continue
            (gold_start, gold_end) = gold[in_gold[0]]
            (pred_start, pred_end) = pred[in_pred[0]]
            dstart, dend = pred_start - gold_start, pred_end - gold_end
            start_delta[edge_delta_bucket(dstart)] += 1
            end_delta[edge_delta_bucket(dend)] += 1
            shapes[boundary_shape(dstart, dend, near)] += 1
            which_edge[
                "both" if dstart and dend else "start only" if dstart else "end only"
            ] += 1
            if pred_start >= gold_start and pred_end <= gold_end:
                inside_gold += 1

    return {
        "buckets": tally,
        "boundary": {
            "n_groups": tally["boundary"]["components"],
            "pred_strictly_inside_gold": inside_gold,
            "start_edge_delta": dict(sorted(start_delta.items())),
            "end_edge_delta": dict(sorted(end_delta.items())),
            "which_edge_wrong": dict(which_edge),
            "shape": dict(shapes),
        },
        "examples": dict(examples),
    }


def reconcile(buckets: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """Derive span_overlap's greedy counts from the structural buckets.

    span_overlap_metrics pairs one prediction with one gold span, so a split
    leaves (pred - gold) surplus predictions counted as false positives and a
    merge leaves (gold - pred) surplus gold spans counted as false negatives.
    Recovering its tp/fp/fn exactly is the check that the decomposition is sound;
    if these do not equal the numbers in metrics.json, something is wrong here
    rather than in the metric.
    """
    matched = {name: min(buckets[name]["gold"], buckets[name]["pred"]) for name in BUCKETS}
    overlap_tp = (
        buckets["exact"]["gold"]
        + buckets["boundary"]["gold"]
        + matched["merge"]
        + matched["split"]
        + matched["tangled"]
    )
    surplus_pred = sum(
        max(0, buckets[name]["pred"] - buckets[name]["gold"])
        for name in ("merge", "split", "tangled")
    )
    surplus_gold = sum(
        max(0, buckets[name]["gold"] - buckets[name]["pred"])
        for name in ("merge", "split", "tangled")
    )
    return {
        "overlap_tp": overlap_tp,
        "overlap_fp": buckets["spurious"]["pred"] + surplus_pred,
        "overlap_fn": buckets["missed"]["gold"] + surplus_gold,
        "exact_tp": buckets["exact"]["gold"],
        "gold_never_touched": buckets["missed"]["gold"],
        "pred_touching_no_gold": buckets["spurious"]["pred"],
        "surplus_pred_from_splits": surplus_pred,
        "surplus_gold_from_merges": surplus_gold,
    }


def tokenisation_ceiling(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Span-exact score a model with a perfect tag sequence could reach here.

    Encodes the gold spans to BIO against this split's own token offsets and
    decodes straight back with the shipped functions, so the answer reflects the
    decoder actually in use rather than a reimplementation of it. Anything below
    this number is the model's doing; the gap up to 1.0 is the tokenizer's.

    Run it on the split being reported. A corpus-wide figure is not the ceiling
    for a subset of it, and on RAGTruth the difference was large enough to change
    the diagnosis.
    """
    perfect: List[Dict[str, Any]] = []
    widths: Counter = Counter()

    for record in records:
        offsets = list(record["answer_offsets"])
        sequence_ids = [ANSWER_SEQUENCE_ID] * len(offsets)
        labels = char_spans_to_bio(offsets, sequence_ids, record["gold_spans"])
        decoded = bio_to_char_spans(
            labels, offsets, sequence_ids, answer=record["answer"]
        )
        for gold_span in record["gold_spans"]:
            hit = next((d for d in decoded if overlaps(gold_span, d)), None)
            if hit is None:
                widths["lost"] += 1
            else:
                widths[str((gold_span[0] - hit[0]) + (hit[1] - gold_span[1]))] += 1
        perfect.append(
            {
                "task_type": record.get("task_type"),
                "gold_spans": list(record["gold_spans"]),
                "pred_spans": decoded,
            }
        )

    tasks = sorted({r["task_type"] for r in perfect if r["task_type"]})
    return {
        "n_gold_spans": sum(len(r["gold_spans"]) for r in perfect),
        "n_decoded_spans": sum(len(r["pred_spans"]) for r in perfect),
        "gold_width_delta_chars": dict(sorted(widths.items())),
        "span_exact": span_char_metrics(perfect),
        "per_task": {
            task: span_char_metrics([r for r in perfect if r["task_type"] == task])
            for task in tasks
        },
    }


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return float(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))])

    return {
        "n": len(ordered),
        "mean": round(statistics.fmean(ordered), 2),
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "max": float(ordered[-1]),
    }


def span_lengths(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Predicted span length against gold span length, overall and per task.

    Ten lines that decide the diagnosis. If predictions are systematically longer
    than gold, the model is running spans together and a decoding fix is worth
    building. If they are shorter, it is under-covering and no decoder will help.
    """
    out: Dict[str, Any] = {}
    tasks = ["ALL"] + sorted({r["task_type"] for r in records if r.get("task_type")})
    for task in tasks:
        rows = [r for r in records if task == "ALL" or r.get("task_type") == task]
        out[task] = {
            "gold_chars": _quantiles([e - s for r in rows for s, e in r["gold_spans"]]),
            "pred_chars": _quantiles([e - s for r in rows for s, e in r["pred_spans"]]),
        }
    return out


def load_probability_dump(path: Path | str) -> List[Dict[str, Any]]:
    """Read evaluate_c1's probability dump into the shape this module expects.

    The dump writes spans as dicts, not tuples: gold spans carry start/end/text,
    predictions also carry the per-span probabilities and `is_hallucinated`.
    That last field is the ground-truth label C2 needs -- whether the prediction
    overlaps any gold span -- and not a filter, so every entry in `pred_spans` is
    a predicted span regardless of its value.
    """
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["gold_spans"] = [(s["start"], s["end"]) for s in row["gold_spans"]]
            row["pred_spans"] = [(s["start"], s["end"]) for s in row["pred_spans"]]
            row["answer_offsets"] = [tuple(o) for o in row.get("answer_offsets", [])]
            rows.append(row)
    return rows


def build_report(
    records: Sequence[Dict[str, Any]], near: int = NEAR_CHARS
) -> Dict[str, Any]:
    """The whole Block A analysis, ready to write as JSON."""
    overall = decompose(records, near=near)
    tasks = sorted({r["task_type"] for r in records if r.get("task_type")})

    per_task = {}
    for task in tasks:
        rows = [r for r in records if r.get("task_type") == task]
        block = decompose(rows, near=near)
        block.pop("examples")
        block["n_gold_spans"] = sum(len(r["gold_spans"]) for r in rows)
        block["n_pred_spans"] = sum(len(r["pred_spans"]) for r in rows)
        per_task[task] = block

    measured = {
        "span_exact": span_char_metrics(records),
        "span_overlap": span_overlap_metrics(records),
    }
    derived = reconcile(overall["buckets"])
    agrees = (
        derived["exact_tp"] == measured["span_exact"]["tp"]
        and derived["overlap_tp"] == measured["span_overlap"]["tp"]
        and derived["overlap_fp"] == measured["span_overlap"]["fp"]
        and derived["overlap_fn"] == measured["span_overlap"]["fn"]
    )

    examples = overall.pop("examples")
    return {
        "n_records": len(records),
        "n_gold_spans": sum(len(r["gold_spans"]) for r in records),
        "n_pred_spans": sum(len(r["pred_spans"]) for r in records),
        "near_threshold_chars": near,
        "measured_metrics": measured,
        "derived_from_buckets": derived,
        "reconciles_with_measured_metrics": agrees,
        "overall": overall,
        "per_task": per_task,
        "tokenisation_ceiling": tokenisation_ceiling(records),
        "span_lengths": span_lengths(records),
        "examples": examples,
    }


def _bucket_table(name: str, buckets: Dict[str, Dict[str, int]], gold: int, pred: int) -> str:
    lines = [
        f"--- {name}: gold {gold:,}  predicted {pred:,} ---",
        f"{'bucket':<10}{'groups':>8}{'gold':>8}{'% gold':>9}{'pred':>8}{'% pred':>9}",
    ]
    for bucket in BUCKETS:
        row = buckets[bucket]
        lines.append(
            f"{bucket:<10}{row['components']:>8,}{row['gold']:>8,}"
            f"{(100 * row['gold'] / gold if gold else 0):>8.1f}%"
            f"{row['pred']:>8,}"
            f"{(100 * row['pred'] / pred if pred else 0):>8.1f}%"
        )
    return "\n".join(lines)


def format_report(report: Dict[str, Any]) -> str:
    """Human-readable summary. Printed at the end of a run and pasted into notes."""
    exact = report["measured_metrics"]["span_exact"]
    overlap = report["measured_metrics"]["span_overlap"]
    derived = report["derived_from_buckets"]
    ceiling = report["tokenisation_ceiling"]

    lines = [
        f"records {report['n_records']:,}  gold spans {report['n_gold_spans']:,}  "
        f"pred spans {report['n_pred_spans']:,}",
        f"span_exact   tp/fp/fn {exact['tp']}/{exact['fp']}/{exact['fn']}  "
        f"F1 {exact['f1']:.4f}",
        f"span_overlap tp/fp/fn {overlap['tp']}/{overlap['fp']}/{overlap['fn']}  "
        f"F1 {overlap['f1']:.4f}",
        "buckets reconcile with the measured metrics: "
        f"{report['reconciles_with_measured_metrics']}",
        "",
        _bucket_table(
            "ALL", report["overall"]["buckets"], report["n_gold_spans"], report["n_pred_spans"]
        ),
    ]

    for task, block in report["per_task"].items():
        lines.append("")
        lines.append(
            _bucket_table(task, block["buckets"], block["n_gold_spans"], block["n_pred_spans"])
        )

    boundary = report["overall"]["boundary"]
    lines += [
        "",
        "--- boundary errors (one gold, one prediction, offsets differ) ---",
        f"  groups {boundary['n_groups']:,}, of which the prediction sits strictly "
        f"inside gold: {boundary['pred_strictly_inside_gold']:,}",
        f"  start edge delta {boundary['start_edge_delta']}",
        f"  end   edge delta {boundary['end_edge_delta']}",
        f"  which edge wrong {boundary['which_edge_wrong']}",
        f"  shape vs {report['near_threshold_chars']} chars {boundary['shape']}",
        "",
        "--- tokenisation ceiling, this split only ---",
        f"  span_exact P {ceiling['span_exact']['precision']:.4f}  "
        f"R {ceiling['span_exact']['recall']:.4f}  "
        f"F1 {ceiling['span_exact']['f1']:.4f}",
        f"  per task {({k: round(v['f1'], 4) for k, v in ceiling['per_task'].items()})}",
        f"  gold span width delta after round trip {ceiling['gold_width_delta_chars']}",
        "",
        "--- span lengths in characters ---",
    ]
    for task, block in report["span_lengths"].items():
        lines.append(
            f"  {task:<14} gold p50 {block['gold_chars'].get('p50')!s:>6}  "
            f"pred p50 {block['pred_chars'].get('p50')!s:>6}  "
            f"gold mean {block['gold_chars'].get('mean')!s:>7}  "
            f"pred mean {block['pred_chars'].get('mean')!s:>7}"
        )
    lines += [
        "",
        "Greedy span_overlap counts are not structural counts:",
        f"  gold spans never touched by any prediction {derived['gold_never_touched']:,} "
        f"(overlap fn is {overlap['fn']:,}, the difference being "
        f"{derived['surplus_gold_from_merges']:,} gold spans inside merge groups)",
        f"  predictions touching no gold span {derived['pred_touching_no_gold']:,} "
        f"(overlap fp is {overlap['fp']:,}, the difference being "
        f"{derived['surplus_pred_from_splits']:,} split fragments)",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decompose C1's span localisation failures (Block A)."
    )
    parser.add_argument(
        "--probs",
        default="results/c1/test/probabilities.jsonl",
        help="probability dump written by evaluate_c1 --dump-probs",
    )
    parser.add_argument(
        "--out-dir",
        default="results/c1/analysis",
        help="where localisation_report.json is written",
    )
    parser.add_argument(
        "--near",
        type=int,
        default=NEAR_CHARS,
        help="characters within which a boundary error counts as near",
    )
    args = parser.parse_args(argv)

    records = load_probability_dump(args.probs)
    report = build_report(records, near=args.near)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "localisation_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(format_report(report))
    print()
    print(f"written: {out_path}")

    if not report["reconciles_with_measured_metrics"]:
        print(
            "WARNING: the bucket counts do not reproduce span_overlap's tp/fp/fn. "
            "Trust the metric, not this decomposition, until that is resolved."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
