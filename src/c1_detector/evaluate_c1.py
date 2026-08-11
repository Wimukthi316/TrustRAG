"""Evaluation for C1: token-level, span-level and example-level, per task.

Three granularities, because they answer three different questions and a good
number at one level can hide a bad one at another.

    token-level     Of the answer tokens the model called hallucinated, how many
                    were? Micro-averaged over tokens, binary (B-HAL and I-HAL
                    collapsed). Insensitive to span boundaries, so it is the
                    most forgiving of the three and the least interesting on its
                    own.

    span-level      Did the model find the right spans, with the right edges?
                    Reported two ways: seqeval in strict IOB2 mode over the tag
                    sequence, and exact character-offset match after decoding.
                    They can disagree -- seqeval works in token space, the
                    character version in the space the UI actually highlights.

    example-level   Did the model correctly say whether this response contains
                    any hallucination at all? Positive-class P/R/F1 over
                    responses. THIS is the number that compares against
                    LettuceDetect's reported 79.22%.

On that comparison, two caveats that have to be stated wherever the number
appears:

  - Whether LettuceDetect computes example-level F1 as positive-class F1 (what
    this file does) or as something else is UNVERIFIED. Check their paper or
    repository before writing the comparison down.
  - build_examples defaults to keeping `implicit_true` spans as hallucinations.
    Whether LettuceDetect does the same is also unverified. If they drop them,
    the two numbers are not measuring the same task.

Span exact match is capped near 99% by tokenisation, not by the model: 69 of the
7,664 gold spans in RAGTruth end inside a subword token, so no token-level model
can reproduce their exact character offsets. STATUS.md records the measurement.
Report the ceiling alongside the score.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.c1_detector.bio import (
    ANSWER_SEQUENCE_ID,
    B_HAL,
    I_HAL,
    IGNORE_INDEX,
    LABEL_NAMES,
    OUTSIDE,
    bio_to_char_spans,
)

Span = Tuple[int, int]


# --------------------------------------------------------------------------
# Counting helpers
# --------------------------------------------------------------------------


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Precision, recall, F1 from raw counts. Zero denominators give 0.0."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def repair_iob2(tags: Sequence[str]) -> List[str]:
    """Turn a possibly-invalid tag sequence into valid IOB2.

    A model can emit I-HAL with no B-HAL before it. seqeval's strict mode
    rejects that, so the leading I- is promoted to B-. This matches what
    bio_to_char_spans does when decoding to characters -- an orphan I- starts a
    span rather than being dropped -- so the two views stay consistent.
    """
    repaired: List[str] = []
    previous = "O"
    for tag in tags:
        if tag.startswith("I-") and (previous == "O" or previous[2:] != tag[2:]):
            tag = "B-" + tag[2:]
        repaired.append(tag)
        previous = tag
    return repaired


def spans_from_tag_ids(ids: Sequence[int]) -> List[Tuple[int, int]]:
    """Token-index spans (start, end_exclusive) from a BIO id sequence."""
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, label in enumerate(ids):
        if label == B_HAL:
            if start is not None:
                spans.append((start, i))
            start = i
        elif label == I_HAL:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i))
                start = None
    if start is not None:
        spans.append((start, len(ids)))
    return spans


def binary_from_hal_mask(mask: Sequence[bool]) -> List[int]:
    """Turn a per-token hallucinated/not mask into BIO label ids."""
    ids: List[int] = []
    previous = False
    for flag in mask:
        if not flag:
            ids.append(OUTSIDE)
        elif previous:
            ids.append(I_HAL)
        else:
            ids.append(B_HAL)
        previous = flag
    return ids


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------


@torch.no_grad()
def predict(
    model: Any,
    loader: Any,
    dataset: Any,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Run the model over a loader and return one dict per example.

    The loader must have been built from a dataset with keep_offsets=True --
    without the offset mapping there is no way to turn predicted labels back
    into character spans.

    `threshold` decides how a token is called hallucinated:
        None  -> argmax over the three logits. This is the plain reading of the
                 model and the default for reporting C1.
        float -> P(B-HAL) + P(I-HAL) >= threshold. Needed by C2, which sweeps
                 the operating point rather than accepting argmax.

    Returned per example:
        index, id, task_type, model, answer
        gold_spans, pred_spans        character offsets into answer
        gold_ids, pred_ids            BIO label ids over answer tokens only
        token_probs                   P(hallucinated) per answer token
        answer_offsets                (start, end) per answer token
        answer_truncated              True if max_length cut the answer short
    """
    model.eval()
    results: List[Dict[str, Any]] = []

    for batch in loader:
        if "offset_mapping" not in batch:
            raise ValueError(
                "the dataset must be built with keep_offsets=True for evaluation"
            )

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        if amp_dtype is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Softmax in float32 regardless of the autocast dtype. fp16 probabilities
        # are what C2 calibrates on; rounding them to half precision here would
        # put a quantisation floor under every ECE number downstream.
        probs = torch.softmax(logits.float(), dim=-1).cpu()
        gold = batch["labels"].cpu()

        for row, index in enumerate(batch["index"]):
            sequence_ids = batch["sequence_ids"][row]
            offsets = batch["offset_mapping"][row]
            meta = dataset.meta(index)

            positions = [
                i for i, sid in enumerate(sequence_ids) if sid == ANSWER_SEQUENCE_ID
            ]
            # encode_example reports (0, 0) for specials; a zero-width offset is
            # never a real answer token.
            positions = [i for i in positions if offsets[i][1] > offsets[i][0]]

            token_probs = [
                float(probs[row, i, B_HAL] + probs[row, i, I_HAL]) for i in positions
            ]
            gold_ids = [int(gold[row, i]) for i in positions]

            if threshold is None:
                pred_ids = [int(probs[row, i].argmax()) for i in positions]
            else:
                pred_ids = binary_from_hal_mask([p >= threshold for p in token_probs])

            # Rebuild full-length sequences so bio_to_char_spans, which walks
            # labels and offsets together, sees the same indexing it expects.
            full_pred = [IGNORE_INDEX] * len(sequence_ids)
            for slot, i in enumerate(positions):
                full_pred[i] = pred_ids[slot]

            pred_spans = bio_to_char_spans(
                full_pred, offsets, sequence_ids, answer=meta["answer"]
            )

            results.append(
                {
                    "index": index,
                    "id": meta["id"],
                    "task_type": meta["task_type"],
                    "model": meta["model"],
                    "answer": meta["answer"],
                    "gold_spans": [tuple(s) for s in meta["gold_spans"]],
                    "pred_spans": [tuple(s) for s in pred_spans],
                    "gold_ids": gold_ids,
                    "pred_ids": pred_ids,
                    "token_probs": token_probs,
                    "answer_offsets": [tuple(offsets[i]) for i in positions],
                    "answer_truncated": bool(batch["answer_truncated"][row]),
                }
            )

    return results


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def token_metrics(predictions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Binary micro P/R/F1 over answer tokens. B-HAL and I-HAL collapse to 1."""
    tp = fp = fn = 0
    for item in predictions:
        for gold, pred in zip(item["gold_ids"], item["pred_ids"]):
            if gold == IGNORE_INDEX:
                continue
            gold_hal = gold in (B_HAL, I_HAL)
            pred_hal = pred in (B_HAL, I_HAL)
            if gold_hal and pred_hal:
                tp += 1
            elif pred_hal:
                fp += 1
            elif gold_hal:
                fn += 1
    return prf(tp, fp, fn)


def span_char_metrics(predictions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Exact character-offset span match.

    A predicted span counts only if some gold span has identical start and end.
    This is the strictest view and the one that matches what a reviewer sees in
    the UI, since the highlight is drawn at exactly these offsets.
    """
    tp = fp = fn = 0
    for item in predictions:
        gold = set(item["gold_spans"])
        pred = set(item["pred_spans"])
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return prf(tp, fp, fn)


def span_overlap_metrics(predictions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Partial-credit span match: a predicted span counts if it overlaps a gold span.

    Reported alongside exact match because exact match is capped near 99% by
    subword granularity, and because a highlight that covers most of a
    hallucination is still useful to a reviewer. Each gold span can be matched
    at most once, so this cannot be gamed by predicting one span per token.
    """
    tp = fp = fn = 0
    for item in predictions:
        gold = list(item["gold_spans"])
        matched = [False] * len(gold)
        for pstart, pend in item["pred_spans"]:
            hit = None
            for i, (gstart, gend) in enumerate(gold):
                if matched[i]:
                    continue
                if gstart < pend and pstart < gend:
                    hit = i
                    break
            if hit is None:
                fp += 1
            else:
                matched[hit] = True
                tp += 1
        fn += matched.count(False)
    return prf(tp, fp, fn)


def span_seqeval_metrics(predictions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Strict IOB2 span F1 via seqeval, over the token tag sequence.

    Returns an empty dict if seqeval is missing, so a smoke run on a machine
    without it still produces the other metrics instead of crashing.
    """
    try:
        from seqeval.metrics import f1_score, precision_score, recall_score
        from seqeval.scheme import IOB2
    except ImportError:
        return {}

    y_true: List[List[str]] = []
    y_pred: List[List[str]] = []
    for item in predictions:
        gold_tags = [
            LABEL_NAMES[g] for g in item["gold_ids"] if g != IGNORE_INDEX
        ]
        pred_tags = [
            LABEL_NAMES[p]
            for g, p in zip(item["gold_ids"], item["pred_ids"])
            if g != IGNORE_INDEX
        ]
        if not gold_tags:
            continue
        y_true.append(repair_iob2(gold_tags))
        y_pred.append(repair_iob2(pred_tags))

    if not y_true:
        return {}

    kwargs = {"mode": "strict", "scheme": IOB2, "zero_division": 0}
    return {
        "precision": float(precision_score(y_true, y_pred, **kwargs)),
        "recall": float(recall_score(y_true, y_pred, **kwargs)),
        "f1": float(f1_score(y_true, y_pred, **kwargs)),
    }


def example_metrics(predictions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Response-level detection: does this answer contain any hallucination?

    Positive class is "contains at least one hallucinated span". This is the
    metric that lines up with LettuceDetect's reported 79.22% -- read the
    caveats in this module's docstring before writing that comparison down.
    """
    tp = fp = fn = tn = 0
    for item in predictions:
        gold_positive = len(item["gold_spans"]) > 0
        pred_positive = len(item["pred_spans"]) > 0
        if gold_positive and pred_positive:
            tp += 1
        elif pred_positive:
            fp += 1
        elif gold_positive:
            fn += 1
        else:
            tn += 1
    out = prf(tp, fp, fn)
    out["tn"] = tn
    total = tp + fp + fn + tn
    out["accuracy"] = (tp + tn) / total if total else 0.0
    return out


def evaluate(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """All metrics, overall and broken down by task type."""

    def block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "n_examples": len(rows),
            "n_gold_spans": sum(len(r["gold_spans"]) for r in rows),
            "n_pred_spans": sum(len(r["pred_spans"]) for r in rows),
            "n_answer_truncated": sum(1 for r in rows if r["answer_truncated"]),
            "token": token_metrics(rows),
            "span_exact": span_char_metrics(rows),
            "span_overlap": span_overlap_metrics(rows),
            "span_seqeval": span_seqeval_metrics(rows),
            "example": example_metrics(rows),
        }

    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in predictions:
        by_task[str(item["task_type"])].append(item)

    return {
        "overall": block(predictions),
        "per_task": {task: block(rows) for task, rows in sorted(by_task.items())},
    }


def format_report(metrics: Dict[str, Any]) -> str:
    """Human-readable summary. Printed at the end of a run and pasted into notes."""
    lines: List[str] = []

    def one(name: str, block: Dict[str, Any]) -> None:
        lines.append(f"--- {name} (n={block['n_examples']:,}) ---")
        lines.append(
            f"  gold spans {block['n_gold_spans']:,}  "
            f"pred spans {block['n_pred_spans']:,}  "
            f"answers truncated {block['n_answer_truncated']:,}"
        )
        for key in ("token", "span_exact", "span_overlap", "span_seqeval", "example"):
            block_metrics = block.get(key) or {}
            if not block_metrics:
                lines.append(f"  {key:<13} unavailable")
                continue
            lines.append(
                f"  {key:<13} P {block_metrics['precision']:.4f}  "
                f"R {block_metrics['recall']:.4f}  "
                f"F1 {block_metrics['f1']:.4f}"
            )

    one("overall", metrics["overall"])
    for task, block in metrics["per_task"].items():
        one(task, block)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Diagnostics: what the model actually believes
# --------------------------------------------------------------------------


def _trim_to_text(answer: str, start: int, end: int) -> Optional[Span]:
    """Shrink a span past leading and trailing whitespace. None if nothing is left."""
    while start < end and answer[start].isspace():
        start += 1
    while end > start and answer[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def spans_from_token_mask(
    mask: Sequence[bool], offsets: Sequence[Tuple[int, int]], answer: str
) -> List[Span]:
    """Merge runs of hallucinated tokens into trimmed character spans.

    The threshold-sweep equivalent of bio_to_char_spans: it works from a
    per-token boolean rather than from BIO ids, so two adjacent spans merge into
    one. That is a real limitation of any threshold view and the reason the
    model is trained with BIO in the first place.
    """
    spans: List[Span] = []
    current: Optional[List[int]] = None
    for flag, (start, end) in zip(mask, offsets):
        if flag:
            if current is None:
                current = [start, end]
            else:
                current[1] = end
        elif current is not None:
            spans.append((current[0], current[1]))
            current = None
    if current is not None:
        spans.append((current[0], current[1]))
    return [t for t in (_trim_to_text(answer, s, e) for s, e in spans) if t is not None]


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[index]


def probability_summary(predictions: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Distribution of P(hallucinated) over answer tokens, gold-positive vs gold-negative.

    This is what tells a degenerate run apart from an undertrained one. A model
    that predicts no spans at argmax may still separate the two classes -- its
    scores are simply all below 0.5. A model whose positive and negative means
    are identical has learned nothing, and no threshold will rescue it.
    """
    positive: List[float] = []
    negative: List[float] = []
    for item in predictions:
        for gold, prob in zip(item["gold_ids"], item["token_probs"]):
            if gold == IGNORE_INDEX:
                continue
            (positive if gold in (B_HAL, I_HAL) else negative).append(prob)

    every = sorted(positive + negative)
    return {
        "n_tokens": len(every),
        "n_gold_positive": len(positive),
        "mean": sum(every) / len(every) if every else 0.0,
        "p50": _quantile(every, 0.50),
        "p95": _quantile(every, 0.95),
        "p99": _quantile(every, 0.99),
        "max": every[-1] if every else 0.0,
        "mean_on_gold_positive": sum(positive) / len(positive) if positive else 0.0,
        "mean_on_gold_negative": sum(negative) / len(negative) if negative else 0.0,
    }


def threshold_sweep(
    predictions: Sequence[Dict[str, Any]],
    thresholds: Sequence[float] = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
) -> List[Dict[str, Any]]:
    """Re-decode spans at each P(hallucinated) cutoff and score them.

    Costs nothing extra -- the probabilities are already in hand, so this is
    arithmetic, not another forward pass. Two uses: it proves the span decoder
    works even when argmax predicts nothing, and it is the operating-point sweep
    C2 needs before it can choose one.
    """
    rows: List[Dict[str, Any]] = []
    for threshold in thresholds:
        rescored = []
        for item in predictions:
            mask = [p >= threshold for p in item["token_probs"]]
            rescored.append(
                {
                    "gold_spans": item["gold_spans"],
                    "pred_spans": spans_from_token_mask(
                        mask, item["answer_offsets"], item["answer"]
                    ),
                }
            )
        rows.append(
            {
                "threshold": threshold,
                "n_pred_spans": sum(len(r["pred_spans"]) for r in rescored),
                "example": example_metrics(rescored),
                "span_overlap": span_overlap_metrics(rescored),
            }
        )
    return rows


def format_diagnostics(predictions: Sequence[Dict[str, Any]]) -> str:
    """Probability distribution plus a threshold sweep, as printable text."""
    summary = probability_summary(predictions)
    lines = [
        "--- P(hallucinated) over answer tokens ---",
        f"  tokens {summary['n_tokens']:,} of which gold-positive "
        f"{summary['n_gold_positive']:,}",
        f"  mean {summary['mean']:.4f}  p50 {summary['p50']:.4f}  "
        f"p95 {summary['p95']:.4f}  p99 {summary['p99']:.4f}  "
        f"max {summary['max']:.4f}",
        f"  mean on gold-positive tokens {summary['mean_on_gold_positive']:.4f}  "
        f"vs gold-negative {summary['mean_on_gold_negative']:.4f}",
        "--- threshold sweep (span decode, not argmax) ---",
        "  thresh   pred spans   example F1   overlap F1",
    ]
    for row in threshold_sweep(predictions):
        lines.append(
            f"  {row['threshold']:<8.2f} {row['n_pred_spans']:>10,} "
            f"{row['example']['f1']:>12.4f} {row['span_overlap']['f1']:>12.4f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Probability dump for C2
# --------------------------------------------------------------------------


def dump_probabilities(
    predictions: Sequence[Dict[str, Any]], path: Path | str
) -> int:
    """Write per-token and per-span probabilities as JSONL, for C2 to calibrate on.

    Written per response, not per span, so C2 can define its own operating point
    and its own span aggregation without re-running the model. Every predicted
    span carries `is_hallucinated`, which is the ground-truth label conformal
    prediction needs: True if the span overlaps any gold span.

    Deliberately the same shape the LettuceDetect adapter will produce. Building
    the C2 harness against the public checkpoint and swapping in C1 later then
    costs a path change and nothing else -- which is the parallelism the plan
    depends on.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in predictions:
            gold = item["gold_spans"]
            spans = []
            for start, end in item["pred_spans"]:
                covered = [
                    p
                    for p, (offset_start, offset_end) in zip(
                        item["token_probs"], item["answer_offsets"]
                    )
                    if offset_start < end and start < offset_end
                ]
                spans.append(
                    {
                        "start": start,
                        "end": end,
                        "text": item["answer"][start:end],
                        "mean_prob": sum(covered) / len(covered) if covered else 0.0,
                        "max_prob": max(covered) if covered else 0.0,
                        "min_prob": min(covered) if covered else 0.0,
                        "n_tokens": len(covered),
                        "is_hallucinated": any(
                            gstart < end and start < gend for gstart, gend in gold
                        ),
                    }
                )

            handle.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "task_type": item["task_type"],
                        "model": item["model"],
                        "answer": item["answer"],
                        "gold_spans": [
                            {"start": s, "end": e, "text": item["answer"][s:e]}
                            for s, e in gold
                        ],
                        "pred_spans": spans,
                        "token_probs": [round(p, 6) for p in item["token_probs"]],
                        "answer_offsets": [list(o) for o in item["answer_offsets"]],
                        "answer_truncated": item["answer_truncated"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained C1 checkpoint on a processed RAGTruth file."
    )
    parser.add_argument("--checkpoint", required=True, help="directory saved by train_c1")
    parser.add_argument("--data", required=True, help="processed RAGTruth JSONL")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "debug only: first N records. The file is grouped by task, so this "
            "gives a single-task sample -- never report a number taken with it"
        ),
    )
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="P(hallucinated) cutoff; omit to use argmax",
    )
    parser.add_argument("--out-dir", default="results/c1/eval")
    parser.add_argument(
        "--dump-probs",
        action="store_true",
        help="also write per-span probabilities for C2",
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit to autodetect")
    args = parser.parse_args()

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from src.c1_detector.dataset import (
        RagTruthTokenDataset,
        build_dataloader,
        read_records,
    )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint).to(device)

    records = read_records(args.data, limit=args.limit)
    dataset = RagTruthTokenDataset(
        records, tokenizer, max_length=args.max_length, keep_offsets=True
    )
    loader, _ = build_dataloader(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=False,
        group_by_length=False,
        num_workers=args.num_workers,
    )

    predictions = predict(model, loader, dataset, device, threshold=args.threshold)
    metrics = evaluate(predictions)
    print(format_report(metrics))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"\nwrote {metrics_path}")

    if args.dump_probs:
        probs_path = out_dir / "probabilities.jsonl"
        n = dump_probabilities(predictions, probs_path)
        print(f"wrote {n:,} records to {probs_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
