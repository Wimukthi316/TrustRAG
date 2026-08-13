"""Out-of-distribution evaluation of C1 on RAGBench (C1-A).

Runs the RAGTruth-trained checkpoint over each RAGBench test subset and reports
the example-level drop. The model is loaded once and reused across all twelve
subsets, because loading a 598 MB checkpoint twelve times is twelve times the
wait for no benefit.

WHAT THIS FILE WILL AND WILL NOT REPORT

Only example-level P/R/F1 goes in the OOD table. RAGBench annotates whole
response sentences; RAGTruth annotates phrases with a median length of 35
characters. Token-level and span-level scores across that mismatch would be
measuring the annotation convention, not the detector. They are still written
into metrics.json so the run is fully recorded, and `format_ood_table` refuses
to print them.

Three columns sit beside the F1 in the table because the F1 is not
interpretable without them:

  positive rate   RAGBench's is 14.2% against RAGTruth's 43.1%, and it varies
                  from 3.5% to 53.2% by subset. Positive-class F1 falls when
                  positives get rarer even with an unchanged classifier.
  trivial         the F1 of the classifier that calls every response
                  hallucinated, 2p/(1+p) at that subset's positive rate. It is
                  the floor a real detector has to clear, and on this corpus
                  most subsets do not clear it. Without this column an F1 of
                  0.69 on expertqa reads as a success when it is in fact below
                  the do-nothing baseline of 0.6945.
  n               expertqa has 203 records, tatqa 3,338. A gap on a small
                  subset is not the same evidence as a gap on a large one.

There is deliberately NO truncation column. `encode_example` truncates the first
sequence only, so `n_answer_truncated` is always zero here and printing it would
imply nothing was cut when cuad and techqa lose most of their context. The real
figure -- 92.4% of techqa, 73.5% of cuad, 19.2% of expertqa, 0.0% of the other
nine, 6.0% overall -- is measured separately from the tokenizer alone and
belongs beside the table, not in it.

The reference row is read from the in-distribution run's metrics.json rather
than typed in, so the gap column cannot drift away from the number it is
subtracting.

Usage:

    python -m src.c1_detector.evaluate_ood \\
        --checkpoint results/c1/modernbert-base/best \\
        --data-dir data/processed/ragbench \\
        --reference results/c1/test/metrics.json \\
        --out-dir results/ood/ragbench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.c1_detector.ragbench import SUBSETS

# The levels this evaluation is allowed to report. See the module docstring.
REPORTABLE_LEVELS = ("example",)


def subset_row(subset: str, metrics: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """One line of the OOD table, built from a subset's metrics block."""
    overall = metrics["overall"]
    example = overall["example"]
    n = overall["n_examples"]
    positive = sum(1 for r in records if r["spans"])
    rate = positive / n if n else 0.0
    return {
        "subset": subset,
        "n": n,
        "positive_rate": rate,
        "precision": example["precision"],
        "recall": example["recall"],
        "f1": example["f1"],
        "accuracy": example["accuracy"],
        "trivial_f1": trivial_f1(rate),
        "n_gold_spans": overall["n_gold_spans"],
        "n_pred_spans": overall["n_pred_spans"],
    }


def trivial_f1(positive_rate: float) -> float:
    """F1 of the classifier that calls every response hallucinated.

    Precision is the positive rate p, recall is 1, so F1 = 2p/(1+p). Cheap, and
    it is the only thing that makes a positive-class F1 readable when the
    positive rate moves between corpora -- RAGBench's is 14.2% against
    RAGTruth's 43.1%, so the same detector scores lower here for free.
    """
    return 2 * positive_rate / (1 + positive_rate) if positive_rate else 0.0


def pooled_row(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """A size-weighted mean of the per-subset F1s.

    NOT a micro-average over pooled predictions -- that would let tatqa and
    finqa, which are 48% of the records between them, decide the headline
    number. Both readings are defensible; this one is stated so nobody has to
    guess which was used.
    """
    total = sum(r["n"] for r in rows) or 1
    rate = sum(r["positive_rate"] * r["n"] for r in rows) / total
    return {
        "subset": "weighted mean",
        "n": total,
        "positive_rate": rate,
        "precision": sum(r["precision"] * r["n"] for r in rows) / total,
        "recall": sum(r["recall"] * r["n"] for r in rows) / total,
        "f1": sum(r["f1"] * r["n"] for r in rows) / total,
        "accuracy": sum(r["accuracy"] * r["n"] for r in rows) / total,
        "trivial_f1": trivial_f1(rate),
        "n_gold_spans": sum(r["n_gold_spans"] for r in rows),
        "n_pred_spans": sum(r["n_pred_spans"] for r in rows),
    }


def reference_row(path: Path) -> Optional[Dict[str, Any]]:
    """Read the in-distribution RAGTruth example-level block for the gap column."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    overall = metrics["overall"]
    example = overall["example"]
    n = overall["n_examples"]
    positive = n - example.get("tn", 0) - example.get("fp", 0)
    rate = positive / n if n else 0.0
    return {
        "subset": "RAGTruth test",
        "n": n,
        "positive_rate": rate,
        "precision": example["precision"],
        "recall": example["recall"],
        "f1": example["f1"],
        "accuracy": example.get("accuracy", 0.0),
        "trivial_f1": trivial_f1(rate),
        "n_gold_spans": overall["n_gold_spans"],
        "n_pred_spans": overall["n_pred_spans"],
    }


def format_ood_table(rows: Sequence[Dict[str, Any]], reference: Optional[Dict[str, Any]]) -> str:
    header = (
        f"{'subset':<15}{'n':>7}{'pos rate':>10}{'P':>8}{'R':>8}{'F1':>8}"
        f"{'trivial':>9}{'clears':>8}{'gold':>8}{'pred':>8}{'gap':>9}"
    )
    lines = [
        "Example-level only. RAGBench annotates whole sentences and RAGTruth",
        "annotates phrases, so span and token scores across the two are not",
        "comparable and are deliberately absent from this table.",
        "RAGBench labels are written by an LLM judge (gpt-4-turbo on 10,742",
        "records, gpt-4o on 1,059), not by human annotators.",
        "'pos rate' is the fraction of responses that are hallucinated. It is",
        "14.2% here against 34.9% on the RAGTruth test split (43.1% across the",
        "whole RAGTruth corpus), and positive-class F1 falls when positives get",
        "rarer even if the detector is unchanged.",
        "'trivial' is the F1 of calling EVERY response hallucinated at that rate.",
        "A row that does not clear it is a null result, not a weak one.",
        "'gold' and 'pred' are span counts: they say whether a low F1 comes from",
        "a detector that fires on everything or one that is silent.",
        "No truncation column: it would always read zero. cuad and techqa lose",
        "most of their context and that is measured separately.",
        "",
        header,
        "-" * len(header),
    ]

    def line(row: Dict[str, Any], gap: Optional[float]) -> str:
        gap_text = f"{gap:>+9.4f}" if gap is not None else f"{'':>9}"
        clears = "yes" if row["f1"] > row["trivial_f1"] else "NO"
        return (
            f"{row['subset']:<15}{row['n']:>7,}{row['positive_rate']:>10.3f}"
            f"{row['precision']:>8.4f}{row['recall']:>8.4f}{row['f1']:>8.4f}"
            f"{row['trivial_f1']:>9.4f}{clears:>8}"
            f"{row['n_gold_spans']:>8,}{row['n_pred_spans']:>8,}{gap_text}"
        )

    if reference is not None:
        lines.append(line(reference, None))
        lines.append("-" * len(header))

    base = reference["f1"] if reference else None
    for row in rows:
        lines.append(line(row, None if base is None else row["f1"] - base))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a RAGTruth-trained C1 checkpoint over the RAGBench test subsets."
    )
    parser.add_argument("--checkpoint", required=True, help="directory saved by train_c1")
    parser.add_argument("--data-dir", default="data/processed/ragbench")
    parser.add_argument("--subsets", nargs="*", default=list(SUBSETS))
    parser.add_argument("--reference", default="results/c1/test/metrics.json")
    parser.add_argument("--out-dir", default="results/ood/ragbench")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--limit", type=int, default=None, help="debug only: first N records per subset"
    )
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit to autodetect")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from src.c1_detector.dataset import (
        RagTruthTokenDataset,
        build_dataloader,
        read_records,
    )
    from src.c1_detector.evaluate_c1 import evaluate, predict

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint).to(device)
    print(f"loaded {args.checkpoint} on {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    all_metrics: Dict[str, Any] = {}

    for subset in args.subsets:
        path = data_dir / f"{subset}.jsonl"
        if not path.exists():
            print(f"{subset:<12} MISSING {path} -- run src.c1_detector.ragbench first")
            continue

        records = read_records(path, limit=args.limit)
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
        predictions = predict(model, loader, dataset, device)
        metrics = evaluate(predictions)
        all_metrics[subset] = metrics
        row = subset_row(subset, metrics, records)
        rows.append(row)
        print(
            f"{subset:<12} n {row['n']:>6,}  example F1 {row['f1']:.4f}  "
            f"trivial {row['trivial_f1']:.4f}  "
            f"pos rate {row['positive_rate']:.3f}  "
            f"pred spans {row['n_pred_spans']:,}"
        )

    if not rows:
        print("no subsets evaluated")
        return 1

    rows.append(pooled_row([r for r in rows]))
    reference = reference_row(Path(args.reference))

    table = format_ood_table(rows, reference)
    print()
    print(table)

    payload = {
        "checkpoint": args.checkpoint,
        "max_length": args.max_length,
        "reference": reference,
        "rows": rows,
        "per_subset_full_metrics": all_metrics,
        "note": (
            "Example-level only. RAGBench labels are generated by gpt-4o and are "
            "sentence-level; RAGTruth labels are human and phrase-level. Span and "
            "token scores are recorded here but are not comparable across the two "
            "corpora and must not be reported."
        ),
    }
    metrics_path = out_dir / "ood_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    table_path = out_dir / "ood_table.txt"
    table_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nwrote {metrics_path}\nwrote {table_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
