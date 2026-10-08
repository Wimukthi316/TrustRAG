"""Medical example-level evaluation of C1 on MedHallu (medical specialization, Phase 1).

Runs a RAGTruth-trained C1 checkpoint over MedHallu and reports whether the
general detector transfers to medicine, at the only level MedHallu supports:
example-level (does the detector flag this answer at all). It then asks C2's
question at that level -- is the detector's confidence calibrated on medical
input, and does it improve after recalibrating on a medical split.

Why a separate evaluator instead of evaluate_c1.evaluate: MedHallu is
answer-level, so a positive record has NO gold spans. evaluate_c1.example_metrics
reads the example label from `len(gold_spans) > 0`, which would call every
MedHallu positive a negative. Here the label is read from the explicit
`gold_positive` field that medhallu.py writes.

WHAT IS AND IS NOT REPORTABLE
  - Example-level P/R/F1 only. No span or token number -- MedHallu has no gold
    spans. The detector inference still produces spans; they are the prediction,
    never scored against a gold span here.
  - The positive rate is 0.5 by construction (one hallucinated + one ground-truth
    answer per row), so the trivial "flag everything" F1 is 0.667. A row that
    does not clear it is a null result. It is printed beside the F1.
  - MedHallu's hallucinations are model-generated; this measures transfer to that
    construction, not to human medical span annotation.

The detector's native flag decision (argmax -> any predicted span) is the
example prediction, i.e. what the product actually does. The calibration score is
the answer-level P(hallucinated) = max over answer tokens, which is the
continuous signal C2 calibrates. The two are reported side by side.

Usage:

    python -m src.c1_detector.medhallu --out-dir data/processed/medhallu
    python -m src.c1_detector.evaluate_medhallu \\
        --checkpoint results/c1/modernbert-base-seed13/best \\
        --data-dir data/processed/medhallu \\
        --out-dir results/medical/medhallu
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from src.c1_detector.medhallu import CONFIGS


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def trivial_f1(positive_rate: float) -> float:
    """F1 of calling every answer hallucinated: 2p/(1+p). The floor to clear."""
    return 2 * positive_rate / (1 + positive_rate) if positive_rate else 0.0


def example_block(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Example-level confusion + P/R/F1 from explicit gold labels."""
    tp = fp = fn = tn = 0
    for r in rows:
        gold = bool(r["gold_positive"])
        pred = bool(r["pred_positive"])
        if gold and pred:
            tp += 1
        elif pred:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    out = prf(tp, fp, fn)
    out["tn"] = tn
    n = tp + fp + fn + tn
    positive = tp + fn
    out["n"] = n
    out["positive_rate"] = positive / n if n else 0.0
    out["accuracy"] = (tp + tn) / n if n else 0.0
    out["trivial_f1"] = trivial_f1(out["positive_rate"])
    out["clears_trivial"] = out["f1"] > out["trivial_f1"]
    return out


def stratified_split(
    labels: Sequence[int], calib_fraction: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Indices split into (calibration, test), keeping each class's rate in both."""
    rng = random.Random(seed)
    calib: List[int] = []
    test: List[int] = []
    for value in (0, 1):
        members = [i for i, lab in enumerate(labels) if lab == value]
        rng.shuffle(members)
        cut = int(round(len(members) * calib_fraction))
        calib.extend(members[:cut])
        test.extend(members[cut:])
    calib.sort()
    test.sort()
    return calib, test


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a RAGTruth-trained C1 checkpoint over MedHallu (medical, example-level)."
    )
    parser.add_argument("--checkpoint", required=True, help="directory saved by train_c1")
    parser.add_argument("--data-dir", default="data/processed/medhallu")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--out-dir", default="results/medical/medhallu")
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--calib-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None, help="debug only: first N records per config")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit to autodetect")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from src.c1_detector.dataset import (
        RagTruthTokenDataset,
        build_dataloader,
        read_records,
    )
    from src.c1_detector.evaluate_c1 import predict
    from src.c2_calibration.calibration import (
        best_calibrator,
        calibration_report,
        format_calibration_table,
    )
    from src.c2_calibration.conformal import (
        area_under_risk_coverage,
        risk_coverage_curve,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint).to(device)
    print(f"loaded {args.checkpoint} on {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-answer rows carrying everything the metrics and C2 need.
    answers: List[Dict[str, Any]] = []

    for config in args.configs:
        path = data_dir / f"{config}.jsonl"
        if not path.exists():
            print(f"{config:<16} MISSING {path} -- run src.c1_detector.medhallu first")
            continue
        records = read_records(path, limit=args.limit)
        meta = {
            r["id"]: {
                "gold_positive": bool(r.get("gold_positive")),
                "difficulty": str(r.get("difficulty") or ""),
                "category": str(r.get("category") or ""),
                "config": config,
            }
            for r in records
        }
        dataset = RagTruthTokenDataset(
            records, tokenizer, max_length=args.max_length, keep_offsets=True
        )
        loader, _ = build_dataloader(
            dataset, tokenizer, batch_size=args.batch_size, shuffle=False,
            group_by_length=False, num_workers=args.num_workers,
        )
        predictions = predict(model, loader, dataset, device)
        for item in predictions:
            info = meta.get(item["id"])
            if info is None:
                continue
            probs = item["token_probs"]
            answers.append(
                {
                    "id": item["id"],
                    "config": info["config"],
                    "difficulty": info["difficulty"],
                    "category": info["category"],
                    "gold_positive": info["gold_positive"],
                    # Native product decision: argmax -> any predicted span.
                    "pred_positive": len(item["pred_spans"]) > 0,
                    # Answer-level hallucination score C2 calibrates.
                    "score": max(probs) if probs else 0.0,
                    "mean_token_prob": (sum(probs) / len(probs)) if probs else 0.0,
                    "n_answer_tokens": len(probs),
                    "n_pred_spans": len(item["pred_spans"]),
                    "answer_truncated": bool(item["answer_truncated"]),
                }
            )
        pos = sum(1 for a in answers if a["config"] == config and a["gold_positive"])
        print(f"{config:<16} {sum(1 for a in answers if a['config']==config):>6,} answers  ({pos:,} positive)")

    if not answers:
        print("no answers evaluated -- did you run src.c1_detector.medhallu?")
        return 1

    # ---- example-level detection -----------------------------------------
    overall = example_block(answers)
    by_difficulty = {
        d: example_block([a for a in answers if a["difficulty"] == d])
        for d in sorted({a["difficulty"] for a in answers if a["difficulty"]})
    }
    by_config = {
        c: example_block([a for a in answers if a["config"] == c])
        for c in sorted({a["config"] for a in answers})
    }

    # Per-category recall on positives only: of the hallucinations of this kind,
    # what fraction did the detector flag. Negatives have no category.
    by_category: Dict[str, Dict[str, Any]] = {}
    for cat in sorted({a["category"] for a in answers if a["gold_positive"] and a["category"]}):
        members = [a for a in answers if a["gold_positive"] and a["category"] == cat]
        flagged = sum(1 for a in members if a["pred_positive"])
        by_category[cat] = {"n": len(members), "flagged": flagged,
                            "recall": flagged / len(members) if members else 0.0}

    # ---- calibration (C2 at the example level) ---------------------------
    scores = [a["score"] for a in answers]
    labels = [1 if a["gold_positive"] else 0 for a in answers]
    calib_idx, test_idx = stratified_split(labels, args.calib_fraction, args.seed)
    calib_scores = [scores[i] for i in calib_idx]
    calib_labels = [labels[i] for i in calib_idx]
    test_scores = [scores[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    raw_report = calibration_report(test_scores, test_labels)
    calibrator, calib_rows = best_calibrator(
        calib_scores, calib_labels, test_scores, test_labels
    )
    calibrated_scores = [float(x) for x in calibrator.transform(test_scores)]
    calibrated_report = calibration_report(calibrated_scores, test_labels)

    rc_curve = risk_coverage_curve(scores, labels)
    aurc = area_under_risk_coverage(rc_curve)

    # ---- report ----------------------------------------------------------
    print()
    print(f"example-level detection (positive rate {overall['positive_rate']:.3f}, "
          f"trivial F1 {overall['trivial_f1']:.4f}):")
    print(f"  overall   P {overall['precision']:.4f}  R {overall['recall']:.4f}  "
          f"F1 {overall['f1']:.4f}  acc {overall['accuracy']:.4f}  "
          f"clears trivial: {'yes' if overall['clears_trivial'] else 'NO'}")
    for d, block in by_difficulty.items():
        print(f"  {d:<8}  F1 {block['f1']:.4f}  (n {block['n']:,})")
    print()
    print("calibration on the medical test split (lower ECE is better):")
    print(f"  raw        ECE {raw_report['ece']:.4f}  Brier {raw_report['brier']:.4f}")
    print(f"  {calibrator.name:<10} ECE {calibrated_report['ece']:.4f}  "
          f"Brier {calibrated_report['brier']:.4f}")
    print(format_calibration_table(calib_rows))
    print(f"  risk-coverage AURC {aurc:.4f}")

    payload = {
        "checkpoint": args.checkpoint,
        "max_length": args.max_length,
        "n_answers": len(answers),
        "example": {
            "overall": overall,
            "by_difficulty": by_difficulty,
            "by_config": by_config,
            "by_category_recall": by_category,
        },
        "calibration": {
            "calib_fraction": args.calib_fraction,
            "seed": args.seed,
            "n_calibration": len(calib_idx),
            "n_test": len(test_idx),
            "raw": raw_report,
            "best_method": calibrator.name,
            "best_params": calibrator.params(),
            "calibrated": calibrated_report,
            "all_methods": calib_rows,
        },
        "risk_coverage": {"aurc": aurc, "curve": rc_curve},
        "answer_truncated": sum(1 for a in answers if a["answer_truncated"]),
        "note": (
            "Example-level only; MedHallu is answer-level with model-generated "
            "hallucinations. Positive rate is 0.5 by construction (trivial F1 "
            "0.667). Example prediction is the detector's native argmax flag; the "
            "calibration score is max answer-token P(hallucinated). Best calibrator "
            "is selected on the test split's ECE (a mild peek, declared)."
        ),
    }
    metrics_path = out_dir / "medhallu_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    scores_path = out_dir / "answer_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as handle:
        for a in answers:
            handle.write(json.dumps(a) + "\n")

    print(f"\nwrote {metrics_path}\nwrote {scores_path} ({len(answers):,} answers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
