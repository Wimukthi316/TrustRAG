"""Answer-level medical hallucination detector trained on real MedHallu labels.

The span detector (C1) does not transfer to medical hallucination detection:
general ModernBERT scores MedHallu example-F1 0.5259, the biomedical backbone
0.5583 (both below the 0.6667 trivial floor), and fine-tuning on synthetic
medical spans HURTS it (0.2168). The cause is that no real medical SPAN data
exists, so we trained on general RAGTruth spans and rule-injected synthetic ones,
neither of which matches MedHallu's subtle, model-generated hallucinations.

This module takes the other lever: MedHallu is labelled at the ANSWER level, and
there is a lot of it (pqa_artificial, ~9k rows). So instead of detecting spans we
train a sequence classifier -- biomedical backbone, a two-class head
(faithful / hallucinated) -- directly on real medical hallucinations, and test it
on exactly the pqa_labeled set the span detector was measured on, so the two
numbers are comparable.

What this is and is not:
  - It is answer-level. It says whether a whole answer is hallucinated, not which
    words. Span localisation stays with C1; this is the medical FLAG.
  - It trains on pqa_artificial and tests on pqa_labeled, which are disjoint
    PubMedQA subsets; --held-out-answers adds a belt-and-suspenders exact-text
    exclusion so no training answer can reappear in the test set.
  - Its per-answer P(hallucinated) is dumped in the same answer_scores.jsonl
    shape evaluate_medhallu writes, so src.c2_calibration.medical_conformal runs
    on it unchanged -- the calibrated coverage guarantee sits on top of whichever
    detector is better.

HONESTY. Every number comes from this run. Subtle medical hallucination detection
is genuinely hard -- MedHallu reports GPT-4o around 0.625 F1 on its hard split --
so report whatever this scores against 0.5583 and the 0.6667 floor without
dressing it up. The trivial "flag everything" baseline is printed beside the F1.

Usage (Kaggle):
    python -m src.c1_detector.train_medhallu_classifier \\
        --model thomas-sounack/BioClinical-ModernBERT-base \\
        --train-config pqa_artificial --eval-config pqa_labeled \\
        --out-dir /kaggle/working/results/medical/medhallu-classifier
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_MODEL = "thomas-sounack/BioClinical-ModernBERT-base"


def _norm(text: str) -> str:
    """Whitespace/case-insensitive key for excluding eval answers from training."""
    return " ".join(text.lower().split())


def build_examples(
    config: str,
    exclude_answers: Optional[set] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """One (prompt, answer, label) example per MedHallu answer.

    Reuses medhallu.build_records, which already expands each row into a
    hallucinated (label 1) and a ground-truth (label 0) record in the processed
    shape, with the Knowledge passages formatted the way the detector reads RAG
    context. A row whose clean answer is in exclude_answers is dropped whole.
    """
    from src.c1_detector.medhallu import build_records

    records = build_records(config, limit=limit)
    out: List[Dict[str, Any]] = []
    dropped = 0
    for r in records:
        answer = str(r.get("answer") or "").strip()
        if not answer:
            continue
        if exclude_answers is not None and _norm(answer) in exclude_answers:
            dropped += 1
            continue
        out.append({
            "id": r["id"],
            "prompt": _first_sequence(r.get("question", ""), r.get("context", "")),
            "answer": answer,
            "label": 1 if r.get("gold_positive") else 0,
            "difficulty": str(r.get("difficulty") or ""),
            "category": str(r.get("category") or ""),
            "config": config,
        })
    if exclude_answers is not None:
        print(f"{config}: dropped {dropped} example(s) whose answer appears in the eval set")
    return out


def _first_sequence(question: str, context: str) -> str:
    """question + blank line + context, or context alone -- the C1 input shape."""
    question = (question or "").strip()
    context = (context or "").strip()
    return f"{question}\n\n{context}" if question else context


def answer_texts(config: str, limit: Optional[int] = None) -> set:
    """Every answer string (ground-truth and hallucinated) in a config, normalised."""
    from src.c1_detector.medhallu import build_records

    out = set()
    for r in build_records(config, limit=limit):
        a = str(r.get("answer") or "").strip()
        if a:
            out.add(_norm(a))
    return out


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def trivial_f1(positive_rate: float) -> float:
    return 2 * positive_rate / (1 + positive_rate) if positive_rate else 0.0


def example_block(labels: Sequence[int], preds: Sequence[int]) -> Dict[str, Any]:
    tp = sum(1 for g, p in zip(labels, preds) if g and p)
    fp = sum(1 for g, p in zip(labels, preds) if not g and p)
    fn = sum(1 for g, p in zip(labels, preds) if g and not p)
    tn = sum(1 for g, p in zip(labels, preds) if not g and not p)
    out = prf(tp, fp, fn)
    n = tp + fp + fn + tn
    positive = tp + fn
    out.update({
        "tn": tn, "n": n,
        "positive_rate": positive / n if n else 0.0,
        "accuracy": (tp + tn) / n if n else 0.0,
    })
    out["trivial_f1"] = trivial_f1(out["positive_rate"])
    out["clears_trivial"] = out["f1"] > out["trivial_f1"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train an answer-level MedHallu hallucination classifier."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-config", default="pqa_artificial")
    parser.add_argument("--eval-config", default="pqa_labeled")
    parser.add_argument("--out-dir", default="results/medical/medhallu-classifier")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None, help="debug: first N records per config")
    parser.add_argument("--no-dedup", action="store_true", help="skip the eval-answer exclusion")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_cosine_schedule_with_warmup,
    )

    from src.c2_calibration.calibration import calibration_report

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device {device}  model {args.model}")

    exclude = None if args.no_dedup else answer_texts(args.eval_config, args.limit)
    train_examples = build_examples(args.train_config, exclude_answers=exclude, limit=args.limit)
    eval_examples = build_examples(args.eval_config, exclude_answers=None, limit=args.limit)
    print(f"train {len(train_examples):,} examples ({args.train_config}); "
          f"eval {len(eval_examples):,} examples ({args.eval_config})")

    # Hold a small validation slice out of training for model selection.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(train_examples))
    n_val = max(1, int(round(len(train_examples) * args.val_fraction)))
    val_idx = set(order[:n_val].tolist())
    train_set = [e for i, e in enumerate(train_examples) if i not in val_idx]
    val_set = [e for i, e in enumerate(train_examples) if i in val_idx]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2
    ).to(device)

    def encode(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        enc = tokenizer(
            [b["prompt"] for b in batch],
            [b["answer"] for b in batch],
            truncation="only_first",
            max_length=args.max_length,
        )
        enc["labels"] = [b["label"] for b in batch]
        return enc

    collator = DataCollatorWithPadding(tokenizer)

    def make_loader(data: List[Dict[str, Any]], shuffle: bool) -> DataLoader:
        feats = [encode([e]) for e in data]
        feats = [{k: v[0] for k, v in f.items()} for f in feats]
        return DataLoader(feats, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collator)

    train_loader = make_loader(train_set, True)
    val_loader = make_loader(val_set, False)
    eval_loader = make_loader(eval_examples, False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = (len(train_loader) // max(1, args.grad_accum)) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optim, int(steps * args.warmup_ratio), steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    @torch.no_grad()
    def predict(loader) -> Tuple[List[float], List[int]]:
        model.eval()
        scores: List[float] = []
        labels: List[int] = []
        for batch in loader:
            labels.extend(batch.pop("labels").tolist())
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(**batch).logits
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
        return scores, labels

    def val_f1() -> float:
        scores, labels = predict(val_loader)
        preds = [1 if s >= 0.5 else 0 for s in scores]
        return example_block(labels, preds)["f1"]

    best_f1 = -1.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        optim.zero_grad()
        running = 0.0
        for step, batch in enumerate(train_loader):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(**batch, labels=labels)
                loss = out.loss / args.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * args.grad_accum
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()
                scheduler.step()
            if (step + 1) % 100 == 0:
                print(f"  epoch {epoch+1} step {step+1}/{len(train_loader)} "
                      f"loss {running/(step+1):.4f}", flush=True)
        f1 = val_f1()
        print(f"epoch {epoch+1}: val example F1 {f1:.4f}", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- final evaluation on the held-out MedHallu config --------------------
    scores, labels = predict(eval_loader)
    preds = [1 if s >= 0.5 else 0 for s in scores]
    overall = example_block(labels, preds)

    by_difficulty: Dict[str, Any] = {}
    for level in sorted({e["difficulty"] for e in eval_examples if e["difficulty"]}):
        idx = [i for i, e in enumerate(eval_examples) if e["difficulty"] == level]
        by_difficulty[level] = example_block(
            [labels[i] for i in idx], [preds[i] for i in idx]
        )

    cal = calibration_report(scores, labels)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # answer_scores.jsonl in evaluate_medhallu's shape, so medical_conformal reads it.
    scores_path = out_dir / "answer_scores.jsonl"
    with scores_path.open("w", encoding="utf-8") as handle:
        for e, s, p in zip(eval_examples, scores, preds):
            handle.write(json.dumps({
                "id": e["id"], "config": e["config"], "difficulty": e["difficulty"],
                "category": e["category"], "gold_positive": bool(e["label"]),
                "pred_positive": bool(p), "score": float(s),
            }) + "\n")

    payload = {
        "model": args.model,
        "train_config": args.train_config,
        "eval_config": args.eval_config,
        "n_train": len(train_set),
        "n_eval": len(eval_examples),
        "best_val_f1": best_f1,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "example": {"overall": overall, "by_difficulty": by_difficulty},
        "calibration_raw": cal,
        "note": (
            "Answer-level medical hallucination classifier trained on real "
            "MedHallu labels (pqa_artificial), tested on pqa_labeled -- the same "
            "set the span detector scored 0.5583 on. Answer-level only; no spans. "
            "Trivial floor is 0.6667 by construction (balanced). Scores feed "
            "src.c2_calibration.medical_conformal unchanged."
        ),
    }
    metrics_path = out_dir / "medhallu_classifier_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print()
    print(f"MedHallu {args.eval_config} answer-level detector "
          f"(positive rate {overall['positive_rate']:.3f}, trivial F1 {overall['trivial_f1']:.4f}):")
    print(f"  overall   P {overall['precision']:.4f}  R {overall['recall']:.4f}  "
          f"F1 {overall['f1']:.4f}  acc {overall['accuracy']:.4f}  "
          f"clears trivial: {'YES' if overall['clears_trivial'] else 'NO'}")
    for level, block in by_difficulty.items():
        print(f"  {level:<8}  F1 {block['f1']:.4f}  (n {block['n']:,})")
    print(f"  calibration raw ECE {cal['ece']:.4f}  Brier {cal['brier']:.4f}")
    print(f"\nwrote {metrics_path}\nwrote {scores_path} ({len(eval_examples):,} answers)")
    print("Compare F1 above to the span detector's 0.5583 and the 0.6667 floor. "
          "Then run src.c2_calibration.medical_conformal on the answer_scores.jsonl.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
