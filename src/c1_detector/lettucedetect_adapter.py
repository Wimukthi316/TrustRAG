"""Run the public LettuceDetect checkpoint and emit our probability format.

Two jobs:

1. **Baseline.** LettuceDetect reports 79.22% example-level F1 on RAGTruth. A
   number quoted from a paper is not a number we measured. This reproduces it
   under our own evaluation code, so the comparison in the report is
   like-for-like rather than our metric definition against theirs.

2. **Unblocking C2.** Split conformal needs scores from a detector on held-out
   data. It does not care which detector. Running C2 against this checkpoint
   while C1 is still training on Kaggle is worth several days, and when C1
   finishes the only thing that changes is which file C2 reads.

Everything about the input format was read from the LettuceDetect source, not
guessed:

    scripts/preprocess_ragtruth.py   prompt = source["prompt"]
    datasets/hallucination_dataset.py
        tokenizer(prompt, answer, truncation="only_first", max_length=4096)
    detectors/transformer.py
        P(hallucinated) = softmax(logits)[..., 1]

That first line is the one that matters. LettuceDetect does not rebuild a prompt
from the passages -- it feeds RAGTruth's own published `prompt` field, the exact
string that was given to the six generator LLMs. Handing this checkpoint a
differently-formatted context would put it off-distribution, and the
probabilities C2 then calibrated would be measuring our formatting mistake
rather than the model. Our own C1 uses a different layout (question + context),
which is fine because C1 is trained on that layout; this file exists precisely
so that difference does not leak.

The checkpoint has two labels, not our three. Nothing downstream needs to know:
this module emits the same per-example dicts `evaluate_c1.predict` produces, so
the metric and dump functions are reused unchanged.

Run:
    python -m src.c1_detector.lettucedetect_adapter --split test --dump-probs
    python -m src.c1_detector.lettucedetect_adapter --split calib --dump-probs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from src.c1_detector.bio import ANSWER_SEQUENCE_ID, encode_example
from src.c1_detector.dataset import read_records, split_records
from src.c1_detector.evaluate_c1 import (
    binary_from_hal_mask,
    dump_probabilities,
    evaluate,
    format_diagnostics,
    format_report,
)
from src.c1_detector.evaluate_c1 import (
    spans_from_token_mask as _spans_from_token_mask,
)

MODEL_ID = "KRLabsOrg/lettucedect-large-modernbert-en-v1"

# LettuceDetect's checkpoint is binary: index 1 is the hallucinated class.
HALLUCINATED_INDEX = 1


def load_raw_prompts(path: Path | str) -> Dict[str, str]:
    """source_id -> RAGTruth's own published prompt string.

    This is the field the dataset authors gave to the generator LLMs, and the
    field LettuceDetect trained on. Read from data/raw, not from our processed
    records, because build_examples deliberately reshapes context and question
    into the layout our C1 wants.
    """
    prompts: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompts[str(record["source_id"])] = str(record["prompt"])
    return prompts


def as_lettucedetect_record(record: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    """Re-point a processed record at LettuceDetect's prompt layout.

    encode_example builds its first sequence as `question\\n\\ncontext`, or just
    `context` when there is no question. Blanking the question and putting the
    raw prompt in the context slot therefore reproduces
    `tokenizer(prompt, answer, ...)` exactly, and every other piece of machinery
    -- offsets, sequence ids, BIO labels, the round-trip guarantee -- is the
    code already tested against the full corpus.
    """
    return {**record, "question": "", "context": prompt}


@torch.no_grad()
def predict_binary(
    model: Any,
    tokenizer: Any,
    records: Sequence[Dict[str, Any]],
    prompts: Dict[str, str],
    device: torch.device,
    max_length: int = 4096,
    amp_dtype: Optional[torch.dtype] = None,
    threshold: Optional[float] = None,
    log_every: int = 200,
) -> List[Dict[str, Any]]:
    """Score every record, returning the same dicts `evaluate_c1.predict` returns.

    One record per forward pass. Batching would need padding across a 400-to-
    4,096-token range for a 396M-parameter model on 6GB, and the whole job is a
    few thousand records -- not worth the memory risk for the time saved.
    """
    model.eval()
    results: List[Dict[str, Any]] = []
    missing_prompt = 0

    for position, record in enumerate(records):
        source_id = str(record.get("source_id"))
        prompt = prompts.get(source_id)
        if prompt is None:
            missing_prompt += 1
            continue

        encoded = encode_example(
            tokenizer, as_lettucedetect_record(record, prompt), max_length=max_length
        )
        input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
        attention_mask = torch.tensor(
            [encoded["attention_mask"]], dtype=torch.long, device=device
        )

        if amp_dtype is not None and device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        else:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        # float32 softmax regardless of the autocast dtype: these probabilities
        # are exactly what C2 calibrates, and half-precision rounding would put
        # a floor under every ECE number computed from them.
        probs = torch.softmax(logits.float(), dim=-1)[0].cpu()

        sequence_ids = encoded["sequence_ids"]
        offsets = encoded["offset_mapping"]
        positions = [
            i
            for i, sid in enumerate(sequence_ids)
            if sid == ANSWER_SEQUENCE_ID and offsets[i][1] > offsets[i][0]
        ]

        token_probs = [float(probs[i, HALLUCINATED_INDEX]) for i in positions]
        gold_ids = [int(encoded["labels"][i]) for i in positions]
        cut = 0.5 if threshold is None else threshold
        mask = [p >= cut for p in token_probs]
        pred_ids = binary_from_hal_mask(mask)

        answer = record["answer"]
        answer_offsets = [tuple(offsets[i]) for i in positions]

        results.append(
            {
                "index": position,
                "id": record.get("id"),
                "task_type": record.get("task_type"),
                "model": record.get("model"),
                "answer": answer,
                "gold_spans": [(s["start"], s["end"]) for s in record.get("spans", [])],
                "pred_spans": _spans_from_token_mask(mask, answer_offsets, answer),
                "gold_ids": gold_ids,
                "pred_ids": pred_ids,
                "token_probs": token_probs,
                "answer_offsets": answer_offsets,
                "answer_truncated": bool(encoded["answer_truncated"]),
            }
        )

        if log_every and (position + 1) % log_every == 0:
            print(f"  {position + 1:,}/{len(records):,}", flush=True)

    if missing_prompt:
        raise SystemExit(
            f"{missing_prompt} records had no matching source_id in source_info.jsonl; "
            "the raw and processed files are out of sync"
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score RAGTruth with the public LettuceDetect checkpoint."
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "train", "val", "calib"],
        help=(
            "test reads the test file; train/val/calib re-derive the split of the "
            "train file, so calib is exactly the set C1 will hold out"
        ),
    )
    parser.add_argument("--train-file", default="data/processed/ragtruth_train.jsonl")
    parser.add_argument("--test-file", default="data/processed/ragtruth_test.jsonl")
    parser.add_argument("--source-info", default="data/raw/source_info.jsonl")
    # Defaults deliberately mirror configs/c1_base.yaml. If they drift apart the
    # calibration set stops being the set C1 held out, and the conformal
    # guarantee quietly stops holding.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--calib-fraction", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=None, help="debug only")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--dump-probs", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    args = parser.parse_args()

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    precision = args.precision
    if precision == "auto":
        precision = "fp32" if device.type != "cuda" else (
            "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        )
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
    print(f"device={device} precision={precision} model={args.model}")

    if args.split == "test":
        records = read_records(args.test_file)
    else:
        records = split_records(
            read_records(args.train_file),
            val_fraction=args.val_fraction,
            calib_fraction=args.calib_fraction,
            seed=args.seed,
        )[args.split]
    if args.limit:
        records = records[: args.limit]
    print(f"split={args.split} records={len(records):,}")

    prompts = load_raw_prompts(args.source_info)
    print(f"loaded {len(prompts):,} raw prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model).to(device)
    if model.config.num_labels != 2:
        raise SystemExit(
            f"expected a 2-label checkpoint, got {model.config.num_labels}; "
            "the hallucinated-class index in this file would be wrong"
        )

    predictions = predict_binary(
        model,
        tokenizer,
        records,
        prompts,
        device,
        max_length=args.max_length,
        amp_dtype=amp_dtype,
        threshold=args.threshold,
    )

    metrics = evaluate(predictions)
    print(format_report(metrics))
    print(format_diagnostics(predictions))

    out_dir = Path(args.out_dir or f"results/lettucedetect/{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"model": args.model, "split": args.split, "n": len(predictions), **metrics},
            handle,
            indent=2,
        )
    print(f"\nwrote {out_dir / 'metrics.json'}")

    if args.dump_probs:
        path = out_dir / "probabilities.jsonl"
        print(f"wrote {dump_probabilities(predictions, path):,} records to {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
