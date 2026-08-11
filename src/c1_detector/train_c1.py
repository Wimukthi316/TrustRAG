"""Fine-tune ModernBERT as a span-level hallucination detector (C1).

Plain PyTorch, deliberately. The installed transformers is 5.x, where the
Trainer/TrainingArguments surface has moved, and a training loop that breaks on
a library upgrade three weeks before a deadline is not worth the convenience.
An explicit loop also makes the two things this project has to defend visible
in one screen: the answer-only loss mask, and exactly which examples the model
never saw.

Everything configurable lives in a YAML file under configs/. Nothing about the
run -- not the learning rate, not the sequence length, not the split seed -- is
hard-coded here, so a result can always be traced back to a config file that is
committed alongside it.

Run:
    python -m src.c1_detector.train_c1 --config configs/c1_smoke.yaml
    python -m src.c1_detector.train_c1 --config configs/c1_base.yaml

The answer-only loss mask needs no special handling in this file: bio.py already
writes IGNORE_INDEX (-100) on every context, question, special and padding
position, and CrossEntropyLoss ignores those. The mask is a property of the
labels, which means it cannot silently fall out of sync with the batching.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.c1_detector.bio import IGNORE_INDEX, LABEL_NAMES
from src.c1_detector.dataset import (
    RagTruthTokenDataset,
    build_dataloader,
    read_records,
    split_records,
    subsample,
    write_split_ids,
)
from src.c1_detector.evaluate_c1 import (
    evaluate,
    format_diagnostics,
    format_report,
    predict,
    probability_summary,
)

# Fast tokenizers warn loudly when used inside forked DataLoader workers. The
# tokenisation here is per-item and single-threaded anyway, so silence it.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "run_name": "c1",
    "seed": 42,
    "data": {
        "train_file": "data/processed/ragtruth_train.jsonl",
        "test_file": "data/processed/ragtruth_test.jsonl",
        "limit": None,
        "val_fraction": 0.05,
        "calib_fraction": 0.10,
        "max_length": 3072,
    },
    "model": {
        "name": "answerdotai/ModernBERT-base",
        "gradient_checkpointing": False,
    },
    "train": {
        "epochs": 3,
        "batch_size": 4,
        "grad_accum": 4,
        "lr": 3.0e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "max_grad_norm": 1.0,
        "precision": "auto",
        "group_by_length": False,
        "num_workers": 0,
        "class_weights": None,
        "log_every": 20,
        "eval_batch_size": 8,
    },
    "output": {
        "dir": "results/c1/run",
        "save_best": True,
        "select_on": "example.f1",
    },
    "wandb": {
        "enabled": False,
        "project": "trustrag-c1",
        "entity": None,
    },
}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge `override` onto a copy of `base`, recursing into nested dicts."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: Path | str) -> Dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    config = deep_merge(DEFAULTS, loaded)
    unknown_top = set(loaded) - set(DEFAULTS)
    if unknown_top:
        raise ValueError(
            f"unknown top-level config keys {sorted(unknown_top)}; "
            f"expected some of {sorted(DEFAULTS)}"
        )
    return config


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_precision(preference: str, device: torch.device) -> str:
    """Pick the autocast dtype.

    bf16 where the hardware has it (Ampere and later, so the RTX 3050 and
    Kaggle's T4 generation onwards), otherwise fp16. The distinction matters on
    Kaggle: the P100 is Pascal and has no bf16 at all, so a config that demands
    bf16 there fails or silently falls back.

    fp16 needs a GradScaler because its dynamic range underflows gradients;
    bf16 does not. Both cases are handled below.
    """
    if device.type != "cuda":
        return "fp32"
    if preference != "auto":
        return preference
    try:
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    except Exception:
        return "fp16"


AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with weight decay off for biases and normalisation parameters.

    Decaying a LayerNorm gain toward zero shrinks the activations it is there to
    stabilise. Standard practice for transformer fine-tuning, and cheap to do.
    """
    no_decay = ("bias", "LayerNorm.weight", "layer_norm", "norm.weight")
    decay_params, plain_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in no_decay):
            plain_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": plain_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def get_metric(metrics: Dict[str, Any], dotted: str) -> float:
    """Pull 'example.f1' style paths out of the metrics dict."""
    node: Any = metrics["overall"]
    for part in dotted.split("."):
        node = node[part]
    return float(node)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


@dataclass
class RunState:
    best_score: float = -1.0
    best_epoch: int = -1
    global_step: int = 0


def train(config: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["train"]
    out_cfg = config["output"]

    set_seed(config["seed"])
    out_dir = Path(out_cfg["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    precision = resolve_precision(train_cfg["precision"], device)
    amp_dtype = AMP_DTYPES[precision]
    print(f"device={device} precision={precision}")

    # --- data ---------------------------------------------------------
    # Read the whole file, then subsample. Reading 15,090 records costs a couple
    # of seconds; taking the first N instead would hand a smoke run 500
    # summarization examples and zero QA, because the file is grouped by task.
    records = read_records(data_cfg["train_file"])
    records = subsample(records, data_cfg["limit"], seed=config["seed"])
    splits = split_records(
        records,
        val_fraction=data_cfg["val_fraction"],
        calib_fraction=data_cfg["calib_fraction"],
        seed=config["seed"],
    )
    write_split_ids(splits, out_dir / "split_ids.json")
    print(
        "records: "
        + "  ".join(f"{name} {len(rows):,}" for name, rows in splits.items())
    )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"], use_fast=True)

    train_set = RagTruthTokenDataset(
        splits["train"], tokenizer, max_length=data_cfg["max_length"]
    )
    val_set = RagTruthTokenDataset(
        splits["val"], tokenizer, max_length=data_cfg["max_length"], keep_offsets=True
    )

    train_loader, train_sampler = build_dataloader(
        train_set,
        tokenizer,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        group_by_length=train_cfg["group_by_length"],
        num_workers=train_cfg["num_workers"],
        seed=config["seed"],
    )
    val_loader, _ = build_dataloader(
        val_set,
        tokenizer,
        batch_size=train_cfg["eval_batch_size"],
        shuffle=False,
        group_by_length=False,
        num_workers=train_cfg["num_workers"],
    )

    # --- model --------------------------------------------------------
    model = AutoModelForTokenClassification.from_pretrained(
        model_cfg["name"],
        num_labels=len(LABEL_NAMES),
        id2label={i: name for i, name in enumerate(LABEL_NAMES)},
        label2id={name: i for i, name in enumerate(LABEL_NAMES)},
    ).to(device)

    if model_cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    # --- optimisation -------------------------------------------------
    grad_accum = max(1, int(train_cfg["grad_accum"]))
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg["warmup_ratio"])

    optimizer = build_optimizer(model, train_cfg["lr"], train_cfg["weight_decay"])

    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    weights = train_cfg["class_weights"]
    weight_tensor = (
        torch.tensor(weights, dtype=torch.float, device=device) if weights else None
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=IGNORE_INDEX)

    print(
        f"steps/epoch {steps_per_epoch:,}  total {total_steps:,}  "
        f"warmup {warmup_steps:,}  effective batch "
        f"{train_cfg['batch_size'] * grad_accum}"
    )

    run = _init_wandb(config)
    state = RunState()
    history: List[Dict[str, Any]] = []

    for epoch in range(train_cfg["epochs"]):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0
        epoch_start = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            if amp_dtype is not None:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    logits = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).logits
                    loss = loss_fn(logits.view(-1, len(LABEL_NAMES)), labels.view(-1))
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits.view(-1, len(LABEL_NAMES)), labels.view(-1))

            # A batch whose labels are all IGNORE_INDEX gives a NaN loss rather
            # than zero. It should not happen -- every record has answer tokens
            # -- but a NaN here would poison every weight in one step, so skip
            # loudly instead of training on it.
            if not torch.isfinite(loss):
                print(f"  non-finite loss at step {step}, batch skipped")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss / grad_accum).backward()
            running_loss += float(loss.detach())
            seen += 1

            is_accum_boundary = (step + 1) % grad_accum == 0
            is_last_batch = (step + 1) == len(train_loader)
            if is_accum_boundary or is_last_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg["max_grad_norm"]
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1

                if state.global_step % train_cfg["log_every"] == 0:
                    mean_loss = running_loss / max(seen, 1)
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"  epoch {epoch} step {state.global_step}/{total_steps} "
                        f"loss {mean_loss:.4f} lr {lr_now:.2e}"
                    )
                    _log(run, {"train/loss": mean_loss, "train/lr": lr_now}, state.global_step)
                    running_loss, seen = 0.0, 0

        train_seconds = time.time() - epoch_start
        print(f"epoch {epoch} trained in {train_seconds/60:.1f} min; evaluating")

        predictions = predict(model, val_loader, val_set, device, amp_dtype=amp_dtype)
        metrics = evaluate(predictions)
        print(format_report(metrics))

        # Argmax can predict nothing at all early in training, which makes every
        # F1 above 0.0000 and hides whether the model has learned anything. The
        # probability distribution and the threshold sweep tell those two cases
        # apart, and cost nothing -- the probabilities are already computed.
        diagnostics = probability_summary(predictions)
        print(format_diagnostics(predictions))

        score = get_metric(metrics, out_cfg["select_on"])
        history.append(
            {
                "epoch": epoch,
                "train_seconds": train_seconds,
                "select_on": out_cfg["select_on"],
                "score": score,
                "metrics": metrics,
                "probability_summary": diagnostics,
            }
        )
        _log(
            run,
            {
                "val/example_f1": get_metric(metrics, "example.f1"),
                "val/token_f1": get_metric(metrics, "token.f1"),
                "val/span_exact_f1": get_metric(metrics, "span_exact.f1"),
                "val/span_overlap_f1": get_metric(metrics, "span_overlap.f1"),
                "epoch": epoch,
            },
            state.global_step,
        )

        if score > state.best_score:
            state.best_score = score
            state.best_epoch = epoch
            if out_cfg["save_best"]:
                best_dir = out_dir / "best"
                model.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                print(f"  new best {out_cfg['select_on']}={score:.4f}, saved to {best_dir}")

        with (out_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    summary = {
        "run_name": config["run_name"],
        "best_epoch": state.best_epoch,
        "best_score": state.best_score,
        "select_on": out_cfg["select_on"],
        "precision": precision,
        "config": config,
        "history": history,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)

    if run is not None:
        run.finish()

    print(
        f"\nbest {out_cfg['select_on']} = {state.best_score:.4f} at epoch "
        f"{state.best_epoch}; wrote {out_dir/'summary.json'}"
    )
    return summary


# --------------------------------------------------------------------------
# W&B
# --------------------------------------------------------------------------


def _init_wandb(config: Dict[str, Any]) -> Optional[Any]:
    """Start a W&B run, or return None and carry on.

    Never fatal. A Kaggle session that dies because the logging service is
    unreachable has wasted GPU quota for nothing.
    """
    wandb_cfg = config["wandb"]
    if not wandb_cfg["enabled"]:
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("wandb enabled in config but WANDB_API_KEY is not set; logging disabled")
        return None
    try:
        import wandb

        return wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg["entity"],
            name=config["run_name"],
            config=config,
        )
    except Exception as exc:
        print(f"wandb init failed ({exc}); continuing without logging")
        return None


def _log(run: Optional[Any], payload: Dict[str, Any], step: int) -> None:
    if run is None:
        return
    try:
        run.log(payload, step=step)
    except Exception as exc:
        print(f"wandb log failed ({exc})")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the C1 span detector.")
    parser.add_argument("--config", required=True, help="YAML config under configs/")
    parser.add_argument("--limit", type=int, default=None, help="override data.limit")
    parser.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    parser.add_argument("--out-dir", default=None, help="override output.dir")
    parser.add_argument("--run-name", default=None, help="override run_name")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device", default=None, help="cuda, cpu, or omit to autodetect")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    config = load_config(args.config)
    if args.limit is not None:
        config["data"]["limit"] = args.limit
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    if args.out_dir:
        config["output"]["dir"] = args.out_dir
    if args.run_name:
        config["run_name"] = args.run_name
    if args.no_wandb:
        config["wandb"]["enabled"] = False

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train(config, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
