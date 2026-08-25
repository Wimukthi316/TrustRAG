"""Train the C3 two-axis type head. One Kaggle run, configs/c3_type.yaml.

Usage on Kaggle, from a notebook that only clones, installs and calls this:

    python -m src.c3_explanation.train_c3_type --config configs/c3_type.yaml

What it does NOT do, deliberately: it never reads or writes C1's checkpoint, and
it never writes anything under results/c1/ or data/. Those are frozen inputs and
every C2 number in the handoff assumes them unchanged. It reuses C1's frozen
split ids so that the calibration split C3's conformal layer uses is the same
1,509 responses C2 calibrates on, and is therefore genuinely held out from C1's
training as well.

Model selection is on axis A macro F1, not on 4-way accuracy. Axis B is dominated
by 'evident' at roughly 88% of the test split, so a 4-way accuracy would select
the epoch that guesses the majority class most confidently.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.c3_explanation.type_eval import classification_report
from src.c3_explanation.type_model import (
    MECHANISMS,
    SEVERITIES,
    TypeDataset,
    TwoAxisTypeModel,
    collate,
    examples_from_rows,
)


def resolve_max_length(requested: int, model_config: Any) -> int:
    """Never ask for more positions than the encoder has.

    ModernBERT supports 8192 so the configured 1024 passes through untouched. A
    smaller encoder -- a tiny model in a smoke test, or a swap to a different
    checkpoint -- would otherwise fail deep inside the embedding layer with a
    shape error that says nothing about the cause.
    """
    limit = int(getattr(model_config, "max_position_embeddings", 0) or 0)
    if limit and requested > limit:
        print(
            f"note: max_length {requested} exceeds this encoder's "
            f"{limit} positions; using {limit}"
        )
        return limit
    return requested


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_split_of(path: Optional[Path]) -> Dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    out: Dict[str, str] = {}
    for name, ids in json.loads(Path(path).read_text(encoding="utf-8")).items():
        for rid in ids:
            out[str(rid)] = name
    return out


def gold_model_rows(
    corpus: Sequence[Dict[str, Any]],
    split_of: Dict[str, str],
    keep: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """One row per annotated gold span, carrying everything the model input needs."""
    from src.c3_explanation.type_baseline import ERROR_TYPES, primary_type

    rows: List[Dict[str, Any]] = []
    for record in corpus:
        rid = str(record["id"])
        split = split_of.get(rid, str(record.get("split", "unknown")))
        if keep is not None and split not in keep:
            continue
        for span in record.get("spans") or []:
            error_type = primary_type(span)
            if error_type not in ERROR_TYPES:
                continue
            rows.append(
                {
                    "response_id": rid,
                    "split": split,
                    "task_type": str(record.get("task_type", "other")),
                    "model": str(record.get("model", "unknown")),
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                    "error_type": error_type,
                    "answer": record.get("answer", ""),
                    "question": record.get("question", ""),
                    "context": record.get("context", ""),
                    "source": "gold",
                }
            )
    return rows


def evaluate(
    model: TwoAxisTypeModel,
    loader: DataLoader,
    device: torch.device,
    examples: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    model.eval()
    mechanism_pred: List[int] = []
    severity_pred: List[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            mechanism_pred += out["mechanism"].argmax(dim=-1).tolist()
            severity_pred += out["severity"].argmax(dim=-1).tolist()
    mechanism_gold = [MECHANISMS[e["mechanism_label"]] for e in examples]
    severity_gold = [SEVERITIES[e["severity_label"]] for e in examples]
    mechanism = classification_report(
        mechanism_gold, [MECHANISMS[i] for i in mechanism_pred], MECHANISMS
    )
    severity = classification_report(
        severity_gold, [SEVERITIES[i] for i in severity_pred], SEVERITIES
    )
    return {
        "mechanism_macro_f1": mechanism["macro_f1"],
        "mechanism_accuracy": mechanism["accuracy"],
        "severity_macro_f1": severity["macro_f1"],
        "severity_accuracy": severity["accuracy"],
        "mechanism": mechanism,
        "severity": severity,
    }


def resolve_precision(name: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "auto":
        return (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Train the C3 two-axis type head.")
    parser.add_argument("--config", default="configs/c3_type.yaml")
    parser.add_argument("--limit", type=int, default=None, help="smoke-test row cap")
    parser.add_argument("--model-name", default=None, help="override for a CPU smoke test")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args(argv)

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(config.get("seed", 42)))

    data = config["data"]
    train_config = config["train"]
    model_name = args.model_name or config["model"]["name"]
    epochs = args.epochs if args.epochs is not None else int(train_config["epochs"])

    corpus = read_jsonl(Path(data["train_file"]))
    split_of = load_split_of(Path(data["split_ids"]) if data.get("split_ids") else None)
    train_rows = gold_model_rows(corpus, split_of, keep=("train",))
    val_rows = gold_model_rows(corpus, split_of, keep=("val",))
    if args.limit:
        train_rows = train_rows[: args.limit]
        val_rows = val_rows[: max(2, args.limit // 4)]
    print(f"train spans {len(train_rows):,}  val spans {len(val_rows):,}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    hidden = model_config.hidden_size
    model = TwoAxisTypeModel(
        encoder, hidden, dropout=float(config["model"].get("dropout", 0.1))
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_examples = examples_from_rows(
        train_rows, data.get("span_open", "<<"), data.get("span_close", ">>")
    )
    val_examples = examples_from_rows(
        val_rows, data.get("span_open", "<<"), data.get("span_close", ">>")
    )
    max_length = resolve_max_length(int(data.get("max_length", 1024)), model_config)

    def loader_for(examples, batch_size, shuffle):
        return DataLoader(
            TypeDataset(examples, tokenizer, max_length),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=lambda batch: collate(batch, tokenizer),
        )

    train_loader = loader_for(train_examples, int(train_config["batch_size"]), True)
    val_loader = loader_for(val_examples, int(train_config["eval_batch_size"]), False)

    grad_accum = max(1, int(train_config.get("grad_accum", 1)))
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = max(1, steps_per_epoch * epochs)

    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    warmup = int(total_steps * float(train_config.get("warmup_ratio", 0.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda step: (
            (step + 1) / max(1, warmup)
            if step < warmup
            else max(0.0, (total_steps - step) / max(1, total_steps - warmup))
        ),
    )

    dtype = resolve_precision(str(train_config.get("precision", "auto")), device)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    out_dir = Path(config["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    select_on = str(config["output"].get("select_on", "mechanism_macro_f1"))

    history: List[Dict[str, Any]] = []
    best = -1.0
    for epoch in range(epochs):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            if dtype is not None:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    out = model(**batch)
                    loss = out["loss"] / grad_accum
            else:
                out = model(**batch)
                loss = out["loss"] / grad_accum
            scaler.scale(loss).backward()
            running += float(out["loss"].detach())
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config.get("max_grad_norm", 1.0))
                )
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad(set_to_none=True)
                scheduler.step()
            if step % int(train_config.get("log_every", 50)) == 0:
                print(
                    f"epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss {float(out['loss'].detach()):.4f}"
                )

        metrics = evaluate(model, val_loader, device, val_examples)
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(1, len(train_loader))
        history.append(metrics)
        print(
            f"epoch {epoch}: mechanism macro-F1 {metrics['mechanism_macro_f1']:.4f}  "
            f"severity macro-F1 {metrics['severity_macro_f1']:.4f}"
        )
        if config["output"].get("save_best", True) and metrics[select_on] > best:
            best = metrics[select_on]
            torch.save(model.state_dict(), out_dir / "type_head.pt")
            tokenizer.save_pretrained(out_dir / "tokenizer")
            (out_dir / "selected.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "select_on": select_on,
                        "value": best,
                        "model_name": model_name,
                        "max_length": max_length,
                        "seed": int(config.get("seed", 42)),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    (out_dir / "history.json").write_text(
        json.dumps({"history": history, "config": config}, indent=2), encoding="utf-8"
    )
    print(f"written: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
