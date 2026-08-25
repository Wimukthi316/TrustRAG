"""Score the trained C3 type head and write the probability dumps Layer 4 reads.

The output files have exactly the shape type_baseline.py writes, so
type_conformal.py and run_c3.py consume either the transformer or the CPU
baseline without knowing which produced them. That is what makes the decision
rule in the plan real rather than aspirational: if the Kaggle run never happens,
the CPU baseline ships as the type model and every downstream layer still runs.

Four evaluation sets, for the same reason type_baseline uses four:

    gold_calib, gold_test           the model's own task, all four classes present
    predicted_calib, predicted_test the deployed setting, where the class mix is
                                    whatever C1 delivers and subtle_conflict has
                                    almost vanished

Reported with the majority-class floor beside every number, per task, and with
response-level cluster bootstrap intervals -- the same shape C2 reports in, via
src/c3_explanation/type_eval.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.c3_explanation.train_c3_type import (
    gold_model_rows,
    load_split_of,
    read_jsonl,
    resolve_max_length,
)
from src.c3_explanation.type_baseline import ERROR_TYPES, MECHANISMS, SEVERITIES
from src.c3_explanation.type_eval import evaluate_axis, format_axis, majority_label
from src.c3_explanation.type_model import (
    TypeDataset,
    TwoAxisTypeModel,
    collate,
    error_type_probabilities,
    examples_from_rows,
)


def predicted_model_rows(
    typed_predictions: Sequence[Dict[str, Any]],
    corpus: Sequence[Dict[str, Any]],
    convention: str = "majority_by_characters",
    split: str = "test",
) -> List[Dict[str, Any]]:
    """Typeable predicted spans, with the type Layer 2's protocol gave them."""
    by_id = {str(r["id"]): r for r in corpus}
    rows: List[Dict[str, Any]] = []
    for row in typed_predictions:
        if not row.get("typeable"):
            continue
        inherited = row.get(convention)
        if inherited not in ERROR_TYPES:
            continue
        record = by_id.get(str(row["response_id"]))
        if record is None:
            continue
        rows.append(
            {
                "response_id": str(row["response_id"]),
                "split": split,
                "task_type": str(record.get("task_type", "other")),
                "model": str(record.get("model", "unknown")),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "error_type": inherited,
                "answer": record.get("answer", ""),
                "question": record.get("question", ""),
                "context": record.get("context", ""),
                "bucket": row.get("bucket"),
                "source": "predicted",
            }
        )
    return rows


def infer(
    model: TwoAxisTypeModel,
    tokenizer: Any,
    rows: Sequence[Dict[str, Any]],
    device: torch.device,
    max_length: int,
    batch_size: int,
    span_open: str,
    span_close: str,
) -> Dict[str, np.ndarray]:
    examples = examples_from_rows(rows, span_open, span_close)
    loader = DataLoader(
        TypeDataset(examples, tokenizer, max_length),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: collate(batch, tokenizer),
    )
    mechanism: List[np.ndarray] = []
    severity: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            mechanism.append(torch.softmax(out["mechanism"], dim=-1).cpu().numpy())
            severity.append(torch.softmax(out["severity"], dim=-1).cpu().numpy())
    if not mechanism:
        return {
            "mechanism": np.empty((0, len(MECHANISMS))),
            "severity": np.empty((0, len(SEVERITIES))),
            "error_type": np.empty((0, len(ERROR_TYPES))),
        }
    mechanism_probs = np.concatenate(mechanism, axis=0)
    severity_probs = np.concatenate(severity, axis=0)
    product = error_type_probabilities(
        torch.from_numpy(mechanism_probs), torch.from_numpy(severity_probs)
    ).numpy()
    return {
        "mechanism": mechanism_probs,
        "severity": severity_probs,
        "error_type": product,
    }


def score(
    rows: Sequence[Dict[str, Any]],
    probs: Dict[str, np.ndarray],
    majority: Dict[str, Optional[str]],
    with_intervals: bool = True,
) -> Dict[str, Any]:
    response_ids = [r["response_id"] for r in rows]
    tasks = [r["task_type"] for r in rows]
    from src.c3_explanation.labels_audit import axes_of_error_type
    from src.common.schema import ErrorType

    gold_axes = [axes_of_error_type(ErrorType(r["error_type"])) for r in rows]
    mechanism_gold = [str(a) for a, _ in gold_axes]
    severity_gold = [str(b) for _, b in gold_axes]

    mechanism_pred = [MECHANISMS[i] for i in probs["mechanism"].argmax(axis=1)]
    severity_pred = [SEVERITIES[i] for i in probs["severity"].argmax(axis=1)]
    type_pred = [ERROR_TYPES[i] for i in probs["error_type"].argmax(axis=1)]

    return {
        "n": len(rows),
        "n_responses": len(set(response_ids)),
        "mechanism": evaluate_axis(
            response_ids, mechanism_gold, mechanism_pred, MECHANISMS,
            tasks=tasks, floor_class=majority.get("mechanism"),
            with_intervals=with_intervals,
        ),
        "severity": evaluate_axis(
            response_ids, severity_gold, severity_pred, SEVERITIES,
            tasks=tasks, floor_class=majority.get("severity"),
            with_intervals=with_intervals,
        ),
        "error_type_axis_product": evaluate_axis(
            response_ids, [r["error_type"] for r in rows], type_pred, ERROR_TYPES,
            tasks=tasks, floor_class=majority.get("error_type"),
            with_intervals=with_intervals,
        ),
    }


def probability_rows(
    rows: Sequence[Dict[str, Any]], probs: Dict[str, np.ndarray]
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        product = [float(v) for v in probs["error_type"][i]]
        out.append(
            {
                "response_id": row["response_id"],
                "task_type": row["task_type"],
                "model": row.get("model", "unknown"),
                "start": row["start"],
                "end": row["end"],
                "source": row["source"],
                "error_type": row["error_type"],
                "bucket": row.get("bucket"),
                "classes": list(ERROR_TYPES),
                # Both keys carry the same distribution for the neural head: it
                # has no flat 4-way output, only the product of its two axes.
                # Keeping both means Layer 4 needs no branch on which model ran.
                "probs_flat": product,
                "probs_axis_product": product,
            }
        )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the C3 type head.")
    parser.add_argument("--config", default="configs/c3_type.yaml")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--out-dir", default="results/c3/type")
    parser.add_argument("--prefix", default="head_probs")
    parser.add_argument("--metrics-name", default="head_metrics.json")
    parser.add_argument(
        "--limit", type=int, default=None, help="row cap, for a CPU smoke test"
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=2000,
        help="cluster-bootstrap resamples; 0 skips the intervals",
    )
    args = parser.parse_args(argv)

    import yaml
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data = config["data"]
    checkpoint_dir = Path(args.checkpoint_dir or config["output"]["dir"])
    selected = json.loads((checkpoint_dir / "selected.json").read_text(encoding="utf-8"))
    model_name = selected["model_name"]
    max_length = int(selected.get("max_length", data.get("max_length", 1024)))

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir / "tokenizer")
    encoder = AutoModel.from_pretrained(model_name)
    model_config = AutoConfig.from_pretrained(model_name)
    hidden = model_config.hidden_size
    max_length = resolve_max_length(max_length, model_config)
    model = TwoAxisTypeModel(encoder, hidden, dropout=0.0)
    model.load_state_dict(torch.load(checkpoint_dir / "type_head.pt", map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_corpus = read_jsonl(Path(data["train_file"]))
    test_corpus = read_jsonl(Path(data["test_file"]))
    split_of = load_split_of(Path(data["split_ids"]) if data.get("split_ids") else None)

    train_rows = gold_model_rows(train_corpus, split_of, keep=("train",))
    majority = {
        "error_type": majority_label([r["error_type"] for r in train_rows]),
        "mechanism": None,
        "severity": None,
    }
    from src.c3_explanation.labels_audit import axes_of_error_type
    from src.common.schema import ErrorType

    axes = [axes_of_error_type(ErrorType(r["error_type"])) for r in train_rows]
    majority["mechanism"] = majority_label([str(a) for a, _ in axes])
    majority["severity"] = majority_label([str(b) for _, b in axes])

    sets: List[tuple] = [
        ("gold_calib", gold_model_rows(train_corpus, split_of, keep=("calib",))),
        ("gold_test", gold_model_rows(test_corpus, split_of)),
    ]
    for name, path_key, corpus, split in (
        ("predicted_test", "typed_predictions_test", test_corpus, "test"),
        ("predicted_calib", "typed_predictions_calib", train_corpus, "calib"),
    ):
        path = Path(data[path_key]) if data.get(path_key) else None
        if path and path.exists():
            sets.append(
                (name, predicted_model_rows(read_jsonl(path), corpus, split=split))
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "model": "c3_type_head",
        "checkpoint": str(checkpoint_dir),
        "selected": selected,
        "evaluations": {},
    }

    span_open = data.get("span_open", "<<")
    span_close = data.get("span_close", ">>")
    batch_size = int(config["train"].get("eval_batch_size", 32))
    for name, rows in sets:
        if args.limit:
            rows = rows[: args.limit]
        if not rows:
            print(f"{name}: no rows, skipped")
            continue
        probs = infer(
            model, tokenizer, rows, device, max_length, batch_size, span_open, span_close
        )
        block = score(rows, probs, majority, with_intervals=args.resamples > 0)
        report["evaluations"][name] = block
        print()
        print(f"=== {name}  (spans {block['n']:,}, responses {block['n_responses']:,}) ===")
        for axis in ("mechanism", "severity", "error_type_axis_product"):
            print(format_axis(axis, block[axis]))
            print()
        with (out_dir / f"{args.prefix}_{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in probability_rows(rows, probs):
                handle.write(json.dumps(row) + "\n")

    out_path = out_dir / args.metrics_name
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
