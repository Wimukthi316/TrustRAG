"""Layer 3.1 of C3: the CPU feature baseline for the error type.

This is built before the transformer, on purpose. It de-risks the component --
if the Kaggle run never happens, this is still a measured result with a floor
beside it -- and it sets the bar the transformer has to clear to be worth its
GPU hour.

**The two-axis formulation.** RAGTruth's four categories are a 2x2, not four flat
classes:

    axis A, mechanism   conflict  the span contradicts something in the context
                        baseless  the context says nothing about it
    axis B, severity    evident   obvious once you look
                        subtle    a small shift in meaning

They are different kinds of question. Mechanism is a grounding question a lexical
feature can answer: does this span share vocabulary with a context sentence, or
is it made of words that appear nowhere. Severity is a human judgement about how
far the meaning moved, and nothing in a bag of overlap features can see it.
Predicting them separately means the report can say which of the two the system
can actually do, instead of hiding both inside one 4-way average.

Three models are fitted, all on the same features and the same split:

    mechanism   binary logistic regression
    severity    binary logistic regression
    flat        4-way multinomial logistic regression

The 4-class distribution Layer 4 needs can be built two ways -- from the flat
model directly, or as the product of the two axis marginals. The product assumes
the axes are independent given the span, which is an assumption and is therefore
measured rather than asserted: both are written out and Layer 4 reports which one
calibrates better.

The fitted model is stored as plain JSON coefficients, not a pickle. It loads
without scikit-learn, so the FastAPI serving path can type a span with a dot
product and no new runtime dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.c3_explanation.grounding import (
    FEATURE_NAMES,
    feature_vector,
    span_features,
    split_sentences,
)
from src.c3_explanation.labels_audit import (
    MECHANISM_BASELESS,
    MECHANISM_CONFLICT,
    SEVERITY_EVIDENT,
    SEVERITY_SUBTLE,
    axes_of_error_type,
)
from src.c3_explanation.type_eval import evaluate_axis, format_axis, majority_label
from src.common.schema import ErrorType

ERROR_TYPES: Tuple[str, ...] = (
    ErrorType.EVIDENT_CONFLICT.value,
    ErrorType.SUBTLE_CONFLICT.value,
    ErrorType.EVIDENT_BASELESS.value,
    ErrorType.SUBTLE_BASELESS.value,
)
MECHANISMS: Tuple[str, ...] = (MECHANISM_BASELESS, MECHANISM_CONFLICT)
SEVERITIES: Tuple[str, ...] = (SEVERITY_EVIDENT, SEVERITY_SUBTLE)

SEED = 42


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def primary_type(span: Dict[str, Any]) -> str:
    """One type per gold span, deterministically, when preprocessing merged two."""
    types = [str(t) for t in (span.get("error_types") or []) if t]
    if not types and span.get("error_type"):
        types = [str(span["error_type"])]
    return sorted(types)[0] if types else ErrorType.UNKNOWN.value


def gold_rows(
    processed: Iterable[Dict[str, Any]],
    split_of: Optional[Dict[str, str]] = None,
    keep_splits: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Feature rows for every annotated gold span, with its type."""
    rows: List[Dict[str, Any]] = []
    for record in processed:
        rid = str(record["id"])
        split = (split_of or {}).get(rid, str(record.get("split", "unknown")))
        if keep_splits is not None and split not in keep_splits:
            continue
        sentences = split_sentences(record.get("context", ""))
        answer = record.get("answer", "")
        for span in record.get("spans") or []:
            error_type = primary_type(span)
            if error_type not in ERROR_TYPES:
                continue
            mechanism, severity = axes_of_error_type(ErrorType(error_type))
            features = span_features(
                span.get("text", answer[span["start"]:span["end"]]),
                answer,
                int(span["start"]),
                sentences,
                str(record.get("task_type", "other")),
            )
            rows.append(
                {
                    "response_id": rid,
                    "split": split,
                    "task_type": str(record.get("task_type", "other")),
                    "model": str(record.get("model", "unknown")),
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                    "text": span.get("text", ""),
                    "error_type": error_type,
                    "mechanism": mechanism,
                    "severity": severity,
                    "implicit_true": bool(span.get("implicit_true", False)),
                    "due_to_null": bool(span.get("due_to_null", False)),
                    "features": features,
                    "source": "gold",
                }
            )
    return rows


def predicted_rows(
    typed_predictions: Iterable[Dict[str, Any]],
    processed: Iterable[Dict[str, Any]],
    convention: str = "majority_by_characters",
) -> List[Dict[str, Any]]:
    """Feature rows for the typeable predicted spans, with their inherited type.

    Spurious predictions are dropped here rather than scored: no gold span means
    no true type exists, so including them would be scoring the type head for
    C1's false positives. Layer 2's report counts them.
    """
    by_id = {str(r["id"]): r for r in processed}
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
        answer = record.get("answer", "")
        sentences = split_sentences(record.get("context", ""))
        start, end = int(row["start"]), int(row["end"])
        text = answer[start:end]
        mechanism, severity = axes_of_error_type(ErrorType(inherited))
        rows.append(
            {
                "response_id": str(row["response_id"]),
                "split": "test",
                "task_type": str(record.get("task_type", "other")),
                "model": str(record.get("model", "unknown")),
                "start": start,
                "end": end,
                "text": text,
                "error_type": inherited,
                "mechanism": mechanism,
                "severity": severity,
                "bucket": row.get("bucket"),
                "crosses_type_boundary": bool(row.get("crosses_type_boundary")),
                "features": span_features(
                    text, answer, start, sentences, str(record.get("task_type", "other"))
                ),
                "source": "predicted",
            }
        )
    return rows


def matrix(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    return np.asarray([feature_vector(r["features"]) for r in rows], dtype=np.float64)


# --------------------------------------------------------------------------
# A logistic model stored as JSON
# --------------------------------------------------------------------------


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_logistic(
    features: np.ndarray, labels: Sequence[str], classes: Sequence[str]
) -> Dict[str, Any]:
    """Standardise, fit multinomial logistic regression, return plain JSON.

    Stored as coefficients so the serving path can score a span without
    scikit-learn or torch. lbfgs on standardised features is deterministic, and
    the seed is set anyway so nothing depends on that being true.
    """
    from sklearn.linear_model import LogisticRegression

    mean = features.mean(axis=0) if features.size else np.zeros(len(FEATURE_NAMES))
    scale = features.std(axis=0) if features.size else np.ones(len(FEATURE_NAMES))
    scale = np.where(scale < 1e-9, 1.0, scale)
    standardised = (features - mean) / scale

    # multi_class is left at the library default: lbfgs already fits a
    # multinomial model for more than two classes, and naming it explicitly is
    # deprecated in newer scikit-learn.
    model = LogisticRegression(max_iter=2000, random_state=SEED)
    model.fit(standardised, list(labels))

    order = list(model.classes_)
    coef = np.asarray(model.coef_, dtype=np.float64)
    intercept = np.asarray(model.intercept_, dtype=np.float64)
    if coef.shape[0] == 1:
        # Binary: sklearn keeps one row and reads it as sigmoid(x.w + b) for the
        # second class. The softmax form of exactly that is a zero row for the
        # first class and the fitted row for the second -- NOT a sign flip, which
        # would double the logit gap. The parity test in
        # tests/test_c3_type_baseline.py is what caught that.
        coef = np.vstack([np.zeros_like(coef[0]), coef[0]])
        intercept = np.asarray([0.0, intercept[0]], dtype=np.float64)

    return {
        "feature_names": list(FEATURE_NAMES),
        "classes": order,
        "declared_classes": list(classes),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": coef.tolist(),
        "intercept": intercept.tolist(),
        "n_train": int(features.shape[0]),
        "seed": SEED,
    }


def predict_proba(model: Dict[str, Any], features: np.ndarray) -> np.ndarray:
    """Class probabilities from a stored model. No scikit-learn needed."""
    if features.size == 0:
        return np.empty((0, len(model["classes"])), dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    standardised = (features - mean) / scale
    return _softmax(standardised @ coef.T + intercept)


def predict_labels(model: Dict[str, Any], features: np.ndarray) -> List[str]:
    probs = predict_proba(model, features)
    classes = model["classes"]
    if probs.size == 0:
        return []
    return [classes[int(i)] for i in probs.argmax(axis=1)]


def axis_product(
    mechanism_probs: np.ndarray,
    mechanism_classes: Sequence[str],
    severity_probs: np.ndarray,
    severity_classes: Sequence[str],
) -> np.ndarray:
    """P(type) = P(mechanism) x P(severity), in ERROR_TYPES order.

    The independence assumption is stated, not hidden: the flat 4-way model is
    fitted alongside precisely so the cost of assuming it can be measured.
    """
    mech_index = {name: i for i, name in enumerate(mechanism_classes)}
    sev_index = {name: i for i, name in enumerate(severity_classes)}
    out = np.zeros((mechanism_probs.shape[0], len(ERROR_TYPES)), dtype=np.float64)
    for column, error_type in enumerate(ERROR_TYPES):
        mechanism, severity = axes_of_error_type(ErrorType(error_type))
        out[:, column] = (
            mechanism_probs[:, mech_index[str(mechanism)]]
            * severity_probs[:, sev_index[str(severity)]]
        )
    total = out.sum(axis=1, keepdims=True)
    return np.divide(out, total, out=np.full_like(out, 1.0 / len(ERROR_TYPES)), where=total > 0)


def reorder(probs: np.ndarray, classes: Sequence[str], target: Sequence[str]) -> np.ndarray:
    """Put a probability matrix into a fixed column order."""
    index = {name: i for i, name in enumerate(classes)}
    out = np.zeros((probs.shape[0], len(target)), dtype=np.float64)
    for column, name in enumerate(target):
        if name in index:
            out[:, column] = probs[:, index[name]]
    return out


# --------------------------------------------------------------------------
# Train and evaluate
# --------------------------------------------------------------------------


def train(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    features = matrix(rows)
    return {
        "mechanism": fit_logistic(
            features, [r["mechanism"] for r in rows], MECHANISMS
        ),
        "severity": fit_logistic(features, [r["severity"] for r in rows], SEVERITIES),
        "flat": fit_logistic(features, [r["error_type"] for r in rows], ERROR_TYPES),
        "majority": {
            "error_type": majority_label([r["error_type"] for r in rows]),
            "mechanism": majority_label([r["mechanism"] for r in rows]),
            "severity": majority_label([r["severity"] for r in rows]),
        },
    }


def score(
    models: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    with_intervals: bool = True,
) -> Dict[str, Any]:
    """Every number C3-3 needs for one evaluation set."""
    features = matrix(rows)
    response_ids = [r["response_id"] for r in rows]
    tasks = [r["task_type"] for r in rows]

    mechanism_pred = predict_labels(models["mechanism"], features)
    severity_pred = predict_labels(models["severity"], features)
    flat_pred = predict_labels(models["flat"], features)

    product = axis_product(
        predict_proba(models["mechanism"], features),
        models["mechanism"]["classes"],
        predict_proba(models["severity"], features),
        models["severity"]["classes"],
    )
    product_pred = [ERROR_TYPES[int(i)] for i in product.argmax(axis=1)] if len(rows) else []

    return {
        "n": len(rows),
        "n_responses": len(set(response_ids)),
        "mechanism": evaluate_axis(
            response_ids,
            [r["mechanism"] for r in rows],
            mechanism_pred,
            MECHANISMS,
            tasks=tasks,
            floor_class=models["majority"]["mechanism"],
            with_intervals=with_intervals,
        ),
        "severity": evaluate_axis(
            response_ids,
            [r["severity"] for r in rows],
            severity_pred,
            SEVERITIES,
            tasks=tasks,
            floor_class=models["majority"]["severity"],
            with_intervals=with_intervals,
        ),
        "error_type_flat": evaluate_axis(
            response_ids,
            [r["error_type"] for r in rows],
            flat_pred,
            ERROR_TYPES,
            tasks=tasks,
            floor_class=models["majority"]["error_type"],
            with_intervals=with_intervals,
        ),
        "error_type_axis_product": evaluate_axis(
            response_ids,
            [r["error_type"] for r in rows],
            product_pred,
            ERROR_TYPES,
            tasks=tasks,
            floor_class=models["majority"]["error_type"],
            with_intervals=with_intervals,
        ),
    }


def probability_rows(
    models: Dict[str, Any], rows: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """One record per span: both 4-class distributions plus the truth.

    This is Layer 4's input. It is written for every evaluation set so the
    conformal layer can calibrate on calib and be tested on test without ever
    touching the type model again.
    """
    features = matrix(rows)
    flat = reorder(
        predict_proba(models["flat"], features), models["flat"]["classes"], ERROR_TYPES
    )
    product = axis_product(
        predict_proba(models["mechanism"], features),
        models["mechanism"]["classes"],
        predict_proba(models["severity"], features),
        models["severity"]["classes"],
    )
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        out.append(
            {
                "response_id": row["response_id"],
                "task_type": row["task_type"],
                "model": row.get("model", "unknown"),
                "start": row["start"],
                "end": row["end"],
                "source": row["source"],
                "error_type": row["error_type"],
                "mechanism": row["mechanism"],
                "severity": row["severity"],
                "implicit_true": bool(row.get("implicit_true", False)),
                "due_to_null": bool(row.get("due_to_null", False)),
                "bucket": row.get("bucket"),
                "classes": list(ERROR_TYPES),
                "probs_flat": [float(v) for v in flat[i]],
                "probs_axis_product": [float(v) for v in product[i]],
            }
        )
    return out


def format_report(name: str, block: Dict[str, Any]) -> str:
    lines = [f"=== {name}  (spans {block['n']:,}, responses {block['n_responses']:,}) ==="]
    for axis in ("mechanism", "severity", "error_type_flat", "error_type_axis_product"):
        lines.append(format_axis(axis, block[axis]))
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="C3 Layer 3.1: the CPU feature baseline for the error type."
    )
    parser.add_argument("--train", default="data/processed/ragtruth_train.jsonl")
    parser.add_argument("--test", default="data/processed/ragtruth_test.jsonl")
    parser.add_argument(
        "--split-ids", default="results/c1/modernbert-base/split_ids.json"
    )
    parser.add_argument(
        "--typed-test",
        default="results/c3/analysis/typed_predictions_test.jsonl",
        help="Layer 2 output; pass '' to skip the predicted-span evaluation",
    )
    parser.add_argument(
        "--typed-calib", default="results/c3/analysis/typed_predictions_calib.jsonl"
    )
    parser.add_argument("--out-dir", default="results/c3/type")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    split_of: Dict[str, str] = {}
    path = Path(args.split_ids)
    if path.exists():
        for name, ids in json.loads(path.read_text(encoding="utf-8")).items():
            for rid in ids:
                split_of[str(rid)] = name

    train_corpus = read_jsonl(Path(args.train))
    test_corpus = read_jsonl(Path(args.test))

    train_gold = gold_rows(train_corpus, split_of, keep_splits=("train",))
    calib_gold = gold_rows(train_corpus, split_of, keep_splits=("calib",))
    test_gold = gold_rows(test_corpus, split_of)

    print(
        f"gold spans: train {len(train_gold):,}  calib {len(calib_gold):,}  "
        f"test {len(test_gold):,}"
    )
    models = train(train_gold)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_model.json").write_text(
        json.dumps(models, indent=2), encoding="utf-8"
    )

    report: Dict[str, Any] = {
        "model": "cpu_logistic_baseline",
        "features": list(FEATURE_NAMES),
        "n_train_gold_spans": len(train_gold),
        "resamples": args.resamples,
        "evaluations": {},
    }

    sets: List[Tuple[str, List[Dict[str, Any]]]] = [
        ("gold_calib", calib_gold),
        ("gold_test", test_gold),
    ]

    typed_test_path = Path(args.typed_test) if args.typed_test else None
    if typed_test_path and typed_test_path.exists():
        sets.append(
            (
                "predicted_test",
                predicted_rows(read_jsonl(typed_test_path), test_corpus),
            )
        )
    typed_calib_path = Path(args.typed_calib) if args.typed_calib else None
    if typed_calib_path and typed_calib_path.exists():
        sets.append(
            (
                "predicted_calib",
                predicted_rows(read_jsonl(typed_calib_path), train_corpus),
            )
        )

    for name, rows in sets:
        if not rows:
            print(f"{name}: no rows, skipped")
            continue
        block = score(models, rows)
        report["evaluations"][name] = block
        print()
        print(format_report(name, block))
        with (out_dir / f"baseline_probs_{name}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in probability_rows(models, rows):
                handle.write(json.dumps(row) + "\n")

    out_path = out_dir / "baseline_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
