"""HHEM-2.1-Open on the RAGTruth test split: the free tool that cannot point.

The first question a panel asks about a detector is "why not use something off
the shelf?" Vectara's HHEM-2.1-Open is the strongest freely available answer to
that: an open-weights factual-consistency classifier under Apache 2.0, no API
key, no quota. Running it here fills the example-level column of the baseline
table with a number rather than an opinion.

It also makes C1's case without arguing. HHEM scores whole responses. It has no
notion of a span, so its span columns are not blank, they are "n/a -- response
level model". That pattern of n/a down the column is the gap this project exists
in: the strongest free tool available cannot tell you which words are wrong.

Facts checked against the model card on 2026-08-16, because getting any of them
backwards would silently invert the result:

  identifier   vectara/hallucination_evaluation_model  (HHEM-2.1-Open)
  licence      Apache 2.0
  direction    a HIGH score means CONSISTENT. So P(hallucinated) = 1 - score,
               and forgetting that inversion produces a beautiful, wrong table
  inputs       (premise, hypothesis) = (context, answer), in that order
  interface    model.predict(pairs) after from_pretrained(trust_remote_code=True)
  threshold    the card publishes none, so one has to be chosen -- see below

Two things that need saying out loud.

trust_remote_code=True runs code from the Hub inside the repo's environment. It
is what the published interface requires, it is the reason this module is opt-in
behind a CLI rather than imported anywhere, and it should be a conscious choice
rather than a default.

No published threshold means one has to be picked, and picking it on the test
split would be exactly the mistake this project has been careful about
everywhere else. It is chosen on the held-out calibration split -- the 1,509
responses C1 never trained on and never selected an epoch against -- and applied
to test once. The argmax-equivalent 0.5 row is reported beside it so the choice
is visible rather than buried.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.c1_detector.ood_operating_point import score_rule

MODEL_ID = "vectara/hallucination_evaluation_model"
THRESHOLDS = tuple(round(0.02 * i, 2) for i in range(1, 50))  # 0.02 .. 0.98


def read_records(path: Path | str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def premise_of(record: Dict[str, Any]) -> str:
    """The evidence HHEM checks the answer against.

    RAGTruth records carry the retrieved passages and the question separately.
    HHEM takes one premise string, and the question is not evidence -- it is the
    request -- so the premise is the context alone. Including the question would
    let the model treat the user's own words as support for the answer.
    """
    for key in ("context", "source_info", "passages"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise KeyError(
        "no context field on this record; expected one of context/source_info/"
        f"passages, saw {sorted(record)}"
    )


def build_pairs(records: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """(premise, hypothesis) pairs in the order the model card specifies."""
    return [(premise_of(record), record["answer"]) for record in records]


def to_rows(
    records: Sequence[Dict[str, Any]], consistency: Sequence[float]
) -> List[Dict[str, Any]]:
    """Turn HHEM's consistency scores into the row shape the scorer expects.

    The inversion lives here and nowhere else: HHEM is confident the answer is
    SUPPORTED when the score is high, and this project's positive class is
    "contains a hallucination".
    """
    if len(records) != len(consistency):
        raise ValueError(
            f"{len(records)} records against {len(consistency)} scores"
        )
    rows: List[Dict[str, Any]] = []
    for record, score in zip(records, consistency):
        rows.append(
            {
                "id": record.get("id"),
                "subset": record.get("task_type", "all"),
                "gold_positive": len(record.get("spans") or []) > 0,
                "consistency": float(score),
                "p_hallucinated": 1.0 - float(score),
            }
        )
    return rows


def calibration_records(
    train_path: Path | str, split_ids_path: Path | str, limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """The processed records behind C1's calibration split.

    HHEM's threshold has to be chosen somewhere, and the only honest somewhere is
    data the reported test split does not overlap. C1 already holds out a
    calibration split and writes its ids, so reuse exactly that set: it keeps the
    two systems comparable and it costs nothing.

    Note this reads the processed records, not the probability dump. The dump
    carries spans and probabilities but not the context, and HHEM needs the
    context as its premise.
    """
    ids = set(json.loads(Path(split_ids_path).read_text(encoding="utf-8"))["calib"])
    rows = [r for r in read_records(train_path) if r["id"] in ids]
    if not rows:
        raise SystemExit(
            f"no records in {train_path} matched the {len(ids):,} calibration ids"
        )
    return rows[:limit] if limit else rows


def decide(threshold: float):
    return lambda row: row["p_hallucinated"] >= threshold


def sweep(
    rows: Sequence[Dict[str, Any]], thresholds: Sequence[float] = THRESHOLDS
) -> List[Dict[str, Any]]:
    return [{"threshold": t, **score_rule(rows, decide(t))} for t in thresholds]


def choose_threshold(
    calib: Sequence[Dict[str, Any]], thresholds: Sequence[float] = THRESHOLDS
) -> Dict[str, Any]:
    """Best calibration F1, ties broken toward the more sensitive setting."""
    rows = sweep(calib, thresholds)
    if not rows:
        raise ValueError("no thresholds to choose from")
    return {
        "chosen": max(rows, key=lambda r: (r["f1"], -r["threshold"])),
        "sweep": rows,
    }


def per_task(rows: Sequence[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    tasks = sorted({row["subset"] for row in rows})
    return {
        task: score_rule([r for r in rows if r["subset"] == task], decide(threshold))
        for task in tasks
    }


def build_report(
    calib_rows: Sequence[Dict[str, Any]],
    test_rows: Sequence[Dict[str, Any]],
    thresholds: Sequence[float] = THRESHOLDS,
) -> Dict[str, Any]:
    chosen = choose_threshold(calib_rows, thresholds)
    threshold = chosen["chosen"]["threshold"]
    return {
        "model": MODEL_ID,
        "licence": "Apache-2.0",
        "score_direction": "high consistency means supported; p_hallucinated = 1 - score",
        "threshold_source": "chosen on the held-out calibration split, applied to test once",
        "chosen_threshold": threshold,
        "n_calibration": len(calib_rows),
        "n_test": len(test_rows),
        "calibration": {"chosen": chosen["chosen"], "sweep": chosen["sweep"]},
        "test": {
            "at_half": score_rule(test_rows, decide(0.5)),
            "adapted": score_rule(test_rows, decide(threshold)),
            "per_task_adapted": per_task(test_rows, threshold),
        },
        "span_level": "n/a - response-level model, HHEM does not produce spans",
    }


def format_report(report: Dict[str, Any]) -> str:
    def line(label: str, row: Dict[str, Any]) -> str:
        return (
            f"{label:<32}{row['n']:>7,}{row['positive_rate']:>9.3f}"
            f"{row['precision']:>9.4f}{row['recall']:>9.4f}{row['f1']:>9.4f}"
            f"{row['trivial_f1']:>9.4f}{('yes' if row['clears_trivial'] else 'NO'):>8}"
        )

    header = (
        f"{'row':<32}{'n':>7}{'pos rate':>9}{'P':>9}{'R':>9}"
        f"{'F1':>9}{'trivial':>9}{'clears':>8}"
    )
    lines = [
        f"{report['model']}  ({report['licence']})",
        report["score_direction"],
        f"threshold {report['chosen_threshold']:.2f}, {report['threshold_source']}",
        "",
        header,
        "-" * len(header),
        line("calibration, chosen", report["calibration"]["chosen"]),
        line("TEST at 0.5", report["test"]["at_half"]),
        line("TEST at chosen threshold", report["test"]["adapted"]),
        "",
        f"span-level: {report['span_level']}",
        "",
        f"{'task':<18}{'n':>7}{'P':>9}{'R':>9}{'F1':>9}{'trivial':>9}",
    ]
    for task, row in report["test"]["per_task_adapted"].items():
        lines.append(
            f"{task:<18}{row['n']:>7,}{row['precision']:>9.4f}"
            f"{row['recall']:>9.4f}{row['f1']:>9.4f}{row['trivial_f1']:>9.4f}"
        )
    return "\n".join(lines)


def load_hhem(device: str):
    """Load HHEM, working around its code predating transformers 5.

    HHEM ships its own modelling code through trust_remote_code, and that code was
    written against transformers 4. Version 5 renamed the attribute it uses to
    decide which missing weights are tied rather than genuinely absent, so
    from_pretrained dies with:

        AttributeError: 'HHEMv2ForSequenceClassification' object has no
        attribute 'all_tied_weights_keys'

    The fix is to fetch the class, copy its transformers-4 tied-weight list onto
    the name transformers 5 looks for, and only then load. Nothing about the
    weights changes; only the bookkeeping that says the T5 encoder embedding is
    tied to the shared embedding rather than missing.

    That distinction is the whole risk. If the tie is not honoured the encoder
    embedding is randomly initialised, the model still loads, still returns
    numbers between 0 and 1, and every one of them is noise. So the tie is
    asserted after loading rather than assumed: the two tensors must share
    storage. A baseline that silently scored garbage would be worse than no
    baseline at all.
    """
    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    model_class = get_class_from_dynamic_module(
        "modeling_hhem_v2.HHEMv2ForSequenceClassification", MODEL_ID
    )
    if not hasattr(model_class, "all_tied_weights_keys"):
        # transformers 5 wants a mapping here, not the list transformers 4 used,
        # and HHEM's own list is empty anyway. An empty mapping is deliberately
        # NOT an assertion that nothing is tied - it just declines to guess at an
        # internal contract. Whether the encoder embedding really ends up tied is
        # then checked below against the tensors themselves, which is the only
        # evidence that actually matters.
        existing = getattr(model_class, "_tied_weights_keys", None)
        model_class.all_tied_weights_keys = existing if isinstance(existing, dict) else {}
        print(
            "shimmed all_tied_weights_keys ="
            f" {model_class.all_tied_weights_keys} for transformers 5"
        )

    model = model_class.from_pretrained(MODEL_ID, trust_remote_code=True).to(device)
    model.eval()

    transformer = model.t5.transformer
    if transformer.shared.weight.data_ptr() != transformer.encoder.embed_tokens.weight.data_ptr():
        # This is the state transformers 5 leaves the model in: shared.weight was
        # loaded from the checkpoint, encoder.embed_tokens.weight was freshly
        # initialised, and nothing tied them. Left alone the encoder reads random
        # embeddings and every score is noise while the model still "works".
        print("encoder embedding was not tied after loading; tying it to shared")
        transformer.encoder.embed_tokens = transformer.shared
        if hasattr(transformer, "decoder"):
            transformer.decoder.embed_tokens = transformer.shared

    shared = transformer.shared.weight
    embed = transformer.encoder.embed_tokens.weight
    if shared.data_ptr() != embed.data_ptr():
        raise SystemExit(
            "the T5 encoder embedding is still NOT tied to the shared embedding, "
            "so it is randomly initialised and every score would be noise. "
            "Refusing to run."
        )
    print(f"embedding tie verified, {shared.shape[0]:,} x {shared.shape[1]}")
    _ = torch  # imported for the caller's device handling
    return model


def sanity_check(model, batch_size: int = 4) -> None:
    """Refuse to run a model that cannot separate an obvious pair.

    Two hand-written pairs, one supported and one flatly contradicted. If the
    supported one does not score higher, the weights or the score direction are
    wrong and the whole run would be wasted.
    """
    supported = ("The Eiffel Tower is in Paris, France.", "The Eiffel Tower is in Paris.")
    contradicted = ("The Eiffel Tower is in Paris, France.", "The Eiffel Tower is in Berlin.")
    scores = model.predict([supported, contradicted])
    values = [float(x) for x in (scores.tolist() if hasattr(scores, "tolist") else scores)]
    print(f"sanity check: supported {values[0]:.4f}  contradicted {values[1]:.4f}")
    if not values[0] > values[1]:
        raise SystemExit(
            "HHEM scores the contradicted pair at least as high as the supported "
            "one. Either the weights are wrong or the score direction is not what "
            "the model card says. Refusing to run."
        )


def score_with_model(
    records: Sequence[Dict[str, Any]],
    model,
    batch_size: int = 8,
) -> List[float]:
    """Consistency score per record, in order."""
    import torch

    pairs = build_pairs(records)
    scores: List[float] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            out = model.predict(batch)
            scores.extend(
                float(x) for x in (out.tolist() if hasattr(out, "tolist") else out)
            )
            if start % (batch_size * 25) == 0:
                print(f"  {start + len(batch):,}/{len(pairs):,}", flush=True)
    return scores


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="HHEM-2.1-Open as a response-level baseline on RAGTruth."
    )
    parser.add_argument("--train", default="data/processed/ragtruth_train.jsonl")
    parser.add_argument("--test", default="data/processed/ragtruth_test.jsonl")
    parser.add_argument(
        "--split-ids",
        default="results/c1/modernbert-base/split_ids.json",
        help="C1's split record; its calib ids define the threshold-choosing set",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug only")
    parser.add_argument("--out-dir", default="results/hhem")
    args = parser.parse_args(argv)

    calib_records = calibration_records(args.train, args.split_ids, args.limit)
    test_records = read_records(args.test, args.limit)
    print(f"calibration {len(calib_records):,} responses, test {len(test_records):,}")

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_hhem(device)
    sanity_check(model)

    calib_rows = to_rows(calib_records, score_with_model(calib_records, model, args.batch_size))
    test_rows = to_rows(test_records, score_with_model(test_records, model, args.batch_size))

    report = build_report(calib_rows, test_rows)
    print()
    print(format_report(report))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hhem_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out_dir / "hhem_scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in test_rows:
            handle.write(json.dumps(row) + "\n")
    print(f"\nwritten: {out_dir / 'hhem_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
