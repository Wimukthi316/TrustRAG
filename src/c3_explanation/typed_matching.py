"""Layer 2 of C3: how a predicted span inherits an error type.

Footnote 6 of "Hallucinated Span Detection with Multi-View Attention Features"
(arXiv:2504.04335) says of RAGTruth's four categories:

    "Precision (and thus F1) is difficult to compute because it is non-trivial to
    decide to which category does detected hallucination belong."

That is the field saying it has no rule for putting a type on a *predicted* span,
so per-type precision has never been reported by anyone. This module writes the
rule down, and then measures how much of the corpus the hard case actually
accounts for -- because "non-trivial" is a claim about frequency and nobody has
checked it.

The rule sits on top of the buckets src/c1_detector/localisation.py already
assigns. Those come from connected components of the gold-to-prediction overlap
graph, so every span is in exactly one component by construction:

    exact / boundary   one gold, one prediction        inherit that gold type
    split              one gold, several predictions   each inherits that gold type
    merge              several gold, one prediction    see below
    tangled            several on both sides           same rule as merge, per prediction
    spurious           a prediction touching no gold   NO type exists. Excluded from
                                                       typed scoring, counted separately
    missed             a gold span nothing touches     nothing to type

The merge case is footnote 6's case, and it only bites when the gold spans a
prediction covers do not all carry the same type. Both conventions are reported
side by side rather than one being chosen quietly:

    (a) majority by overlapping characters -- the predicted span takes the type
        that owns the most of the characters it shares with gold. Ties break on
        the sorted type name so the result does not depend on dict order.
    (b) typed-merge failure -- the span is counted as unassignable and scored as
        an error against every type.

There is a second source of ambiguity that the published discussion does not
mention, because it is created by preprocessing rather than by the detector:
build_examples.py merges overlapping raw annotations, and when two merged
annotations carry different types the resulting GOLD span already holds a list.
Those are counted separately as `gold_multi_type` so the two ambiguities are
never pooled.

Nothing here uses a GPU, a network, or the detector. It reads C1's frozen
probability dump and the frozen processed corpus, and writes its own artefacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.c1_detector.localisation import (
    classify,
    load_probability_dump,
    overlap_components,
)

Span = Tuple[int, int]

# A predicted span in one of these buckets has no gold span to inherit from.
UNTYPEABLE_BUCKETS = ("spurious",)

MAJORITY = "majority_by_characters"
FAILURE = "typed_merge_failure"

# What convention (b) writes when it refuses to assign a type. It is deliberately
# not a member of ErrorType: a scorer that treated it as a fifth class would be
# rewarding the system for a case it declined to answer.
UNASSIGNED = "unassigned"


def overlap_chars(a: Span, b: Span) -> int:
    """Characters two half-open spans share."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def majority_type_by_characters(
    pred: Span,
    gold: Sequence[Span],
    gold_types: Sequence[str],
) -> Optional[str]:
    """The type owning the most characters `pred` shares with the gold spans.

    Ties break on the sorted type name. That is arbitrary but deterministic, and
    a deterministic arbitrary rule is reportable where a dict-order one is not.
    """
    weights: Dict[str, int] = defaultdict(int)
    for span, error_type in zip(gold, gold_types):
        shared = overlap_chars(pred, span)
        if shared > 0:
            weights[error_type] += shared
    if not weights:
        return None
    best = max(weights.values())
    return sorted(name for name, weight in weights.items() if weight == best)[0]


def assign_types_for_record(
    gold: Sequence[Span],
    gold_types: Sequence[Sequence[str]],
    pred: Sequence[Span],
) -> Dict[str, Any]:
    """Bucket one record's spans and give every predicted span its inherited type.

    `gold_types[i]` is the list of raw error types on gold span i -- normally one
    entry, more when preprocessing merged annotations of different types.
    """
    if len(gold) != len(gold_types):
        raise ValueError(
            f"{len(gold)} gold spans but {len(gold_types)} type lists"
        )

    primary = [sorted(types)[0] if types else UNASSIGNED for types in gold_types]

    predictions: List[Dict[str, Any]] = []
    gold_rows: List[Dict[str, Any]] = []
    components: List[Dict[str, Any]] = []

    for in_gold, in_pred in overlap_components(list(gold), list(pred)):
        identical = (
            len(in_gold) == 1
            and len(in_pred) == 1
            and gold[in_gold[0]] == pred[in_pred[0]]
        )
        bucket = classify(len(in_gold), len(in_pred), identical)
        covered_types = sorted({primary[i] for i in in_gold})
        crosses = len(covered_types) > 1
        components.append(
            {
                "bucket": bucket,
                "n_gold": len(in_gold),
                "n_pred": len(in_pred),
                "covered_types": covered_types,
                "crosses_type_boundary": crosses,
            }
        )

        for i in in_gold:
            gold_rows.append(
                {
                    "start": gold[i][0],
                    "end": gold[i][1],
                    "error_type": primary[i],
                    "error_types": list(gold_types[i]),
                    "gold_multi_type": len(set(gold_types[i])) > 1,
                    "bucket": bucket,
                    "n_pred_in_component": len(in_pred),
                }
            )

        for j in in_pred:
            if not in_gold:
                predictions.append(
                    {
                        "start": pred[j][0],
                        "end": pred[j][1],
                        "bucket": bucket,
                        "typeable": False,
                        "covered_types": [],
                        "crosses_type_boundary": False,
                        MAJORITY: None,
                        FAILURE: None,
                    }
                )
                continue
            majority = majority_type_by_characters(
                pred[j], [gold[i] for i in in_gold], [primary[i] for i in in_gold]
            )
            # A prediction can sit inside a component without literally overlapping
            # every gold span in it; if it overlaps none, fall back to the
            # component's own single type when there is one.
            if majority is None:
                majority = covered_types[0] if len(covered_types) == 1 else None
            predictions.append(
                {
                    "start": pred[j][0],
                    "end": pred[j][1],
                    "bucket": bucket,
                    "typeable": True,
                    "covered_types": covered_types,
                    "crosses_type_boundary": crosses,
                    MAJORITY: majority,
                    FAILURE: (UNASSIGNED if crosses else majority),
                }
            )

    return {"predictions": predictions, "gold": gold_rows, "components": components}


# --------------------------------------------------------------------------
# Joining the frozen artefacts
# --------------------------------------------------------------------------


def gold_types_by_record(processed: Iterable[Dict[str, Any]]) -> Dict[str, Dict[Span, List[str]]]:
    """(start, end) -> the error types on that gold span, per response id."""
    out: Dict[str, Dict[Span, List[str]]] = {}
    for row in processed:
        table: Dict[Span, List[str]] = {}
        for span in row.get("spans") or []:
            key = (int(span["start"]), int(span["end"]))
            types = list(span.get("error_types") or [])
            if not types and span.get("error_type"):
                types = [str(span["error_type"])]
            table[key] = [str(t) for t in types]
        out[str(row["id"])] = table
    return out


def build_rows(
    records: Sequence[Dict[str, Any]],
    types_by_record: Dict[str, Dict[Span, List[str]]],
) -> Dict[str, Any]:
    """Run the assignment over every record and flatten it."""
    predictions: List[Dict[str, Any]] = []
    gold_rows: List[Dict[str, Any]] = []
    components: List[Dict[str, Any]] = []
    unmatched_gold = 0

    for record in records:
        rid = str(record["id"])
        table = types_by_record.get(rid, {})
        gold = [(int(a), int(b)) for a, b in record["gold_spans"]]
        pred = [(int(a), int(b)) for a, b in record["pred_spans"]]
        types: List[List[str]] = []
        for span in gold:
            found = table.get(span)
            if found is None:
                unmatched_gold += 1
                found = []
            types.append(found)

        assigned = assign_types_for_record(gold, types, pred)
        stamp = {
            "response_id": rid,
            "task_type": record.get("task_type", "unknown"),
            "model": record.get("model", "unknown"),
        }
        for row in assigned["predictions"]:
            predictions.append({**stamp, **row})
        for row in assigned["gold"]:
            gold_rows.append({**stamp, **row})
        for row in assigned["components"]:
            components.append({**stamp, **row})

    return {
        "predictions": predictions,
        "gold": gold_rows,
        "components": components,
        "unmatched_gold_spans": unmatched_gold,
    }


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def build_report(rows: Dict[str, Any]) -> Dict[str, Any]:
    predictions = rows["predictions"]
    gold_rows = rows["gold"]
    components = rows["components"]

    by_bucket: Counter = Counter(p["bucket"] for p in predictions)
    typeable = [p for p in predictions if p["typeable"]]
    untypeable = [p for p in predictions if not p["typeable"]]

    merge_like = [c for c in components if c["bucket"] in ("merge", "tangled")]
    crossing = [c for c in merge_like if c["crosses_type_boundary"]]
    multi_gold = [c for c in components if c["n_gold"] > 1]
    multi_gold_crossing = [c for c in multi_gold if c["crosses_type_boundary"]]

    crossing_predictions = [p for p in typeable if p["crosses_type_boundary"]]

    def type_mix(key: str) -> Dict[str, int]:
        counter: Counter = Counter(
            (p[key] if p[key] is not None else UNASSIGNED) for p in typeable
        )
        return {k: counter[k] for k in sorted(counter)}

    per_task: Dict[str, Any] = {}
    for task in sorted({p["task_type"] for p in predictions}):
        subset = [p for p in predictions if p["task_type"] == task]
        typeable_task = [p for p in subset if p["typeable"]]
        crossing_task = [p for p in typeable_task if p["crosses_type_boundary"]]
        per_task[task] = {
            "n_predicted": len(subset),
            "n_typeable": len(typeable_task),
            "typeable_rate": len(typeable_task) / len(subset) if subset else 0.0,
            "n_crossing": len(crossing_task),
            "crossing_rate_of_typeable": (
                len(crossing_task) / len(typeable_task) if typeable_task else 0.0
            ),
        }

    gold_multi = [g for g in gold_rows if g["gold_multi_type"]]

    return {
        "n_records_with_spans": len(
            {p["response_id"] for p in predictions}
            | {g["response_id"] for g in gold_rows}
        ),
        "n_gold_spans": len(gold_rows),
        "n_predicted_spans": len(predictions),
        "unmatched_gold_spans": rows["unmatched_gold_spans"],
        "predicted_by_bucket": {k: by_bucket[k] for k in sorted(by_bucket)},
        "gold_by_bucket": {
            k: v
            for k, v in sorted(Counter(g["bucket"] for g in gold_rows).items())
        },
        "typeable": {
            "n": len(typeable),
            "rate_of_predicted": len(typeable) / len(predictions) if predictions else 0.0,
            "n_excluded_spurious": len(untypeable),
            "excluded_rate": len(untypeable) / len(predictions) if predictions else 0.0,
        },
        "footnote_6": {
            "n_merge_like_components": len(merge_like),
            "n_merge_like_crossing_a_type_boundary": len(crossing),
            "crossing_rate_of_merge_like": (
                len(crossing) / len(merge_like) if merge_like else 0.0
            ),
            "n_multi_gold_components": len(multi_gold),
            "n_multi_gold_crossing": len(multi_gold_crossing),
            "crossing_rate_of_multi_gold": (
                len(multi_gold_crossing) / len(multi_gold) if multi_gold else 0.0
            ),
            "n_typeable_predictions_affected": len(crossing_predictions),
            "affected_rate_of_typeable": (
                len(crossing_predictions) / len(typeable) if typeable else 0.0
            ),
        },
        "gold_multi_type": {
            "n": len(gold_multi),
            "rate_of_gold": len(gold_multi) / len(gold_rows) if gold_rows else 0.0,
            "note": (
                "created by preprocessing merging overlapping annotations of "
                "different types, not by the detector"
            ),
        },
        "type_mix": {
            MAJORITY: type_mix(MAJORITY),
            FAILURE: type_mix(FAILURE),
        },
        "conventions_disagree_on": sum(
            1 for p in typeable if p[MAJORITY] != p[FAILURE]
        ),
        "per_task": per_task,
    }


def format_report(report: Dict[str, Any]) -> str:
    lines = ["=== C3 Layer 2: typed span matching ==="]
    lines.append(
        f"records carrying at least one span {report['n_records_with_spans']:,}  "
        f"gold spans {report['n_gold_spans']:,}  "
        f"predicted spans {report['n_predicted_spans']:,}"
    )
    if report["unmatched_gold_spans"]:
        lines.append(
            f"WARNING: {report['unmatched_gold_spans']:,} gold spans in the "
            "probability dump had no matching span in the processed corpus"
        )
    lines += ["", "--- predicted spans by localisation bucket ---"]
    for bucket, n in report["predicted_by_bucket"].items():
        lines.append(f"  {bucket:<10} {n:>7,}")

    t = report["typeable"]
    lines += ["", "--- how many predicted spans can carry a type at all ---"]
    lines.append(
        f"  typeable {t['n']:,} ({t['rate_of_predicted']:.4f})   "
        f"excluded as spurious {t['n_excluded_spurious']:,} ({t['excluded_rate']:.4f})"
    )
    lines.append(
        "  a spurious prediction touches no gold span, so no true type exists "
        "for it. It is excluded from typed scoring and counted here instead."
    )

    f6 = report["footnote_6"]
    lines += ["", "--- the footnote-6 case: does a merge cross a type boundary? ---"]
    lines.append(
        f"  components covering more than one gold span: "
        f"{f6['n_multi_gold_components']:,}"
    )
    lines.append(
        f"  of those, covering more than one TYPE: {f6['n_multi_gold_crossing']:,} "
        f"({f6['crossing_rate_of_multi_gold']:.4f})"
    )
    lines.append(
        f"  merge and tangled components only: "
        f"{f6['n_merge_like_crossing_a_type_boundary']:,}/"
        f"{f6['n_merge_like_components']:,} "
        f"({f6['crossing_rate_of_merge_like']:.4f})"
    )
    lines.append(
        f"  typeable predictions affected: {f6['n_typeable_predictions_affected']:,} "
        f"({f6['affected_rate_of_typeable']:.4f} of typeable)"
    )
    lines.append(
        f"  the two conventions disagree on "
        f"{report['conventions_disagree_on']:,} predicted spans"
    )

    gm = report["gold_multi_type"]
    lines += ["", "--- gold spans that already hold more than one type ---"]
    lines.append(f"  {gm['n']:,} ({gm['rate_of_gold']:.4f} of gold). {gm['note']}.")

    lines += ["", "--- inherited type mix ---"]
    for convention, mix in report["type_mix"].items():
        cells = "  ".join(f"{k}={v:,}" for k, v in mix.items())
        lines.append(f"  {convention:<22} {cells}")

    lines += ["", "--- per task ---"]
    for task, block in report["per_task"].items():
        lines.append(
            f"  {task:<14} predicted {block['n_predicted']:>6,}  "
            f"typeable {block['n_typeable']:>6,} ({block['typeable_rate']:.4f})  "
            f"crossing {block['n_crossing']:>4,} "
            f"({block['crossing_rate_of_typeable']:.4f})"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="C3 Layer 2: give predicted spans a type, and measure the hard case."
    )
    parser.add_argument("--probs", default="results/c1/test/probabilities.jsonl")
    parser.add_argument("--processed", default="data/processed/ragtruth_test.jsonl")
    parser.add_argument("--split", default="test", help="name used in the output files")
    parser.add_argument("--out-dir", default="results/c3/analysis")
    args = parser.parse_args(argv)

    records = load_probability_dump(Path(args.probs))
    processed = [
        json.loads(line)
        for line in Path(args.processed).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = build_rows(records, gold_types_by_record(processed))
    report = build_report(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"typed_matching_{args.split}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_rows(out_dir / f"typed_predictions_{args.split}.jsonl", rows["predictions"])
    write_rows(out_dir / f"typed_gold_{args.split}.jsonl", rows["gold"])

    print(format_report(report))
    print()
    print(f"written: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
