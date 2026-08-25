"""Layer 1 of C3: measure RAGTruth's error-type label before modelling it.

C1 printed its tokenisation ceiling beside a 0.1485 span-exact F1 and C2 printed a
constant-base-rate floor beside every ECE. This module does the same job for the
label C3 is about to predict. It answers four questions, none of which anybody has
published for this corpus:

    L1.1  Does ragtruth_labels.py's mapping actually cover the real file?
          The four normalised strings in that module were read off the published
          documentation, not off the downloaded corpus, and assert_label_coverage
          had never been run against data/raw/response.jsonl. Until it has, every
          error_type C3 emits is unverified.

    L1.2  Do the annotators' own words agree with the published label?
          Every annotated span carries `meta`, described by the corpus README as
          "the comment of the hallucination given by annotators". Those comments
          open with a prefix in the annotators' vocabulary -- EVIDENT CONFLICT,
          HIGH INTRO OF NEW INFO -- which is not the same vocabulary as
          `label_type`. Cross-tabulating the two gives a label-noise ceiling that
          caps any type classifier. Agreement is a result (the benchmark validates)
          and disagreement is a result (the classifier's headroom is bounded).

          The prefix vocabulary below is a HYPOTHESIS about what the annotators
          wrote, stated explicitly so it can be argued with. Anything it does not
          recognise is counted as unmapped and reported, never quietly folded into
          a category.

    L1.3  How much of the corpus carries a confound? `implicit_true` marks a span
          that is "correct while the info is not mentioned in the context" -- a
          span annotated as hallucinated that is factually true. `due_to_null` marks
          a data-to-text artefact. `quality` marks a whole response as an incorrect
          refusal or truncated. All three are inherited by every binary detector
          trained on this corpus, C1 included, and none of them is reported anywhere.

    L1.4  What is the floor? Per-type base rates on each split, and the accuracy and
          macro-F1 of a predictor that always answers the training majority class.
          That row is printed beside every C3 classifier number.

The module reads the raw corpus, not the preprocessed jsonl, because preprocessing
drops `meta` and `quality`.

Nothing here needs a GPU, a network or a dependency the repo does not already have.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.c1_detector.ragtruth_labels import (
    assert_label_coverage,
    error_type_of,
    normalise,
    task_type_of,
    unmapped_error_types,
)
from src.common.schema import ErrorType

# --------------------------------------------------------------------------
# Axes
# --------------------------------------------------------------------------
#
# RAGTruth's four categories are a 2x2, not four flat classes:
#
#     mechanism   conflict  (contradicts something in the context)
#                 baseless  (absent from the context altogether)
#     severity    evident   (obvious once you look)
#                 subtle    (a small shift in meaning)
#
# C3 predicts and reports the two axes separately because they are different kinds
# of question: mechanism is a grounding question, severity is a human judgement.

MECHANISM_CONFLICT = "conflict"
MECHANISM_BASELESS = "baseless"
SEVERITY_EVIDENT = "evident"
SEVERITY_SUBTLE = "subtle"

Axes = Tuple[Optional[str], Optional[str]]

_AXES_BY_ERROR_TYPE: Dict[ErrorType, Axes] = {
    ErrorType.EVIDENT_CONFLICT: (MECHANISM_CONFLICT, SEVERITY_EVIDENT),
    ErrorType.SUBTLE_CONFLICT: (MECHANISM_CONFLICT, SEVERITY_SUBTLE),
    ErrorType.EVIDENT_BASELESS: (MECHANISM_BASELESS, SEVERITY_EVIDENT),
    ErrorType.SUBTLE_BASELESS: (MECHANISM_BASELESS, SEVERITY_SUBTLE),
    ErrorType.UNKNOWN: (None, None),
}


def axes_of_error_type(error_type: ErrorType) -> Axes:
    """Split one of the four categories into (mechanism, severity)."""
    return _AXES_BY_ERROR_TYPE.get(error_type, (None, None))


# --------------------------------------------------------------------------
# The annotator vocabulary in `meta`
# --------------------------------------------------------------------------
#
# These patterns are read off the prefixes that actually occur in
# data/raw/response.jsonl. They are deliberately narrow. A prefix that matches
# neither mechanism pattern is reported as unmapped rather than guessed at, and the
# unmapped share is part of the result: a cross-tabulation that quietly absorbs its
# own long tail would report an agreement rate that means nothing.
#
# The severity words differ by mechanism, which is itself worth saying out loud:
# the annotators wrote EVIDENT/SUBTLE about conflicts and HIGH/LOW about introduced
# information, while `label_type` uses EVIDENT/SUBTLE for both.

_CONFLICT_PREFIX = re.compile(r"\bconflicts?\b", re.IGNORECASE)
_BASELESS_PREFIX = re.compile(
    r"\bintro(?:duction)?\s+of\s+(?:new\s+)?info(?:rmation)?\b", re.IGNORECASE
)
_EVIDENT_PREFIX = re.compile(r"\b(?:evident|high)\b", re.IGNORECASE)
_SUBTLE_PREFIX = re.compile(r"\b(?:subtle|low)\b", re.IGNORECASE)

# The severity word the annotator actually typed. This matters more than the
# derived axis: `evident` and `subtle` are the published label's own words, so
# agreement on those is partly tautological, while `high` and `low` are a
# vocabulary that appears nowhere in `label_type`. Only the second is independent
# evidence about the label, so the two are reported separately and never pooled
# into one agreement number.
_SEVERITY_WORD = re.compile(r"\b(evident|subtle|high|low)\b", re.IGNORECASE)

INDEPENDENT_SEVERITY_WORDS = ("high", "low")

# Structural markers inside `meta`. The annotator often quotes the source text they
# judged the span against; L5.2's snippet metric depends on how often they did.
_ORIGINAL_MARKER = re.compile(r"^\s*original\s*:", re.IGNORECASE | re.MULTILINE)
_GENERATED_MARKER = re.compile(r"^\s*generated\s*:", re.IGNORECASE | re.MULTILINE)
_AIGC_MARKER = re.compile(r"^\s*aigc\s*:", re.IGNORECASE | re.MULTILINE)

_PREFIX_SPLIT = re.compile(r"[\n:]")


def meta_prefix(meta: Optional[str]) -> str:
    """The leading label an annotator wrote, up to the first newline or colon.

    Returns "" for an absent or blank comment, which is counted separately rather
    than treated as a category.
    """
    if not meta:
        return ""
    return _PREFIX_SPLIT.split(meta.strip(), 1)[0].strip()


def axes_of_meta_prefix(prefix: str) -> Axes:
    """Read (mechanism, severity) out of an annotator's prefix.

    Either element is None when the prefix does not name it. A prefix naming no
    mechanism is unmapped: the vocabulary above is not widened to swallow it.
    """
    if not prefix:
        return (None, None)
    if _CONFLICT_PREFIX.search(prefix):
        mechanism: Optional[str] = MECHANISM_CONFLICT
    elif _BASELESS_PREFIX.search(prefix):
        mechanism = MECHANISM_BASELESS
    else:
        return (None, None)
    if _EVIDENT_PREFIX.search(prefix):
        severity: Optional[str] = SEVERITY_EVIDENT
    elif _SUBTLE_PREFIX.search(prefix):
        severity = SEVERITY_SUBTLE
    else:
        severity = None
    return (mechanism, severity)


def severity_word(prefix: str) -> Optional[str]:
    """The literal severity word in an annotator prefix, lowercased."""
    if not prefix:
        return None
    match = _SEVERITY_WORD.search(prefix)
    return match.group(1).lower() if match else None


def meta_structure(meta: Optional[str]) -> Dict[str, bool]:
    """Which quoting conventions this comment uses."""
    text = meta or ""
    return {
        "original": bool(_ORIGINAL_MARKER.search(text)),
        "generated": bool(_GENERATED_MARKER.search(text)),
        "aigc": bool(_AIGC_MARKER.search(text)),
    }


def original_snippet(meta: Optional[str]) -> Optional[str]:
    """The text after an `Original:` marker, up to the next line break.

    This is the source text the human annotator pointed at. L5.2 scores C3's
    explanation against it. Returns None when the comment has no such marker.
    """
    if not meta:
        return None
    match = _ORIGINAL_MARKER.search(meta)
    if match is None:
        return None
    rest = meta[match.end():]
    line = rest.split("\n", 1)[0].strip()
    return line or None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def task_type_by_source_id(source_info: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """source_id -> our TaskType value. Raises if a raw task_type is unrecognised."""
    out: Dict[str, str] = {}
    for row in source_info:
        out[str(row["source_id"])] = task_type_of(str(row["task_type"])).value
    return out


def split_by_response_id(
    responses: Iterable[Dict[str, Any]],
    split_ids: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, str]:
    """response id -> split name.

    The corpus ships `train` and `test`. C1 then carved train into train/val/calib
    and froze the ids in results/c1/<run>/split_ids.json; C2 calibrates on that
    calib split. Passing it here makes the base rates line up with the splits every
    other number in the project is reported on. Without it the corpus split is used
    unchanged, and the report records which happened.
    """
    lookup: Dict[str, str] = {}
    if split_ids:
        for name, ids in split_ids.items():
            for rid in ids:
                lookup[str(rid)] = name
    out: Dict[str, str] = {}
    for row in responses:
        rid = str(row["id"])
        out[rid] = lookup.get(rid, str(row.get("split", "unknown")))
    return out


# --------------------------------------------------------------------------
# Floors
# --------------------------------------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def constant_predictor_floor(
    gold: Sequence[str], predicted_class: str
) -> Dict[str, Any]:
    """Score a predictor that answers `predicted_class` for every item.

    This is the row that must sit beside every C3 classifier number. It ignores its
    input entirely, exactly as the constant-base-rate row does for C2's ECE.
    """
    total = len(gold)
    counts = Counter(gold)
    per_class: Dict[str, Dict[str, float]] = {}
    for label in sorted(counts):
        tp = counts[label] if label == predicted_class else 0
        fp = (total - counts[label]) if label == predicted_class else 0
        fn = 0 if label == predicted_class else counts[label]
        per_class[label] = _prf(tp, fp, fn)
    macro_f1 = (
        sum(block["f1"] for block in per_class.values()) / len(per_class)
        if per_class
        else 0.0
    )
    return {
        "predicted_class": predicted_class,
        "n": total,
        "accuracy": (counts.get(predicted_class, 0) / total) if total else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "ignores_the_input": True,
    }


def _rates(counts: Counter) -> Dict[str, Any]:
    total = sum(counts.values())
    return {
        "n": total,
        "counts": {k: counts[k] for k in sorted(counts)},
        "rates": {k: (counts[k] / total if total else 0.0) for k in sorted(counts)},
    }


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


def build_report(
    responses: Sequence[Dict[str, Any]],
    task_by_source: Optional[Dict[str, str]] = None,
    split_ids: Optional[Dict[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    task_by_source = task_by_source or {}
    split_of = split_by_response_id(responses, split_ids)

    raw_label_types: List[str] = []
    spans: List[Dict[str, Any]] = []
    quality_counter: Counter = Counter()
    model_counter: Counter = Counter()
    responses_with_spans = 0

    for row in responses:
        rid = str(row["id"])
        labels = row.get("labels") or []
        quality_counter[str(row.get("quality", "unknown"))] += 1
        model_counter[str(row.get("model", "unknown"))] += 1
        if labels:
            responses_with_spans += 1
        task = task_by_source.get(str(row.get("source_id")), "unknown")
        for label in labels:
            raw = str(label.get("label_type", ""))
            raw_label_types.append(raw)
            error_type = error_type_of(raw)
            mechanism, severity = axes_of_error_type(error_type)
            prefix = meta_prefix(label.get("meta"))
            p_mechanism, p_severity = axes_of_meta_prefix(prefix)
            structure = meta_structure(label.get("meta"))
            spans.append(
                {
                    "response_id": rid,
                    "split": split_of.get(rid, "unknown"),
                    "task_type": task,
                    "model": str(row.get("model", "unknown")),
                    "quality": str(row.get("quality", "unknown")),
                    "raw_label_type": raw,
                    "error_type": error_type.value,
                    "mechanism": mechanism,
                    "severity": severity,
                    "meta_prefix": prefix,
                    "prefix_mechanism": p_mechanism,
                    "prefix_severity": p_severity,
                    "prefix_severity_word": severity_word(prefix),
                    "implicit_true": bool(label.get("implicit_true", False)),
                    "due_to_null": bool(label.get("due_to_null", False)),
                    "structure": structure,
                    "has_meta": bool((label.get("meta") or "").strip()),
                }
            )

    report: Dict[str, Any] = {
        "n_responses": len(responses),
        "n_responses_with_spans": responses_with_spans,
        "n_spans": len(spans),
        "used_frozen_split_ids": bool(split_ids),
        "l1_1_label_coverage": _l1_1(raw_label_types),
        "l1_2_meta_agreement": _l1_2(spans),
        "l1_3_census": _l1_3(spans, quality_counter, model_counter, responses),
        "l1_4_floors": _l1_4(spans),
    }
    return report


# --- L1.1 -----------------------------------------------------------------


def _l1_1(raw_label_types: Sequence[str]) -> Dict[str, Any]:
    distinct = Counter(raw_label_types)
    missing = unmapped_error_types(raw_label_types)
    passed = True
    error = None
    try:
        assert_label_coverage(raw_label_types)
    except ValueError as exc:  # pragma: no cover - exercised by the test suite
        passed = False
        error = str(exc)
    return {
        "distinct_raw_label_types": {k: distinct[k] for k in sorted(distinct)},
        "n_distinct": len(distinct),
        "normalised_forms": sorted({normalise(r) for r in distinct}),
        "assert_label_coverage_passed": passed,
        "unmapped_raw_strings": missing,
        "error": error,
    }


# --- L1.2 -----------------------------------------------------------------


def _l1_2(spans: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    prefix_counts = Counter(s["meta_prefix"] for s in spans)
    mapped = [s for s in spans if s["prefix_mechanism"] is not None]
    unmapped = [s for s in spans if s["prefix_mechanism"] is None]

    mechanism_table: Dict[str, Counter] = defaultdict(Counter)
    for s in mapped:
        mechanism_table[s["prefix_mechanism"]][str(s["mechanism"])] += 1
    mechanism_agree = sum(
        1 for s in mapped if s["prefix_mechanism"] == s["mechanism"]
    )

    with_severity = [s for s in mapped if s["prefix_severity"] is not None]
    severity_table: Dict[str, Counter] = defaultdict(Counter)
    for s in with_severity:
        severity_table[s["prefix_severity"]][str(s["severity"])] += 1
    severity_agree = sum(
        1 for s in with_severity if s["prefix_severity"] == s["severity"]
    )

    joint_table: Dict[str, Counter] = defaultdict(Counter)
    for s in with_severity:
        key = f"{s['prefix_severity']}_{s['prefix_mechanism']}"
        joint_table[key][str(s["error_type"])] += 1
    joint_agree = sum(
        1
        for s in with_severity
        if s["prefix_mechanism"] == s["mechanism"]
        and s["prefix_severity"] == s["severity"]
    )

    # Split the severity agreement by the word the annotator used. Pooling them
    # would let the tautological half carry the independent half.
    word_table: Dict[str, Counter] = defaultdict(Counter)
    for s in with_severity:
        word = s["prefix_severity_word"]
        if word:
            word_table[word][str(s["severity"])] += 1
    independent = [
        s
        for s in with_severity
        if s["prefix_severity_word"] in INDEPENDENT_SEVERITY_WORDS
    ]
    independent_agree = sum(
        1 for s in independent if s["prefix_severity"] == s["severity"]
    )
    tautological = [
        s
        for s in with_severity
        if s["prefix_severity_word"] not in INDEPENDENT_SEVERITY_WORDS
    ]
    tautological_agree = sum(
        1 for s in tautological if s["prefix_severity"] == s["severity"]
    )

    structure_counts = {
        name: sum(1 for s in spans if s["structure"][name])
        for name in ("original", "generated", "aigc")
    }
    structure_by_type: Dict[str, Dict[str, Any]] = {}
    for error_type in sorted({s["error_type"] for s in spans}):
        subset = [s for s in spans if s["error_type"] == error_type]
        structure_by_type[error_type] = {
            "n": len(subset),
            "original": sum(1 for s in subset if s["structure"]["original"]),
            "original_rate": (
                sum(1 for s in subset if s["structure"]["original"]) / len(subset)
                if subset
                else 0.0
            ),
        }
    structure_by_task: Dict[str, Dict[str, Any]] = {}
    for task in sorted({s["task_type"] for s in spans}):
        subset = [s for s in spans if s["task_type"] == task]
        structure_by_task[task] = {
            "n": len(subset),
            "original": sum(1 for s in subset if s["structure"]["original"]),
            "original_rate": (
                sum(1 for s in subset if s["structure"]["original"]) / len(subset)
                if subset
                else 0.0
            ),
        }

    return {
        "n_spans": len(spans),
        "n_with_meta": sum(1 for s in spans if s["has_meta"]),
        "n_blank_meta": sum(1 for s in spans if not s["has_meta"]),
        "n_distinct_prefixes": len(prefix_counts),
        "top_prefixes": [
            {"prefix": p, "n": n} for p, n in prefix_counts.most_common(25)
        ],
        "n_prefix_mapped": len(mapped),
        "n_prefix_unmapped": len(unmapped),
        "prefix_mapped_rate": len(mapped) / len(spans) if spans else 0.0,
        "top_unmapped_prefixes": [
            {"prefix": p, "n": n}
            for p, n in Counter(
                s["meta_prefix"] for s in unmapped
            ).most_common(25)
        ],
        "mechanism": {
            "n_comparable": len(mapped),
            "n_agree": mechanism_agree,
            "agreement": mechanism_agree / len(mapped) if mapped else 0.0,
            "table": {k: dict(v) for k, v in mechanism_table.items()},
        },
        "severity": {
            "n_comparable": len(with_severity),
            "n_agree": severity_agree,
            "agreement": severity_agree / len(with_severity) if with_severity else 0.0,
            "table": {k: dict(v) for k, v in severity_table.items()},
        },
        "severity_by_annotator_word": {
            "table": {k: dict(v) for k, v in sorted(word_table.items())},
            "independent_words": list(INDEPENDENT_SEVERITY_WORDS),
            "independent": {
                "n_comparable": len(independent),
                "n_agree": independent_agree,
                "agreement": (
                    independent_agree / len(independent) if independent else 0.0
                ),
            },
            "shares_label_vocabulary": {
                "n_comparable": len(tautological),
                "n_agree": tautological_agree,
                "agreement": (
                    tautological_agree / len(tautological) if tautological else 0.0
                ),
                "note": (
                    "the annotator wrote the same word the published label uses, so "
                    "this row is not independent evidence about the label"
                ),
            },
        },
        "joint": {
            "n_comparable": len(with_severity),
            "n_agree": joint_agree,
            "agreement": joint_agree / len(with_severity) if with_severity else 0.0,
            "table": {k: dict(v) for k, v in joint_table.items()},
        },
        "structure": {
            "counts": structure_counts,
            "rates": {
                name: (count / len(spans) if spans else 0.0)
                for name, count in structure_counts.items()
            },
            "by_error_type": structure_by_type,
            "by_task": structure_by_task,
        },
    }


# --- L1.3 -----------------------------------------------------------------


def _l1_3(
    spans: Sequence[Dict[str, Any]],
    quality_counter: Counter,
    model_counter: Counter,
    responses: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    def flag_block(flag: str) -> Dict[str, Any]:
        marked = [s for s in spans if s[flag]]
        by_type = Counter(s["error_type"] for s in marked)
        by_task = Counter(s["task_type"] for s in marked)
        totals_by_type = Counter(s["error_type"] for s in spans)
        totals_by_task = Counter(s["task_type"] for s in spans)
        return {
            "n": len(marked),
            "rate": len(marked) / len(spans) if spans else 0.0,
            "by_error_type": {
                k: {
                    "n": by_type.get(k, 0),
                    "share_of_type": (
                        by_type.get(k, 0) / totals_by_type[k]
                        if totals_by_type[k]
                        else 0.0
                    ),
                }
                for k in sorted(totals_by_type)
            },
            "by_task": {
                k: {
                    "n": by_task.get(k, 0),
                    "share_of_task": (
                        by_task.get(k, 0) / totals_by_task[k]
                        if totals_by_task[k]
                        else 0.0
                    ),
                }
                for k in sorted(totals_by_task)
            },
            "by_split": dict(Counter(s["split"] for s in marked)),
        }

    quality_by_split: Dict[str, Counter] = defaultdict(Counter)
    for row in responses:
        quality_by_split[str(row.get("split", "unknown"))][
            str(row.get("quality", "unknown"))
        ] += 1

    model_by_type: Dict[str, Counter] = defaultdict(Counter)
    for s in spans:
        model_by_type[s["model"]][s["error_type"]] += 1

    return {
        "implicit_true": flag_block("implicit_true"),
        "due_to_null": flag_block("due_to_null"),
        "quality": {
            "responses": _rates(quality_counter),
            "by_corpus_split": {k: dict(v) for k, v in quality_by_split.items()},
        },
        "generator_model": {
            "responses": _rates(model_counter),
            "spans_by_model_and_type": {
                k: dict(v) for k, v in sorted(model_by_type.items())
            },
        },
    }


# --- L1.4 -----------------------------------------------------------------


def _l1_4(spans: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in spans:
        by_split[s["split"]].append(s)

    base_rates: Dict[str, Any] = {}
    for split in sorted(by_split):
        subset = by_split[split]
        base_rates[split] = {
            "error_type": _rates(Counter(s["error_type"] for s in subset)),
            "mechanism": _rates(Counter(str(s["mechanism"]) for s in subset)),
            "severity": _rates(Counter(str(s["severity"]) for s in subset)),
            "task_type": _rates(Counter(s["task_type"] for s in subset)),
        }

    train = by_split.get("train", [])
    fitted: Dict[str, Optional[str]] = {}
    for axis in ("error_type", "mechanism", "severity"):
        counts = Counter(str(s[axis]) for s in train)
        fitted[axis] = counts.most_common(1)[0][0] if counts else None

    floors: Dict[str, Any] = {}
    for split in sorted(by_split):
        subset = by_split[split]
        floors[split] = {}
        for axis in ("error_type", "mechanism", "severity"):
            if fitted[axis] is None:
                floors[split][axis] = None
                continue
            floors[split][axis] = constant_predictor_floor(
                [str(s[axis]) for s in subset], fitted[axis]
            )

    return {
        "base_rates": base_rates,
        "majority_class_fitted_on_train": fitted,
        "majority_class_floor": floors,
    }


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------


def format_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("=== C3 Layer 1: RAGTruth error-type label audit ===")
    lines.append(
        f"responses {report['n_responses']:,}  "
        f"with >=1 annotated span {report['n_responses_with_spans']:,}  "
        f"spans {report['n_spans']:,}"
    )
    lines.append(
        "splits from results/c1/.../split_ids.json"
        if report["used_frozen_split_ids"]
        else "splits from the corpus field only (frozen split ids not supplied)"
    )

    cov = report["l1_1_label_coverage"]
    lines += ["", "--- L1.1 label_type coverage ---"]
    lines.append(
        f"  assert_label_coverage: "
        f"{'PASS' if cov['assert_label_coverage_passed'] else 'FAIL'}"
    )
    for raw, n in cov["distinct_raw_label_types"].items():
        lines.append(f"    {raw:<24} {n:>7,}")
    if cov["unmapped_raw_strings"]:
        lines.append(f"    unmapped: {cov['unmapped_raw_strings']}")

    meta = report["l1_2_meta_agreement"]
    lines += ["", "--- L1.2 annotator comment vs published label ---"]
    lines.append(
        f"  distinct meta prefixes {meta['n_distinct_prefixes']:,}   "
        f"blank meta {meta['n_blank_meta']:,}"
    )
    lines.append(
        f"  prefix names a mechanism for {meta['n_prefix_mapped']:,} spans "
        f"({meta['prefix_mapped_rate']:.4f} of all spans); "
        f"{meta['n_prefix_unmapped']:,} unmapped"
    )
    for axis in ("mechanism", "severity", "joint"):
        block = meta[axis]
        lines.append(
            f"  {axis:<10} agreement {block['agreement']:.4f} "
            f"({block['n_agree']:,}/{block['n_comparable']:,})"
        )
    for axis in ("mechanism", "severity"):
        lines.append(f"  {axis} table (annotator prefix -> published label):")
        for key in sorted(meta[axis]["table"]):
            row = meta[axis]["table"][key]
            cells = "  ".join(f"{k}={v:,}" for k, v in sorted(row.items()))
            lines.append(f"    {key:<10} {cells}")
    st = meta["structure"]
    word_block = meta["severity_by_annotator_word"]
    lines.append("  severity by the word the annotator typed:")
    for word in sorted(word_block["table"]):
        cells = "  ".join(
            f"{k}={v:,}" for k, v in sorted(word_block["table"][word].items())
        )
        tag = (
            "independent of the label's vocabulary"
            if word in word_block["independent_words"]
            else "same word as the label"
        )
        lines.append(f"    {word:<8} {cells:<34} ({tag})")
    lines.append(
        f"    independent-vocabulary agreement "
        f"{word_block['independent']['agreement']:.4f} "
        f"({word_block['independent']['n_agree']:,}/"
        f"{word_block['independent']['n_comparable']:,})"
    )
    lines.append(
        "  quoting conventions: "
        + "  ".join(
            f"{k}={st['counts'][k]:,} ({st['rates'][k]:.4f})" for k in sorted(st["counts"])
        )
    )
    lines.append("  top unmapped prefixes:")
    for row in meta["top_unmapped_prefixes"][:8]:
        shown = row["prefix"][:60]
        lines.append(f"    {row['n']:>6,}  {shown!r}")

    census = report["l1_3_census"]
    lines += ["", "--- L1.3 census ---"]
    for flag in ("implicit_true", "due_to_null"):
        block = census[flag]
        lines.append(f"  {flag}: {block['n']:,} spans ({block['rate']:.4f})")
        for key in sorted(block["by_error_type"]):
            cell = block["by_error_type"][key]
            lines.append(
                f"    {key:<22} {cell['n']:>6,}  "
                f"share of that type {cell['share_of_type']:.4f}"
            )
    quality = census["quality"]["responses"]
    lines.append(
        "  response quality: "
        + "  ".join(
            f"{k}={quality['counts'][k]:,} ({quality['rates'][k]:.4f})"
            for k in sorted(quality["counts"])
        )
    )

    floors = report["l1_4_floors"]
    lines += ["", "--- L1.4 base rates and the floor ---"]
    lines.append(
        "  majority class fitted on train: "
        + "  ".join(
            f"{axis}={value}"
            for axis, value in floors["majority_class_fitted_on_train"].items()
        )
    )
    for split in sorted(floors["base_rates"]):
        block = floors["base_rates"][split]["error_type"]
        lines.append(f"  [{split}] n={block['n']:,}")
        for key in sorted(block["counts"]):
            lines.append(
                f"    {key:<22} {block['counts'][key]:>7,}  {block['rates'][key]:.4f}"
            )
        for axis in ("error_type", "mechanism", "severity"):
            floor = floors["majority_class_floor"][split].get(axis)
            if floor:
                lines.append(
                    f"    floor {axis:<12} always '{floor['predicted_class']}': "
                    f"accuracy {floor['accuracy']:.4f}  macro-F1 {floor['macro_f1']:.4f}"
                )
    lines.append("")
    lines.append(
        "Every C3 classifier number is read against the floor rows above and "
        "against the L1.2 agreement rates."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="C3 Layer 1: audit RAGTruth's error-type label before modelling it."
    )
    parser.add_argument("--responses", default="data/raw/response.jsonl")
    parser.add_argument("--source-info", default="data/raw/source_info.jsonl")
    parser.add_argument(
        "--split-ids",
        default="results/c1/modernbert-base/split_ids.json",
        help="frozen train/val/calib ids from the C1 run; pass '' to skip",
    )
    parser.add_argument("--out-dir", default="results/c3/analysis")
    args = parser.parse_args(argv)

    responses = read_jsonl(Path(args.responses))
    task_by_source = task_type_by_source_id(read_jsonl(Path(args.source_info)))

    split_ids = None
    if args.split_ids:
        path = Path(args.split_ids)
        if path.exists():
            split_ids = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"note: {path} not found, falling back to the corpus split field")

    report = build_report(responses, task_by_source=task_by_source, split_ids=split_ids)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "label_audit.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(format_report(report))
    print()
    print(f"written: {out_path}")

    if not report["l1_1_label_coverage"]["assert_label_coverage_passed"]:
        print(
            "assert_label_coverage FAILED. Add the missing strings to "
            "src/c1_detector/ragtruth_labels.py explicitly. Do not widen the "
            "fallback to UNKNOWN."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
