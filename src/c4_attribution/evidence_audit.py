"""Layer 1 -- an audit of RAGTruth's own evidence annotations, before any model runs.

RAGTruth ships an evidence label nobody has used: in most annotator comments
there is an `Original:` line quoting the source text the span was judged
against. Modelling against a label without first measuring it is how a project
ends up scoring a system against a reference that moves. This module measures
it, on C1's frozen splits, and writes one artefact.

It answers, with counts rather than adjectives:

* how much evidence supervision exists, and how much of it is a quote at all;
* how often the quote is verbatim in the context -- the number that decides
  which metric the rest of C4 may use;
* how that splits by task and by error type, because the three tasks are three
  different problems;
* how big the candidate pool is, which is the floor every later number is read
  against;
* how much the two reasonable extraction conventions disagree;
* how much the naive sentence splitter costs, measured instead of admitted.

Nothing here needs a GPU, a model, or a dependency. It is a reportable result on
its own, and it exists before anything can fail.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.c4_attribution import evidence_labels as el
from src.c4_attribution.dataset import (
    REPORTED_SPLITS,
    SpanRecord,
    load_responses,
    span_records,
)
from src.c4_attribution.lexical import best_containment, content_tokens

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/c4/analysis"
REPORT = OUT_DIR / "evidence_audit.json"
SAMPLE = OUT_DIR / "no_evidence_sample.json"
SAMPLE_SHEET = OUT_DIR / "no_evidence_sample.md"

TASKS = ("qa", "data2text", "summarization")
ERROR_TYPES = (
    "Evident Conflict",
    "Evident Baseless Info",
    "Subtle Conflict",
    "Subtle Baseless Info",
)
HAND_CHECK_N = 50
HAND_CHECK_SEED = 42


def _share(count: int, total: int) -> float:
    return count / total if total else 0.0


def _quantiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def at(q: float) -> float:
        if len(ordered) == 1:
            return float(ordered[0])
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        weight = position - low
        return float(ordered[low] * (1 - weight) + ordered[high] * weight)

    return {
        "n": len(ordered),
        "min": float(ordered[0]),
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "max": float(ordered[-1]),
        "mean": float(statistics.fmean(ordered)),
    }


# --------------------------------------------------------------------------
# The per-span measurements
# --------------------------------------------------------------------------


def measure(row: SpanRecord, sentences: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Everything Layer 1 asks about one annotated span.

    `sentences` overrides the record's own pool so the segmentation ablation can
    re-run the identical measurement under a different splitter.
    """
    pool = list(row.sentences if sentences is None else sentences)
    snippet = row.label.snippet or ""
    normalised_snippet = el.normalise(snippet)
    normalised_pool = [el.normalise(s) for s in pool]
    joined = el.normalise(" ".join(pool))

    verbatim = bool(normalised_snippet) and normalised_snippet in joined
    inside_one = bool(normalised_snippet) and any(
        normalised_snippet in s for s in normalised_pool
    )
    n_hits = sum(1 for s in normalised_pool if normalised_snippet and normalised_snippet in s)
    best_index, best_value = best_containment(snippet, pool)

    return {
        "response_id": row.response_id,
        "task": row.task,
        "split": row.response.split,
        "label_type": row.label.label_type,
        "implicit_true": row.label.implicit_true,
        "due_to_null": row.label.due_to_null,
        "has_original": row.label.has_original,
        "no_evidence": row.label.no_evidence,
        "attributable": row.label.attributable,
        "snippet_content_tokens": len(content_tokens(snippet)),
        "n_sentences": len(pool),
        "verbatim": verbatim,
        "inside_exactly_one": inside_one and n_hits == 1,
        "best_index": best_index,
        "best_containment": best_value,
    }


def summarise(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The block of rates reported for any slice of attributable spans."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    containments = [r["best_containment"] for r in rows]
    return {
        "n": n,
        "verbatim": _share(sum(1 for r in rows if r["verbatim"]), n),
        "inside_exactly_one": _share(sum(1 for r in rows if r["inside_exactly_one"]), n),
        "containment_ge_0.9": _share(sum(1 for r in rows if r["best_containment"] >= 0.9), n),
        "containment_ge_0.8": _share(sum(1 for r in rows if r["best_containment"] >= 0.8), n),
        "mean_containment": float(statistics.fmean(containments)),
    }


def supervision_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Section 9.1's table: how much evidence supervision this slice carries."""
    n = len(rows)
    with_original = [r for r in rows if r["has_original"]]
    non_empty = [r for r in rows if r["snippet_content_tokens"] > 0 or r["no_evidence"]]
    no_ev = [r for r in rows if r["no_evidence"]]
    attributable = [r for r in rows if r["attributable"]]
    return {
        "n_annotated_spans": n,
        "n_with_original_marker": len(with_original),
        "share_with_original_marker": _share(len(with_original), n),
        "n_snippet_non_empty": len(non_empty),
        "n_no_evidence": len(no_ev),
        "share_no_evidence_of_snippets": _share(len(no_ev), len(with_original)),
        "n_attributable": len(attributable),
        "share_attributable_of_snippets": _share(len(attributable), len(with_original)),
    }


def _group(rows: Sequence[Dict[str, Any]], key: str, values: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for value in values:
        slice_all = [r for r in rows if r[key] == value]
        slice_attr = [r for r in slice_all if r["attributable"]]
        block = summarise(slice_attr)
        block["n_annotated"] = len(slice_all)
        block["share_no_evidence"] = _share(
            sum(1 for r in slice_all if r["no_evidence"]),
            sum(1 for r in slice_all if r["has_original"]),
        )
        out[value] = block
    return out


# --------------------------------------------------------------------------
# The two extraction conventions, and C3's third one
# --------------------------------------------------------------------------


def convention_disagreement(records: Sequence[SpanRecord]) -> Dict[str, Any]:
    """Until-next-marker against first-line-only, on the same comments.

    Two components reading the same field with different rules and reporting
    different numbers, without either saying so, is the kind of silent
    inconsistency that costs a viva. C3 uses first-line-only. C4 uses
    until-next-marker, because a quote that wraps onto a second line is a quote
    and not two. The disagreement is measured here and reported in the artefact.
    """
    try:  # C3 lives on a branch C4 does not depend on; absence is not a failure.
        from src.c3_explanation.labels_audit import original_snippet as c3_snippet
    except Exception:  # pragma: no cover - exercised only when C3 is absent
        c3_snippet = None

    same = 0
    differ = 0
    c3_matches_first_line = 0
    c3_checked = 0
    examples: List[Dict[str, str]] = []

    for row in records:
        label = row.label
        if not label.has_original:
            continue
        until = label.snippet or ""
        first = el.original_snippet_first_line(label.raw_meta) or ""
        if el.normalise(until) == el.normalise(first):
            same += 1
        else:
            differ += 1
            if len(examples) < 10:
                examples.append(
                    {"until_marker": until[:300], "first_line": first[:300]}
                )
        if c3_snippet is not None:
            c3_checked += 1
            if el.normalise(c3_snippet(label.raw_meta) or "") == el.normalise(first):
                c3_matches_first_line += 1

    total = same + differ
    return {
        "n_comments_with_original": total,
        "identical": same,
        "different": differ,
        "share_different": _share(differ, total),
        "c3_importable": c3_snippet is not None,
        "c3_checked": c3_checked,
        "c3_reproduces_first_line_convention": c3_matches_first_line,
        "examples": examples,
        "note": (
            "C3's labels_audit.original_snippet takes the first line after the "
            "marker; C4 takes everything up to the next annotator marker. Both "
            "are defensible. The rates they produce are not the same number and "
            "must not be quoted interchangeably. The report says which one each "
            "figure came from."
        ),
    }


# --------------------------------------------------------------------------
# Layer 1c -- the segmentation ablation
# --------------------------------------------------------------------------


def segmentation_ablation(records: Sequence[SpanRecord]) -> Dict[str, Any]:
    """What the naive splitter costs, measured rather than admitted.

    The splitter is frozen: `evidence_index` is in the serving contract and two
    other test files assert the three copies are identical. So instead of
    replacing it, Layer 1 runs twice -- once with the naive splitter, once with a
    variant that also breaks on blank lines and RAGTruth's `passage N:` markers
    -- and reports the difference. If the two agree, the weakness is small and
    the report says so with a number. If they do not, the number is the finding.
    """
    attributable = [r for r in records if r.label.attributable]
    naive = [measure(r) for r in attributable]
    variant = [measure(r, r.response.variant_sentences) for r in attributable]

    moved = sum(1 for a, b in zip(naive, variant) if a["best_index"] != b["best_index"])
    pool_naive = [a["n_sentences"] for a in naive]
    pool_variant = [b["n_sentences"] for b in variant]
    pool_differs = sum(
        1
        for row in attributable
        if row.response.sentences != row.response.variant_sentences
    )
    return {
        "n_scored": len(naive),
        "naive": summarise(naive),
        "variant": summarise(variant),
        "delta_verbatim": summarise(variant).get("verbatim", 0.0)
        - summarise(naive).get("verbatim", 0.0),
        "delta_containment_ge_0.9": summarise(variant).get("containment_ge_0.9", 0.0)
        - summarise(naive).get("containment_ge_0.9", 0.0),
        "best_index_moved": moved,
        "share_best_index_moved": _share(moved, len(naive)),
        "n_spans_whose_pool_differs": pool_differs,
        "share_spans_whose_pool_differs": _share(pool_differs, len(naive)),
        "pool_size_naive": _quantiles(pool_naive),
        "pool_size_variant": _quantiles(pool_variant),
    }


# --------------------------------------------------------------------------
# Layer 1b -- the hand-check sample
# --------------------------------------------------------------------------


def _cohens_kappa(r1: Sequence[bool], r2: Sequence[bool]) -> Dict[str, Any]:
    """Compute raw inter-annotator agreement P_o and Cohen's Kappa (kappa)."""
    n = len(r1)
    if n == 0 or len(r2) != n:
        return {
            "inter_annotator_agreement": None,
            "cohens_kappa": None,
            "confusion_matrix": None,
        }
    both_true = sum(1 for a, b in zip(r1, r2) if a and b)
    r1_t_r2_f = sum(1 for a, b in zip(r1, r2) if a and not b)
    r1_f_r2_t = sum(1 for a, b in zip(r1, r2) if not a and b)
    both_false = sum(1 for a, b in zip(r1, r2) if not a and not b)

    p_o = (both_true + both_false) / n
    p_r1_true = (both_true + r1_t_r2_f) / n
    p_r2_true = (both_true + r1_f_r2_t) / n
    p_r1_false = 1.0 - p_r1_true
    p_r2_false = 1.0 - p_r2_true
    p_e = (p_r1_true * p_r2_true) + (p_r1_false * p_r2_false)
    kappa = (p_o - p_e) / (1.0 - p_e) if (1.0 - p_e) > 1e-12 else 1.0
    return {
        "inter_annotator_agreement": round(p_o, 4),
        "cohens_kappa": round(kappa, 4),
        "confusion_matrix": {
            "both_true": both_true,
            "r1_true_r2_false": r1_t_r2_f,
            "r1_false_r2_true": r1_f_r2_t,
            "both_false": both_false,
        },
    }


def no_evidence_sample(
    records: Sequence[SpanRecord], n: int = HAND_CHECK_N, seed: int = HAND_CHECK_SEED
) -> List[Dict[str, Any]]:
    """A seeded sample of "no evidence" snippets for a human to label.

    The no-evidence split is a regex over a phrase list. Until somebody reads
    fifty of them and says whether the regex was right, it is a heuristic and
    every count that rests on it inherits that. This builds the sheet; the
    agreement rate goes back into the artefact once it is filled in.
    """
    import numpy as np

    candidates = [r for r in records if r.label.snippet and r.label.no_evidence]
    if not candidates:
        return []
    rng = np.random.default_rng(seed)
    picks = rng.permutation(len(candidates))[: min(n, len(candidates))]
    out = []
    for position in sorted(int(p) for p in picks):
        row = candidates[position]
        out.append(
            {
                "response_id": row.response_id,
                "split": row.response.split,
                "task": row.task,
                "label_type": row.label.label_type,
                "span_text": row.label.span_text,
                "snippet": row.label.snippet,
                "human_says_no_evidence": None,
                "human_rater_2_no_evidence": None,
            }
        )
    return out


def hand_check_agreement(path: Path = SAMPLE) -> Dict[str, Any]:
    """Read back the labelled sample and turn it into the agreement rate.

    Until a human has filled in `human_says_no_evidence`, this returns a TODO
    and every count resting on the phrase list is reported as a heuristic. Once
    it is filled in, the rate lands in the artefact and Layer 7's precision
    claim is either supported or downgraded -- and if the regex turns out to be
    wrong, the label is downgraded rather than quietly kept.
    """
    if not path.exists():
        return {"labelled": 0, "agreement": None, "status": "TODO -- sample not built"}
    rows = json.loads(path.read_text(encoding="utf-8"))
    labelled = [r for r in rows if r.get("human_says_no_evidence") is not None]
    if not labelled:
        return {
            "n_sampled": len(rows),
            "labelled": 0,
            "agreement": None,
            "status": (
                "TODO -- nobody has filled in the sheet yet, so the no-evidence "
                "label is a phrase-list heuristic and is reported as one."
            ),
        }
    agree = sum(1 for r in labelled if bool(r["human_says_no_evidence"]))
    rate = agree / len(labelled)
    result: Dict[str, Any] = {
        "n_sampled": len(rows),
        "labelled": len(labelled),
        "agreement": rate,
        "status": (
            "the phrase list agrees with a human on "
            f"{agree} of {len(labelled)} sampled snippets"
        ),
        "verdict": (
            "The label may be quoted as a label."
            if rate >= 0.9
            else
            "Agreement is below 0.90. Downgrade the no-evidence label to a "
            "heuristic in every table that uses it, or drop Layer 7's precision "
            "claim and keep only the recall analysis. Do not quietly keep the "
            "regex."
        ),
    }
    dual = [
        r for r in labelled if r.get("human_rater_2_no_evidence") is not None
    ]
    if len(dual) == len(labelled):
        r1 = [bool(r["human_says_no_evidence"]) for r in dual]
        r2 = [bool(r["human_rater_2_no_evidence"]) for r in dual]
        agree_r2 = sum(1 for v in r2 if v)
        irr = _cohens_kappa(r1, r2)
        result["rater_2_agreement"] = round(agree_r2 / len(dual), 4)
        result.update(irr)
    return result


def write_sample_sheet(sample: Sequence[Dict[str, Any]], path: Path = SAMPLE_SHEET) -> None:
    """A markdown sheet with one line per sampled snippet, for hand labelling."""
    lines = [
        "# Layer 1b -- hand check of the 'no evidence' regex",
        "",
        "Fifty snippets the phrase list called 'no evidence exists'. For each,",
        "write `yes` if the annotator really is saying the information is absent",
        "from the source, and `no` if the regex misfired on a real quote.",
        "",
        "Agreement goes into results/c4/analysis/evidence_audit.json. Until it",
        "does, every count resting on this label is a heuristic and is reported",
        "as one.",
        "",
        "| # | task | label type | annotator's Original: text | rater 1 | rater 2 |",
        "|---|---|---|---|---|---|",
    ]
    for index, row in enumerate(sample, 1):
        # Read with .get: the sheet is regenerated from a file a human may have
        # edited by hand, and a missing column should not stop the rebuild.
        snippet = (row.get("snippet") or "").replace("|", "\\|").replace("\n", " ")
        verdict = row.get("human_says_no_evidence")
        verdict_2 = row.get("human_rater_2_no_evidence")
        yn = "yes" if verdict is True else ("no" if verdict is False else "")
        yn2 = "yes" if verdict_2 is True else ("no" if verdict_2 is False else "")
        lines.append(
            f"| {index} | {row.get('task', '?')} | {row.get('label_type', '?')} "
            f"| {snippet[:200]} | {yn} | {yn2} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# The artefact
# --------------------------------------------------------------------------


def build_report(records: Sequence[SpanRecord]) -> Dict[str, Any]:
    rows = [measure(r) for r in records]
    attributable = [r for r in rows if r["attributable"]]

    per_split: Dict[str, Any] = {}
    for split in REPORTED_SPLITS:
        slice_rows = [r for r in rows if r["split"] == split]
        slice_attr = [r for r in slice_rows if r["attributable"]]
        per_split[split] = {
            "counts": supervision_counts(slice_rows),
            "rates": summarise(slice_attr),
        }

    metas = [row.label.raw_meta for row in records]
    return {
        "provenance": {
            "source": "data/raw/response.jsonl + data/raw/source_info.jsonl",
            "splits": list(REPORTED_SPLITS),
            "split_ids": "results/c1/modernbert-base/split_ids.json",
            "context_rule": "src.c1_detector.build_examples.build_context_and_question",
            "sentence_splitter": "naive (?<=[.!?])\\s+, identical to the serving path",
            "snippet_convention": "until the next annotator marker",
            "note": (
                "These are C1's frozen splits, not the whole corpus. A "
                "corpus-wide number and a frozen-split number are different "
                "quantities and are never mixed in the report."
            ),
        },
        "counts": supervision_counts(rows),
        "rates": summarise(attributable),
        "per_split": per_split,
        "by_task": _group(rows, "task", TASKS),
        "by_error_type": _group(rows, "label_type", ERROR_TYPES),
        "candidate_pool": {
            "all": _quantiles([r["n_sentences"] for r in attributable]),
            **{
                task: _quantiles(
                    [r["n_sentences"] for r in attributable if r["task"] == task]
                )
                for task in TASKS
            },
        },
        "snippet_length_content_words": _quantiles(
            [r["snippet_content_tokens"] for r in attributable]
        ),
        "markers": el.marker_counts(metas),
        "confounds": {
            "implicit_true": {
                "n": sum(1 for r in rows if r["implicit_true"]),
                "share": _share(sum(1 for r in rows if r["implicit_true"]), len(rows)),
                "share_of_subtle_baseless": _share(
                    sum(
                        1
                        for r in rows
                        if r["implicit_true"] and r["label_type"] == "Subtle Baseless Info"
                    ),
                    sum(1 for r in rows if r["label_type"] == "Subtle Baseless Info"),
                ),
            },
            "due_to_null": {
                "n": sum(1 for r in rows if r["due_to_null"]),
                "share": _share(sum(1 for r in rows if r["due_to_null"]), len(rows)),
            },
        },
        "convention": convention_disagreement(records),
        "segmentation_ablation": segmentation_ablation(records),
        "hand_check": {
            "n_sampled": HAND_CHECK_N,
            "seed": HAND_CHECK_SEED,
            "sheet": str(SAMPLE_SHEET),
            "sample": str(SAMPLE),
            **hand_check_agreement(),
            "note": (
                "The no-evidence label is a phrase-list heuristic until this "
                "agreement rate exists. Layer 7's precision claim is gated on "
                "it. Fill in human_says_no_evidence in the sample JSON (or the "
                "markdown sheet, then copy the answers across) and rerun."
            ),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    records = span_records(load_responses())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # The sample is written first and only when it does not already exist, so a
    # rerun cannot wipe out labels somebody has already entered by hand.
    if not SAMPLE.exists():
        sample = no_evidence_sample(records)
        SAMPLE.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
    else:
        sample = json.loads(SAMPLE.read_text(encoding="utf-8"))
    write_sample_sheet(sample)

    report = build_report(records)
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    counts = report["counts"]
    rates = report["rates"]
    print(f"annotated spans            {counts['n_annotated_spans']}")
    print(
        f"with an Original: marker   {counts['n_with_original_marker']} "
        f"({counts['share_with_original_marker']:.4f})"
    )
    print(
        f"  of those, no evidence    {counts['n_no_evidence']} "
        f"({counts['share_no_evidence_of_snippets']:.4f})"
    )
    print(f"  of those, attributable   {counts['n_attributable']}")
    if rates.get("n"):
        print(f"verbatim in context        {rates['verbatim']:.4f}")
        print(f"containment >= 0.9         {rates['containment_ge_0.9']:.4f}")
        print(f"mean containment           {rates['mean_containment']:.4f}")
    else:
        print("verbatim / containment     no attributable spans in this slice")
    print(f"wrote {REPORT} and {SAMPLE_SHEET}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
