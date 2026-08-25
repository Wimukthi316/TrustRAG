"""Layer 5.2 of C3: score the explanations instead of demoing them.

RAGTruth's `meta` field is described by the corpus README as "the comment of the
hallucination given by annotators" -- a human-written rationale on every
annotated span. Layer 1 measured what is inside it: 63.72% of spans carry an
explicit `Original:` line quoting the source text the annotator judged the span
against. That line is what makes a real metric possible.

Three numbers come out of here, in decreasing order of how much they are worth.

**Snippet match (primary).** Does C3 quote the same piece of source the human
quoted? This is a grounding question with a right answer, and it is essentially
free. Reported as mean containment of the annotator's snippet in C3's quoted
sentence, plus the rate at several thresholds so no single cut-off has to be
defended.

**Self-faithfulness.** Does the sentence C3 quotes actually appear in the
context, verbatim? It should be 1.0 by construction. It is measured anyway,
because a template bug or, later, a generative model inventing its own evidence
would show up here and nowhere else.

**ROUGE-L against the whole `meta` text (weakest).** It measures phrasing
overlap against a single reference written in a different register -- an
annotator's shorthand note, not a user-facing sentence. It is reported with a
trivial-constant-explanation floor beside it, and the caption has to say that a
higher ROUGE-L here does not mean a more correct explanation.

The LLM comparison row of the plan is not run. The free-tier quota measured in
the D3 runbook is roughly twenty judgements per model per day, the two exposed
API keys are still outstanding, and the standing rule is to report the n obtained
and stop rather than pad. So the row says n = 0 and why.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.c3_explanation.explanations import explain
from src.c3_explanation.grounding import content_tokens, split_sentences, tokenise
from src.c3_explanation.labels_audit import original_snippet
from src.c3_explanation.type_baseline import primary_type

# The floor: an explanation that ignores the span, the type and the context.
CONSTANT_EXPLANATION = "This span is not supported by the source."

THRESHOLDS: Tuple[float, ...] = (0.5, 0.8, 1.0)

_WHITESPACE = re.compile(r"\s+")


def normalise_for_substring(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "")).strip().lower()


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length, the quantity ROUGE-L is built on."""
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(candidate: str, reference: str) -> Dict[str, float]:
    """ROUGE-L precision, recall and F1 over lowercased word tokens."""
    c = tokenise(candidate)
    r = tokenise(reference)
    lcs = lcs_length(c, r)
    precision = lcs / len(c) if c else 0.0
    recall = lcs / len(r) if r else 0.0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def snippet_containment(snippet: str, quoted: Optional[str]) -> Optional[float]:
    """Fraction of the annotator's snippet tokens that appear in C3's quote.

    Returns None when the annotator quoted nothing, so those spans are excluded
    from the mean rather than scored as zero -- the metric has no opinion about a
    span it has no reference for, and averaging in a zero would invent one.
    """
    reference = set(content_tokens(snippet))
    if not reference:
        return None
    if not quoted:
        return 0.0
    return len(reference & set(content_tokens(quoted))) / len(reference)


def quoted_is_verbatim(quoted: Optional[str], context: str) -> Optional[bool]:
    """Self-faithfulness: is the quoted sentence really in the context?"""
    if not quoted:
        return None
    return normalise_for_substring(quoted) in normalise_for_substring(context)


# --------------------------------------------------------------------------
# Joining the corpus
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def meta_by_span(raw_responses: Iterable[Dict[str, Any]]) -> Dict[str, Dict[Tuple[int, int], str]]:
    """response id -> (start, end) -> the annotator's comment."""
    out: Dict[str, Dict[Tuple[int, int], str]] = defaultdict(dict)
    for row in raw_responses:
        rid = str(row["id"])
        for label in row.get("labels") or []:
            out[rid][(int(label["start"]), int(label["end"]))] = str(
                label.get("meta") or ""
            )
    return out


def lookup_meta(
    table: Dict[Tuple[int, int], str], start: int, end: int
) -> Tuple[str, str]:
    """The comment for a span, exact if possible, else the best overlapping one.

    Preprocessing merged some overlapping raw annotations, so a processed gold
    span does not always carry the exact offsets an annotator used. Falling back
    to the annotation sharing the most characters keeps those spans in the
    measurement, and the join mode is returned so the report can say how many
    were exact.
    """
    if (start, end) in table:
        return table[(start, end)], "exact"
    best_key = None
    best_overlap = 0
    for (a, b), _ in table.items():
        shared = max(0, min(end, b) - max(start, a))
        if shared > best_overlap:
            best_key, best_overlap = (a, b), shared
    if best_key is None:
        return "", "none"
    return table[best_key], "overlap"


# --------------------------------------------------------------------------
# Scoring one system
# --------------------------------------------------------------------------


def score_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the per-span measurements into the numbers C3-7 reports."""
    containments = [r["snippet_containment"] for r in rows if r["snippet_containment"] is not None]
    faithful = [r["quoted_is_verbatim"] for r in rows if r["quoted_is_verbatim"] is not None]
    rouge = [r["rouge_l_f1"] for r in rows if r["rouge_l_f1"] is not None]

    block: Dict[str, Any] = {
        "n_spans": len(rows),
        "snippet": {
            "n_with_reference": len(containments),
            "reference_rate": len(containments) / len(rows) if rows else 0.0,
            "mean_containment": (
                sum(containments) / len(containments) if containments else float("nan")
            ),
            "rate_at_threshold": {
                str(t): (
                    sum(1 for v in containments if v >= t) / len(containments)
                    if containments
                    else float("nan")
                )
                for t in THRESHOLDS
            },
        },
        "self_faithfulness": {
            "n_quoting_a_sentence": len(faithful),
            "quote_rate": len(faithful) / len(rows) if rows else 0.0,
            "verbatim_rate": (
                sum(1 for v in faithful if v) / len(faithful)
                if faithful
                else float("nan")
            ),
        },
        "rouge_l_vs_meta": {
            "n": len(rouge),
            "mean_f1": sum(rouge) / len(rouge) if rouge else float("nan"),
            "caption": (
                "single reference, written by an annotator as a shorthand note "
                "rather than as a user-facing sentence. A higher value here is "
                "phrasing overlap, not correctness."
            ),
        },
    }

    for key, group in (("by_task", "task_type"), ("by_error_type", "error_type")):
        cells: Dict[str, Any] = {}
        for name in sorted({str(r[group]) for r in rows}):
            subset = [r for r in rows if str(r[group]) == name]
            values = [
                r["snippet_containment"]
                for r in subset
                if r["snippet_containment"] is not None
            ]
            cells[name] = {
                "n": len(subset),
                "n_with_reference": len(values),
                "mean_containment": (
                    sum(values) / len(values) if values else float("nan")
                ),
            }
        block[key] = cells
    return block


def build_rows(
    processed: Sequence[Dict[str, Any]],
    metas: Dict[str, Dict[Tuple[int, int], str]],
    type_source: str = "gold",
    predicted_types: Optional[Dict[Tuple[str, int, int], str]] = None,
    constant: bool = False,
) -> List[Dict[str, Any]]:
    """Explain every gold span and measure the explanation.

    `type_source` picks which type the template is built from: "gold" scores the
    generator on its own, "predicted" scores the generator downstream of the type
    model, so the difference between the two rows is the cost of a type error.
    """
    rows: List[Dict[str, Any]] = []
    for record in processed:
        rid = str(record["id"])
        context = record.get("context", "")
        sentences = split_sentences(context)
        table = metas.get(rid, {})
        for span in record.get("spans") or []:
            start, end = int(span["start"]), int(span["end"])
            gold_type = primary_type(span)
            if type_source == "predicted" and predicted_types is not None:
                error_type = predicted_types.get((rid, start, end), gold_type)
            else:
                error_type = gold_type
            meta, join_mode = lookup_meta(table, start, end)
            snippet = original_snippet(meta)

            if constant:
                generated: Dict[str, Any] = {
                    "explanation": CONSTANT_EXPLANATION,
                    "quoted_sentence": None,
                    "quoted_sentence_index": None,
                    "committed_to": "none",
                }
            else:
                generated = explain(
                    str(span.get("text", "")), sentences, error_type=error_type
                )

            rows.append(
                {
                    "response_id": rid,
                    "task_type": record.get("task_type", "unknown"),
                    "error_type": gold_type,
                    "explained_as": error_type,
                    "start": start,
                    "end": end,
                    "meta_join": join_mode,
                    "has_meta": bool(meta.strip()),
                    "explanation": generated["explanation"],
                    "quoted_sentence": generated["quoted_sentence"],
                    "snippet_containment": (
                        snippet_containment(snippet, generated["quoted_sentence"])
                        if snippet
                        else None
                    ),
                    "quoted_is_verbatim": quoted_is_verbatim(
                        generated["quoted_sentence"], context
                    ),
                    "rouge_l_f1": (
                        rouge_l(generated["explanation"], meta)["f1"]
                        if meta.strip()
                        else None
                    ),
                }
            )
    return rows


def format_report(report: Dict[str, Any]) -> str:
    lines = ["=== C3 Layer 5.2: explanations, scored ==="]
    for name in report["systems"]:
        block = report["systems"][name]
        lines.append("")
        lines.append(f"--- {name}  ({block['n_spans']:,} spans) ---")
        snippet = block["snippet"]
        lines.append(
            f"  snippet match: mean containment {snippet['mean_containment']:.4f} "
            f"over {snippet['n_with_reference']:,} spans "
            f"({snippet['reference_rate']:.4f} of spans carry an Original: line)"
        )
        for threshold, rate in snippet["rate_at_threshold"].items():
            lines.append(f"    at containment >= {threshold}: {rate:.4f}")
        faith = block["self_faithfulness"]
        lines.append(
            f"  self-faithfulness: quoted a sentence on {faith['quote_rate']:.4f} "
            f"of spans, verbatim in the context {faith['verbatim_rate']:.4f}"
        )
        rouge = block["rouge_l_vs_meta"]
        lines.append(f"  ROUGE-L vs meta: {rouge['mean_f1']:.4f} (n={rouge['n']:,})")
        lines.append("  snippet containment by task:")
        for task, cell in block["by_task"].items():
            lines.append(
                f"    {task:<16} n={cell['n']:>6,}  with reference "
                f"{cell['n_with_reference']:>6,}  {cell['mean_containment']:.4f}"
            )
        lines.append("  snippet containment by error type:")
        for name_, cell in block["by_error_type"].items():
            lines.append(
                f"    {name_:<24} n={cell['n']:>6,}  {cell['mean_containment']:.4f}"
            )
    lines.append("")
    lines.append(f"ROUGE-L caption: {report['systems']['template_gold_type']['rouge_l_vs_meta']['caption']}")
    lines.append("")
    lines.append(f"Snippet metric: {report['snippet_metric_note']}")
    lines.append("")
    join = report["join"]
    lines.append(
        f"meta join: exact {join['exact']:,}  by overlap {join['overlap']:,}  "
        f"none {join['none']:,} of {join['n']:,}"
    )
    lines.append("")
    lines.append(f"LLM comparison: {report['llm_comparison']['status']}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="C3 Layer 5.2: score template explanations against the annotators' notes."
    )
    parser.add_argument("--processed", default="data/processed/ragtruth_test.jsonl")
    parser.add_argument("--raw", default="data/raw/response.jsonl")
    parser.add_argument(
        "--baseline-probs",
        default="results/c3/type/baseline_probs_gold_test.jsonl",
        help="type-model predictions, for the downstream row; '' to skip",
    )
    parser.add_argument("--out-dir", default="results/c3/explanation")
    args = parser.parse_args(argv)

    processed = read_jsonl(Path(args.processed))
    metas = meta_by_span(read_jsonl(Path(args.raw)))

    predicted_types: Optional[Dict[Tuple[str, int, int], str]] = None
    probs_path = Path(args.baseline_probs) if args.baseline_probs else None
    if probs_path and probs_path.exists():
        predicted_types = {}
        for row in read_jsonl(probs_path):
            classes = row["classes"]
            best = max(range(len(classes)), key=lambda i: row["probs_flat"][i])
            predicted_types[
                (str(row["response_id"]), int(row["start"]), int(row["end"]))
            ] = classes[best]

    gold_rows_scored = build_rows(processed, metas, "gold")
    systems: Dict[str, Any] = {
        "constant_floor": score_rows(build_rows(processed, metas, constant=True)),
        "template_gold_type": score_rows(gold_rows_scored),
    }
    if predicted_types is not None:
        systems["template_predicted_type"] = score_rows(
            build_rows(processed, metas, "predicted", predicted_types)
        )

    report = {
        "systems": systems,
        "constant_explanation": CONSTANT_EXPLANATION,
        "llm_comparison": {
            "n": 0,
            "status": (
                "not run. The free-tier quota measured in the D3 runbook is about "
                "twenty judgements per model per day and the two exposed API keys "
                "are still outstanding. The standing rule is to report the n "
                "obtained and stop, so this row is n = 0 rather than an estimate."
            ),
        },
        "join": {
            "n": len(gold_rows_scored),
            "exact": sum(1 for r in gold_rows_scored if r["meta_join"] == "exact"),
            "overlap": sum(1 for r in gold_rows_scored if r["meta_join"] == "overlap"),
            "none": sum(1 for r in gold_rows_scored if r["meta_join"] == "none"),
            "note": (
                "preprocessing merged some overlapping raw annotations, so a gold "
                "span does not always carry an annotator's exact offsets; those "
                "fall back to the annotation sharing the most characters"
            ),
        },
        "snippet_metric_note": (
            "The quoted sentence is chosen by grounding, not by predicted type, "
            "so snippet containment is identical for the gold-type and "
            "predicted-type rows by construction. What a type error changes is "
            "the WORDING, which is why ROUGE-L moves between those two rows and "
            "containment does not. Reported this way rather than dropped, "
            "because the invariance is a property of the design worth stating."
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "explanation_metrics.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(format_report(report))
    print()
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
