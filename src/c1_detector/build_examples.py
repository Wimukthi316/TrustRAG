"""Join RAGTruth's two JSONL files into flat, per-response training examples.

This is the tokenizer-free half of preprocessing. It does the joining, the text
assembly and the offset validation, and produces records that a human can read
and check without loading a model. Tokenisation and BIO tagging live in bio.py.

The split exists on purpose. Almost every way this pipeline can silently produce
garbage is an offset problem, and offset problems are far easier to catch here --
where a span is still a slice of a plain Python string -- than after a subword
tokenizer has been applied.

Input   data/raw/response.jsonl, data/raw/source_info.jsonl
Output  data/processed/ragtruth_train.jsonl, data/processed/ragtruth_test.jsonl

Usage:
    python -m src.c1_detector.build_examples
    python -m src.c1_detector.build_examples --stats-only
    python -m src.c1_detector.build_examples --drop-implicit-true --quality good

Record shape (one JSON object per line):
    {
      "id": "1472",
      "source_id": "11316",
      "model": "mistral-7B-instruct",
      "task_type": "summarization",
      "source": "CNN/DM",
      "split": "train",
      "quality": "good",
      "question": "...",
      "context": "...",
      "answer": "...",
      "spans": [
        {"start": 219, "end": 229, "text": "Gaza Strip",
         "error_type": "evident_baseless_info",
         "implicit_true": false, "due_to_null": false}
      ]
    }

`start` and `end` are character offsets into `answer`, half-open, exactly as
RAGTruth publishes them and exactly as src/common/schema.py defines them. This
was checked against the sample record in the RAGTruth README: response "1472",
label [219:229], slices out to "Gaza Strip". They are plain Python string
offsets, not byte offsets and not token offsets.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c1_detector.ragtruth_labels import (  # noqa: E402
    assert_label_coverage,
    error_type_of,
    task_type_of,
)
from src.common.schema import TaskType  # noqa: E402

# --------------------------------------------------------------------------
# Text assembly
# --------------------------------------------------------------------------
#
# The detector reads (context, question, answer). RAGTruth stores the context in
# `source_info`, whose type depends on the task, so each task needs its own
# unpacking rule.
#
# CAVEAT, and it matters for how the baseline comparison is worded: LettuceDetect
# reports 79.22% example-level F1 on RAGTruth, but the exact string formatting it
# feeds the encoder is not something this file reproduces -- it was not verified.
# Different context formatting changes tokenisation and therefore the numbers.
# Report our result as "our formatting, our training run", and if a like-for-like
# comparison is needed, evaluate the public LettuceDetect checkpoint through this
# same pipeline rather than quoting its paper number as if it were measured here.

# Stand-in questions for the two tasks that have no natural user question. Kept
# short and constant so they add the same token overhead to every example.
_QA_LESS_QUESTION = {
    TaskType.DATA2TEXT: "Write an objective overview of this business based only on the structured data.",
    TaskType.SUMMARIZATION: "Summarize the following news article.",
}


def build_context_and_question(task: TaskType, source_info: Any) -> tuple[str, str]:
    """Turn a RAGTruth `source_info` value into a (context, question) pair.

    QA           source_info is a dict with "question" and "passages".
    Data2txt     source_info is a dict describing a business; serialise it.
    Summary      source_info is already the article string.
    """
    if task is TaskType.QA:
        if not isinstance(source_info, dict):
            raise TypeError(f"QA source_info should be a dict, got {type(source_info).__name__}")
        return str(source_info.get("passages", "")), str(source_info.get("question", ""))

    if task is TaskType.DATA2TEXT:
        if not isinstance(source_info, dict):
            raise TypeError(
                f"Data2txt source_info should be a dict, got {type(source_info).__name__}"
            )
        # Stable key order so the same record always produces the same string.
        context = json.dumps(source_info, ensure_ascii=False, sort_keys=True)
        return context, _QA_LESS_QUESTION[TaskType.DATA2TEXT]

    if task is TaskType.SUMMARIZATION:
        return str(source_info), _QA_LESS_QUESTION[TaskType.SUMMARIZATION]

    raise ValueError(f"no context rule for task type {task}")


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run: python -m src.c1_detector.download_ragtruth"
        )
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} is not valid JSON: {exc}") from exc


def load_sources(path: Path) -> Dict[str, Dict[str, Any]]:
    sources: Dict[str, Dict[str, Any]] = {}
    for rec in read_jsonl(path):
        sources[str(rec["source_id"])] = rec
    return sources


# --------------------------------------------------------------------------
# Span handling
# --------------------------------------------------------------------------


class OffsetMismatch(Exception):
    """A published label's offsets do not slice out its own text."""


def clean_spans(
    answer: str,
    labels: List[Dict[str, Any]],
    response_id: str,
    drop_implicit_true: bool = False,
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Validate, normalise and sort one response's hallucination labels.

    Returns (spans, problems). `problems` is a list of human-readable strings;
    an empty list means every label checked out. Bad labels are dropped, never
    silently repaired -- a repaired offset is a fabricated one.
    """
    spans: List[Dict[str, Any]] = []
    problems: List[str] = []
    n = len(answer)

    for i, lab in enumerate(labels or []):
        start, end = lab.get("start"), lab.get("end")
        text = lab.get("text", "")

        if start is None or end is None:
            problems.append(f"{response_id} label {i}: missing start/end")
            continue
        start, end = int(start), int(end)

        if not (0 <= start < end <= n):
            problems.append(
                f"{response_id} label {i}: [{start}:{end}] outside a {n}-char response"
            )
            continue
        if answer[start:end] != text:
            problems.append(
                f"{response_id} label {i}: [{start}:{end}] slices out "
                f"{answer[start:end]!r} but text field says {text!r}"
            )
            continue

        implicit_true = bool(lab.get("implicit_true", False))
        if drop_implicit_true and implicit_true:
            continue

        spans.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "error_type": error_type_of(str(lab.get("label_type", ""))).value,
                "implicit_true": implicit_true,
                "due_to_null": bool(lab.get("due_to_null", False)),
            }
        )

    spans.sort(key=lambda s: (s["start"], s["end"]))
    return spans, problems


def merge_overlapping(
    answer: str, spans: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], int]:
    """Collapse overlapping labels into maximal non-overlapping spans.

    About 0.9% of RAGTruth's spans overlap another span on the same response:
    exact duplicates, a short label nested inside a longer one, and a handful of
    partial overlaps. These are annotation artefacts, not a parsing bug -- every
    one of them still slices out its own text correctly.

    They have to be merged anyway. The BIO tagger unions overlapping spans by
    construction, so ground truth that still contains overlaps would report more
    spans than the labels can ever decode back to, and span-level recall would be
    capped below 1.0 for reasons that have nothing to do with the model.

    Counting is unaffected: the reproduction check against the published
    statistics table runs on the raw labels, before this.

    When merged spans disagree on category -- 31 pairs do -- the longest
    contributor wins `error_type`, and every contributing category is kept in
    `error_types` so the C3 taxonomy work is not blocked by this choice.
    """
    if not spans:
        return [], 0

    merged: List[Dict[str, Any]] = []
    group: List[Dict[str, Any]] = [spans[0]]

    def flush(members: List[Dict[str, Any]]) -> Dict[str, Any]:
        start = min(m["start"] for m in members)
        end = max(m["end"] for m in members)
        primary = max(members, key=lambda m: m["end"] - m["start"])
        return {
            "start": start,
            "end": end,
            "text": answer[start:end],
            "error_type": primary["error_type"],
            "error_types": sorted({m["error_type"] for m in members}),
            "implicit_true": all(m["implicit_true"] for m in members),
            "due_to_null": any(m["due_to_null"] for m in members),
        }

    for span in spans[1:]:
        if span["start"] < max(m["end"] for m in group):
            group.append(span)
        else:
            merged.append(flush(group))
            group = [span]
    merged.append(flush(group))

    return merged, len(spans) - len(merged)


# --------------------------------------------------------------------------
# Main build
# --------------------------------------------------------------------------


def build(
    raw_dir: Path,
    drop_implicit_true: bool = False,
    quality_filter: Optional[str] = None,
    keep_overlaps: bool = False,
) -> tuple[List[Dict[str, Any]], List[str], Counter, int]:
    """Join the two files into flat records.

    Returns (records, problems, raw_counter, merged_away). `raw_counter` is
    computed before any filtering or merging so the published statistics table
    can be reproduced. `merged_away` is how many spans were absorbed into another
    by overlap merging.
    """
    sources = load_sources(raw_dir / "source_info.jsonl")
    records: List[Dict[str, Any]] = []
    problems: List[str] = []
    raw_counter: Counter = Counter()
    raw_label_types: List[str] = []
    seen_source_ids: set = set()
    merged_away = 0

    for rec in read_jsonl(raw_dir / "response.jsonl"):
        source_id = str(rec["source_id"])
        src = sources.get(source_id)
        if src is None:
            problems.append(f"response {rec.get('id')}: source_id {source_id} not found")
            continue

        task = task_type_of(str(src["task_type"]))
        source_name = str(src.get("source", ""))
        answer = str(rec.get("response", ""))
        labels = rec.get("labels") or []

        # Unfiltered tallies, for the reproduction check against the README table.
        bucket = _stats_bucket(task, source_name)
        raw_counter[(bucket, "responses")] += 1
        raw_counter[(bucket, "spans")] += len(labels)
        if labels:
            raw_counter[(bucket, "hallucinated_responses")] += 1
        if source_id not in seen_source_ids:
            seen_source_ids.add(source_id)
            raw_counter[(bucket, "instances")] += 1
        raw_label_types.extend(str(lab.get("label_type", "")) for lab in labels)

        if quality_filter is not None and str(rec.get("quality", "")) != quality_filter:
            continue

        try:
            context, question = build_context_and_question(task, src.get("source_info"))
        except (TypeError, ValueError) as exc:
            problems.append(f"response {rec.get('id')}: {exc}")
            continue

        spans, span_problems = clean_spans(
            answer, labels, str(rec.get("id")), drop_implicit_true=drop_implicit_true
        )
        problems.extend(span_problems)

        if not keep_overlaps:
            spans, collapsed = merge_overlapping(answer, spans)
            merged_away += collapsed
        else:
            for a, b in zip(spans, spans[1:]):
                if b["start"] < a["end"]:
                    problems.append(
                        f"{rec.get('id')}: spans [{a['start']}:{a['end']}] and "
                        f"[{b['start']}:{b['end']}] overlap"
                    )

        records.append(
            {
                "id": str(rec.get("id")),
                "source_id": source_id,
                "model": str(rec.get("model", "")),
                "task_type": task.value,
                "source": source_name,
                "split": str(rec.get("split", "")),
                "quality": str(rec.get("quality", "")),
                "question": question,
                "context": context,
                "answer": answer,
                "spans": spans,
            }
        )

    assert_label_coverage(raw_label_types)
    return records, problems, raw_counter, merged_away


def _stats_bucket(task: TaskType, source: str) -> str:
    """Bucket a record the way the RAGTruth README's statistics table does.

    The README splits summarization into CNN/DM and recent news, so reproducing
    its numbers requires the same split.
    """
    if task is TaskType.SUMMARIZATION:
        return "summarization_cnndm" if "cnn" in source.lower() else "summarization_recent"
    if task is TaskType.QA:
        return "qa"
    return "data2text"


# Published in the RAGTruth README, "Data Statistics" section, read 2026-08-11.
# These are the authors' numbers, not ours. If our join reproduces them exactly,
# the join and the label parsing are correct.
PUBLISHED = {
    "summarization_cnndm": (628, 3768, 1165, 1474),
    "summarization_recent": (315, 1890, 521, 598),
    "qa": (989, 5934, 1724, 2927),
    "data2text": (1033, 6198, 4254, 9290),
}


def print_stats(counter: Counter) -> bool:
    """Print ours vs published side by side. Returns True if everything matches."""
    cols = ("instances", "responses", "hallucinated_responses", "spans")
    header = f"{'bucket':<24}" + "".join(f"{c:>26}" for c in cols)
    print(header)
    print("-" * len(header))

    totals = defaultdict(int)
    published_totals = [0, 0, 0, 0]
    all_match = True

    for bucket, expected in PUBLISHED.items():
        cells = []
        for i, col in enumerate(cols):
            got = counter[(bucket, col)]
            want = expected[i]
            totals[col] += got
            published_totals[i] += want
            mark = "" if got == want else "  MISMATCH"
            all_match &= got == want
            cells.append(f"{got:>10,} / {want:<10,}{mark:<5}")
        print(f"{bucket:<24}" + "".join(f"{c:>26}" for c in cells))

    print("-" * len(header))
    overall = []
    for i, col in enumerate(cols):
        overall.append(f"{totals[col]:>10,} / {published_totals[i]:<10,}     ")
    print(f"{'overall':<24}" + "".join(f"{c:>26}" for c in overall))

    if all_match:
        print("\nEvery bucket matches the published statistics table.")
    else:
        print(
            "\nAt least one bucket does not match the published table. The join or "
            "the label parsing is wrong. Do not train until this is resolved."
        )
    return all_match


def write_splits(records: List[Dict[str, Any]], out_dir: Path) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_split[rec["split"] or "unknown"].append(rec)

    written: Dict[str, int] = {}
    for split, rows in sorted(by_split.items()):
        path = out_dir / f"ragtruth_{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[split] = len(rows)
        print(f"wrote {len(rows):,} records to {path}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "processed")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="reproduce the published statistics table and exit without writing",
    )
    parser.add_argument(
        "--drop-implicit-true",
        action="store_true",
        help=(
            "drop spans flagged implicit_true (factually correct but absent from "
            "the context). Default is to keep them, because a span unsupported by "
            "the context is exactly what this detector is supposed to flag."
        ),
    )
    parser.add_argument(
        "--quality",
        default=None,
        help=(
            "keep only responses with this quality value, e.g. 'good'. Default "
            "keeps everything; the field is preserved on each record so it can be "
            "filtered later without re-running."
        ),
    )
    parser.add_argument(
        "--keep-overlaps",
        action="store_true",
        help=(
            "leave overlapping labels as-is and report them instead of merging. "
            "Default merges, because the BIO tagger unions overlaps anyway and "
            "overlapping ground truth caps span-level recall below 1.0."
        ),
    )
    parser.add_argument(
        "--max-problems", type=int, default=20, help="how many problems to print"
    )
    args = parser.parse_args()

    records, problems, raw_counter, merged_away = build(
        args.raw_dir,
        drop_implicit_true=args.drop_implicit_true,
        quality_filter=args.quality,
        keep_overlaps=args.keep_overlaps,
    )

    print("\nReproduction check against the published RAGTruth statistics")
    print("(computed before any filtering, so flags do not affect it)\n")
    matched = print_stats(raw_counter)

    if merged_away:
        kept = sum(len(r["spans"]) for r in records)
        print(
            f"\n{merged_away:,} overlapping labels merged into a neighbour "
            f"({kept:,} non-overlapping spans remain in the written records). "
            "Run with --keep-overlaps to see them listed individually instead."
        )

    if problems:
        print(f"\n{len(problems):,} problems found. First {args.max_problems}:")
        for p in problems[: args.max_problems]:
            print(f"  {p}")
    else:
        print("\nNo offset or join problems: every published label slices out its own text.")

    if args.stats_only:
        return 0 if matched and not problems else 1

    print()
    write_splits(records, args.out_dir)
    print("\nNext: python -m src.c1_detector.inspect_examples --n 10")
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
