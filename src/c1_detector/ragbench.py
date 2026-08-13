"""RAGBench -> processed-record loader for the out-of-distribution test (C1-A).

RAGBench (Friel, Belyi & Sanyal, arXiv:2407.11005, CC-BY-4.0) is the
cross-domain testbed. We NEVER train on it. The checkpoint trained on RAGTruth
is run over RAGBench's test splits and the drop is reported.

Four properties of this corpus decide what may honestly be reported from it,
and all four were measured on the real parquet files, not assumed:

1.  THE LABELS ARE MODEL-GENERATED. `annotating_model_name` is
    "gpt-4-turbo-2024-04-09" on 10,570 of the 11,802 test records and "gpt-4o"
    on the other 1,057 -- counted across all twelve subsets, not read off one
    of them. RAGTruth's spans were written by two human annotators with 91.8%
    response-level agreement. A gap measured here is a gap against an LLM
    judge, not against human ground truth, and every sentence in the report
    that uses these numbers has to say so. The two annotator models are also
    not evenly spread across subsets, so a per-subset difference could be an
    annotator difference; the CLI prints the breakdown.

2.  THE ANNOTATION IS SENTENCE-LEVEL. RAGBench marks whole response sentences
    as unsupported (`unsupported_response_sentence_keys`). RAGTruth's gold
    spans are short phrases -- median 35 characters, only 11.1% of them a full
    sentence. The two are not the same object, so span-exact and span-overlap
    F1 computed across them would compare a phrase detector against a sentence
    annotation and mean nothing. ONLY EXAMPLE-LEVEL F1 IS REPORTABLE HERE.
    The sentence offsets are still written onto each record, because that is
    what carries the example-level label through `evaluate_c1.example_metrics`,
    and because a later reader may want them -- but do not put a span number
    from this file in a table.

3.  THE BASE RATE IS DIFFERENT. 1,677 of 11,802 test responses are
    non-adherent, 14.2%, against RAGTruth's 43.1%; per subset it runs from
    3.5% (tatqa) to 53.2% (expertqa). Positive-class F1 falls when the
    positive class gets rarer even if the classifier is unchanged, so a raw
    F1 drop is NOT by itself evidence of domain transfer failure. Report the
    per-subset positive rate in the same table as the F1.

4.  THE CONTEXTS ARE LONGER. Median context is 25,881 characters on cuad and
    17,582 on techqa, against a `max_length` of 3,072 tokens. Those subsets
    truncate. `encode_example` truncates the first sequence only, so the
    answer survives and the labels stay valid, but the model sees less
    evidence than the annotator did. The evaluator reports truncation counts
    per subset; read them before reading the F1.

Label mapping, and why it is unambiguous. `adherence_score` is a boolean:
False means the response contains unsupported content. Measured over all
11,802 test records, `adherence_score is False` holds if and only if
`unsupported_response_sentence_keys` is non-empty -- 0 disagreements either
way. So the example-level gold label is exactly `not adherence_score`, and no
threshold or heuristic enters the mapping.

Two data defects found while measuring, both handled explicitly rather than
silently:

  - 305 unsupported sentence keys do not appear verbatim in
    `response_sentences`. 303 of them are a trailing-punctuation mismatch --
    the key is written "a." in one field and "a" in the other -- so keys are
    compared after stripping to lowercase alphanumerics. That normalisation
    was checked for collisions across all 11,802 records and merges two
    distinct sentence keys in zero of them. The 2 that remain are genuine
    corruption in the released file, one of them an unescaped JSON fragment
    that leaked into a key string. They are skipped and counted.
    NOTE: without the key normalisation this defect silently deleted 175
    hallucinated responses, 10.4% of the positive class, and moved the
    measured positive rate from 14.2% to 12.9%. Positive-class F1 depends on
    the positive rate, so that would have been a real error in the reported
    OOD number.
  - 6 records have a sentence that cannot be located in their own response
    even with whitespace-tolerant matching. Same treatment.

Sentence offsets need whitespace-tolerant matching. `response_sentences`
collapses the newlines in `response`, so a plain `str.find` locates only
1,346 of 2,450 pubmedqa records; matching each whitespace run as `\\s+`
locates 2,449. The regex runs against the ORIGINAL response, so the offsets it
returns are real offsets into `answer` and satisfy schema.py's
`answer[start:end] == text` check.

Context formatting deliberately mirrors RAGTruth QA: the documents are joined
as `passage 1:...`, blank line, `passage 2:...`. C1 trained on that exact
shape. Feeding RAGBench in some other layout would measure prompt-format
shift on top of domain shift and there would be no way to separate the two.

Usage:

    python -m src.c1_detector.ragbench --out-dir data/processed/ragbench
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ID = "galileo-ai/ragbench"

# The 12 sub-datasets, in the order the RAGBench paper lists them.
SUBSETS: Tuple[str, ...] = (
    "covidqa",
    "cuad",
    "delucionqa",
    "emanual",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
)

# Columns we actually use. Naming them keeps the parquet read small -- the full
# schema carries eleven float columns of other systems' scores (trulens, ragas,
# gpt3) that are irrelevant here.
COLUMNS: Tuple[str, ...] = (
    "id",
    "question",
    "documents",
    "response",
    "generation_model_name",
    "annotating_model_name",
    "response_sentences",
    "unsupported_response_sentence_keys",
    "adherence_score",
)


class BuildProblems:
    """Counters for everything that did not map cleanly. Printed, never hidden."""

    def __init__(self) -> None:
        self.missing_sentence_key = 0
        self.unlocatable_sentence = 0
        self.empty_response = 0
        self.null_adherence = 0
        self.positive_without_spans = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "missing_sentence_key": self.missing_sentence_key,
            "unlocatable_sentence": self.unlocatable_sentence,
            "empty_response": self.empty_response,
            "null_adherence": self.null_adherence,
            "positive_without_spans": self.positive_without_spans,
        }

    def total(self) -> int:
        return sum(self.as_dict().values())


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def test_split_path(subset: str, cache_dir: Optional[Path] = None) -> Path:
    """Fetch one subset's test parquet from the Hub and return the local path.

    Uses `huggingface_hub`, which is already a dependency, rather than
    `datasets.load_dataset`. One file, one call, cached on disk, and no loading
    script to go stale.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"{subset}/test-00000-of-00001.parquet",
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    return Path(path)


def read_test_split(subset: str, cache_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Download if needed and return the test rows as plain dicts."""
    import pyarrow.parquet as pq

    table = pq.read_table(test_split_path(subset, cache_dir), columns=list(COLUMNS))
    return table.to_pylist()


# --------------------------------------------------------------------------
# Sentence localisation
# --------------------------------------------------------------------------


def normalise_key(key: Any) -> str:
    """Reduce a sentence key to lowercase alphanumerics.

    RAGBench writes the same sentence key two ways: "a" in `response_sentences`
    and "a." in `unsupported_response_sentence_keys`. 303 of the 305 keys that
    fail an exact match across the whole test set are exactly this mismatch.
    Checked on all 11,802 records, this normalisation never merges two distinct
    keys within one record, so it cannot move a label onto the wrong sentence.
    """
    return re.sub(r"[^0-9a-z]", "", str(key).strip().lower())


def sentence_pattern(sentence: str) -> Optional[re.Pattern[str]]:
    """Compile a sentence into a regex that tolerates different whitespace.

    Every run of whitespace in the sentence becomes `\\s+`; everything else is
    escaped literally. Matching happens against the untouched response, so the
    span this yields is a real offset into the answer string.

    Returns None for a sentence that is empty once stripped -- there is nothing
    to locate and `re.search("")` would match at position 0 and produce a
    zero-width span that schema.py rejects.
    """
    stripped = sentence.strip()
    if not stripped:
        return None
    parts = re.split(r"\s+", stripped)
    return re.compile(r"\s+".join(re.escape(part) for part in parts))


def locate_sentences(
    response: str, sentences: Sequence[Sequence[str]]
) -> Tuple[Dict[str, Tuple[int, int]], List[str]]:
    """Find each (key, sentence) pair inside `response`, left to right.

    `sentences` is RAGBench's `response_sentences`: a list of [key, text] pairs
    in document order. Searching forward from the end of the previous match
    keeps the spans non-overlapping and in order, which is what the BIO tagger
    requires, and it also disambiguates a sentence that repeats verbatim.

    Returns (offsets by NORMALISED key, keys that could not be located). Callers
    must look up with `normalise_key`; see that function for why.
    """
    found: Dict[str, Tuple[int, int]] = {}
    missing: List[str] = []
    cursor = 0
    for pair in sentences:
        key, text = normalise_key(pair[0]), str(pair[1])
        pattern = sentence_pattern(text)
        if pattern is None:
            missing.append(key)
            continue
        match = pattern.search(response, cursor)
        if match is None:
            # Fall back to searching the whole response: a sentence list that is
            # out of order should still resolve rather than losing everything
            # after the first surprise.
            match = pattern.search(response)
        if match is None:
            missing.append(key)
            continue
        found[key] = (match.start(), match.end())
        cursor = max(cursor, match.end())
    return found, missing


# --------------------------------------------------------------------------
# Record building
# --------------------------------------------------------------------------


def build_context(documents: Sequence[str]) -> str:
    """Join retrieved documents the way RAGTruth QA formats its passages.

    RAGTruth writes `passage 1:<text>` blocks separated by a blank line and C1
    trained on exactly that. Keeping the layout identical means the OOD number
    measures domain shift and not a change of input format.
    """
    blocks = [f"passage {i}:{doc}" for i, doc in enumerate(documents or [], start=1)]
    return "\n\n".join(blocks)


def build_record(row: Dict[str, Any], subset: str, problems: BuildProblems) -> Optional[Dict[str, Any]]:
    """One RAGBench row -> one record in build_examples' output format.

    Returns None for a row that cannot produce a valid record at all, which so
    far means only an empty response.
    """
    answer = row.get("response") or ""
    if not answer.strip():
        problems.empty_response += 1
        return None

    adherence = row.get("adherence_score")
    if adherence is None:
        problems.null_adherence += 1
        return None

    keys = [normalise_key(k) for k in (row.get("unsupported_response_sentence_keys") or [])]
    found, unlocatable = locate_sentences(answer, row.get("response_sentences") or [])
    problems.unlocatable_sentence += len(unlocatable)

    spans: List[Dict[str, Any]] = []
    for key in dict.fromkeys(keys):
        if key not in found:
            problems.missing_sentence_key += 1
            continue
        start, end = found[key]
        spans.append(
            {
                "start": start,
                "end": end,
                "text": answer[start:end],
                # RAGBench has no error taxonomy. schema.ErrorType.UNKNOWN is the
                # honest value; do not map its TRACe dimensions onto RAGTruth's
                # four categories, they are different frameworks.
                "error_type": "unknown",
                "implicit_true": False,
                "due_to_null": False,
            }
        )
    spans.sort(key=lambda s: (s["start"], s["end"]))

    # The example-level gold label comes from adherence_score, which is the
    # authoritative field. If the keys failed to resolve on a non-adherent
    # response the record would silently become a negative, so it is dropped and
    # counted instead of quietly weakening the positive class.
    if adherence is False and not spans:
        problems.positive_without_spans += 1
        return None

    return {
        "id": f"{subset}:{row.get('id')}",
        "source_id": str(row.get("id")),
        "model": str(row.get("generation_model_name") or ""),
        # Every RAGBench record is a question answered over retrieved documents,
        # which is structurally RAGTruth's QA task. It is the closest of the four
        # TaskType values and it is the comparison the OOD table draws. The
        # per-subset breakdown is what carries the domain information.
        "task_type": "qa",
        "source": subset,
        "split": "test",
        "quality": "",
        "annotator": str(row.get("annotating_model_name") or ""),
        "adherence_score": bool(adherence),
        "question": str(row.get("question") or ""),
        "context": build_context(row.get("documents") or []),
        "answer": answer,
        "spans": spans,
    }


def build_subset(
    subset: str, cache_dir: Optional[Path] = None
) -> Tuple[List[Dict[str, Any]], BuildProblems]:
    problems = BuildProblems()
    rows = read_test_split(subset, cache_dir)
    records = [r for r in (build_record(row, subset, problems) for row in rows) if r]
    return records, problems


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def subset_stats(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    positive = sum(1 for r in records if r["spans"])
    return {
        "n": n,
        "positive": positive,
        "positive_rate": positive / n if n else 0.0,
        "n_spans": sum(len(r["spans"]) for r in records),
        "median_context_chars": _median([len(r["context"]) for r in records]),
        "median_answer_chars": _median([len(r["answer"]) for r in records]),
    }


def _median(values: Sequence[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def format_stats_table(stats: Dict[str, Dict[str, Any]]) -> str:
    header = (
        f"{'subset':<12}{'n':>7}{'positive':>10}{'rate':>8}"
        f"{'spans':>8}{'ctx chars':>11}{'ans chars':>11}"
    )
    lines = [header, "-" * len(header)]
    total_n = total_pos = total_spans = 0
    for name in sorted(stats):
        block = stats[name]
        total_n += block["n"]
        total_pos += block["positive"]
        total_spans += block["n_spans"]
        lines.append(
            f"{name:<12}{block['n']:>7,}{block['positive']:>10,}"
            f"{block['positive_rate']:>8.3f}{block['n_spans']:>8,}"
            f"{block['median_context_chars']:>11,}{block['median_answer_chars']:>11,}"
        )
    lines.append("-" * len(header))
    rate = total_pos / total_n if total_n else 0.0
    lines.append(
        f"{'overall':<12}{total_n:>7,}{total_pos:>10,}{rate:>8.3f}{total_spans:>8,}"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download RAGBench test splits and convert them to the processed "
            "record format for the out-of-distribution evaluation. Never used "
            "for training."
        )
    )
    parser.add_argument("--out-dir", default="data/processed/ragbench")
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=list(SUBSETS),
        help="subset names; default is all 12",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="huggingface_hub cache directory; omit for the default",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    stats: Dict[str, Dict[str, Any]] = {}
    all_problems: Dict[str, Dict[str, int]] = {}
    annotators: Dict[str, int] = {}

    for subset in args.subsets:
        if subset not in SUBSETS:
            print(f"unknown subset {subset!r}; known: {', '.join(SUBSETS)}")
            return 2
        records, problems = build_subset(subset, cache_dir)
        written = write_jsonl(records, out_dir / f"{subset}.jsonl")
        stats[subset] = subset_stats(records)
        all_problems[subset] = problems.as_dict()
        for record in records:
            annotators[record["annotator"]] = annotators.get(record["annotator"], 0) + 1
        print(f"{subset:<12} wrote {written:,} records")

    print()
    print(format_stats_table(stats))

    print()
    print("annotating models:", ", ".join(f"{k} {v:,}" for k, v in sorted(annotators.items())))
    print(
        "These labels are model-generated. Any number derived from this file is "
        "measured against an LLM judge, not human annotation, and the judge is "
        "not the same model on every subset."
    )

    dropped = {k: v for k, v in all_problems.items() if sum(v.values())}
    if dropped:
        print()
        print("records that did not map cleanly (skipped and counted, never guessed):")
        for subset, counts in sorted(dropped.items()):
            detail = ", ".join(f"{k} {v}" for k, v in counts.items() if v)
            print(f"  {subset:<12} {detail}")

    manifest = out_dir / "manifest.json"
    with manifest.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "repo_id": REPO_ID,
                "split": "test",
                "stats": stats,
                "problems": all_problems,
                "annotators": annotators,
            },
            handle,
            indent=2,
        )
    print(f"\nwrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
