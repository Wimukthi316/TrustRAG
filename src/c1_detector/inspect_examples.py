"""Print processed RAGTruth examples so the span offsets can be checked by hand.

Do this before any training run. A silent off-by-one in the offset handling
produces a model that trains happily, reports a plausible F1, and is measuring
the wrong thing. Ten examples read carefully costs ten minutes and is the only
check that actually catches it.

What to look for, in order:
  1. The bracketed text reads like a hallucination given the context above it.
  2. The brackets sit on word boundaries, not one character early or late.
  3. Data-to-text examples, which carry roughly two thirds of all spans in the
     corpus, look right -- they are the ones most likely to expose a bug.
  4. With --tokenizer, the BIO round trip recovers the same spans.

Usage:
    python -m src.c1_detector.inspect_examples --n 10
    python -m src.c1_detector.inspect_examples --task data2text --n 5
    python -m src.c1_detector.inspect_examples --id 1472
    python -m src.c1_detector.inspect_examples --n 10 --tokenizer answerdotai/ModernBERT-base
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c1_detector.bio import (  # noqa: E402
    ANSWER_SEQUENCE_ID,
    LABEL_NAMES,
    assert_round_trip,
    encode_example,
)

RULE = "=" * 78
CONTEXT_PREVIEW = 700


def load(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run: python -m src.c1_detector.build_examples"
        )
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def bracket(answer: str, spans: List[Dict[str, Any]]) -> str:
    """Rewrite the answer with each hallucinated span wrapped in >>> <<<.

    Built by walking backwards so that inserting markers never shifts an offset
    that has not been used yet. Doing it forwards is the classic way to
    introduce the exact bug this tool exists to find.
    """
    out = answer
    for span in sorted(spans, key=lambda s: s["start"], reverse=True):
        s, e = span["start"], span["end"]
        out = f"{out[:s]}>>>{out[s:e]}<<<{out[e:]}"
    return out


def show(rec: Dict[str, Any], index: int, total: int) -> None:
    print(f"\n{RULE}")
    print(
        f"[{index}/{total}]  id={rec['id']}  source_id={rec['source_id']}  "
        f"task={rec['task_type']}  model={rec['model']}  split={rec['split']}  "
        f"quality={rec['quality']}"
    )
    print(RULE)

    context = rec["context"]
    print(f"\nQUESTION\n  {rec['question']}")
    print(f"\nCONTEXT ({len(context):,} chars)")
    preview = context[:CONTEXT_PREVIEW]
    print("  " + preview.replace("\n", "\n  "))
    if len(context) > CONTEXT_PREVIEW:
        print(f"  ... [{len(context) - CONTEXT_PREVIEW:,} more chars]")

    print(f"\nANSWER ({len(rec['answer']):,} chars), hallucinated spans in >>> <<<")
    print("  " + bracket(rec["answer"], rec["spans"]).replace("\n", "\n  "))

    if not rec["spans"]:
        print("\nSPANS: none -- this response was annotated as fully supported")
        return

    print(f"\nSPANS ({len(rec['spans'])})")
    for span in rec["spans"]:
        flags = []
        if span["implicit_true"]:
            flags.append("implicit_true")
        if span["due_to_null"]:
            flags.append("due_to_null")
        suffix = f"  [{', '.join(flags)}]" if flags else ""
        print(
            f"  [{span['start']:>5}:{span['end']:<5}] {span['error_type']:<22} "
            f"{span['text']!r}{suffix}"
        )
        # Independent re-check: slice the answer again, right here, rather than
        # trusting the text field that build_examples already compared against.
        sliced = rec["answer"][span["start"] : span["end"]]
        if sliced != span["text"]:
            print(f"    OFFSET MISMATCH: answer slices out {sliced!r}")


def show_tokenisation(rec: Dict[str, Any], tokenizer: Any, max_length: int) -> None:
    enc = encode_example(tokenizer, rec, max_length=max_length)
    labels = enc["labels"]
    offsets = enc["offset_mapping"]
    seq_ids = enc["sequence_ids"]

    n_tagged = sum(1 for label in labels if label in (1, 2))
    print(
        f"\nTOKENISATION  {len(labels):,} tokens total, "
        f"{enc['n_answer_tokens']:,} in the answer, {n_tagged:,} tagged hallucinated"
    )
    if enc["answer_truncated"]:
        print(
            "  WARNING: the answer itself was truncated. Labels past the cut are "
            "lost. Raise --max-length."
        )

    if rec["spans"]:
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        print("  answer tokens carrying a hallucination label:")
        for token, label, off, sid in zip(tokens, labels, offsets, seq_ids):
            if sid == ANSWER_SEQUENCE_ID and label in (1, 2):
                print(
                    f"    {LABEL_NAMES[label]:<6} {token!r:<20} "
                    f"chars [{off[0]}:{off[1]}] = {rec['answer'][off[0]:off[1]]!r}"
                )

        try:
            assert_round_trip(
                [(s["start"], s["end"]) for s in rec["spans"]],
                labels,
                offsets,
                seq_ids,
                rec["answer"],
                tolerance=0,
            )
            print("  round trip: exact")
        except AssertionError as exc:
            # Widening by a character or two is normal when a published span
            # starts mid-token. Anything else is a bug.
            print(f"  round trip: NOT exact -- {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "ragtruth_train.jsonl",
    )
    parser.add_argument("--n", type=int, default=10, help="how many examples to print")
    parser.add_argument("--task", default=None, help="qa, data2text or summarization")
    parser.add_argument("--id", default=None, help="print one specific response id")
    parser.add_argument(
        "--with-spans",
        action="store_true",
        help="only show responses that actually have hallucination spans",
    )
    parser.add_argument("--seed", type=int, default=0, help="sampling seed")
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="model id to tokenize with, e.g. answerdotai/ModernBERT-base",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    records = load(args.file)

    if args.id is not None:
        records = [r for r in records if r["id"] == args.id]
        if not records:
            raise SystemExit(f"no record with id {args.id} in {args.file}")
    else:
        if args.task:
            records = [r for r in records if r["task_type"] == args.task]
        if args.with_spans:
            records = [r for r in records if r["spans"]]
        if not records:
            raise SystemExit("no records match those filters")
        random.Random(args.seed).shuffle(records)
        records = records[: args.n]

    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer  # imported lazily; heavy

        print(f"loading tokenizer {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    for i, rec in enumerate(records, 1):
        show(rec, i, len(records))
        if tokenizer is not None:
            show_tokenisation(rec, tokenizer, args.max_length)

    print(f"\n{RULE}")
    print(
        f"{len(records)} example(s) shown. Read them. Confirm the brackets land on\n"
        "word boundaries and the bracketed text is genuinely unsupported by the\n"
        "context above it before starting a training run."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
