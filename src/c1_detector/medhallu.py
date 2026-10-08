"""MedHallu -> processed-record loader for the medical example-level evaluation.

MedHallu (Pandit et al., arXiv:2502.14302, MIT licence) is a medical
hallucination benchmark built on PubMedQA. Each row carries a question, a
knowledge passage (the PubMedQA evidence), a ground-truth answer, and a
model-written hallucinated answer tagged with a hallucination category. We never
train on it; it is the medical example-level testbed, the counterpart of
RAGBench (see ragbench.py) for the medical domain.

What this corpus is, and what may honestly be reported from it:

1.  IT IS ANSWER-LEVEL, NOT SPAN-LEVEL. MedHallu labels a whole answer as
    hallucinated or faithful; there are no gold character spans. So the only
    reportable detection metric here is EXAMPLE-LEVEL F1 (does the detector flag
    this answer at all). Do not put a span or token number from this file in a
    table -- there is no gold span to score against.

2.  THE LABEL IS EXPLICIT, NOT SPAN-DERIVED. Every row yields two evaluation
    records with a known example-level label: the Hallucinated Answer (positive)
    and the Ground Truth answer (negative). That is why each record carries a
    `gold_positive` field and evaluate_medhallu reads THAT, not the presence of
    spans the way evaluate_c1.example_metrics does -- a positive MedHallu record
    has no spans, so the span-presence rule would silently call every positive a
    negative.

3.  THE HALLUCINATIONS ARE MODEL-GENERATED. The hallucinated answers were
    produced and filtered by an automated pipeline over PubMedQA; the questions
    and passages are human. A gap measured here is against that construction, not
    against human span annotation. Say so wherever the number appears.

4.  BASE RATE IS 0.5 BY CONSTRUCTION. One positive and one negative per row, so
    the positive rate is 0.5 and the trivial "flag everything" F1 is 0.667. That
    floor is printed beside the F1.

Context formatting mirrors RAGTruth QA -- `passage 1:...`, blank line,
`passage 2:...` -- the exact shape C1 trained on, so the number measures domain
shift and not a change of input layout. Same choice ragbench.py makes.

Usage:

    python -m src.c1_detector.medhallu --out-dir data/processed/medhallu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ID = "UTAustin-AIHealth/MedHallu"

# pqa_labeled = 1,000 rows built on the human-curated PubMedQA subset;
# pqa_artificial = 9,000 rows built on the generated subset. Verified from the
# dataset's HF config list on 2026-10-08.
CONFIGS: Tuple[str, ...] = ("pqa_labeled", "pqa_artificial")

# Exact MedHallu column names, verified from the HF datasets-server schema.
COL_QUESTION = "Question"
COL_KNOWLEDGE = "Knowledge"
COL_TRUTH = "Ground Truth"
COL_HALLUCINATED = "Hallucinated Answer"
COL_CATEGORY = "Category of Hallucination"
COL_DIFFICULTY = "Difficulty Level"


def build_context(knowledge: Any) -> str:
    """Join MedHallu's Knowledge passages the way RAGTruth QA formats passages.

    `Knowledge` is a list of strings; a bare string is tolerated. The layout is
    identical to ragbench.build_context so the two out-of-distribution
    evaluations feed the detector the same shape.
    """
    if isinstance(knowledge, str):
        passages = [knowledge]
    else:
        passages = [str(p) for p in (knowledge or [])]
    blocks = [f"passage {i}:{p}" for i, p in enumerate(passages, start=1)]
    return "\n\n".join(blocks)


def _record(
    row_id: str,
    kind: str,
    gold_positive: bool,
    question: str,
    context: str,
    answer: str,
    category: str,
    difficulty: str,
    config: str,
) -> Dict[str, Any]:
    """One evaluation record in build_examples' output shape (spans always empty)."""
    return {
        "id": f"medhallu:{config}:{row_id}:{kind}",
        "source_id": row_id,
        "model": "",
        # Every MedHallu item is a question answered over a passage: structurally
        # RAGTruth's QA task, the same mapping ragbench.py uses.
        "task_type": "qa",
        "source": f"medhallu/{config}",
        "split": "train",
        "quality": difficulty,
        "annotator": "",
        # The example-level gold label. evaluate_medhallu reads this, never the
        # (always empty) span list.
        "gold_positive": gold_positive,
        "category": category if gold_positive else "",
        "difficulty": difficulty,
        "question": question,
        "context": context,
        "answer": answer,
        "spans": [],
    }


def build_records(config: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load one MedHallu config and expand each row into a positive + a negative.

    Uses datasets.load_dataset: MedHallu ships plain parquet with no loading
    script, so there is nothing to go stale, and `datasets` is already a project
    dependency.
    """
    from datasets import load_dataset

    data = load_dataset(REPO_ID, config, split="train")
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(data):
        if limit is not None and idx >= limit:
            break
        question = str(row.get(COL_QUESTION) or "")
        context = build_context(row.get(COL_KNOWLEDGE))
        category = str(row.get(COL_CATEGORY) or "")
        difficulty = str(row.get(COL_DIFFICULTY) or "")
        row_id = str(row.get("id", idx))

        truth = str(row.get(COL_TRUTH) or "").strip()
        hallucinated = str(row.get(COL_HALLUCINATED) or "").strip()

        # Skip, never fabricate: a row missing an answer on one side still
        # contributes its other side rather than being dropped whole.
        if hallucinated:
            records.append(
                _record(row_id, "hall", True, question, context, hallucinated,
                        category, difficulty, config)
            )
        if truth:
            records.append(
                _record(row_id, "true", False, question, context, truth,
                        category, difficulty, config)
            )
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    positives = sum(1 for r in records if r["gold_positive"])
    return {
        "n": len(records),
        "positive": positives,
        "negative": len(records) - positives,
        "rows": len(records) // 2 if records else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download MedHallu and convert it to the processed record format for "
            "the medical example-level evaluation. Never used for training."
        )
    )
    parser.add_argument("--out-dir", default="data/processed/medhallu")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument(
        "--limit", type=int, default=None, help="debug only: first N rows per config"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest: Dict[str, Any] = {"repo_id": REPO_ID, "configs": {}}

    for config in args.configs:
        if config not in CONFIGS:
            print(f"unknown config {config!r}; known: {', '.join(CONFIGS)}")
            return 2
        records = build_records(config, limit=args.limit)
        written = write_jsonl(records, out_dir / f"{config}.jsonl")
        stats = _stats(records)
        manifest["configs"][config] = stats
        print(
            f"{config:<16} wrote {written:,} records "
            f"({stats['rows']:,} rows -> {stats['positive']:,} pos / "
            f"{stats['negative']:,} neg)"
        )

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nwrote {manifest_path}")
    print(
        "MedHallu hallucinations are model-generated and answer-level. Only "
        "example-level F1 is reportable from this file; there are no gold spans."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
