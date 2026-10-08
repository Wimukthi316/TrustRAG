"""Synthesise a medical SPAN-labelled hallucination set from MedHallu (phase 3).

No large human span-annotated medical corpus exists, so the span-level
domain-adaptation lever needs data we build ourselves. This takes MedHallu's
clean "Ground Truth" answers (the PubMedQA long answers, grounded by the
"Knowledge" passage) and injects ONE error per positive example with the exact
character offset recorded, so the result is a span-labelled set in the same
processed-record shape train_c1 consumes.

HONESTY, say this wherever a number from here appears: the spans are SYNTHETIC.
Errors are inserted by rule, the labels are correct by construction, and that is
both the strength (offsets are exact, no annotation noise) and the weakness (the
distribution is not natural hallucination). It is training/ablation data, not a
gold benchmark; MedHallu (answer-level) and, if obtained, MedHalu/MedHal-Loc are
the real medical evals.

Three injectors, tried in this order so most positives are in-place conflicts
rather than canned insertions:

  1. number        - change a measurement/dosage/percentage/count to a clearly
                     different value. evident_conflict.
  2. directionality- flip a direction/polarity word to its antonym
                     (increased<->decreased, significant<->insignificant, ...).
                     evident_conflict.
  3. fabricated    - insert one unsupported clause after the first sentence, only
                     when neither in-place edit applies. evident_baseless_info.
                     Kept last and few because repeated canned clauses are an
                     easy, unrealistic signal.

For every row: one NEGATIVE record (the clean Ground Truth, no spans) and, when
an injector applies, one POSITIVE record (the edited answer with its one span).
Offsets are asserted to round-trip (answer[start:end] == span text) before a
record is emitted; a row whose answer admits no injector is emitted as a negative
only, and counted.

Usage:
    python -m src.c1_detector.synth_medical_spans --out-dir data/processed/synth_medical
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ID = "UTAustin-AIHealth/MedHallu"
CONFIGS = ("pqa_labeled", "pqa_artificial")

# A whole word -> its opposite. Capitalisation of the original is preserved.
ANTONYMS = {
    "increased": "decreased", "decreased": "increased",
    "increase": "decrease", "decrease": "increase",
    "increases": "decreases", "decreases": "increases",
    "higher": "lower", "lower": "higher",
    "elevated": "reduced", "reduced": "elevated",
    "greater": "lesser", "lesser": "greater",
    "significant": "insignificant", "significantly": "insignificantly",
    "effective": "ineffective", "beneficial": "harmful",
    "positive": "negative", "negative": "positive",
    "improved": "worsened", "improves": "worsens",
    "associated": "unassociated", "present": "absent",
    "activated": "inhibited", "activates": "inhibits",
    "promotes": "suppresses", "induced": "prevented",
}

# Inserted only as a fallback. Unsupported relative to any specific abstract.
FABRICATED = (
    "This effect was mediated primarily through calcium channel modulation.",
    "The association remained significant after adjusting for every comorbidity.",
    "These findings were confirmed in a subsequent randomized controlled trial.",
    "The underlying mechanism involves direct repair of mitochondrial DNA.",
    "This result was consistent across all age groups and both sexes.",
)

Injection = Tuple[str, int, int, str, str]  # new_answer, start, end, span_text, error_type


def _inject_number(text: str, rng: random.Random) -> Optional[Injection]:
    # Standalone integers/decimals, optional trailing %, not part of a longer
    # token (so not the "19" in COVID-19 or a word-internal digit).
    matches = [m for m in re.finditer(r"(?<![\w.-])\d{1,4}(?:\.\d+)?%?", text)]
    if not matches:
        return None
    m = rng.choice(matches)
    orig = m.group()
    pct = orig.endswith("%")
    core = orig[:-1] if pct else orig
    if "." in core:
        new_core = f"{float(core) * 2 + 1.5:.1f}"
    else:
        new_core = str(int(core) * 3 + 7)
    if new_core == core:
        new_core = str(int(float(core)) + 13)
    new = new_core + ("%" if pct else "")
    start = m.start()
    return text[:start] + new + text[m.end():], start, start + len(new), new, "evident_conflict"


def _inject_direction(text: str, rng: random.Random) -> Optional[Injection]:
    candidates = [
        (m, m.group().lower())
        for m in re.finditer(r"[A-Za-z]+", text)
        if m.group().lower() in ANTONYMS
    ]
    if not candidates:
        return None
    m, word = rng.choice(candidates)
    repl = ANTONYMS[word]
    if m.group()[0].isupper():
        repl = repl.capitalize()
    start = m.start()
    return text[:start] + repl + text[m.end():], start, start + len(repl), repl, "evident_conflict"


def _inject_fabricated(text: str, rng: random.Random) -> Optional[Injection]:
    boundary = re.search(r"[.!?]\s", text)
    pos = boundary.end() if boundary else len(text)
    clause = rng.choice(FABRICATED)
    insert = (clause + " ") if (boundary and pos < len(text)) else (" " + clause)
    if boundary and pos < len(text):
        new_text = text[:pos] + insert + text[pos:]
        start, span_text = pos, clause
    else:
        new_text = text[:pos] + insert
        start, span_text = pos + 1, clause
    return new_text, start, start + len(span_text), span_text, "evident_baseless_info"


INJECTORS = (_inject_number, _inject_direction, _inject_fabricated)


def inject_one(text: str, rng: random.Random) -> Optional[Injection]:
    """First applicable injector, in the order defined above."""
    for injector in INJECTORS:
        result = injector(text, rng)
        if result is not None:
            new_text, start, end, span_text, _etype = result
            # The guarantee the whole file rests on. Never emit a span that does
            # not slice back to its own text.
            if new_text[start:end] != span_text or end <= start:
                continue
            return result
    return None


def build_context(knowledge: Any) -> str:
    passages = [knowledge] if isinstance(knowledge, str) else [str(p) for p in (knowledge or [])]
    return "\n\n".join(f"passage {i}:{p}" for i, p in enumerate(passages, start=1))


def _record(row_id: str, kind: str, question: str, context: str, answer: str,
            spans: List[Dict[str, Any]], difficulty: str, config: str) -> Dict[str, Any]:
    return {
        "id": f"synthmed:{config}:{row_id}:{kind}",
        "source_id": row_id,
        "model": "",
        "task_type": "qa",
        "source": f"synth-medical/{config}",
        "split": "train",
        "quality": difficulty,
        "annotator": "synthetic",
        "question": question,
        "context": context,
        "answer": answer,
        "spans": spans,
    }


def build_records(config: str, seed: int, limit: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    from datasets import load_dataset

    data = load_dataset(REPO_ID, config, split="train")
    rng = random.Random(seed)
    records: List[Dict[str, Any]] = []
    stats = {"rows": 0, "positive": 0, "negative": 0, "no_injector": 0,
             "evident_conflict": 0, "evident_baseless_info": 0}

    for idx, row in enumerate(data):
        if limit is not None and idx >= limit:
            break
        answer = str(row.get("Ground Truth") or "").strip()
        if not answer:
            continue
        stats["rows"] += 1
        question = str(row.get("Question") or "")
        context = build_context(row.get("Knowledge"))
        difficulty = str(row.get("Difficulty Level") or "")
        row_id = str(row.get("id", idx))

        # Clean negative.
        records.append(_record(row_id, "clean", question, context, answer, [], difficulty, config))
        stats["negative"] += 1

        injected = inject_one(answer, rng)
        if injected is None:
            stats["no_injector"] += 1
            continue
        new_answer, start, end, span_text, etype = injected
        span = {
            "start": start, "end": end, "text": span_text,
            "error_type": etype, "implicit_true": False, "due_to_null": False,
        }
        records.append(_record(row_id, "hall", question, context, new_answer, [span], difficulty, config))
        stats["positive"] += 1
        stats[etype] += 1

    return records, stats


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthesise a medical span-labelled hallucination set from MedHallu."
    )
    parser.add_argument("--out-dir", default="data/processed/synth_medical")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None, help="debug: first N rows per config")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest: Dict[str, Any] = {"source": REPO_ID, "seed": args.seed, "configs": {}}

    for config in args.configs:
        if config not in CONFIGS:
            print(f"unknown config {config!r}; known: {', '.join(CONFIGS)}")
            return 2
        records, stats = build_records(config, args.seed, args.limit)
        # Every emitted span must round-trip. Checked again here across the whole
        # set, not just per injection, so a later edit to the record shape cannot
        # silently break the contract.
        bad = 0
        for r in records:
            for s in r["spans"]:
                if r["answer"][s["start"]:s["end"]] != s["text"]:
                    bad += 1
        assert bad == 0, f"{bad} spans did not round-trip in {config}"
        written = write_jsonl(records, out_dir / f"{config}.jsonl")
        manifest["configs"][config] = stats
        print(f"{config:<16} wrote {written:,} records  "
              f"({stats['positive']:,} pos / {stats['negative']:,} neg; "
              f"conflict {stats['evident_conflict']:,}, baseless {stats['evident_baseless_info']:,}, "
              f"no-injector {stats['no_injector']:,})")

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nwrote {out_dir / 'manifest.json'}")
    print("Spans are SYNTHETIC (rule-injected). Training/ablation data, not a gold "
          "benchmark. Say so wherever a number derived from this set is reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
