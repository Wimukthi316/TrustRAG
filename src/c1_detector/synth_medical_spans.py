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

Four injectors, tried in this order so most positives are grounded in-place
conflicts rather than canned insertions:

  1. number        - change a measurement/dosage/percentage/count to a clearly
                     different value. evident_conflict.
  2. directionality- flip a direction/polarity word to its antonym
                     (increased<->decreased, significant<->insignificant, ...).
                     evident_conflict.
  3. entity        - replace an entity in the answer with a DIFFERENT entity that
                     appears in the answer's own context passage, so the result
                     is a grounded wrong-entity contradiction rather than a
                     generic one. evident_conflict.
  4. fabricated    - insert one unsupported clause after the first sentence, only
                     when none of the in-place edits apply. evident_baseless_info.
                     Last and few because repeated canned clauses are an easy,
                     unrealistic signal; the first three keep it a fallback.

For every row: one NEGATIVE record (the clean Ground Truth, no spans) and, when
an injector applies, one POSITIVE record (the edited answer with its one span).
Offsets are asserted to round-trip (answer[start:end] == span text) before a
record is emitted.

LEAKAGE CONTROL (phase 3b). This set is TRAINING data; the medical example-level
eval is MedHallu pqa_labeled. To keep them disjoint:
  * `--held-out-answers-config CONF` drops any row whose clean answer text also
    appears as a Ground Truth in CONF, so no answer trained on reappears in the
    eval even if the two configs overlap.
  * `--test-fraction` carves a deterministic, row-hashed slice into a separate
    `{config}.test.jsonl` so there is an IN-DOMAIN synthetic span test the model
    never trained on. The hash is over (split_seed, config, row_id), so the split
    is stable across machines and independent of row order.
Both default to off, so the old single-file behaviour is unchanged.

Usage:
    python -m src.c1_detector.synth_medical_spans --out-dir data/processed/synth_medical
    python -m src.c1_detector.synth_medical_spans \\
        --configs pqa_artificial --held-out-answers-config pqa_labeled \\
        --test-fraction 0.12 --out-dir data/processed/synth_medical
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ID = "UTAustin-AIHealth/MedHallu"
CONFIGS = ("pqa_labeled", "pqa_artificial")

# A whole word -> its opposite. Capitalisation of the original is preserved.
ANTONYMS = {
    "increased": "decreased", "decreased": "increased",
    "increase": "decrease", "decrease": "increase",
    "increases": "decreases", "decreases": "increases",
    "increasing": "decreasing", "decreasing": "increasing",
    "higher": "lower", "lower": "higher", "high": "low", "low": "high",
    "elevated": "reduced", "reduced": "elevated",
    "greater": "lesser", "lesser": "greater",
    "significant": "insignificant", "significantly": "insignificantly",
    "effective": "ineffective", "beneficial": "harmful", "harmful": "beneficial",
    "positive": "negative", "negative": "positive",
    "improved": "worsened", "improves": "worsens", "worsened": "improved",
    "associated": "unassociated", "present": "absent", "absent": "present",
    "activated": "inhibited", "activates": "inhibits",
    "promotes": "suppresses", "suppresses": "promotes",
    "induced": "prevented", "prevented": "induced",
    "upregulated": "downregulated", "downregulated": "upregulated",
    "overexpressed": "underexpressed",
    "acute": "chronic", "chronic": "acute",
    "benign": "malignant", "malignant": "benign",
    "stable": "unstable", "unstable": "stable",
    "normal": "abnormal", "abnormal": "normal",
    "safe": "unsafe", "unsafe": "safe",
    "sensitive": "resistant", "resistant": "sensitive",
    "responsive": "unresponsive", "susceptible": "resistant",
    "soluble": "insoluble", "early": "late", "late": "early",
    "better": "worse", "worse": "better",
    "more": "less", "less": "more", "faster": "slower", "slower": "faster",
    "rapid": "slow", "correlated": "uncorrelated", "related": "unrelated",
}

# Sentence-initial / structural capitals that are not entities. Excluded from the
# entity injector so it swaps drug/gene/disease-like terms, not "Results".
ENTITY_STOP = {
    "The", "This", "These", "That", "Those", "It", "In", "On", "At", "As", "An",
    "A", "We", "Our", "Their", "His", "Her", "Its", "Results", "Result",
    "Conclusion", "Conclusions", "Background", "Methods", "Method", "Objective",
    "Objectives", "Patients", "Patient", "However", "Although", "There", "While",
    "When", "Thus", "Therefore", "Here", "Both", "All", "No", "Not", "After",
    "Before", "During", "First", "Second", "Study", "Studies", "Group", "Groups",
}

ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z]{2,}|[A-Z]{2,})\b")

# Inserted only as a fallback. Unsupported relative to any specific abstract.
FABRICATED = (
    "This effect was mediated primarily through calcium channel modulation.",
    "The association remained significant after adjusting for every comorbidity.",
    "These findings were confirmed in a subsequent randomized controlled trial.",
    "The underlying mechanism involves direct repair of mitochondrial DNA.",
    "This result was consistent across all age groups and both sexes.",
)

Injection = Tuple[str, int, int, str, str]  # new_answer, start, end, span_text, error_type


def _inject_number(text: str, context: str, rng: random.Random) -> Optional[Injection]:
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


def _inject_direction(text: str, context: str, rng: random.Random) -> Optional[Injection]:
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


def _entities(s: str) -> Set[str]:
    return {e for e in ENTITY_RE.findall(s) if e not in ENTITY_STOP}


def _inject_entity(text: str, context: str, rng: random.Random) -> Optional[Injection]:
    """Swap an entity in the answer for a different one from its own context.

    Grounded by construction: both the removed and the inserted entity are terms
    the passage actually uses, so the edit is a wrong-entity contradiction rather
    than a free-floating one. Returns None when the answer shares fewer than the
    two distinct context entities it needs.
    """
    ctx_ents = _entities(context)
    if len(ctx_ents) < 2:
        return None
    hits = [m for m in ENTITY_RE.finditer(text) if m.group() in ctx_ents]
    if not hits:
        return None
    m = rng.choice(hits)
    orig = m.group()
    alts = sorted(e for e in ctx_ents if e.lower() != orig.lower())
    if not alts:
        return None
    repl = rng.choice(alts)
    start = m.start()
    return text[:start] + repl + text[m.end():], start, start + len(repl), repl, "evident_conflict"


def _inject_fabricated(text: str, context: str, rng: random.Random) -> Optional[Injection]:
    boundary = re.search(r"[.!?]\s", text)
    pos = boundary.end() if boundary else len(text)
    clause = rng.choice(FABRICATED)
    if boundary and pos < len(text):
        new_text = text[:pos] + clause + " " + text[pos:]
        start, span_text = pos, clause
    else:
        new_text = text[:pos] + " " + clause
        start, span_text = pos + 1, clause
    return new_text, start, start + len(span_text), span_text, "evident_baseless_info"


INJECTORS = (_inject_number, _inject_direction, _inject_entity, _inject_fabricated)


def inject_one(text: str, context: str, rng: random.Random) -> Optional[Injection]:
    """First applicable injector, in the order defined above."""
    for injector in INJECTORS:
        result = injector(text, context, rng)
        if result is None:
            continue
        new_text, start, end, span_text, _etype = result
        # The guarantee the whole file rests on. Never emit a span that does not
        # slice back to its own text.
        if end > start and new_text[start:end] == span_text:
            return result
    return None


def _norm(text: str) -> str:
    """Whitespace- and case-insensitive key for matching the same answer text."""
    return " ".join(text.lower().split())


def _row_bucket(config: str, row_id: str, split_seed: int) -> int:
    """Deterministic 0..999 bucket for a row, independent of its position.

    Keyed on (split_seed, config, row_id) so the train/test carve is identical on
    any machine and does not depend on dataset iteration order.
    """
    digest = hashlib.sha1(f"{split_seed}:{config}:{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 1000


def build_context(knowledge: Any) -> str:
    passages = [knowledge] if isinstance(knowledge, str) else [str(p) for p in (knowledge or [])]
    return "\n\n".join(f"passage {i}:{p}" for i, p in enumerate(passages, start=1))


def _record(row_id: str, kind: str, question: str, context: str, answer: str,
            spans: List[Dict[str, Any]], difficulty: str, config: str,
            split: str = "train") -> Dict[str, Any]:
    return {
        "id": f"synthmed:{config}:{row_id}:{kind}",
        "source_id": row_id,
        "model": "",
        "task_type": "qa",
        "source": f"synth-medical/{config}",
        "split": split,
        "quality": difficulty,
        "annotator": "synthetic",
        "question": question,
        "context": context,
        "answer": answer,
        "spans": spans,
    }


def _answers_of(config: str, limit: Optional[int] = None) -> Set[str]:
    """Normalised Ground Truth answer texts of a config, for leakage exclusion."""
    from datasets import load_dataset

    data = load_dataset(REPO_ID, config, split="train")
    out: Set[str] = set()
    for idx, row in enumerate(data):
        if limit is not None and idx >= limit:
            break
        answer = str(row.get("Ground Truth") or "").strip()
        if answer:
            out.add(_norm(answer))
    return out


def build_records(
    config: str,
    seed: int,
    limit: Optional[int] = None,
    exclude_answers: Optional[Set[str]] = None,
    test_fraction: float = 0.0,
    split_seed: int = 0,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    """Build (train_records, test_records, stats) for one config.

    A row goes to the test list when its hashed bucket falls under test_fraction;
    with test_fraction == 0.0 every row is a train row and test_records is empty.
    A row whose clean answer is in exclude_answers is dropped entirely (both its
    negative and its positive), and counted under stats["excluded"].
    """
    from datasets import load_dataset

    data = load_dataset(REPO_ID, config, split="train")
    rng = random.Random(seed)
    train: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    cutoff = int(round(max(0.0, min(1.0, test_fraction)) * 1000))
    stats = {"rows": 0, "positive": 0, "negative": 0, "no_injector": 0,
             "excluded": 0, "test_rows": 0,
             "evident_conflict": 0, "evident_baseless_info": 0}

    for idx, row in enumerate(data):
        if limit is not None and idx >= limit:
            break
        answer = str(row.get("Ground Truth") or "").strip()
        if not answer:
            continue
        if exclude_answers is not None and _norm(answer) in exclude_answers:
            stats["excluded"] += 1
            continue
        stats["rows"] += 1
        question = str(row.get("Question") or "")
        context = build_context(row.get("Knowledge"))
        difficulty = str(row.get("Difficulty Level") or "")
        row_id = str(row.get("id", idx))

        is_test = _row_bucket(config, row_id, split_seed) < cutoff
        bucket = test if is_test else train
        split = "test" if is_test else "train"
        if is_test:
            stats["test_rows"] += 1

        bucket.append(_record(row_id, "clean", question, context, answer, [], difficulty, config, split))
        stats["negative"] += 1

        injected = inject_one(answer, context, rng)
        if injected is None:
            stats["no_injector"] += 1
            continue
        new_answer, start, end, span_text, etype = injected
        span = {
            "start": start, "end": end, "text": span_text,
            "error_type": etype, "implicit_true": False, "due_to_null": False,
        }
        bucket.append(_record(row_id, "hall", question, context, new_answer, [span], difficulty, config, split))
        stats["positive"] += 1
        stats[etype] += 1

    return train, test, stats


def write_jsonl(records: List[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def _check_roundtrip(records: List[Dict[str, Any]]) -> int:
    bad = 0
    for r in records:
        for s in r["spans"]:
            if r["answer"][s["start"]:s["end"]] != s["text"]:
                bad += 1
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synthesise a medical span-labelled hallucination set from MedHallu."
    )
    parser.add_argument("--out-dir", default="data/processed/synth_medical")
    parser.add_argument("--configs", nargs="*", default=list(CONFIGS))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--limit", type=int, default=None, help="debug: first N rows per config")
    parser.add_argument(
        "--held-out-answers-config", default=None,
        help="drop any row whose clean answer also appears in this config's Ground "
             "Truth, so training never sees an answer used by that eval set",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.0,
        help="deterministic fraction of rows carved into {config}.test.jsonl",
    )
    parser.add_argument("--split-seed", type=int, default=0, help="hash seed for the train/test carve")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest: Dict[str, Any] = {
        "source": REPO_ID, "seed": args.seed,
        "held_out_answers_config": args.held_out_answers_config,
        "test_fraction": args.test_fraction, "split_seed": args.split_seed,
        "configs": {},
    }

    exclude_answers: Optional[Set[str]] = None
    if args.held_out_answers_config:
        if args.held_out_answers_config not in CONFIGS:
            print(f"unknown held-out config {args.held_out_answers_config!r}; known: {', '.join(CONFIGS)}")
            return 2
        exclude_answers = _answers_of(args.held_out_answers_config, args.limit)
        print(f"excluding {len(exclude_answers):,} answer(s) seen in {args.held_out_answers_config}")

    for config in args.configs:
        if config not in CONFIGS:
            print(f"unknown config {config!r}; known: {', '.join(CONFIGS)}")
            return 2
        train, test, stats = build_records(
            config, args.seed, args.limit,
            exclude_answers=exclude_answers,
            test_fraction=args.test_fraction, split_seed=args.split_seed,
        )
        bad = _check_roundtrip(train) + _check_roundtrip(test)
        assert bad == 0, f"{bad} spans did not round-trip in {config}"
        written = write_jsonl(train, out_dir / f"{config}.jsonl")
        test_written = 0
        if test:
            test_written = write_jsonl(test, out_dir / f"{config}.test.jsonl")
        manifest["configs"][config] = {**stats, "train_records": written, "test_records": test_written}
        print(f"{config:<16} train {written:,} rec / test {test_written:,} rec  "
              f"({stats['positive']:,} pos / {stats['negative']:,} neg; "
              f"conflict {stats['evident_conflict']:,}, baseless {stats['evident_baseless_info']:,}, "
              f"no-injector {stats['no_injector']:,}, excluded {stats['excluded']:,})")

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nwrote {out_dir / 'manifest.json'}")
    print("Spans are SYNTHETIC (rule-injected). Training/ablation data, not a gold "
          "benchmark. Say so wherever a number derived from this set is reported.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
