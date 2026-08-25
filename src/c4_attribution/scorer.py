"""Layer 4 -- the neural attribution scorers, their pre-filter, and their ablations.

Four things are decided here, and each one is measured rather than assumed.

**The cost control, and its ceiling.** Scoring every span against every context
sentence is roughly 2,400 spans by 19 sentences on the test split alone, which
is hours of CPU. Instead a MiniLM embedding pre-filter keeps the top k sentences
per span and only those are scored by the expensive model. That is a real speed
-up and a real error floor: if the pre-filter drops the gold sentence, no later
stage can recover it. So `prefilter_recall` is reported as its own row. A stage
that hides its own truncation is the failure this project refuses to make.

**The hypothesis question (4b).** A flagged span is often a fragment -- "in
2019" -- and a fragment is not a proposition. FactScore rewrites units into
standalone facts, which needs an LLM this project does not have. AlignScore and
MiniCheck both work at sentence granularity and never go below it. C4 takes that
path, and then measures the two alternatives it did not take, because it is
cheap and it is a result: the bare span, the parent answer sentence, and the
parent sentence with the span marked with `<< >>` -- the same input format
`src/c3_explanation/type_model.py` already uses.

**The direction question (4c), and getting it right decides whether the whole
component is well specified.** For a *hallucinated* span, the sentence the
annotator quoted is the sentence they judged it **against**. For an Evident
Conflict that is a *contradicting* sentence; for Evident Baseless Info it is
usually nothing at all. So C4's task is not "find the sentence that entails the
span" -- for a hallucinated span an entailing sentence mostly does not exist.
C4's task is *find the sentence a human would point at to justify the flag*.
That is why the entailment head alone is the wrong score and
`max(P_entail, P_contradict)` is reported beside it.

`Span.entailment_score` in the frozen contract is therefore defined, once, as
**the model's confidence that this sentence is the one a human would cite for
this span** -- not as P(entailment). The field name is inherited, and calling it
what it is beats renaming a frozen contract.

**Licences, checked on the model pages 2026-08-24.** MiniLM Apache-2.0, the
MoritzLaurer DeBERTa NLI family MIT, HHEM-2.1-Open Apache-2.0.
`bespokelabs/Bespoke-MiniCheck-7B` is CC BY-NC 4.0 and is not used: a
non-commercial dependency would make the product unsellable, which is an LO5
claim this project would rather be able to make.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.c4_attribution.dataset import SpanRecord
from src.c4_attribution.lexical import idf_overlap_scores

OUT_DIR = Path("results/c4/scores")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
HHEM_MODEL = "vectara/hallucination_evaluation_model"

TOP_K = 5
TOP_K_SWEEP: Tuple[int, ...] = (1, 3, 5, 10)
MAX_LENGTH = 256
BATCH = 32

HYPOTHESES: Tuple[str, ...] = ("span", "sentence", "marked")

# Every score a backend can produce for one (span, context sentence) pair.
SCORE_KEYS: Tuple[str, ...] = (
    "lexical",
    "embed",
    "nli_entail",
    "nli_signed",
    "hhem_support",
    "hhem_decisive",
)


def hypothesis_text(row: SpanRecord, kind: str) -> str:
    """The three constructions of Layer 4b, in one place so nothing drifts."""
    if kind == "span":
        return row.span_text
    if kind == "sentence":
        return row.answer_sentence
    if kind == "marked":
        return row.marked_answer_sentence
    raise ValueError(f"unknown hypothesis {kind!r}; expected one of {HYPOTHESES}")


def _threads() -> None:
    import torch

    torch.set_num_threads(2)


# --------------------------------------------------------------------------
# The embedding pre-filter, which is also a backend in its own right
# --------------------------------------------------------------------------


class EmbeddingScorer:
    """MiniLM cosine similarity. Fast enough to score every sentence, so it does.

    Used twice: as the pre-filter that decides which pairs the expensive models
    see, and as a backend of its own, because "an embedding model with no
    entailment training" is the right control for "does the entailment model
    earn its keep".
    """

    def __init__(self, model_name: str = EMBED_MODEL) -> None:
        _threads()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = model_name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        return self.model.encode(
            list(texts),
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def scores(self, hypothesis: str, sentences: Sequence[str]) -> List[float]:
        if not sentences:
            return []
        vectors = self.encode([hypothesis] + list(sentences))
        return [float(v) for v in vectors[1:] @ vectors[0]]


def prefilter_indices(scores: Sequence[float], k: int) -> Tuple[int, ...]:
    """Top-k sentence indices by pre-filter score, ties to the lowest index."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return tuple(order[:k])


def prefilter_recall(
    kept: Sequence[Sequence[int]], gold: Sequence[Optional[int]], k: int
) -> Dict[str, Any]:
    """How often the gold sentence survives the pre-filter at this k.

    This is the ceiling on everything downstream, and it is reported as a row
    rather than described in prose.
    """
    scored = [(list(keep)[:k], target) for keep, target in zip(kept, gold) if target is not None]
    if not scored:
        return {"k": k, "n": 0, "recall": float("nan")}
    hits = sum(1 for keep, target in scored if target in keep)
    return {
        "k": k,
        "n": len(scored),
        "recall": hits / len(scored),
        "error_floor": 1.0 - hits / len(scored),
    }


# --------------------------------------------------------------------------
# The entailment backends
# --------------------------------------------------------------------------


@dataclass
class PairBatch:
    """One (premise, hypothesis) pair queued for a cross-encoder."""

    row_index: int
    sentence_index: int
    premise: str
    hypothesis: str


class NLIScorer:
    """DeBERTa-v3-base MNLI/FEVER/ANLI. Signed on purpose: it reports both heads.

    The premise is the context sentence and the hypothesis is the flagged text,
    which is the direction the NLI literature uses. Both P(entail) and
    P(contradict) are kept, because for a hallucinated span the sentence a human
    would cite is often the one it contradicts.
    """

    def __init__(self, model_name: str = NLI_MODEL) -> None:
        _threads()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.name = model_name
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail_id = labels["entailment"]
        self.contradict_id = labels["contradiction"]

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[Tuple[float, float]]:
        """(P_entail, P_contradict) per (premise, hypothesis) pair.

        Batches are formed over length-sorted pairs. Padding is what a batched
        cross-encoder actually spends its time on -- one 250-token premise drags
        thirty-one short ones up to its length -- and sorting first removes most
        of that without touching a single score, because the attention mask
        already makes padding a no-op numerically.
        """
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        results: List[Optional[Tuple[float, float]]] = [None] * len(pairs)
        for start in range(0, len(order), BATCH):
            positions = order[start:start + BATCH]
            chunk = [pairs[i] for i in positions]
            encoded = self.tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            )
            with self.torch.inference_mode():
                logits = self.model(**encoded).logits
            probs = self.torch.softmax(logits, dim=-1).numpy()
            for position, row in zip(positions, probs):
                results[position] = (
                    float(row[self.entail_id]),
                    float(row[self.contradict_id]),
                )
        return [r if r is not None else (0.0, 0.0) for r in results]


class HHEMScorer:
    """Vectara HHEM-2.1-Open, loaded through the repo's own hardened loader.

    `src/c1_detector/hhem_baseline.load_hhem` verifies that the T5 encoder
    embedding really is tied to the shared embedding after loading. Without that
    check the model still returns numbers between 0 and 1 and every one of them
    is noise, so it is reused here rather than re-written.

    Score direction: HHEM returns P(the hypothesis is supported by the premise),
    so a *high* score means supported. `hhem_decisive` = |2p - 1| is the
    two-headed reading -- how far from "no opinion" the model is -- which is the
    HHEM analogue of max(P_entail, P_contradict).
    """

    def __init__(self) -> None:
        _threads()
        from src.c1_detector.hhem_baseline import load_hhem, sanity_check

        self.model = load_hhem("cpu")
        sanity_check(self.model)
        self.name = HHEM_MODEL

    def score(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """P(supported) per pair, batched over length-sorted inputs. See NLIScorer."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        results: List[float] = [0.0] * len(pairs)
        for start in range(0, len(order), BATCH):
            positions = order[start:start + BATCH]
            values = self.model.predict([pairs[i] for i in positions])
            for position, value in zip(positions, values):
                results[position] = float(value)
        return results


# --------------------------------------------------------------------------
# The dump: one record per span, every backend's scores over the same pairs
# --------------------------------------------------------------------------


def score_path(split: str) -> Path:
    return OUT_DIR / f"pair_scores_{split}.jsonl"


def _key(row: SpanRecord) -> str:
    return f"{row.response_id}:{row.label.start}:{row.label.end}"


def existing_keys(path: Path) -> set:
    """Keys already written, so an interrupted run resumes instead of restarting."""
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    keys.add(json.loads(line)["key"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return keys


def build_pair_scores(
    rows: Sequence[SpanRecord],
    matches: Sequence[Any],
    split: str,
    top_k: int = TOP_K,
    backends: Sequence[str] = ("lexical", "embed", "nli", "hhem"),
    chunk_size: int = 256,
    progress_every: int = 320,
) -> Path:
    """Score every span in `rows` and append one JSON record per span.

    One pre-filter, one candidate set, every backend scored on exactly those
    pairs -- so a backend comparison is a comparison of models and not of which
    sentences each one happened to see. The embedding backend is the exception
    and is scored over *all* sentences, because it is the pre-filter and costs
    nothing extra; that is stated in the artefact rather than left for a reader
    to infer.

    Spans are processed in chunks and every cross-encoder call is batched across
    the whole chunk. Five pairs at a time leaves a CPU mostly idle; a thousand
    does not, and on this machine that is the difference between two hours and
    twenty minutes.
    """
    path = score_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_keys(path)

    todo = [
        (row, gold)
        for row, gold in zip(rows, matches)
        if _key(row) not in done
    ]
    if not todo:
        print(f"[{split}] nothing to do; {len(done)} records already on disk")
        return path

    embedder = EmbeddingScorer() if "embed" in backends else None
    nli = NLIScorer() if "nli" in backends else None
    hhem = HHEMScorer() if "hhem" in backends else None

    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for offset in range(0, len(todo), chunk_size):
            chunk = todo[offset:offset + chunk_size]
            records: List[Dict[str, Any]] = []
            embed_cache: Dict[str, np.ndarray] = {}

            for row, gold in chunk:
                sentences = row.sentences
                record: Dict[str, Any] = {
                    "key": _key(row),
                    "response_id": row.response_id,
                    "start": row.label.start,
                    "end": row.label.end,
                    "split": split,
                    "task": row.task,
                    "label_type": row.label.label_type,
                    "implicit_true": row.label.implicit_true,
                    "match_kind": gold.kind,
                    "gold_index": gold.convention_a,
                    "gold_indices": list(gold.gold_indices),
                    "n_sentences": len(sentences),
                    "top_k": top_k,
                }
                record["lexical"] = [
                    round(v, 6) for v in idf_overlap_scores(row.span_text, sentences)
                ]

                parent = hypothesis_text(row, "sentence")
                if embedder is not None:
                    if row.response_id not in embed_cache:
                        embed_cache[row.response_id] = embedder.encode(sentences)
                    sentence_vectors = embed_cache[row.response_id]
                    hypothesis_vector = embedder.encode([parent])[0]
                    scores = (
                        [float(v) for v in sentence_vectors @ hypothesis_vector]
                        if len(sentences)
                        else []
                    )
                    record["embed"] = [round(v, 6) for v in scores]
                    keep = prefilter_indices(scores, top_k)
                else:
                    keep = prefilter_indices(record["lexical"], top_k)
                record["topk"] = list(keep)
                records.append(record)

            if nli is not None:
                for kind in HYPOTHESES:
                    pairs: List[Tuple[str, str]] = []
                    owners: List[Tuple[int, int]] = []
                    for position, (record, (row, _)) in enumerate(zip(records, chunk)):
                        hypothesis = hypothesis_text(row, kind)
                        for index in record["topk"]:
                            pairs.append((row.sentences[index], hypothesis))
                            owners.append((position, index))
                    scored = nli.score(pairs) if pairs else []
                    for record in records:
                        record.setdefault("nli", {})[kind] = []
                    for (position, index), (entail, contradict) in zip(owners, scored):
                        records[position]["nli"][kind].append(
                            [int(index), round(entail, 6), round(contradict, 6)]
                        )

            if hhem is not None:
                pairs = []
                owners = []
                for position, (record, (row, _)) in enumerate(zip(records, chunk)):
                    parent = hypothesis_text(row, "sentence")
                    for index in record["topk"]:
                        pairs.append((row.sentences[index], parent))
                        owners.append((position, index))
                scored = hhem.score(pairs) if pairs else []
                for record in records:
                    record["hhem"] = []
                for (position, index), value in zip(owners, scored):
                    records[position]["hhem"].append([int(index), round(value, 6)])

            for record in records:
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            written += len(records)
            if progress_every and written % progress_every < chunk_size:
                print(f"[{split}] {written}/{len(todo)} scored", flush=True)

    print(f"[{split}] wrote {written} new records to {path}")
    return path


def load_pair_scores(split: str) -> List[Dict[str, Any]]:
    path = score_path(split)
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------------
# Turning a dump record into one backend's per-sentence scores
# --------------------------------------------------------------------------


def backend_scores(
    record: Dict[str, Any], backend: str, hypothesis: str = "sentence"
) -> List[float]:
    """A backend's score for every sentence of this record's context.

    Sentences the pre-filter dropped score 0.0. That is the truncation the
    pre-filter recall row quantifies; it is not hidden and it is not smoothed.
    """
    n = int(record["n_sentences"])
    if backend == "lexical":
        return list(record.get("lexical") or [0.0] * n)
    if backend == "embed":
        return list(record.get("embed") or [0.0] * n)

    out = [0.0] * n
    if backend in ("nli_entail", "nli_signed"):
        for index, entail, contradict in (record.get("nli") or {}).get(hypothesis, []):
            if 0 <= index < n:
                out[index] = entail if backend == "nli_entail" else max(entail, contradict)
        return out
    if backend in ("hhem_support", "hhem_decisive"):
        for index, value in record.get("hhem") or []:
            if 0 <= index < n:
                out[index] = value if backend == "hhem_support" else abs(2.0 * value - 1.0)
        return out
    raise ValueError(f"unknown backend {backend!r}; expected one of {SCORE_KEYS}")


# --------------------------------------------------------------------------
# Layer 4's analysis: the pre-filter ceiling, the backends, and the two ablations
# --------------------------------------------------------------------------

BACKENDS: Tuple[str, ...] = (
    "lexical",
    "embed",
    "nli_entail",
    "nli_signed",
    "hhem_support",
    "hhem_decisive",
)

# Only the NLI backend is scored under all three hypothesis constructions. The
# lexical backend reads the span text and has no hypothesis to vary; the
# embedding and HHEM backends are scored once, with the parent answer sentence,
# because the ablation's question -- can a bare fragment be scored at all -- is
# answered by running it on one cross-encoder rather than three. Listing HHEM
# three times with identical numbers would look like an ablation that found no
# difference, which is not what happened.
HYPOTHESIS_VARIED: Tuple[str, ...] = ("nli_entail", "nli_signed")


def hypotheses_for(backend: str) -> Tuple[str, ...]:
    return HYPOTHESES if backend in HYPOTHESIS_VARIED else ("sentence",)


def prefilter_report(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Recall@k of the pre-filter, which is a ceiling on everything after it.

    A pipeline whose first stage drops the right sentence 8% of the time has an
    8% error floor no later stage can recover. Reporting it as its own row is
    the difference between a cost control and a silent truncation.

    The IDF ranking is scored at the same k for free, because it is already in
    the dump. That comparison is the honest way to say whether the embedding
    model was the right choice of pre-filter -- with a number, rather than by
    rerunning the whole thing once the answer is known and presenting the second
    attempt as the design.
    """
    gold = [r.get("gold_index") for r in records]
    out: Dict[str, Any] = {"top_k_used": TOP_K, "candidates": {}}
    for name, key in (("embedding_used", "embed"), ("idf_lexical_not_used", "lexical")):
        ranked = [
            prefilter_indices(record.get(key) or [], max(TOP_K_SWEEP))
            for record in records
        ]
        out["candidates"][name] = [prefilter_recall(ranked, gold, k) for k in TOP_K_SWEEP]

    rows = out["candidates"]["embedding_used"]
    used = next((r for r in rows if r["k"] == TOP_K), rows[-1])
    alternative = next(
        (r for r in out["candidates"]["idf_lexical_not_used"] if r["k"] == TOP_K),
        None,
    )
    out["rows"] = rows
    out["error_floor_at_k_used"] = used.get("error_floor")
    out["alternative_error_floor_at_k_used"] = (
        alternative.get("error_floor") if alternative else None
    )
    out["note"] = (
        "The embedding backend is scored over ALL sentences and is not affected "
        f"by this truncation. The entailment and HHEM backends see only the top "
        f"{TOP_K}, so every one of their numbers carries this error floor. The "
        "second candidate row is the pre-filter that was NOT used, scored on the "
        "same rows; if it is better, that is a measured design mistake and it is "
        "reported as one."
    )
    return out
