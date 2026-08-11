"""torch Dataset, collation and splitting over the preprocessed RAGTruth JSONL.

This sits directly on top of bio.encode_example: that function owns the
tokenisation and the char-span -> BIO conversion, this file owns batching,
padding and the train/val/calib split. Nothing here re-implements offset logic.

Three things are load-bearing:

1. Label padding is IGNORE_INDEX (-100), not 0. Padding with 0 would train the
   model that every pad position is a supported token, which both dilutes the
   loss and lets the model score well by predicting "O" everywhere.

2. Only answer tokens carry a real label. encode_example already sets
   IGNORE_INDEX on the context, the question and the specials, so the
   answer-only loss mask is a property of the labels themselves -- there is no
   separate mask tensor to keep in sync.

3. The split is stratified by task_type. RAGTruth's three tasks have very
   different hallucination densities (data-to-text is roughly 69% hallucinated
   responses, QA roughly 29%), so an unstratified random split gives a
   validation set whose task mix drifts from training and makes the per-task
   breakdown noisy.

The calibration split exists for C2. Conformal prediction needs a set that the
model never saw during training or model selection, and it has to be disjoint
from the test set the final numbers are reported on. Carving it out of train
now, with the ids written to disk, means C2 can start against the public
LettuceDetect checkpoint on exactly the same records and swap in C1's
probabilities later without re-splitting anything.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler

from src.c1_detector.bio import IGNORE_INDEX, encode_example

# Keys a collated batch is guaranteed to carry. The tensor keys go to the model;
# the rest are python objects the evaluator needs to decode spans.
TENSOR_KEYS = ("input_ids", "attention_mask", "labels")


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


def read_records(path: Path | str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load build_examples output. `limit` takes the first N lines.

    The file is one JSON object per line; 44.6% of answers contain a newline, so
    a line-oriented reader is only safe because the newlines are escaped inside
    the JSON strings. Do not switch this to a naive split.

    WARNING: the processed file is grouped by task -- the first 3,000+ lines of
    ragtruth_train.jsonl are all summarization. `limit` here is a head, so it
    gives a single-task sample. Use subsample() for anything that has to
    represent the corpus, which is every smoke run.
    """
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def subsample(
    records: Sequence[Dict[str, Any]],
    limit: Optional[int],
    seed: int = 42,
    stratify_key: str = "task_type",
) -> List[Dict[str, Any]]:
    """Take `limit` records while keeping the corpus's task mix.

    This exists because of a trap that cost a wasted smoke run: build_examples
    writes records grouped by task, so the first 500 lines of
    ragtruth_train.jsonl are 500 summarization examples and nothing else. A
    smoke test on that sample never touches QA or data-to-text, never exercises
    the per-task breakdown, and would happily pass while a task-specific bug sat
    untouched.

    Sampling is seeded, so a given (limit, seed) always yields the same subset
    and two smoke runs are comparable.
    """
    if limit is None or limit >= len(records):
        return list(records)

    buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record.get(stratify_key)].append(record)

    rng = random.Random(seed)
    keys = sorted(buckets, key=lambda k: str(k))
    for key in keys:
        rng.shuffle(buckets[key])

    # Proportional quota, then hand out the rounding remainder one at a time so
    # the total is exactly `limit`.
    total = len(records)
    quotas = {key: int(len(buckets[key]) * limit / total) for key in keys}
    order = sorted(keys, key=lambda k: -len(buckets[k]))
    i = 0
    while sum(quotas.values()) < limit:
        key = order[i % len(order)]
        if quotas[key] < len(buckets[key]):
            quotas[key] += 1
        i += 1

    sampled: List[Dict[str, Any]] = []
    for key in keys:
        sampled.extend(buckets[key][: quotas[key]])
    rng.shuffle(sampled)
    return sampled


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def split_records(
    records: Sequence[Dict[str, Any]],
    val_fraction: float = 0.05,
    calib_fraction: float = 0.10,
    seed: int = 42,
    stratify_key: str = "task_type",
) -> Dict[str, List[Dict[str, Any]]]:
    """Split into train / val / calib, stratified and reproducible.

    Returns a dict with keys "train", "val", "calib". Fractions are of the whole
    input; the remainder is train. Stratification is per `stratify_key` value,
    so each output split holds roughly the same task mix as the input.

    Deterministic given `seed`: the same seed always produces the same three
    sets, which is what makes the C2 calibration set stable across reruns.
    """
    if val_fraction < 0 or calib_fraction < 0:
        raise ValueError("fractions must be non-negative")
    if val_fraction + calib_fraction >= 1.0:
        raise ValueError(
            f"val_fraction ({val_fraction}) + calib_fraction ({calib_fraction}) "
            "must leave something for training"
        )

    buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record.get(stratify_key)].append(record)

    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "calib": []}
    rng = random.Random(seed)

    for key in sorted(buckets, key=lambda k: str(k)):
        rows = list(buckets[key])
        rng.shuffle(rows)
        n = len(rows)
        n_val = int(round(n * val_fraction))
        n_calib = int(round(n * calib_fraction))
        # Guard the degenerate case where a tiny stratum rounds away the train set.
        while n_val + n_calib >= n and (n_val or n_calib):
            if n_calib >= n_val:
                n_calib -= 1
            else:
                n_val -= 1
        out["val"].extend(rows[:n_val])
        out["calib"].extend(rows[n_val : n_val + n_calib])
        out["train"].extend(rows[n_val + n_calib :])

    for name in out:
        rng.shuffle(out[name])
    return out


def write_split_ids(splits: Dict[str, List[Dict[str, Any]]], path: Path | str) -> None:
    """Record which record id landed in which split.

    C2 reads this so its calibration set is provably the same set of responses
    the detector never trained on. Without it, "the calibration set" is whatever
    the last run happened to shuffle up, and the coverage guarantee is not
    reproducible.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: [r["id"] for r in rows] for name, rows in splits.items()}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------


class RagTruthTokenDataset(Dataset):
    """One preprocessed RAGTruth record per item, tokenised on access.

    Tokenisation is done lazily rather than cached up front. A fast tokenizer
    handles a record in single-digit milliseconds, so a 15k-record epoch costs
    well under a minute of CPU, while caching every input_ids list would hold
    several hundred MB resident -- which matters on Kaggle's 13GB of RAM more
    than the CPU time does.

    `keep_offsets` carries the offset mapping and sequence ids through to the
    batch. Training does not need them; evaluation does, because decoding
    predicted labels back into character spans is what produces the span-level
    and example-level numbers.
    """

    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        tokenizer: Any,
        max_length: int = 3072,
        keep_offsets: bool = False,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise TypeError(
                "a fast tokenizer is required; load it with "
                "AutoTokenizer.from_pretrained(..., use_fast=True)"
            )
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.keep_offsets = keep_offsets

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        encoded = encode_example(self.tokenizer, record, max_length=self.max_length)

        item: Dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["labels"],
            "index": index,
            "answer_truncated": encoded["answer_truncated"],
            "n_answer_tokens": encoded["n_answer_tokens"],
        }
        if self.keep_offsets:
            item["offset_mapping"] = encoded["offset_mapping"]
            item["sequence_ids"] = encoded["sequence_ids"]
        return item

    def meta(self, index: int) -> Dict[str, Any]:
        """Everything the evaluator needs about an item that is not a tensor."""
        record = self.records[index]
        return {
            "id": record.get("id"),
            "task_type": record.get("task_type"),
            "model": record.get("model"),
            "answer": record["answer"],
            "gold_spans": [(s["start"], s["end"]) for s in record.get("spans", [])],
        }

    def char_lengths(self) -> List[int]:
        """Cheap length proxy for the length-grouped sampler.

        Character count, not token count, so it costs nothing. It only has to
        rank records, and the character/token ratio is stable enough across
        English prose for the ranking to hold.
        """
        return [
            len(r.get("question") or "") + len(r["context"]) + len(r["answer"])
            for r in self.records
        ]


# --------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------


def _pad_to(length: int, multiple_of: Optional[int]) -> int:
    if not multiple_of:
        return length
    return int(math.ceil(length / multiple_of) * multiple_of)


def collate_batch(
    features: Sequence[Dict[str, Any]],
    pad_token_id: int,
    pad_to_multiple_of: Optional[int] = 8,
) -> Dict[str, Any]:
    """Pad a list of dataset items into one batch.

    Dynamic padding: the batch is padded to its own longest member, not to
    max_length. RAGTruth sequence lengths run from 171 to 2,628 tokens, so
    padding everything to a fixed 3,072 would waste most of the compute on pad
    tokens. Rounding up to a multiple of 8 keeps tensor cores happy.

    Labels pad with IGNORE_INDEX so the loss never sees a padded position.
    """
    if not features:
        raise ValueError("cannot collate an empty batch")

    batch_size = len(features)
    max_len = _pad_to(max(len(f["input_ids"]) for f in features), pad_to_multiple_of)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)

    for row, feature in enumerate(features):
        n = len(feature["input_ids"])
        input_ids[row, :n] = torch.tensor(feature["input_ids"], dtype=torch.long)
        attention_mask[row, :n] = torch.tensor(feature["attention_mask"], dtype=torch.long)
        labels[row, :n] = torch.tensor(feature["labels"], dtype=torch.long)

    batch: Dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "index": [f["index"] for f in features],
        "answer_truncated": [f["answer_truncated"] for f in features],
    }

    # Offsets and sequence ids stay as ragged python lists. Padding them into a
    # tensor would mean inventing offsets for pad positions, and the evaluator
    # would then have to strip them back out again.
    if "offset_mapping" in features[0]:
        batch["offset_mapping"] = [f["offset_mapping"] for f in features]
        batch["sequence_ids"] = [f["sequence_ids"] for f in features]

    return batch


def make_collate_fn(
    tokenizer: Any, pad_to_multiple_of: Optional[int] = 8
) -> Callable[[Sequence[Dict[str, Any]]], Dict[str, Any]]:
    """Bind the tokenizer's pad id into a collate function for the DataLoader."""
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError(
            f"{tokenizer.name_or_path} has no pad_token_id; set one before batching"
        )

    def _collate(features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return collate_batch(features, pad_token_id, pad_to_multiple_of)

    return _collate


# --------------------------------------------------------------------------
# Length-grouped batching
# --------------------------------------------------------------------------


class LengthGroupedBatchSampler(Sampler[List[int]]):
    """Batch similar-length examples together, while staying random.

    Fully random batches pair a 200-token record with a 2,600-token one and the
    short one is padded out to the long one's length. Fully sorted batches fix
    that but destroy shuffling, which hurts optimisation.

    The usual compromise, and what this does: shuffle, cut into mega-batches of
    `mega_batch_mult` batches, sort each mega-batch by length, emit batches from
    it, then shuffle the order the batches are yielded in. Randomness survives,
    padding waste mostly does not.

    Measured benefit is unverified on this corpus -- it depends on the length
    distribution within a mega-batch. Turn it off in the config if a run needs
    to be compared step-for-step against a non-grouped run.
    """

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        mega_batch_mult: int = 50,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mega_batch_size = max(batch_size, batch_size * mega_batch_mult)
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle differently each epoch. Call this at the top of every epoch."""
        self.epoch = epoch

    def __len__(self) -> int:
        n = len(self.lengths)
        if self.drop_last:
            return n // self.batch_size
        return math.ceil(n / self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(indices)

        batches: List[List[int]] = []
        for start in range(0, len(indices), self.mega_batch_size):
            mega = indices[start : start + self.mega_batch_size]
            mega.sort(key=lambda i: self.lengths[i], reverse=True)
            for offset in range(0, len(mega), self.batch_size):
                batch = mega[offset : offset + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)


def build_dataloader(
    dataset: RagTruthTokenDataset,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    group_by_length: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    pad_to_multiple_of: Optional[int] = 8,
) -> Tuple[torch.utils.data.DataLoader, Optional[LengthGroupedBatchSampler]]:
    """Assemble a DataLoader, returning the batch sampler so set_epoch can be called.

    num_workers defaults to 0 because Windows spawns rather than forks worker
    processes, and a spawned worker re-imports this module and re-pickles the
    tokenizer. On Kaggle (Linux) raising it is worthwhile.
    """
    collate_fn = make_collate_fn(tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    if group_by_length:
        sampler = LengthGroupedBatchSampler(
            dataset.char_lengths(), batch_size, shuffle=shuffle, seed=seed
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        return loader, sampler

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, None
