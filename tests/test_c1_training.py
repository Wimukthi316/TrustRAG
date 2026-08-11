"""Tests for the C1 dataset, batching and evaluation maths.

None of these need a GPU, a network connection or a downloaded model. The
tokenizer is a whitespace fake that satisfies the same contract as a real fast
tokenizer: character offsets per token and sequence ids that separate the
context from the answer. That is enough to exercise the whole path from a
build_examples record to a padded batch, which is where the silent bugs live.

The metric functions are checked against counts worked out by hand rather than
against another library, so a wrong metric cannot pass by agreeing with a wrong
reference.
"""

from __future__ import annotations

import json
import re

import pytest

torch = pytest.importorskip("torch")

from src.c1_detector.bio import B_HAL, I_HAL, IGNORE_INDEX, OUTSIDE  # noqa: E402
from src.c1_detector.dataset import (  # noqa: E402
    LengthGroupedBatchSampler,
    RagTruthTokenDataset,
    collate_batch,
    make_collate_fn,
    read_records,
    split_records,
    subsample,
    write_split_ids,
)
from src.c1_detector.evaluate_c1 import (  # noqa: E402
    dump_probabilities,
    evaluate,
    example_metrics,
    prf,
    repair_iob2,
    span_char_metrics,
    span_overlap_metrics,
    spans_from_tag_ids,
    token_metrics,
)
from src.c1_detector.train_c1 import deep_merge, get_metric, load_config  # noqa: E402

WORD = re.compile(r"\S+")


class FakeEncoding(dict):
    """Minimal stand-in for transformers' BatchEncoding."""

    def __init__(self, data, sequence_ids):
        super().__init__(data)
        self._sequence_ids = sequence_ids

    def sequence_ids(self, _index: int = 0):
        return self._sequence_ids


class FakeFastTokenizer:
    """Whitespace tokenizer with the fast-tokenizer surface encode_example needs.

    Produces [CLS] first [SEP] second [SEP] with per-token character offsets
    relative to whichever sequence the token came from, exactly as a real fast
    tokenizer does for a sequence pair. truncation="only_first" drops tokens off
    the end of the first sequence, never the second.
    """

    is_fast = True
    pad_token_id = 0
    name_or_path = "fake-fast-tokenizer"

    def __call__(
        self,
        first,
        second=None,
        truncation=None,
        max_length=None,
        return_offsets_mapping=False,
    ):
        first_tokens = [(m.group(), m.start(), m.end()) for m in WORD.finditer(first)]
        second_tokens = (
            [(m.group(), m.start(), m.end()) for m in WORD.finditer(second)]
            if second is not None
            else []
        )

        if truncation == "only_first" and max_length is not None:
            # 3 specials: CLS, SEP, SEP.
            budget = max_length - 3 - len(second_tokens)
            if budget < 0:
                budget = 0
            first_tokens = first_tokens[:budget]

        offsets = [(0, 0)]
        sequence_ids = [None]
        input_ids = [101]

        for i, (_, start, end) in enumerate(first_tokens):
            offsets.append((start, end))
            sequence_ids.append(0)
            input_ids.append(1000 + i)

        offsets.append((0, 0))
        sequence_ids.append(None)
        input_ids.append(102)

        for i, (_, start, end) in enumerate(second_tokens):
            offsets.append((start, end))
            sequence_ids.append(1)
            input_ids.append(2000 + i)

        offsets.append((0, 0))
        sequence_ids.append(None)
        input_ids.append(102)

        data = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
        }
        if return_offsets_mapping:
            data["offset_mapping"] = offsets
        return FakeEncoding(data, sequence_ids)


def make_record(record_id, task_type, answer, spans=(), context="the context here"):
    return {
        "id": str(record_id),
        "source_id": "s",
        "model": "gpt-4-0613",
        "task_type": task_type,
        "source": "test",
        "split": "train",
        "quality": "good",
        "question": "a question",
        "context": context,
        "answer": answer,
        "spans": [
            {"start": s, "end": e, "text": answer[s:e], "error_type": "evident_conflict"}
            for s, e in spans
        ],
    }


# --------------------------------------------------------------------------
# Dataset and collation
# --------------------------------------------------------------------------


def test_dataset_labels_only_the_answer():
    tokenizer = FakeFastTokenizer()
    answer = "alpha beta gamma delta"
    record = make_record(1, "qa", answer, spans=[(6, 15)])  # "beta gamma"[6:15]
    dataset = RagTruthTokenDataset([record], tokenizer, max_length=64, keep_offsets=True)
    item = dataset[0]

    answer_positions = [i for i, s in enumerate(item["sequence_ids"]) if s == 1]
    non_answer = [i for i, s in enumerate(item["sequence_ids"]) if s != 1]

    assert all(item["labels"][i] == IGNORE_INDEX for i in non_answer)
    assert all(item["labels"][i] != IGNORE_INDEX for i in answer_positions)

    answer_labels = [item["labels"][i] for i in answer_positions]
    assert answer_labels == [OUTSIDE, B_HAL, I_HAL, OUTSIDE]


def test_dataset_rejects_slow_tokenizer():
    class Slow:
        is_fast = False

    with pytest.raises(TypeError):
        RagTruthTokenDataset([make_record(1, "qa", "a b")], Slow())


def test_collate_pads_labels_with_ignore_index_not_zero():
    """The single most damaging thing this file can get wrong.

    Padding labels with 0 means "supported token", which would train the model
    on thousands of invented negatives per batch.
    """
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [0, 1, 2],
         "index": 0, "answer_truncated": False, "n_answer_tokens": 3},
        {"input_ids": [4], "attention_mask": [1], "labels": [0],
         "index": 1, "answer_truncated": False, "n_answer_tokens": 1},
    ]
    batch = collate_batch(features, pad_token_id=99, pad_to_multiple_of=8)

    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"][1].tolist() == [0] + [IGNORE_INDEX] * 7
    assert batch["input_ids"][1].tolist() == [4] + [99] * 7
    assert batch["attention_mask"][1].tolist() == [1] + [0] * 7
    assert batch["index"] == [0, 1]


def test_collate_without_multiple_of_uses_longest():
    features = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [0, 0, 0],
         "index": 0, "answer_truncated": False, "n_answer_tokens": 3},
        {"input_ids": [4], "attention_mask": [1], "labels": [0],
         "index": 1, "answer_truncated": False, "n_answer_tokens": 1},
    ]
    batch = collate_batch(features, pad_token_id=0, pad_to_multiple_of=None)
    assert batch["input_ids"].shape == (2, 3)


def test_collate_empty_batch_raises():
    with pytest.raises(ValueError):
        collate_batch([], pad_token_id=0)


def test_collate_carries_offsets_only_when_present():
    tokenizer = FakeFastTokenizer()
    records = [make_record(i, "qa", "alpha beta gamma") for i in range(3)]

    with_offsets = RagTruthTokenDataset(records, tokenizer, 64, keep_offsets=True)
    without = RagTruthTokenDataset(records, tokenizer, 64, keep_offsets=False)
    collate = make_collate_fn(tokenizer)

    assert "offset_mapping" in collate([with_offsets[i] for i in range(3)])
    assert "offset_mapping" not in collate([without[i] for i in range(3)])


def test_read_records_respects_limit(tmp_path):
    path = tmp_path / "records.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for i in range(10):
            handle.write(json.dumps(make_record(i, "qa", "a b c")) + "\n")
    assert len(read_records(path)) == 10
    assert len(read_records(path, limit=4)) == 4


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def _corpus():
    records = []
    for i in range(200):
        task = ["qa", "data2text", "summarization"][i % 3]
        records.append(make_record(i, task, "alpha beta"))
    return records


def test_subsample_keeps_every_task():
    """The processed file is grouped by task, so a head-of-file sample is single-task.

    500 records off the top of ragtruth_train.jsonl are 500 summarization
    examples and nothing else, which would make a smoke run pass without ever
    touching QA or data-to-text.
    """
    records = []
    for i in range(300):
        records.append(make_record(i, "summarization", "alpha beta"))
    for i in range(300, 500):
        records.append(make_record(i, "qa", "alpha beta"))

    head = records[:50]
    assert {r["task_type"] for r in head} == {"summarization"}, "fixture no longer ordered"

    sampled = subsample(records, 50, seed=42)
    assert len(sampled) == 50
    assert {r["task_type"] for r in sampled} == {"summarization", "qa"}


def test_subsample_is_proportional_and_seeded():
    records = [make_record(i, "qa" if i < 400 else "data2text", "a b") for i in range(500)]
    sampled = subsample(records, 100, seed=1)
    counts = {}
    for record in sampled:
        counts[record["task_type"]] = counts.get(record["task_type"], 0) + 1
    assert counts == {"qa": 80, "data2text": 20}
    assert [r["id"] for r in subsample(records, 100, seed=1)] == [r["id"] for r in sampled]


def test_subsample_passes_through_when_limit_is_none_or_large():
    records = [make_record(i, "qa", "a b") for i in range(10)]
    assert len(subsample(records, None)) == 10
    assert len(subsample(records, 99)) == 10


def test_split_is_disjoint_and_complete():
    splits = split_records(_corpus(), val_fraction=0.05, calib_fraction=0.10, seed=42)
    ids = {name: {r["id"] for r in rows} for name, rows in splits.items()}

    assert ids["train"] & ids["val"] == set()
    assert ids["train"] & ids["calib"] == set()
    assert ids["val"] & ids["calib"] == set()
    assert len(ids["train"] | ids["val"] | ids["calib"]) == 200


def test_split_is_stratified_by_task():
    splits = split_records(_corpus(), val_fraction=0.10, calib_fraction=0.20, seed=1)
    for name, rows in splits.items():
        tasks = {r["task_type"] for r in rows}
        assert tasks == {"qa", "data2text", "summarization"}, f"{name} lost a task"


def test_split_is_deterministic_given_the_seed():
    a = split_records(_corpus(), 0.05, 0.10, seed=7)
    b = split_records(_corpus(), 0.05, 0.10, seed=7)
    c = split_records(_corpus(), 0.05, 0.10, seed=8)
    assert [r["id"] for r in a["calib"]] == [r["id"] for r in b["calib"]]
    assert [r["id"] for r in a["calib"]] != [r["id"] for r in c["calib"]]


def test_split_rejects_fractions_that_leave_no_training_data():
    with pytest.raises(ValueError):
        split_records(_corpus(), val_fraction=0.6, calib_fraction=0.5)


def test_write_split_ids_round_trips(tmp_path):
    splits = split_records(_corpus(), 0.05, 0.10, seed=42)
    path = tmp_path / "split_ids.json"
    write_split_ids(splits, path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert set(loaded) == {"train", "val", "calib"}
    assert loaded["calib"] == [r["id"] for r in splits["calib"]]


# --------------------------------------------------------------------------
# Length-grouped sampler
# --------------------------------------------------------------------------


def test_length_grouped_sampler_covers_every_index_once():
    lengths = [10, 500, 20, 900, 30, 700, 40, 100, 60, 800]
    sampler = LengthGroupedBatchSampler(lengths, batch_size=3, seed=0)
    batches = list(iter(sampler))
    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == list(range(10))
    assert len(batches) == len(sampler) == 4


def test_length_grouped_sampler_groups_similar_lengths():
    lengths = list(range(100))
    sampler = LengthGroupedBatchSampler(lengths, batch_size=4, seed=0, mega_batch_mult=25)
    spreads = [max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in sampler]
    # Random batches of 4 drawn from 0..99 average a spread near 60. Grouping
    # should be far tighter than that.
    assert sum(spreads) / len(spreads) < 20


def test_length_grouped_sampler_set_epoch_changes_order():
    sampler = LengthGroupedBatchSampler(list(range(50)), batch_size=5, seed=0)
    first = list(iter(sampler))
    sampler.set_epoch(1)
    assert list(iter(sampler)) != first


# --------------------------------------------------------------------------
# Evaluation maths
# --------------------------------------------------------------------------


def test_prf_handles_zero_denominators():
    assert prf(0, 0, 0)["f1"] == 0.0
    exact = prf(2, 2, 2)
    assert exact["precision"] == 0.5 and exact["recall"] == 0.5 and exact["f1"] == 0.5


def test_repair_iob2_promotes_orphan_continuations():
    assert repair_iob2(["I-HAL", "I-HAL", "O", "I-HAL"]) == [
        "B-HAL", "I-HAL", "O", "B-HAL"
    ]
    assert repair_iob2(["B-HAL", "I-HAL", "O"]) == ["B-HAL", "I-HAL", "O"]


def test_spans_from_tag_ids_splits_adjacent_spans():
    """Two spans that touch must stay two spans. This is why BIO exists."""
    ids = [OUTSIDE, B_HAL, I_HAL, B_HAL, OUTSIDE, B_HAL]
    assert spans_from_tag_ids(ids) == [(1, 3), (3, 4), (5, 6)]


def test_token_metrics_counts_by_hand():
    predictions = [
        {
            # gold: O B I O   pred: O B O B
            "gold_ids": [OUTSIDE, B_HAL, I_HAL, OUTSIDE],
            "pred_ids": [OUTSIDE, B_HAL, OUTSIDE, B_HAL],
        }
    ]
    result = token_metrics(predictions)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["f1"] == pytest.approx(0.5)


def test_span_exact_and_overlap_disagree_as_expected():
    predictions = [
        {"gold_spans": [(0, 5), (10, 15)], "pred_spans": [(0, 5), (10, 14)]}
    ]
    exact = span_char_metrics(predictions)
    overlap = span_overlap_metrics(predictions)
    assert (exact["tp"], exact["fp"], exact["fn"]) == (1, 1, 1)
    assert (overlap["tp"], overlap["fp"], overlap["fn"]) == (2, 0, 0)


def test_span_overlap_matches_each_gold_span_at_most_once():
    """Predicting many tiny spans over one gold span must not inflate recall."""
    predictions = [
        {"gold_spans": [(0, 10)], "pred_spans": [(0, 2), (3, 5), (6, 8)]}
    ]
    overlap = span_overlap_metrics(predictions)
    assert (overlap["tp"], overlap["fp"], overlap["fn"]) == (1, 2, 0)


def test_example_metrics_is_response_level():
    predictions = [
        {"gold_spans": [(0, 1)], "pred_spans": [(0, 1)]},   # tp
        {"gold_spans": [], "pred_spans": [(0, 1)]},          # fp
        {"gold_spans": [(0, 1)], "pred_spans": []},          # fn
        {"gold_spans": [], "pred_spans": []},                # tn
    ]
    result = example_metrics(predictions)
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)


def _prediction(record_id, task, answer, gold, pred, probs=None):
    offsets = [(i, i + 1) for i in range(len(answer))]
    return {
        "index": 0,
        "id": record_id,
        "task_type": task,
        "model": "gpt-4-0613",
        "answer": answer,
        "gold_spans": gold,
        "pred_spans": pred,
        "gold_ids": [OUTSIDE] * len(answer),
        "pred_ids": [OUTSIDE] * len(answer),
        "token_probs": probs or [0.1] * len(answer),
        "answer_offsets": offsets,
        "answer_truncated": False,
    }


def test_evaluate_breaks_down_per_task():
    predictions = [
        _prediction("1", "qa", "abcde", [(0, 2)], [(0, 2)]),
        _prediction("2", "data2text", "abcde", [(1, 3)], []),
    ]
    metrics = evaluate(predictions)
    assert set(metrics["per_task"]) == {"qa", "data2text"}
    assert metrics["overall"]["n_examples"] == 2
    assert metrics["per_task"]["qa"]["example"]["f1"] == pytest.approx(1.0)
    assert metrics["per_task"]["data2text"]["example"]["f1"] == pytest.approx(0.0)
    assert get_metric(metrics, "example.tp") == 1


def test_dump_probabilities_labels_spans_and_aggregates(tmp_path):
    answer = "abcdef"
    probs = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1]
    predictions = [
        _prediction("1", "qa", answer, gold=[(0, 2)], pred=[(0, 2), (4, 6)], probs=probs)
    ]
    path = tmp_path / "probabilities.jsonl"
    assert dump_probabilities(predictions, path) == 1

    row = json.loads(path.read_text(encoding="utf-8").strip())
    hit, miss = row["pred_spans"]
    assert hit["is_hallucinated"] is True
    assert hit["mean_prob"] == pytest.approx(0.85)
    assert hit["max_prob"] == pytest.approx(0.9)
    assert hit["n_tokens"] == 2
    assert miss["is_hallucinated"] is False
    assert row["gold_spans"] == [{"start": 0, "end": 2, "text": "ab"}]


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_deep_merge_recurses_into_nested_dicts():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    assert deep_merge(base, {"a": {"y": 9}}) == {"a": {"x": 1, "y": 9}, "b": 3}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}, "deep_merge mutated its input"


def test_shipped_configs_load_and_keep_the_answer_untruncated():
    smoke = load_config("configs/c1_smoke.yaml")
    base = load_config("configs/c1_base.yaml")

    assert smoke["wandb"]["enabled"] is False, "a smoke run must not log to W&B"
    assert smoke["data"]["limit"] == 500

    # The longest RAGTruth sequence measured 2,628 tokens on 2026-08-11.
    assert base["data"]["max_length"] >= 2628
    assert base["data"]["limit"] is None
    assert base["output"]["select_on"] == "example.f1"


def test_load_config_rejects_unknown_top_level_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("trian:\n  epochs: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown top-level config keys"):
        load_config(path)
