"""Tests for the serving-time exchangeability check.

Four things have to hold, and they matter in this order.

**Parity.** The offline reference and the online check must compute the same
features from the same response. Two implementations of "how long is this
answer" that drift apart would make the reference meaningless in exactly the
silent way this project keeps finding, so they share one function and this
suite asserts it on real records.

**No false alarm on the demo.** The record the demo opens on must never trip
the warning. A false alarm in front of a panel is worse than no alarm at all,
and this is the test that stops a threshold change from causing one.

**A real alarm on obviously odd input.** An answer where every character is
highlighted, or one ten times longer than anything in RAGTruth, has to trip.

**A controlled false-alarm rate.** The threshold claims to be a false-alarm
rate. On held-out data it has to behave like one, or the number on the screen
is decoration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from src.c2_calibration.exchangeability import (  # noqa: E402
    DEFAULT_THRESHOLD,
    FEATURE_NAMES,
    ExchangeabilityReference,
    _tail_depth,
    features_from_record,
    response_features,
)

REPO = Path(__file__).resolve().parents[1]
CALIB = REPO / "results" / "c1" / "calib" / "probabilities.jsonl"
TEST = REPO / "results" / "c1" / "test" / "probabilities.jsonl"
# The record the demo opens on. Its offsets and text are quoted verbatim in
# backend/app/main.py, so it is the one a panel will actually see.
DEMO_RECORD_ID = "16121"


def make_record(n_tokens=40, spans=((0, 12, 3),), answer_chars=200, prob=0.05, seed=0):
    """A probability-dump record with controllable shape.

    `spans` is a list of (start, end, n_tokens). The answer is filler of the
    requested length -- nothing here reads it except the length.
    """
    rng = np.random.default_rng(seed)
    return {
        "id": f"synthetic-{seed}",
        "task_type": "qa",
        "model": "test",
        "answer": "x" * answer_chars,
        "gold_spans": [],
        "pred_spans": [
            {
                "start": start,
                "end": end,
                "text": "x" * (end - start),
                "mean_prob": prob,
                "max_prob": prob,
                "min_prob": prob,
                "n_tokens": n,
                "is_hallucinated": False,
            }
            for start, end, n in spans
        ],
        "token_probs": [float(np.clip(rng.normal(prob, 0.02), 0, 1)) for _ in range(n_tokens)],
        "answer_offsets": [[i, i + 1] for i in range(n_tokens)],
        "answer_truncated": False,
    }


def read(path, limit=None):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


needs_dumps = pytest.mark.skipif(
    not (CALIB.exists() and TEST.exists()),
    reason="probability dumps are gitignored; run the C1 evaluation first",
)


# --------------------------------------------------------------------------
# Parity: one implementation, two callers
# --------------------------------------------------------------------------


def test_offline_and_online_features_agree_on_a_synthetic_record():
    record = make_record(n_tokens=33, spans=((0, 10, 3), (20, 45, 7)), answer_chars=180)
    offline = features_from_record(record)
    online = response_features(
        record["token_probs"],
        [(span["start"], span["end"]) for span in record["pred_spans"]],
        [span["n_tokens"] for span in record["pred_spans"]],
        record["answer"],
    )
    assert offline == pytest.approx(online)


@needs_dumps
def test_offline_and_online_features_agree_on_real_records():
    for record in read(TEST, limit=200):
        offline = features_from_record(record)
        online = response_features(
            record["token_probs"],
            [(span["start"], span["end"]) for span in record["pred_spans"]],
            [span["n_tokens"] for span in record["pred_spans"]],
            record["answer"],
        )
        assert offline == pytest.approx(online), record["id"]


def test_every_named_feature_is_produced():
    features = features_from_record(make_record())
    assert set(features) == set(FEATURE_NAMES)
    assert all(isinstance(value, float) for value in features.values())


def test_a_response_with_no_spans_is_not_an_error():
    features = features_from_record(make_record(spans=()))
    assert features["log_candidate_spans"] == 0.0
    assert features["answer_fraction_covered"] == 0.0


def test_a_response_with_no_scored_tokens_is_not_an_error():
    record = make_record(n_tokens=0, spans=())
    record["token_probs"] = []
    record["answer_offsets"] = []
    features = features_from_record(record)
    assert features["log_answer_tokens"] == 0.0
    assert features["max_token_prob"] == 0.0


# --------------------------------------------------------------------------
# The depth statistic
# --------------------------------------------------------------------------


def test_tail_depth_is_half_at_the_median_and_small_at_the_extremes():
    values = np.arange(0.0, 100.0)
    middle, _ = _tail_depth(values, 49.5)
    low, low_pct = _tail_depth(values, -1000.0)
    high, high_pct = _tail_depth(values, 1000.0)
    assert middle == pytest.approx(0.5, abs=0.02)
    assert low == pytest.approx(0.0)
    assert high == pytest.approx(0.0)
    assert low_pct == pytest.approx(0.0)
    assert high_pct == pytest.approx(1.0)


def test_the_p_value_carries_the_same_finite_sample_correction_as_everything_else():
    # The smallest p-value a reference of n points can return is 1/(n+1) -- the
    # same (n+1) that appears in the conformal quantile. If this ever becomes
    # 1/n, the correction has been dropped somewhere.
    records = [make_record(seed=i, n_tokens=40 + i) for i in range(50)]
    reference = ExchangeabilityReference.build(records)
    absurd = features_from_record(
        make_record(n_tokens=5000, spans=((0, 9000, 2000),), answer_chars=9000)
    )
    assert reference.p_value(absurd) == pytest.approx(1.0 / (reference.n + 1))


# --------------------------------------------------------------------------
# The alarm, on real data
# --------------------------------------------------------------------------


@needs_dumps
def test_the_demo_record_never_trips_the_alarm():
    """A false alarm on the record the demo opens on is unacceptable.

    If this fails, do not relax the threshold to make it pass. Find out what
    changed about the record or the features first -- the demo record is a real
    RAGTruth response and it should be unremarkable by construction.
    """
    reference = ExchangeabilityReference.build(read(CALIB))
    demo = [r for r in read(TEST) if str(r["id"]) == DEMO_RECORD_ID]
    assert demo, f"record {DEMO_RECORD_ID} is not in the test dump"
    result = reference.check(features_from_record(demo[0]))
    assert result["in_distribution"], (
        f"the demo record tripped the alarm, p={result['p_value']:.4f}, "
        f"most unusual: {result['most_unusual']}"
    )
    # A canary, not the requirement. The requirement is the line above.
    #
    # The demo record sits at p = 0.053, five times the threshold rather than
    # fifty, and the reason is measured rather than mysterious: 42% of its
    # answer is highlighted, which is the 98.9th percentile of the calibration
    # split, and its mean token probability is the 98.5th. A record chosen to be
    # worth looking at is by construction a record the detector flagged heavily.
    # Three times the threshold is the bar for "not marginal"; if it ever drops
    # to that, something has changed and it needs looking at before the demo.
    assert result["p_value"] > 3 * DEFAULT_THRESHOLD, (
        "the demo record is only just clearing the threshold; it should be "
        f"comfortably ordinary, p={result['p_value']:.4f}"
    )


@needs_dumps
def test_an_answer_where_everything_is_highlighted_trips_the_alarm():
    # The failure mode the demo's own hand-written example shows: a short
    # out-of-distribution answer that the detector covers end to end.
    reference = ExchangeabilityReference.build(read(CALIB))
    saturated = make_record(
        n_tokens=45, spans=((0, 200, 45),), answer_chars=200, prob=0.95
    )
    result = reference.check(features_from_record(saturated))
    assert not result["in_distribution"]


@needs_dumps
def test_an_answer_far_longer_than_anything_in_the_corpus_trips_the_alarm():
    reference = ExchangeabilityReference.build(read(CALIB))
    enormous = make_record(n_tokens=6000, spans=(), answer_chars=30000)
    result = reference.check(features_from_record(enormous))
    assert not result["in_distribution"]
    assert result["most_unusual"] == "log_answer_tokens"


@needs_dumps
def test_the_threshold_behaves_like_a_false_alarm_rate_on_held_out_data():
    """The threshold claims to be a false-alarm rate. Check it on real data.

    A factor of two either way is acceptable -- the p-value is approximate,
    because the calibration depths are computed against a distribution that
    contains them and a new response is not in it. An order of magnitude is not.
    """
    reference = ExchangeabilityReference.build(read(CALIB))
    validation = reference.validate(read(TEST), label="test split")
    assert validation["n"] > 1000
    assert 0.2 * DEFAULT_THRESHOLD <= validation["measured_false_alarm_rate"] <= 2.5 * DEFAULT_THRESHOLD, (
        f"measured {validation['measured_false_alarm_rate']:.4f} against a "
        f"claimed {DEFAULT_THRESHOLD}"
    )


# --------------------------------------------------------------------------
# The artifact
# --------------------------------------------------------------------------


def test_the_reference_survives_a_json_round_trip():
    records = [make_record(seed=i, n_tokens=30 + i) for i in range(60)]
    reference = ExchangeabilityReference.build(records)
    restored = ExchangeabilityReference.from_dict(
        json.loads(json.dumps(reference.to_dict()))
    )
    probe = features_from_record(make_record(seed=999, n_tokens=52))
    assert restored.n == reference.n
    assert restored.p_value(probe) == pytest.approx(reference.p_value(probe))


def test_a_reference_missing_a_feature_is_refused_not_guessed():
    records = [make_record(seed=i) for i in range(20)]
    payload = ExchangeabilityReference.build(records).to_dict()
    payload["sorted_values"].pop("max_token_prob")
    with pytest.raises(ValueError, match="regenerated"):
        ExchangeabilityReference.from_dict(payload)


def test_building_from_an_empty_dump_is_refused():
    with pytest.raises(ValueError):
        ExchangeabilityReference.build([])


def test_check_reports_which_feature_was_most_unusual():
    records = [make_record(seed=i, n_tokens=40) for i in range(80)]
    reference = ExchangeabilityReference.build(records)
    result = reference.check(features_from_record(make_record(n_tokens=4000, seed=1)))
    assert result["most_unusual"] == "log_answer_tokens"
    named = {row["name"] for row in result["features"]}
    assert named == set(FEATURE_NAMES)
    assert all(row["label"] for row in result["features"]), "every feature needs a label"
