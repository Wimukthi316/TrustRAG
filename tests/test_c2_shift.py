"""Tests for the shift study.

The single most important property here is that both weighted methods reduce
*exactly* to plain split conformal when every weight is equal. Weighted
conformal is easy to get subtly wrong -- the test point's own weight belongs in
the denominator, the class-size ratio has to be divided back out of a
classifier's odds, and either mistake produces a threshold that looks sensible
and is not. If the uniform-weight case reproduces `conformal_quantile` to the
bit, the fencepost is right.

The rest checks that the diagnosis can recognise a shift it was handed on
purpose, and that it refuses to report an estimate it cannot make.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from src.c2_calibration.conformal import conformal_quantile  # noqa: E402
from src.c2_calibration.shift import (  # noqa: E402
    FEATURE_NAMES,
    build_span_table,
    clip_weights,
    covariate_shift_conformal,
    estimate_target_priors,
    fit_domain_classifier,
    label_shift_conformal,
    split_target,
    summarise_decisions,
    unweighted_transfer,
    weighted_thresholds,
)


def make_records(n=80, subset="covidqa", positive_rate=0.5, shift=0.0, seed=0):
    """Probability-dump records with a controllable span positive rate.

    `shift` moves the score distribution of both classes together, so a caller
    can build pure label shift (shift = 0, different positive_rate) and separate
    it from covariate shift (same positive_rate, non-zero shift). The diagnosis
    is supposed to tell those two apart, and it cannot be tested on data where
    they are tangled.
    """
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        n_spans = int(rng.integers(1, 4))
        n_tokens = int(rng.integers(6, 20))
        spans = []
        for j in range(n_spans):
            label = bool(rng.random() < positive_rate)
            centre = (0.75 if label else 0.35) + shift
            score = float(np.clip(rng.normal(centre, 0.10), 0.01, 0.99))
            spans.append(
                {
                    "start": 2 * j,
                    "end": 2 * j + 2,
                    "text": "xx",
                    "mean_prob": score,
                    "max_prob": min(0.99, score + 0.02),
                    "min_prob": max(0.01, score - 0.02),
                    "n_tokens": 1,
                    "is_hallucinated": label,
                }
            )
        records.append(
            {
                "id": f"{subset}-{i}",
                "task_type": "qa",
                "subset": subset,
                "model": "test",
                "answer": "x" * (2 * n_tokens),
                "gold_spans": [{"start": 0, "end": 2, "text": "xx"}],
                "pred_spans": spans,
                "token_probs": [float(rng.random()) for _ in range(n_tokens)],
                "answer_offsets": [[2 * k, 2 * k + 2] for k in range(n_tokens)],
                "answer_truncated": False,
            }
        )
    return records


# --------------------------------------------------------------------------
# The weighted quantile must reduce to the unweighted one
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.3, 0.5])
def test_uniform_weights_reproduce_the_conformal_quantile(alpha):
    rng = np.random.default_rng(4)
    scores = np.sort(rng.random(211))
    weights = np.ones_like(scores)
    result = weighted_thresholds(
        scores, np.cumsum(weights), float(weights.sum()), np.array([1.0]), alpha
    )
    assert result[0] == conformal_quantile(scores, alpha)


def test_the_weighted_quantile_is_infinite_when_the_weights_cannot_reach():
    scores = np.sort(np.random.random(5))
    weights = np.ones_like(scores)
    result = weighted_thresholds(
        scores, np.cumsum(weights), float(weights.sum()), np.array([1.0]), 0.05
    )
    assert math.isinf(result[0])


def test_a_heavier_test_point_gets_a_higher_threshold():
    # More mass reserved for the test point means the calibration scores have to
    # reach further to cover 1 - alpha, so the threshold can only rise.
    scores = np.sort(np.random.default_rng(0).random(200))
    weights = np.ones_like(scores)
    thresholds = weighted_thresholds(
        scores, np.cumsum(weights), float(weights.sum()), np.array([0.5, 1.0, 10.0]), 0.1
    )
    assert thresholds[0] <= thresholds[1] <= thresholds[2]


def test_label_shift_with_unit_weights_equals_plain_split_conformal():
    source = build_span_table(make_records(seed=1), group_key="subset")
    target = build_span_table(make_records(seed=2, subset="finqa"), group_key="subset")
    nonconformity = np.where(source.labels == 1, 1 - source.scores, source.scores)
    plain = unweighted_transfer(
        np.sort(nonconformity),
        target.scores,
        target.labels,
        target.groups,
        alphas=(0.1, 0.2),
    )
    weighted = label_shift_conformal(
        nonconformity,
        source.labels,
        [1.0, 1.0],
        target.scores,
        target.labels,
        target.groups,
        alphas=(0.1, 0.2),
    )
    for left, right in zip(plain, weighted):
        assert left["threshold"] == right["threshold_label_0"]
        assert left["threshold"] == right["threshold_label_1"]
        assert left["overall"]["empirical_coverage"] == pytest.approx(
            right["overall"]["empirical_coverage"]
        )


def test_covariate_shift_with_unit_weights_equals_plain_split_conformal():
    source = build_span_table(make_records(seed=3), group_key="subset")
    target = build_span_table(make_records(seed=4, subset="finqa"), group_key="subset")
    nonconformity = np.where(source.labels == 1, 1 - source.scores, source.scores)
    plain = unweighted_transfer(
        np.sort(nonconformity),
        target.scores,
        target.labels,
        target.groups,
        alphas=(0.1, 0.2),
    )
    weighted = covariate_shift_conformal(
        nonconformity,
        np.ones(source.n),
        target.scores,
        np.ones(target.n),
        target.labels,
        target.groups,
        alphas=(0.1, 0.2),
    )
    for left, right in zip(plain, weighted):
        assert right["threshold_mean"] == pytest.approx(left["threshold"])
        assert left["overall"]["empirical_coverage"] == pytest.approx(
            right["overall"]["empirical_coverage"]
        )


# --------------------------------------------------------------------------
# Decisions
# --------------------------------------------------------------------------


def test_summarise_decisions_counts_the_four_set_shapes():
    keep_zero = np.array([True, False, True, False])
    keep_one = np.array([False, True, True, False])
    labels = np.array([0, 1, 1, 1])
    result = summarise_decisions(keep_zero, keep_one, labels)["overall"]
    # rows 0, 1 and 2 are covered; row 3 has an empty set and covers nothing.
    assert result["empirical_coverage"] == pytest.approx(0.75)
    assert result["abstention_rate"] == pytest.approx(0.5)  # one {0,1}, one {}
    assert result["empty_set_rate"] == pytest.approx(0.25)
    assert result["flag_rate"] == pytest.approx(0.25)


def test_per_group_blocks_partition_the_test_set():
    source = build_span_table(
        make_records(n=40, subset="a", seed=5) + make_records(n=40, subset="b", seed=6),
        group_key="subset",
    )
    result = summarise_decisions(
        np.ones(source.n, dtype=bool),
        np.zeros(source.n, dtype=bool),
        source.labels,
        source.groups,
    )
    assert sorted(result["per_group"]) == ["a", "b"]
    assert sum(block["n"] for block in result["per_group"].values()) == source.n


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------


def test_span_table_carries_one_feature_row_per_span():
    records = make_records(n=25, seed=7)
    table = build_span_table(records, group_key="subset")
    assert table.features.shape == (table.n, len(FEATURE_NAMES))
    assert table.response_of_row.size == table.n
    assert table.response_of_row.max() < len(records)


def test_domain_classifier_is_at_chance_on_two_identical_corpora():
    # Same generator, different seed: nothing distinguishes them but noise.
    left = build_span_table(make_records(n=200, seed=11), group_key="subset")
    right = build_span_table(make_records(n=200, seed=12), group_key="subset")
    classifier = fit_domain_classifier(left, right, seed=0)
    assert classifier.held_out_auc == pytest.approx(0.5, abs=0.12)


def test_domain_classifier_finds_a_score_distribution_that_moved():
    left = build_span_table(make_records(n=200, seed=13), group_key="subset")
    right = build_span_table(
        make_records(n=200, seed=14, shift=0.20), group_key="subset"
    )
    classifier = fit_domain_classifier(left, right, seed=0)
    assert classifier.held_out_auc > 0.70


def test_bbse_recovers_a_prior_it_was_shifted_by():
    # Pure label shift by construction: identical score distributions per class,
    # only the mixing proportion differs.
    source = build_span_table(
        make_records(n=600, positive_rate=0.60, seed=15), group_key="subset"
    )
    target = build_span_table(
        make_records(n=600, positive_rate=0.20, seed=16, subset="finqa"),
        group_key="subset",
    )
    result = estimate_target_priors(
        source.scores, source.labels, target.scores, target.labels
    )
    assert result["usable"], result["problems"]
    assert result["estimated_target_prior"][1] == pytest.approx(
        target.positive_rate, abs=0.05
    )


def test_bbse_refuses_a_constant_predictor():
    # This is the real failure mode on span data: every predicted span already
    # scores above 0.5, so a raw-score predictor says 1 to all of them, the
    # confusion matrix has a zero row and there is nothing to invert.
    source = build_span_table(make_records(n=300, seed=17), group_key="subset")
    target = build_span_table(
        make_records(n=300, seed=18, subset="finqa"), group_key="subset"
    )
    source.scores[:] = 0.9
    target.scores[:] = 0.9
    result = estimate_target_priors(
        source.scores, source.labels, target.scores, target.labels
    )
    assert not result["usable"]
    assert result["source_predicted_positive_rate"] == pytest.approx(1.0)
    assert any("constant" in problem for problem in result["problems"])


def test_bbse_refuses_when_the_predictor_carries_no_information():
    # Scores are pure noise, so the confusion matrix is close to singular and
    # the target's predicted-label distribution says little about its true one.
    rng = np.random.default_rng(0)
    source = build_span_table(make_records(n=300, seed=19), group_key="subset")
    target = build_span_table(
        make_records(n=300, seed=20, subset="finqa"), group_key="subset"
    )
    source.scores[:] = rng.random(source.n)
    target.scores[:] = rng.random(target.n)
    result = estimate_target_priors(
        source.scores, source.labels, target.scores, target.labels
    )
    assert not result["usable"] or abs(
        result["estimation_error_on_positive_rate"]
    ) < 0.5


# --------------------------------------------------------------------------
# Weights and splitting
# --------------------------------------------------------------------------


def test_clip_weights_reports_what_it_clipped():
    weights = np.array([0.001, 1.0, 5.0, 1000.0])
    result = clip_weights(weights, limit=20.0)
    assert result["fraction_at_ceiling"] == pytest.approx(0.25)
    assert result["fraction_at_floor"] == pytest.approx(0.25)
    assert result["max"] == pytest.approx(20.0)
    assert result["min"] == pytest.approx(0.05)
    # One weight of 20 against three small ones: far fewer than four effective points.
    assert result["effective_sample_size"] < 4.0


def test_effective_sample_size_is_n_when_every_weight_is_equal():
    result = clip_weights(np.ones(50))
    assert result["effective_sample_size"] == pytest.approx(50.0)
    assert result["fraction_clipped"] == pytest.approx(0.0)


def test_target_split_is_disjoint_and_covers_every_subset():
    records = make_records(n=60, subset="a", seed=19) + make_records(
        n=60, subset="b", seed=20
    )
    estimation, evaluation = split_target(records, fraction=0.5, seed=42)
    assert len(estimation) + len(evaluation) == len(records)
    left = {r["id"] for r in estimation}
    right = {r["id"] for r in evaluation}
    assert not (left & right)
    assert {r["subset"] for r in estimation} == {"a", "b"}
    assert {r["subset"] for r in evaluation} == {"a", "b"}


def test_target_split_is_reproducible():
    records = make_records(n=40, seed=21)
    first, _ = split_target(records, seed=42)
    second, _ = split_target(records, seed=42)
    assert [r["id"] for r in first] == [r["id"] for r in second]


def test_target_split_keeps_both_answers_to_one_question_on_the_same_side():
    # RAGBench pairs two model answers per question under one id. If the split
    # separated them, the half the weights are estimated on and the half they
    # are judged on would share a question and a context, and the repair would
    # be scored on data it had already partly seen.
    records = []
    for model in ("gpt-3.5-turbo-0125", "claude-3-haiku-20240307"):
        for record in make_records(n=50, seed=22, subset="delucionqa"):
            records.append({**record, "model": model})

    estimation, evaluation = split_target(records, fraction=0.5, seed=42)
    left = {r["id"] for r in estimation}
    right = {r["id"] for r in evaluation}
    assert not (left & right), "a question straddled the split"
    assert len(estimation) + len(evaluation) == len(records)
    # Both models are present on both sides, so grouping by question has not
    # accidentally turned into grouping by model.
    assert {r["model"] for r in estimation} == {r["model"] for r in evaluation}
    # Every question contributed both of its responses to whichever side it fell on.
    for half in (estimation, evaluation):
        counts = {}
        for record in half:
            counts[record["id"]] = counts.get(record["id"], 0) + 1
        assert set(counts.values()) == {2}


def test_target_split_rejects_a_fraction_outside_the_unit_interval():
    records = make_records(n=10, seed=23)
    with pytest.raises(ValueError):
        split_target(records, fraction=0.0)
    with pytest.raises(ValueError):
        split_target(records, fraction=1.0)
