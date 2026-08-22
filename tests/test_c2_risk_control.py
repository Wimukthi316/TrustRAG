"""Tests for conformal risk control.

The arithmetic that matters is the finite-sample correction. (n R + B)/(n + 1)
is not the same as R, and the difference is what makes the bound hold at the n we
actually have. It is checked against a hand-computed value rather than against
itself.

The other thing worth a test is the guard: a rule that satisfies any
false-negative bound by flagging the entire answer is not a detector, and the
suite refuses to let one be reported as a win.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from src.c2_calibration.risk_control import (  # noqa: E402
    choose_threshold,
    evaluate_threshold,
    false_negative_matrix,
    flag_rates,
    gold_token_mask,
    positive_probabilities,
    run_risk_control,
)


def make_record(probs, gold_token_indices, record_id="r0"):
    """A probability record whose gold spans cover exactly the named tokens.

    Tokens are two characters wide and contiguous, so a gold span over token k
    is the character range [2k, 2k + 2).
    """
    offsets = [[2 * i, 2 * i + 2] for i in range(len(probs))]
    gold = [
        {"start": 2 * i, "end": 2 * i + 2, "text": "xx"} for i in gold_token_indices
    ]
    return {
        "id": record_id,
        "task_type": "qa",
        "model": "test",
        "answer": "x" * (2 * len(probs)),
        "gold_spans": gold,
        "pred_spans": [],
        "token_probs": list(probs),
        "answer_offsets": offsets,
        "answer_truncated": False,
    }


# --------------------------------------------------------------------------
# The loss
# --------------------------------------------------------------------------


def test_gold_token_mask_marks_exactly_the_covered_tokens():
    record = make_record([0.1, 0.2, 0.3, 0.4], [1, 3])
    assert gold_token_mask(record).tolist() == [False, True, False, True]


def test_a_record_with_no_gold_spans_has_no_positive_tokens():
    record = make_record([0.1, 0.9], [])
    assert not gold_token_mask(record).any()
    _, n_positive, _ = positive_probabilities([record])
    assert n_positive.tolist() == [0]


def test_false_negative_rate_counts_positives_below_the_threshold():
    # Positive tokens score 0.2, 0.4 and 0.8.
    record = make_record([0.2, 0.4, 0.8, 0.05], [0, 1, 2])
    sorted_positives, _, _ = positive_probabilities([record])
    grid = np.array([0.0, 0.3, 0.5, 0.9])
    losses = false_negative_matrix(sorted_positives, grid)[0]
    #   t=0.0 -> nothing missed; t=0.3 -> the 0.2 missed; t=0.5 -> 0.2 and 0.4;
    #   t=0.9 -> all three.
    assert losses.tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])


def test_a_token_exactly_on_the_threshold_counts_as_flagged():
    record = make_record([0.5], [0])
    sorted_positives, _, _ = positive_probabilities([record])
    losses = false_negative_matrix(sorted_positives, np.array([0.5]))[0]
    assert losses[0] == 0.0


def test_a_clean_response_contributes_a_zero_row():
    records = [make_record([0.9, 0.9], []), make_record([0.1], [0])]
    sorted_positives, n_positive, _ = positive_probabilities(records)
    losses = false_negative_matrix(sorted_positives, np.array([0.5]))
    assert losses[0, 0] == 0.0
    assert losses[1, 0] == 1.0
    assert n_positive.tolist() == [0, 1]


def test_positive_probabilities_rejects_a_length_mismatch():
    record = make_record([0.1, 0.2], [0])
    record["token_probs"] = [0.1]
    with pytest.raises(ValueError):
        positive_probabilities([record])


# --------------------------------------------------------------------------
# The flag cost
# --------------------------------------------------------------------------


def test_flag_rates_count_tokens_and_responses():
    records = [make_record([0.1, 0.9], [0]), make_record([0.2, 0.3], [1])]
    grid = np.array([0.0, 0.5, 1.01])
    rates = flag_rates(records, grid)
    assert rates["n_tokens"] == 4
    assert rates["token_flag_rate"].tolist() == pytest.approx([1.0, 0.25, 0.0])
    assert rates["response_flag_rate"].tolist() == pytest.approx([1.0, 0.5, 0.0])


# --------------------------------------------------------------------------
# The finite-sample correction
# --------------------------------------------------------------------------


def test_the_correction_is_not_the_plain_mean():
    # Four responses, all with loss 0.10 at this threshold. R = 0.10, but the
    # corrected risk is (4 * 0.10 + 1) / 5 = 0.28, so alpha = 0.20 must reject
    # a threshold that the uncorrected mean would have accepted.
    losses = np.full((4, 1), 0.10)
    thresholds = np.array([0.5])
    assert choose_threshold(losses, thresholds, alpha=0.30)["feasible"]
    strict = choose_threshold(losses, thresholds, alpha=0.20)
    assert not strict["feasible"]
    assert strict["smallest_corrected_risk"] == pytest.approx(0.28)


def test_the_correction_floor_makes_small_alphas_unreachable():
    # With n = 4 the correction alone contributes B / (n + 1) = 0.20, so no
    # threshold can ever satisfy alpha = 0.10 however good the detector is.
    losses = np.zeros((4, 3))
    result = choose_threshold(losses, np.array([0.1, 0.5, 0.9]), alpha=0.10)
    assert not result["feasible"]
    assert result["correction_floor"] == pytest.approx(0.20)


def test_the_largest_feasible_threshold_is_chosen():
    # Loss rises with the threshold, so the bound holds on a prefix of the grid
    # and the answer is the last index of that prefix.
    thresholds = np.array([0.1, 0.2, 0.3, 0.4])
    losses = np.array([[0.0, 0.0, 0.5, 0.9]] * 100)
    result = choose_threshold(losses, thresholds, alpha=0.10)
    assert result["feasible"]
    assert result["threshold"] == pytest.approx(0.2)
    assert not result["on_grid_edge"]


def test_a_threshold_on_the_grid_edge_is_reported_as_such():
    thresholds = np.array([0.1, 0.2, 0.3])
    losses = np.array([[0.0, 0.9, 0.9]] * 100)
    result = choose_threshold(losses, thresholds, alpha=0.05)
    assert result["threshold"] == pytest.approx(0.1)
    assert result["on_grid_edge"]
    assert "grid" in result["grid_edge_warning"]


# --------------------------------------------------------------------------
# End to end, and the guard
# --------------------------------------------------------------------------


def _corpus(n, seed):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n):
        length = int(rng.integers(6, 14))
        probs = rng.beta(2, 5, size=length)
        positives = sorted(rng.choice(length, size=int(rng.integers(0, 4)), replace=False))
        # Hallucinated tokens score higher, but not by much -- which is the real
        # situation and the reason this block is expected to be expensive.
        for index in positives:
            probs[index] = float(np.clip(probs[index] + 0.35, 0.0, 1.0))
        records.append(make_record(probs.tolist(), positives, f"r{i}"))
    return records


def test_end_to_end_produces_a_row_per_alpha_with_its_flag_cost():
    calib, test = _corpus(400, seed=1), _corpus(400, seed=2)
    result = run_risk_control(calib, test, alphas=(0.10, 0.30))
    assert len(result["rows"]) == 2
    for row in result["rows"]:
        if not row["chosen"]["feasible"]:
            continue
        assert 0.0 <= row["test"]["token_flag_rate"] <= 1.0
        assert row["test"]["n_responses"] == 400


def test_a_rule_that_flags_almost_everything_is_not_a_result():
    # The guard C1's D1 sweep needed. If satisfying the bound costs flagging
    # over 90% of tokens, the rule has converged on "highlight the whole answer"
    # and its risk number says nothing about the detector.
    calib, test = _corpus(300, seed=3), _corpus(300, seed=4)
    result = run_risk_control(calib, test, alphas=(0.05, 0.10, 0.20, 0.40))
    for row in result["rows"]:
        if not row["chosen"]["feasible"]:
            continue
        flagged = row["test"]["token_flag_rate"]
        if flagged > 0.90:
            assert row["chosen"]["threshold"] < 0.05, (
                "a rule flagging over 90% of tokens should only arise from a "
                "threshold at the bottom of the grid; if it arises elsewhere the "
                "loss is being computed wrongly"
            )


def test_flag_everything_has_zero_false_negatives():
    calib, test = _corpus(200, seed=5), _corpus(200, seed=6)
    result = run_risk_control(calib, test, alphas=(0.10,))
    baseline = result["flag_everything_baseline"]
    assert baseline["fnr_over_responses_with_hallucinations"] == pytest.approx(0.0)
    assert baseline["token_flag_rate"] == pytest.approx(1.0)


def test_evaluate_threshold_separates_the_two_conventions():
    records = [make_record([0.9, 0.9], []), make_record([0.1], [0])]
    result = evaluate_threshold(records, 0.5)
    # One clean response and one that misses its only hallucinated token.
    assert result["fnr_over_all_responses"] == pytest.approx(0.5)
    assert result["fnr_over_responses_with_hallucinations"] == pytest.approx(1.0)
    assert result["n_responses_with_hallucinations"] == 1
