"""Layer 10: the compression arithmetic, and the policies it is compared against."""

from __future__ import annotations

import numpy as np
import pytest

from src.c4_attribution import evidence_conformal as cf
from src.c4_attribution.cost import curve, headline


def _rows(n=300, seed=0, responses_of=3):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        size = int(rng.integers(8, 24))
        gold = int(rng.integers(0, size))
        scores = rng.random(size)
        scores[gold] += 1.5
        out.append(
            cf.ConformalRow(f"r{i // responses_of}", "qa", cf.normalise_scores(scores), gold)
        )
    return out


def test_sentences_read_never_exceeds_the_pool():
    rows = curve(_rows(seed=1), _rows(seed=2))
    for row in rows:
        assert row["mean_sentences_per_span"] <= row["mean_pool_per_span"]
        assert 0.0 <= row["compression_per_span"] <= 1.0


def test_reduction_is_one_minus_compression():
    for row in curve(_rows(seed=3), _rows(seed=4)):
        assert row["reduction_in_reading_per_span"] == pytest.approx(
            1.0 - row["compression_per_span"]
        )


def test_a_reviewer_opens_each_sentence_once_per_response_not_once_per_span():
    rows = curve(_rows(seed=5), _rows(seed=6, responses_of=5))
    for row in rows:
        assert row["mean_sentences_per_response"] >= row["mean_sentences_per_span"]
        assert row["mean_sentences_per_response"] <= row["mean_pool_per_response"]


def test_the_two_naive_policies_are_priced_in_the_same_table():
    row = curve(_rows(seed=7), _rows(seed=8))[0]
    assert row["policy_one_guess_sentences_per_span"] == 1.0
    assert row["policy_return_everything_sentences_per_span"] == pytest.approx(
        row["mean_pool_per_span"]
    )


def test_reading_cost_falls_as_alpha_rises():
    rows = curve(_rows(seed=9), _rows(seed=10))
    sizes = [r["mean_sentences_per_span"] for r in rows]
    assert sizes == sorted(sizes, reverse=True)


def test_the_headline_sentence_carries_its_own_numbers():
    rows = curve(_rows(seed=11), _rows(seed=12))
    block = headline(rows, 0.10)
    assert block["available"] is True
    assert "reduction" in block["sentence"]
    assert f"{block['coverage']:.4f}" in block["sentence"]


def test_a_missing_alpha_says_so_rather_than_inventing_a_row():
    assert headline(curve(_rows(seed=13), _rows(seed=14)), 0.99)["available"] is False
