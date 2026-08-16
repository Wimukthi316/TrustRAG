"""Tests for the RAGBench operating-point adaptation.

The thing that would quietly ruin this analysis is choosing the threshold on the
data it is then scored on, so the split and the "chosen on calibration only"
property are what most of these tests are about.
"""

from __future__ import annotations

import pytest

from src.c1_detector.ood_operating_point import (
    argmax_rule,
    build_report,
    choose_threshold,
    format_report,
    per_subset,
    score_rule,
    split_by_subset,
    sweep,
    threshold_rule,
    trivial_f1,
)


def row(subset, rid, gold, prob, argmax=None):
    return {
        "subset": subset,
        "id": rid,
        "gold_positive": gold,
        "argmax_positive": (prob >= 0.5) if argmax is None else argmax,
        "max_token_prob": prob,
        "mean_token_prob": prob / 2,
        "n_answer_tokens": 50,
        "n_pred_spans": 1 if prob >= 0.5 else 0,
    }


@pytest.fixture
def corpus():
    """Two subsets with very different base rates, as RAGBench has."""
    rows = []
    for i in range(100):  # 20% positive, positives score higher
        gold = i % 5 == 0
        rows.append(row("common", f"c{i}", gold, 0.8 if gold else 0.2))
    for i in range(40):  # 50% positive
        gold = i % 2 == 0
        rows.append(row("rare", f"r{i}", gold, 0.7 if gold else 0.3))
    return rows


def test_trivial_baseline_matches_the_formula():
    assert trivial_f1(0.0) == 0.0
    assert trivial_f1(1.0) == pytest.approx(1.0)
    assert trivial_f1(0.142) == pytest.approx(2 * 0.142 / 1.142)


def test_a_detector_that_says_yes_to_everything_scores_the_trivial_baseline(corpus):
    result = score_rule(corpus, lambda r: True)
    assert result["f1"] == pytest.approx(result["trivial_f1"])
    assert result["clears_trivial"] is False


def test_split_keeps_every_subset_on_both_sides(corpus):
    calib, test = split_by_subset(corpus)
    assert {r["subset"] for r in calib} == {"common", "rare"}
    assert {r["subset"] for r in test} == {"common", "rare"}


def test_split_is_disjoint_and_complete(corpus):
    calib, test = split_by_subset(corpus)
    calib_ids = {r["id"] for r in calib}
    test_ids = {r["id"] for r in test}
    assert not (calib_ids & test_ids)
    assert calib_ids | test_ids == {r["id"] for r in corpus}


def test_split_respects_the_requested_fraction(corpus):
    calib, test = split_by_subset(corpus, calib_fraction=0.25)
    assert len(calib) == pytest.approx(0.25 * len(corpus), abs=2)
    assert len(test) + len(calib) == len(corpus)


def test_split_is_reproducible(corpus):
    first = [r["id"] for r in split_by_subset(corpus, seed=7)[0]]
    second = [r["id"] for r in split_by_subset(corpus, seed=7)[0]]
    assert first == second


def test_a_different_seed_gives_a_different_split(corpus):
    first = [r["id"] for r in split_by_subset(corpus, seed=1)[0]]
    second = [r["id"] for r in split_by_subset(corpus, seed=2)[0]]
    assert first != second


def test_an_impossible_fraction_is_refused(corpus):
    with pytest.raises(ValueError):
        split_by_subset(corpus, calib_fraction=0.0)
    with pytest.raises(ValueError):
        split_by_subset(corpus, calib_fraction=1.0)


def test_a_lower_threshold_never_lowers_recall(corpus):
    rows = sweep(corpus, thresholds=[0.1, 0.5, 0.9])
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)


def test_the_threshold_is_chosen_on_the_data_it_was_given(corpus):
    """Perfectly separable at 0.5, so the sweep has to find it."""
    clean = [row("s", str(i), i % 2 == 0, 0.9 if i % 2 == 0 else 0.1) for i in range(20)]
    chosen = choose_threshold(clean, thresholds=[0.2, 0.5, 0.95])
    assert chosen["chosen"]["f1"] == pytest.approx(1.0)
    assert chosen["chosen"]["threshold"] in (0.2, 0.5)


def test_ties_break_toward_the_more_sensitive_setting():
    clean = [row("s", str(i), i % 2 == 0, 0.9 if i % 2 == 0 else 0.1) for i in range(20)]
    chosen = choose_threshold(clean, thresholds=[0.2, 0.5])
    assert chosen["chosen"]["threshold"] == 0.2


def test_the_report_never_chooses_on_the_test_half(corpus):
    """The threshold must come from calibration, even when test prefers another."""
    report = build_report(corpus, calib_fraction=0.3, seed=42)
    calib, test = split_by_subset(corpus, 0.3, 42)
    best_on_test = max(sweep(test), key=lambda r: (r["f1"], -r["threshold"]))
    chosen_on_calib = choose_threshold(calib)["chosen"]["threshold"]
    assert report["chosen_threshold"] == chosen_on_calib
    assert report["test"]["adapted"]["f1"] <= best_on_test["f1"] + 1e-9


def test_report_scores_both_rules_on_the_same_test_half(corpus):
    report = build_report(corpus)
    assert report["test"]["argmax"]["n"] == report["test"]["adapted"]["n"]
    assert report["n_calibration"] + report["n_test"] == len(corpus)


def test_report_keeps_the_whole_corpus_argmax_row_for_comparison(corpus):
    report = build_report(corpus)
    assert report["whole_corpus_argmax"]["n"] == len(corpus)
    assert report["whole_corpus_argmax"] == score_rule(corpus, argmax_rule)


def test_per_subset_totals_add_up(corpus):
    breakdown = per_subset(corpus, threshold_rule(0.5))
    assert sum(v["n"] for v in breakdown.values()) == len(corpus)


def test_the_printed_report_states_the_baseline_verdict(corpus):
    text = format_report(build_report(corpus))
    assert "threshold chosen on calibration" in text
    assert "do-nothing baseline" in text
    assert "TEST half, adapted" in text


def test_a_rule_that_flags_everything_is_visible_as_such(corpus):
    result = score_rule(corpus, lambda r: True)
    assert result["flagged_rate"] == pytest.approx(1.0)
    assert result["margin_over_trivial_points"] == pytest.approx(0.0, abs=1e-9)


def test_the_margin_over_trivial_is_reported_in_points(corpus):
    result = score_rule(corpus, threshold_rule(0.5))
    expected = 100 * (result["f1"] - result["trivial_f1"])
    assert result["margin_over_trivial_points"] == pytest.approx(expected)


def test_the_report_refuses_to_call_a_near_total_flag_rule_a_win():
    """Barely beating the floor while saying yes to everything is not a win."""
    rows = [row("s", str(i), i % 7 == 0, 0.99) for i in range(140)]
    text = format_report(build_report(rows))
    assert "EFFECTIVELY THE TRIVIAL CLASSIFIER" in text
    assert "never as clearing it" in text
