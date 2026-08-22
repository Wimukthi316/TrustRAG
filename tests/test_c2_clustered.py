"""Tests for the response-level uncertainty layer.

Two kinds of test here and they carry different weight.

The first kind checks that the fast arithmetic in `clustered.py` agrees exactly
with the readable reference implementations in `calibration.py` and
`conformal.py`. Those matter most: the fast versions exist only to make a
bootstrap affordable, and a fast version that quietly disagrees would put a
wrong interval beside a right number.

The second kind checks the resampling itself -- that responses, not spans, are
the unit being drawn, and that a draw of one row per response really does take
one row from each response.
"""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from src.c2_calibration.calibration import (  # noqa: E402
    expected_calibration_error,
)
from src.c2_calibration.clustered import (  # noqa: E402
    Units,
    _nonconformity,
    _one_per_response,
    _resample_responses,
    auroc,
    cluster_bootstrap,
    ece_fast,
    evaluate_at_threshold,
    extract,
    floor_rows,
    one_row_per_response,
    percentile_interval,
    quantile_from_sorted,
    select_method,
    unit_counts,
)
from src.c2_calibration.conformal import SplitConformal, conformal_quantile  # noqa: E402


def make_records(n_responses=60, seed=0):
    """Probability-dump records in exactly the shape evaluate_c1 writes.

    Span counts vary between responses, including responses with no predicted
    span at all, because that variation is the thing the cluster bootstrap has
    to reproduce and a fixture with a fixed span count would hide a bug in it.
    """
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_responses):
        n_spans = int(rng.integers(0, 5))
        n_tokens = int(rng.integers(4, 12))
        answer = "x" * (n_tokens * 2)
        spans = []
        for j in range(n_spans):
            label = bool(rng.random() < 0.5)
            # Overconfident scores, so calibration has something to correct.
            score = float(rng.beta(5, 2) if label else rng.beta(3, 3))
            spans.append(
                {
                    "start": 2 * j,
                    "end": 2 * j + 2,
                    "text": "xx",
                    "mean_prob": score,
                    "max_prob": min(1.0, score + 0.05),
                    "min_prob": max(0.0, score - 0.05),
                    "n_tokens": 1,
                    "is_hallucinated": label,
                }
            )
        records.append(
            {
                "id": f"r{i}",
                "task_type": ["qa", "data2text", "summarization"][i % 3],
                "model": "test",
                "answer": answer,
                "gold_spans": [{"start": 0, "end": 2, "text": "xx"}] if n_spans else [],
                "pred_spans": spans,
                "token_probs": [float(rng.random()) for _ in range(n_tokens)],
                "answer_offsets": [[2 * k, 2 * k + 2] for k in range(n_tokens)],
                "answer_truncated": False,
            }
        )
    return records


# --------------------------------------------------------------------------
# The fast arithmetic must equal the readable arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_ece_fast_equals_the_reference_implementation(seed):
    rng = np.random.default_rng(seed)
    scores = rng.random(500)
    labels = (rng.random(500) < scores).astype(int)
    assert ece_fast(scores, labels) == pytest.approx(
        expected_calibration_error(scores, labels), abs=1e-12
    )


def test_ece_fast_handles_scores_at_the_bin_edges():
    # 0.0 and 1.0 sit on the outer edges, and 1/15 sits exactly on an inner one.
    scores = np.array([0.0, 1.0, 1.0 / 15.0, 0.5])
    labels = np.array([0, 1, 1, 0])
    assert ece_fast(scores, labels) == pytest.approx(
        expected_calibration_error(scores, labels), abs=1e-12
    )


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.5, 0.9])
def test_quantile_from_sorted_equals_conformal_quantile(alpha):
    rng = np.random.default_rng(7)
    scores = rng.random(137)
    assert quantile_from_sorted(np.sort(scores), alpha) == conformal_quantile(
        scores, alpha
    )


def test_quantile_returns_infinity_when_calibration_is_too_small():
    # ceil((n+1)(1-alpha)) > n at n = 5, alpha = 0.05.
    assert math.isinf(quantile_from_sorted(np.sort(np.random.random(5)), 0.05))


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2, 0.4])
def test_evaluate_at_threshold_equals_split_conformal(alpha):
    records = make_records(seed=11)
    units = extract(records, "span")
    half = units.n_rows // 2
    conformal = SplitConformal(alpha=alpha).fit(
        units.scores[:half], units.labels[:half]
    )
    reference = conformal.evaluate(units.scores[half:], units.labels[half:])
    fast = evaluate_at_threshold(
        conformal.threshold, units.scores[half:], units.labels[half:]
    )
    for key in ("empirical_coverage", "abstention_rate", "empty_set_rate", "flag_rate"):
        assert fast[key] == pytest.approx(reference[key])


def test_nonconformity_is_one_minus_probability_of_the_true_label():
    scores = np.array([0.3, 0.8])
    labels = np.array([1, 0])
    assert _nonconformity(scores, labels).tolist() == pytest.approx([0.7, 0.8])


# --------------------------------------------------------------------------
# The grouping, and the unit being resampled
# --------------------------------------------------------------------------


def test_extract_groups_every_row_exactly_once_and_in_order():
    records = make_records(seed=3)
    units = extract(records, "span")
    concatenated = np.concatenate([rows for rows in units.index if rows.size])
    assert concatenated.tolist() == list(range(units.n_rows))
    assert units.n_responses == len(records)
    assert unit_counts(records, "span") == [len(r["pred_spans"]) for r in records]


def test_extract_keeps_responses_that_produced_no_span():
    records = make_records(seed=3)
    empty = [i for i, r in enumerate(records) if not r["pred_spans"]]
    assert empty, "fixture should contain at least one span-less response"
    units = extract(records, "span")
    for i in empty:
        assert units.index[i].size == 0


def test_token_extraction_matches_the_token_count():
    records = make_records(seed=5)
    units = extract(records, "token")
    assert units.n_rows == sum(len(r["token_probs"]) for r in records)


def test_resampling_draws_whole_responses_not_spans():
    # Every response here contributes exactly two spans, so any sample drawn at
    # the response level must have an even number of rows and must never contain
    # a lone member of a pair. Drawing spans independently would break both.
    index = [np.array([2 * i, 2 * i + 1]) for i in range(20)]
    rng = np.random.default_rng(0)
    for _ in range(50):
        rows = _resample_responses(index, rng)
        assert rows.size == 40
        pairs = rows.reshape(-1, 2)
        assert np.all(pairs[:, 1] == pairs[:, 0] + 1)
        assert np.all(pairs[:, 0] % 2 == 0)


def test_one_per_response_takes_exactly_one_row_from_each_nonempty_response():
    records = make_records(seed=9)
    units = extract(records, "span")
    rng = np.random.default_rng(0)
    picked = _one_per_response(units.index, rng)
    nonempty = [rows for rows in units.index if rows.size]
    assert picked.size == len(nonempty)
    for row, rows in zip(picked, nonempty):
        assert row in rows


def test_percentile_interval_reports_infinities_separately():
    result = percentile_interval([0.1, 0.2, 0.3, float("inf")])
    assert result["n_finite"] == 3
    assert result["n_infinite"] == 1
    assert result["ci_low"] <= 0.2 <= result["ci_high"]


# --------------------------------------------------------------------------
# The three analyses
# --------------------------------------------------------------------------


def test_cluster_bootstrap_interval_brackets_the_point_estimate():
    records = make_records(n_responses=200, seed=21)
    calib = extract(records[:100], "span")
    test = extract(records[100:], "span")
    method, _ = select_method(calib, test)
    result = cluster_bootstrap(
        calib, test, method, alphas=(0.1, 0.2), resamples=80, seed=1
    )
    for row in result["coverage"]:
        assert row["ci_low"] <= row["bootstrap_mean"] <= row["ci_high"]
        assert row["ci_width"] > 0
    assert result["ece_before"]["ci_low"] <= result["ece_before"]["ci_high"]


def test_the_test_only_bootstrap_is_narrower_than_the_full_procedure():
    # Refitting the calibrator and the threshold on a resampled calibration set
    # adds a source of variation, so the full-procedure interval cannot be the
    # narrower of the two. If it ever is, the calibration resample is not
    # actually reaching the conformal threshold.
    records = make_records(n_responses=240, seed=31)
    calib = extract(records[:120], "span")
    test = extract(records[120:], "span")
    method, _ = select_method(calib, test)
    full = cluster_bootstrap(
        calib, test, method, alphas=(0.1,), resamples=120, seed=2,
        refit_calibration=True,
    )
    frozen = cluster_bootstrap(
        calib, test, method, alphas=(0.1,), resamples=120, seed=2,
        refit_calibration=False,
    )
    assert full["coverage"][0]["ci_width"] >= frozen["coverage"][0]["ci_width"]


def test_one_row_per_response_uses_one_calibration_row_per_response():
    records = make_records(n_responses=200, seed=41)
    calib = extract(records[:100], "span")
    test = extract(records[100:], "span")
    method, _ = select_method(calib, test)
    result = one_row_per_response(
        calib, test, method, alphas=(0.1, 0.2), draws=20, seed=3
    )
    assert result["n_calibration_rows_per_draw"] == sum(
        1 for rows in calib.index if rows.size
    )
    assert result["n_calibration_rows_per_draw"] < result["n_calibration_rows_pooled"]
    for row in result["rows"]:
        assert 0.0 <= row["subsampled_mean_coverage"] <= 1.0
        assert row["pooled_threshold"] == pytest.approx(
            row["pooled_threshold"]
        )  # finite, not NaN


def test_floor_row_is_uninformative_but_well_calibrated():
    records = make_records(n_responses=300, seed=51)
    calib = extract(records[:150], "span")
    test = extract(records[150:], "span")
    method, _ = select_method(calib, test)
    result = floor_rows(calib, test, method)
    constant = result["constant_base_rate"]
    # A single repeated score cannot rank anything, so AUROC is exactly chance.
    assert constant["auroc"] == pytest.approx(0.5)
    # And the two ECE implementations agree on it.
    assert constant["ece"] == pytest.approx(
        constant["ece_reference_implementation"], abs=1e-12
    )
    # Its ECE is just the gap between the two splits' positive rates.
    assert constant["ece"] == pytest.approx(
        abs(result["calibration_positive_rate"] - result["test_positive_rate"]),
        abs=1e-9,
    )


def test_a_monotone_calibrator_cannot_move_auroc():
    records = make_records(n_responses=300, seed=61)
    calib = extract(records[:150], "span")
    test = extract(records[150:], "span")
    result = floor_rows(calib, test, "platt")
    assert result["detector"]["ranking_preserved"]


def test_auroc_is_nan_when_one_class_is_missing():
    assert math.isnan(auroc(np.array([0.1, 0.9]), np.array([1, 1])))


def test_units_dataclass_reports_its_own_sizes():
    units = Units(
        scores=np.array([0.1, 0.2, 0.3]),
        labels=np.array([0, 1, 0]),
        groups=["qa", "qa", "qa"],
        index=[np.array([0, 1]), np.array([2])],
    )
    assert units.n_rows == 3
    assert units.n_responses == 2
