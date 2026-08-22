"""Honest uncertainty for C2, measured at the unit the corpus actually samples.

Every C2 number reported so far treats predicted spans as independent. They are
not. RAGTruth samples **responses**; the spans inside one response come from one
answer written by one model about one context, and one bad sentence produces a
run of hallucinated spans together. Two spans from the same answer carry far
less than two spans' worth of information.

`conformal.py` already admits this in caveat 3 and calls the noise band
optimistic. This module replaces the admission with a number.

Three things are measured here.

**B1 -- cluster bootstrap.** Resample *responses* with replacement, never spans,
and recompute everything on the resampled corpus. Two variants, and the
difference between them is worth understanding rather than picking one:

    test-only     the calibrator and the conformal threshold stay exactly as
                  fitted on the real calibration split; only the test corpus is
                  resampled. This answers "how well do 2,390 spans pin down the
                  number I am reporting?"
    full          the calibration split is resampled too and the calibrator and
                  threshold are refitted on it every draw. This answers "if I
                  ran this whole procedure again on fresh data, how much would
                  the answer move?" It is always the wider of the two, and it is
                  the one that belongs beside a claim about the method rather
                  than about this particular fitted artifact.

**B2 -- one span per response.** The cluster bootstrap widens the interval but
does not repair the exchangeability assumption underneath split conformal, which
is a statement about rows. Drawing exactly one predicted span per response
restores exchangeability by construction: the drawn rows are one per independent
sampling unit, so the finite-sample guarantee applies to them exactly, with no
approximation and no correction (Dunn, Wasserman, Ramdas, JASA 2022,
arXiv:1809.07441). The price is a calibration set of a few hundred rather than a
few thousand, so the threshold is noisier.

Which span gets drawn is arbitrary, so the draw is repeated and the spread over
repetitions is reported. Note what is and is not being claimed: **each single
draw is exactly valid**; the average over draws is reported to show how much the
arbitrary choice matters, and is not itself claimed to carry the guarantee.
Their p-value-aggregation variant, which would recover a guarantee for the
combination, is not implemented here.

**B3 -- the floor.** An ECE near zero is not on its own evidence of anything. A
predictor that ignores its input and returns the calibration positive rate for
every span scores an excellent ECE and is worthless: its AUROC is 0.5 and it can
rank nothing. That row is printed beside every ECE so the reader can see the
difference between "well calibrated" and "well calibrated and informative".

AUROC before and after calibration is printed for the same reason and doubles as
a correctness check: a strictly monotone calibrator cannot change the ranking, so
the two AUROCs must agree to within floating-point noise. If they do not, the
calibrator is not doing what this project says it does.

Run:

    python -m src.c2_calibration.clustered \\
        --calib results/c1/calib/probabilities.jsonl \\
        --test  results/c1/test/probabilities.jsonl \\
        --out   results/c2/c1/c2_uncertainty.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.c2_calibration.calibration import (
    Calibrator,
    all_calibrators,
    brier_score,
    compare_calibrators,
    expected_calibration_error,
    maximum_calibration_error,
)
from src.c2_calibration.conformal import coverage_tolerance
from src.c2_calibration.run_c2 import read_probability_file, span_units, token_units

RESAMPLES = 2000
SUBSAMPLE_DRAWS = 200
SEED = 42
ALPHAS: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


# --------------------------------------------------------------------------
# Rows, grouped by the response they came from
# --------------------------------------------------------------------------


@dataclass
class Units:
    """The same flat rows run_c2 analyses, plus the response each row came from.

    `index[i]` holds the row positions belonging to response i. A response that
    the detector proposed no spans for contributes an empty array, and it is
    kept rather than dropped: the number of spans a response yields is itself
    random, and a resample that could not draw an empty response would
    understate the variability of that count.
    """

    scores: np.ndarray
    labels: np.ndarray
    groups: List[str]
    index: List[np.ndarray]

    @property
    def n_rows(self) -> int:
        return int(self.scores.size)

    @property
    def n_responses(self) -> int:
        return len(self.index)


def unit_counts(records: Sequence[Dict[str, Any]], unit: str) -> List[int]:
    """How many analysis rows each response contributes, in file order."""
    if unit == "span":
        return [len(record.get("pred_spans", [])) for record in records]
    if unit == "token":
        return [len(record["token_probs"]) for record in records]
    raise ValueError(f"unknown unit {unit!r}")


def extract(
    records: Sequence[Dict[str, Any]], unit: str, score_key: str = "mean_prob"
) -> Units:
    """Flat rows exactly as run_c2 builds them, plus the response grouping.

    The rows themselves come from run_c2's own `span_units` / `token_units`
    rather than from a second copy of that logic here, so this module cannot
    drift into analysing a different population from the one the headline
    tables were computed on. The grouping is derived from per-response counts
    and checked against the flat length.
    """
    if unit == "span":
        scores, labels, groups = span_units(records, score_key)
    else:
        scores, labels, groups = token_units(records)

    counts = unit_counts(records, unit)
    if sum(counts) != len(scores):
        raise ValueError(
            f"grouping disagrees with the flat rows: counts sum to {sum(counts)} "
            f"but {unit}_units produced {len(scores)}"
        )

    offsets = np.cumsum([0] + counts)
    index = [
        np.arange(offsets[i], offsets[i + 1], dtype=np.int64) for i in range(len(counts))
    ]
    return Units(
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        groups=list(groups),
        index=index,
    )


# --------------------------------------------------------------------------
# The conformal arithmetic again, vectorised
# --------------------------------------------------------------------------
#
# conformal.py is the readable reference implementation and stays that way. A
# bootstrap runs it a few tens of thousands of times, so these two helpers do
# the same arithmetic without rebuilding Python tuples per row. `test_clustered`
# asserts they agree with SplitConformal on real data; if that test ever fails,
# the reference implementation is right and this one is wrong.


def _nonconformity(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """1 - p(true label): p where the truth is 0, 1 - p where the truth is 1."""
    return np.where(labels == 1, 1.0 - scores, scores)


def quantile_from_sorted(sorted_scores: np.ndarray, alpha: float) -> float:
    """conformal_quantile on an already-sorted array. Same ceil((n+1)(1-a)) rule."""
    n = int(sorted_scores.size)
    if n == 0:
        raise ValueError("cannot calibrate on an empty score set")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(sorted_scores[k - 1])


def evaluate_at_threshold(
    threshold: float, scores: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Coverage and decision rates for one threshold, over whole arrays at once."""
    keep_one = (1.0 - scores) <= threshold
    keep_zero = scores <= threshold
    covered = np.where(labels == 1, keep_one, keep_zero)
    neither = ~keep_zero & ~keep_one
    n = max(int(scores.size), 1)
    return {
        "empirical_coverage": float(covered.sum()) / n,
        "abstention_rate": float(((keep_zero & keep_one) | neither).sum()) / n,
        "empty_set_rate": float(neither.sum()) / n,
        "flag_rate": float((keep_one & ~keep_zero).sum()) / n,
    }


def _resample_responses(index: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Row positions of a bootstrap sample drawn at the response level."""
    n = len(index)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    picks = rng.integers(0, n, size=n)
    return np.concatenate([index[i] for i in picks])


def percentile_interval(values: Sequence[float]) -> Dict[str, float]:
    """2.5th and 97.5th percentile, plus the spread, of a bootstrap distribution.

    Infinities are kept in the count but excluded from the percentiles, because
    an infinite conformal threshold is a real outcome -- it means the resampled
    calibration set was too small for that alpha -- and averaging it into a
    number would be nonsense. The count is reported so it cannot hide.
    """
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"ci_low": float("nan"), "ci_high": float("nan"), "sd": float("nan"),
                "n_finite": 0, "n_infinite": int(array.size)}
    return {
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "sd": float(np.std(finite)),
        "n_finite": int(finite.size),
        "n_infinite": int(array.size - finite.size),
    }


def ece_fast(scores: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Equal-width ECE in one pass, for use inside a bootstrap loop.

    Numerically identical to `calibration.expected_calibration_error` with the
    default equal-width scheme -- `test_clustered` asserts that on real data --
    but it bins with a single digitize instead of building fifteen boolean masks,
    which is the difference between a token-level bootstrap taking seconds and
    taking an hour.

    Assumes scores lie in [0, 1]. Every calibrator in this project returns a
    sigmoid or a clipped isotonic fit, so they do; a score outside the range
    would be clamped into the end bin here and dropped entirely by the reference
    implementation, and the two would disagree.
    """
    if scores.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.digitize(scores, edges[1:-1], right=False)
    count = np.bincount(which, minlength=n_bins).astype(np.float64)
    score_sum = np.bincount(which, weights=scores, minlength=n_bins)
    label_sum = np.bincount(which, weights=labels.astype(np.float64), minlength=n_bins)
    nonempty = count > 0
    gap = np.abs(
        np.divide(score_sum, count, out=np.zeros_like(count), where=nonempty)
        - np.divide(label_sum, count, out=np.zeros_like(count), where=nonempty)
    )
    return float((count * gap).sum() / scores.size)


def fit_calibrator(method: str, scores: np.ndarray, labels: np.ndarray) -> Calibrator:
    """A freshly fitted calibrator of the named kind."""
    calibrator = next((c for c in all_calibrators() if c.name == method), None)
    if calibrator is None:
        raise ValueError(f"unknown calibrator {method!r}")
    calibrator.fit(scores, labels)
    return calibrator


def select_method(calib: Units, test: Units) -> Tuple[str, List[Dict[str, Any]]]:
    """Replicate run_c2's choice: lowest test ECE across the four candidates.

    Chosen once, on the real split, and then held fixed for every bootstrap
    draw. Re-selecting inside the loop would fold the selection's own variance
    into the interval, which is a different and larger quantity than "how
    precise is the number I am reporting for this method".

    The selection peeks at test ECE. That is declared in the report; repeating
    the same peek here rather than inventing a cleaner rule keeps this interval
    attached to the number actually being published.
    """
    rows = compare_calibrators(calib.scores, calib.labels, test.scores, test.labels)
    winner = min(rows, key=lambda row: row["ece"])
    return str(winner["method"]), rows


# --------------------------------------------------------------------------
# B1 -- the cluster bootstrap
# --------------------------------------------------------------------------


def cluster_bootstrap(
    calib: Units,
    test: Units,
    method: str,
    alphas: Sequence[float] = ALPHAS,
    resamples: int = RESAMPLES,
    seed: int = SEED,
    refit_calibration: bool = True,
) -> Dict[str, Any]:
    """Response-level bootstrap over ECE and over coverage, on shared draws.

    Calibration and coverage are resampled together rather than in two separate
    loops so that both intervals describe the same corpus draws. It is also
    three times faster, but that is not the reason.

    `refit_calibration=True` resamples the calibration split as well and refits
    the calibrator and the conformal threshold on it every draw -- the interval
    on the whole procedure. `False` freezes both at their real fitted values and
    resamples only the test corpus -- the interval on this specific artifact.
    Both are reported; the wider one is the one that belongs beside a claim
    about the method.
    """
    rng = np.random.default_rng(seed)

    base_calibrator = fit_calibrator(method, calib.scores, calib.labels)
    base_sorted = np.sort(
        _nonconformity(
            np.asarray(base_calibrator.transform(calib.scores)), calib.labels
        )
    )

    ece_before: List[float] = []
    ece_after: List[float] = []
    ece_delta: List[float] = []
    per_alpha: Dict[float, Dict[str, List[float]]] = {
        alpha: {"empirical_coverage": [], "abstention_rate": [], "flag_rate": [],
                "threshold": []}
        for alpha in alphas
    }

    for _ in range(resamples):
        rows = _resample_responses(test.index, rng)
        test_scores = test.scores[rows]
        test_labels = test.labels[rows]

        if refit_calibration:
            calib_rows = _resample_responses(calib.index, rng)
            calib_scores = calib.scores[calib_rows]
            calib_labels = calib.labels[calib_rows]
            calibrator = fit_calibrator(method, calib_scores, calib_labels)
            sorted_nonconformity = np.sort(
                _nonconformity(
                    np.asarray(calibrator.transform(calib_scores)), calib_labels
                )
            )
        else:
            calibrator = base_calibrator
            sorted_nonconformity = base_sorted

        calibrated = np.asarray(calibrator.transform(test_scores))
        before = ece_fast(test_scores, test_labels)
        after = ece_fast(calibrated, test_labels)
        ece_before.append(before)
        ece_after.append(after)
        ece_delta.append(before - after)

        for alpha in alphas:
            threshold = quantile_from_sorted(sorted_nonconformity, alpha)
            row = evaluate_at_threshold(threshold, calibrated, test_labels)
            bucket = per_alpha[alpha]
            bucket["threshold"].append(threshold)
            for key in ("empirical_coverage", "abstention_rate", "flag_rate"):
                bucket[key].append(row[key])

    coverage_rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        bucket = per_alpha[alpha]
        interval = percentile_interval(bucket["empirical_coverage"])
        band = coverage_tolerance(alpha, calib.n_rows, test.n_rows)
        # Compare standard deviations, not a 95% interval against a 3-sigma band.
        # coverage_tolerance returns 3 sigma of the independent-rows model, so
        # band / 3 is the standard deviation that model predicts, and the ratio
        # below is the only like-for-like reading of "how much did treating
        # spans as independent understate the noise".
        theoretical_sd = band / 3.0
        coverage_rows.append(
            {
                "alpha": alpha,
                "target_coverage": 1.0 - alpha,
                "bootstrap_mean": float(np.mean(bucket["empirical_coverage"])),
                **interval,
                "ci_width": interval["ci_high"] - interval["ci_low"],
                "pooled_3sigma_band": band,
                "theoretical_sd_independent_rows": theoretical_sd,
                "sd_ratio_clustered_over_independent": (
                    interval["sd"] / theoretical_sd if theoretical_sd > 0 else float("nan")
                ),
                "abstention": percentile_interval(bucket["abstention_rate"]),
                "flag": percentile_interval(bucket["flag_rate"]),
                "threshold": percentile_interval(bucket["threshold"]),
            }
        )

    return {
        "method": method,
        "resamples": resamples,
        "seed": seed,
        "refit_calibration": refit_calibration,
        "n_calibration_responses": calib.n_responses,
        "n_test_responses": test.n_responses,
        "n_calibration_rows": calib.n_rows,
        "n_test_rows": test.n_rows,
        "ece_before": percentile_interval(ece_before),
        "ece_after": percentile_interval(ece_after),
        "ece_reduction": {
            **percentile_interval(ece_delta),
            "crosses_zero": bool(
                percentile_interval(ece_delta)["ci_low"]
                <= 0.0
                <= percentile_interval(ece_delta)["ci_high"]
            ),
        },
        "coverage": coverage_rows,
    }


# --------------------------------------------------------------------------
# B2 -- one row per response, so exchangeability holds by construction
# --------------------------------------------------------------------------


def _one_per_response(index: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """One uniformly chosen row from each response that has at least one."""
    picks = [rows[rng.integers(0, rows.size)] for rows in index if rows.size]
    return np.asarray(picks, dtype=np.int64)


def one_row_per_response(
    calib: Units,
    test: Units,
    method: str,
    alphas: Sequence[float] = ALPHAS,
    draws: int = SUBSAMPLE_DRAWS,
    seed: int = SEED,
) -> Dict[str, Any]:
    """Split conformal on one span per response, repeated to show the spread.

    Every row in a single draw comes from a different response, so the rows are
    exchangeable in the sense split conformal actually requires and the
    finite-sample guarantee applies to that draw exactly. The pooled analysis
    only approximates this.

    The calibrator stays as fitted on the full calibration split and is not
    refitted per draw. Calibration is a per-row map with no exchangeability
    requirement of its own, it is what the served artifact holds fixed, and
    freezing it isolates what the subsampling does to the conformal layer.

    What the numbers mean: each draw is a valid conformal run on a smaller
    calibration set, so the drop in calibration size is why the threshold gets
    noisy. If the mean coverage across draws sits at target while the pooled
    analysis also sat at target, the pooling was harmless here and that is worth
    saying. If it does not, the pooled table was optimistic and by how much.
    """
    rng = np.random.default_rng(seed)
    calibrator = fit_calibrator(method, calib.scores, calib.labels)
    calib_calibrated = np.asarray(calibrator.transform(calib.scores))
    test_calibrated = np.asarray(calibrator.transform(test.scores))

    n_calib_used = sum(1 for rows in calib.index if rows.size)
    n_test_used = sum(1 for rows in test.index if rows.size)

    per_alpha: Dict[float, Dict[str, List[float]]] = {
        alpha: {"empirical_coverage": [], "abstention_rate": [], "flag_rate": [],
                "threshold": []}
        for alpha in alphas
    }

    for _ in range(draws):
        calib_rows = _one_per_response(calib.index, rng)
        test_rows = _one_per_response(test.index, rng)
        sorted_nonconformity = np.sort(
            _nonconformity(calib_calibrated[calib_rows], calib.labels[calib_rows])
        )
        scores = test_calibrated[test_rows]
        labels = test.labels[test_rows]
        for alpha in alphas:
            threshold = quantile_from_sorted(sorted_nonconformity, alpha)
            row = evaluate_at_threshold(threshold, scores, labels)
            bucket = per_alpha[alpha]
            bucket["threshold"].append(threshold)
            for key in ("empirical_coverage", "abstention_rate", "flag_rate"):
                bucket[key].append(row[key])

    # The pooled counterpart, computed here so the JSON carries both readings
    # side by side and nobody has to line up two files to compare them.
    pooled_sorted = np.sort(_nonconformity(calib_calibrated, calib.labels))

    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        bucket = per_alpha[alpha]
        pooled_threshold = quantile_from_sorted(pooled_sorted, alpha)
        pooled = evaluate_at_threshold(pooled_threshold, test_calibrated, test.labels)
        band = coverage_tolerance(alpha, n_calib_used, n_test_used)
        rows.append(
            {
                "alpha": alpha,
                "target_coverage": 1.0 - alpha,
                "subsampled_mean_coverage": float(np.mean(bucket["empirical_coverage"])),
                "coverage": percentile_interval(bucket["empirical_coverage"]),
                "abstention": percentile_interval(bucket["abstention_rate"]),
                "flag": percentile_interval(bucket["flag_rate"]),
                "threshold": percentile_interval(bucket["threshold"]),
                "band_at_subsampled_size": band,
                "inside_band": bool(
                    (1.0 - alpha) - float(np.mean(bucket["empirical_coverage"]))
                    <= max(band, 0.005)
                ),
                "pooled_coverage": pooled["empirical_coverage"],
                "pooled_abstention_rate": pooled["abstention_rate"],
                "pooled_threshold": pooled_threshold,
            }
        )

    return {
        "method": method,
        "draws": draws,
        "seed": seed,
        "n_calibration_rows_per_draw": n_calib_used,
        "n_test_rows_per_draw": n_test_used,
        "n_calibration_rows_pooled": calib.n_rows,
        "n_test_rows_pooled": test.n_rows,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# B3 -- the floor, and the ranking check
# --------------------------------------------------------------------------


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, or NaN when one class is missing.

    NaN rather than a made-up 0.5: with one class present the quantity is
    undefined, and a printed 0.5 would be indistinguishable from a real measured
    chance-level result.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def floor_rows(calib: Units, test: Units, method: str) -> Dict[str, Any]:
    """The constant-base-rate baseline, and AUROC before and after calibration.

    The constant predictor returns the calibration positive rate for every
    single row. It cannot possibly be useful -- it has not looked at the input --
    and yet its ECE is excellent, because ECE only asks whether the average
    score matches the average outcome. Printing this row beside every reported
    ECE is what stops a good ECE being read as a good detector. The gap between
    its ECE and ours is not the result; the gap between its AUROC and ours is.

    The AUROC pair is a self-check. Platt and temperature scaling are strictly
    monotone, so they cannot reorder anything and AUROC must come out identical.
    A difference beyond floating-point noise means either the calibrator is not
    monotone on this data -- which isotonic, being only weakly monotone, can
    manage by tying distinct scores together -- or something is wrong.
    """
    calibrator = fit_calibrator(method, calib.scores, calib.labels)
    calibrated = np.asarray(calibrator.transform(test.scores))
    base_rate = float(calib.labels.mean())
    constant = np.full(test.scores.shape, base_rate, dtype=np.float64)

    raw_auroc = auroc(test.scores, test.labels)
    calibrated_auroc = auroc(calibrated, test.labels)
    return {
        "method": method,
        "constant_base_rate": {
            "predictor": "always returns the calibration positive rate",
            "value": base_rate,
            "ece": ece_fast(constant, test.labels),
            "ece_reference_implementation": expected_calibration_error(
                constant, test.labels
            ),
            "mce": maximum_calibration_error(constant, test.labels),
            "brier": brier_score(constant, test.labels),
            "auroc": auroc(constant, test.labels),
        },
        "detector": {
            "ece_raw": ece_fast(test.scores, test.labels),
            "ece_calibrated": ece_fast(calibrated, test.labels),
            "auroc_raw": raw_auroc,
            "auroc_calibrated": calibrated_auroc,
            "auroc_difference": abs(raw_auroc - calibrated_auroc),
            "ranking_preserved": bool(abs(raw_auroc - calibrated_auroc) < 1e-9),
        },
        "test_positive_rate": float(test.labels.mean()),
        "calibration_positive_rate": base_rate,
    }


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------


def format_bootstrap(result: Dict[str, Any], title: str) -> str:
    lines = [
        f"\n{title}",
        f"  {result['resamples']:,} draws, seed {result['seed']}, resampling "
        f"{result['n_test_responses']:,} test responses"
        + (
            f" and {result['n_calibration_responses']:,} calibration responses"
            if result["refit_calibration"]
            else " only (calibrator and threshold frozen)"
        ),
    ]
    before, after, delta = (
        result["ece_before"],
        result["ece_after"],
        result["ece_reduction"],
    )
    lines.append(
        f"  ECE before  95% CI [{before['ci_low']:.4f}, {before['ci_high']:.4f}]  "
        f"sd {before['sd']:.4f}"
    )
    lines.append(
        f"  ECE after   95% CI [{after['ci_low']:.4f}, {after['ci_high']:.4f}]  "
        f"sd {after['sd']:.4f}"
    )
    verdict = (
        "NOT DECIDABLE AT THIS n" if delta["crosses_zero"] else "reduction is real"
    )
    lines.append(
        f"  ECE reduction 95% CI [{delta['ci_low']:+.4f}, {delta['ci_high']:+.4f}]"
        f"   {verdict}"
    )
    lines.append(
        "\n  alpha  target   boot mean   95% CI clustered       sd cluster"
        "   sd indep   ratio"
    )
    lines.append("  " + "-" * 78)
    for row in result["coverage"]:
        lines.append(
            f"  {row['alpha']:<6.2f} {row['target_coverage']:<8.3f} "
            f"{row['bootstrap_mean']:<11.4f} "
            f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]   "
            f"{row['sd']:<12.5f}{row['theoretical_sd_independent_rows']:<11.5f}"
            f"{row['sd_ratio_clustered_over_independent']:.2f}x"
        )
    lines.append(
        "  'sd indep' is the standard deviation conformal.py's noise band assumes,\n"
        "  which is its 3-sigma tolerance divided by 3. 'ratio' above 1.00x means\n"
        "  treating spans as independent understated the real spread by that much."
    )
    return "\n".join(lines)


def format_subsample(result: Dict[str, Any]) -> str:
    lines = [
        "\nONE SPAN PER RESPONSE -- exchangeable by construction",
        f"  {result['draws']} draws, seed {result['seed']}; "
        f"{result['n_calibration_rows_per_draw']:,} calibration rows per draw "
        f"(pooled {result['n_calibration_rows_pooled']:,}), "
        f"{result['n_test_rows_per_draw']:,} test rows per draw "
        f"(pooled {result['n_test_rows_pooled']:,})",
        "\n  alpha  target   subsampled  95% CI over draws     band    pooled   in band",
        "  " + "-" * 76,
    ]
    for row in result["rows"]:
        coverage = row["coverage"]
        lines.append(
            f"  {row['alpha']:<6.2f} {row['target_coverage']:<8.3f} "
            f"{row['subsampled_mean_coverage']:<11.4f} "
            f"[{coverage['ci_low']:.4f}, {coverage['ci_high']:.4f}]   "
            f"{row['band_at_subsampled_size']:<8.4f}{row['pooled_coverage']:<9.4f}"
            f"{'yes' if row['inside_band'] else 'NO':>7}"
        )
    return "\n".join(lines)


def format_floor(result: Dict[str, Any]) -> str:
    constant = result["constant_base_rate"]
    detector = result["detector"]
    ranking = (
        "identical, as a monotone map requires"
        if detector["ranking_preserved"]
        else "MOVED -- the calibrator is not strictly monotone on this data"
    )
    return "\n".join(
        [
            "\nTHE FLOOR -- what an uninformative predictor scores",
            f"  constant {constant['value']:.4f} for every row: "
            f"ECE {constant['ece']:.4f}  MCE {constant['mce']:.4f}  "
            f"Brier {constant['brier']:.4f}  AUROC {constant['auroc']:.4f}",
            f"  our detector, raw:        ECE {detector['ece_raw']:.4f}  "
            f"AUROC {detector['auroc_raw']:.4f}",
            f"  our detector, calibrated: ECE {detector['ece_calibrated']:.4f}  "
            f"AUROC {detector['auroc_calibrated']:.4f}",
            f"  AUROC before vs after: {ranking} "
            f"(difference {detector['auroc_difference']:.2e})",
            "  Read the ECE column against the constant row, not against zero.",
        ]
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def analyse_unit(
    calib_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    unit: str,
    score_key: str,
    alphas: Sequence[float],
    resamples: int,
    draws: int,
    seed: int,
) -> Dict[str, Any]:
    calib = extract(calib_records, unit, score_key)
    test = extract(test_records, unit, score_key)
    method, calibrator_rows = select_method(calib, test)

    return {
        "unit": unit,
        "score_key": score_key if unit == "span" else "token_prob",
        "method": method,
        "calibrator_rows": calibrator_rows,
        "floor": floor_rows(calib, test, method),
        "bootstrap_full": cluster_bootstrap(
            calib, test, method, alphas, resamples, seed, refit_calibration=True
        ),
        "bootstrap_test_only": cluster_bootstrap(
            calib, test, method, alphas, resamples, seed, refit_calibration=False
        ),
        "one_row_per_response": one_row_per_response(
            calib, test, method, alphas, draws, seed
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Response-level uncertainty for C2's calibration and coverage."
    )
    parser.add_argument("--calib", required=True, help="calibration probabilities.jsonl")
    parser.add_argument("--test", required=True, help="test probabilities.jsonl")
    parser.add_argument("--out", default="results/c2/c1/c2_uncertainty.json")
    parser.add_argument("--units", default="span,token")
    parser.add_argument(
        "--score-key", default="mean_prob", choices=["mean_prob", "max_prob", "min_prob"]
    )
    parser.add_argument("--alphas", default="0.05,0.10,0.15,0.20,0.30,0.40")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--subsample-draws", type=int, default=SUBSAMPLE_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    alphas = [float(a) for a in args.alphas.split(",")]
    units = [u.strip() for u in args.units.split(",") if u.strip()]

    calib_records = read_probability_file(args.calib)
    test_records = read_probability_file(args.test)
    print(
        f"calibration {len(calib_records):,} responses | "
        f"test {len(test_records):,} responses"
    )

    payload: Dict[str, Any] = {
        "calib_file": args.calib,
        "test_file": args.test,
        "alphas": alphas,
        "resamples": args.resamples,
        "subsample_draws": args.subsample_draws,
        "seed": args.seed,
    }

    for unit in units:
        result = analyse_unit(
            calib_records,
            test_records,
            unit,
            args.score_key,
            alphas,
            args.resamples,
            args.subsample_draws,
            args.seed,
        )
        payload[unit] = result
        print(f"\n{'=' * 78}\nUNIT: {unit}   calibrator: {result['method']}\n{'=' * 78}")
        print(format_floor(result["floor"]))
        print(
            format_bootstrap(
                result["bootstrap_full"],
                "CLUSTER BOOTSTRAP -- whole procedure (calibration resampled too)",
            )
        )
        print(
            format_bootstrap(
                result["bootstrap_test_only"],
                "CLUSTER BOOTSTRAP -- this fitted artifact (test resampled only)",
            )
        )
        print(format_subsample(result["one_row_per_response"]))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
