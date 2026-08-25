"""Layer 6 -- conformal evidence sets over a variable-size candidate pool. The novelty.

Instead of one guess, return a *set* of context sentences with a coverage
guarantee: "at 90% coverage, the sentence the annotator quoted is one of these
two, out of sixteen". Multi-class LAC exactly as `type_conformal.py` does it for
C3's four error types, with the sentences of *this* context as the classes.

**The question a panel will ask, and the honest answer.**

> Split conformal assumes a fixed label space. Yours has four sentences here and
> a hundred there. Is the guarantee real?

Split-conformal validity requires only that the non-conformity score of the
**true label** be exchangeable between calibration and test. It does not require
the label space to have equal cardinality across instances. For each calibration
span we compute one scalar -- the non-conformity of its gold sentence within its
own context, however many sentences that context has -- take the
`ceil((n+1)(1-alpha))/n` quantile, and at test time include every sentence of
the test context clearing that same global threshold. Marginal coverage >= 1-alpha
holds regardless of per-example pool size, and the set size is naturally
instance-adaptive: a long context yields a bigger set, which is the method
telling the truth about difficulty.

**And the half nobody asks for.** A search of the conformal literature on
2026-08-24 found no named theorem stated for instance-dependent label spaces in
a retrieval or evidence-selection setting. The closest formal treatment is
*Conformal Prediction with Partially Labeled Data* (Javanmardi et al., PMLR
v204, 2023), which has per-instance candidate sets of varying size but for
partial-label learning rather than retrieval. The applied conformal-IR papers
(arXiv:2404.17769, arXiv:2410.02914, arXiv:2511.17908) all work over
variable-size pools and sidestep the question by using conformal risk control
(arXiv:2208.02814), which is stated over a monotone loss and never assumes a
shared label space.

So C4 does both. LAC is the primary result and a conformal-risk-control
formulation of the same statement is the cross-check, with C2's
`choose_threshold` doing the arithmetic unchanged and the loss defined as "the
gold sentence is not in the returned set". If the two agree, the guarantee does
not depend on which formalism the examiner prefers. If they disagree, that is
the most interesting figure in the component.

**Written down before measuring (6b).** Summarization contexts have a median of
23 sentences against QA's 14 on these splits, so one global threshold should not
cover both tasks equally. Marginal coverage will hold and at least one task will
sit below it. The per-task table is then reported as the repair, not as a
footnote -- marginal coverage plus the table that shows why marginal was not
enough is a stronger result than either alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.c2_calibration.conformal import conformal_quantile, coverage_tolerance
from src.c2_calibration.risk_control import choose_threshold
from src.c4_attribution.attribution_eval import bootstrap_by_response

OUT = Path("results/c4/conformal/c4_conformal.json")

ALPHAS: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
RISK_GRID = np.linspace(0.0, 1.0, 201)

PREDICTION_PER_TASK = (
    "Written before running: summarization contexts hold a median of 23 "
    "sentences against QA's 14 on these splits, so one global threshold should "
    "not cover both tasks equally. Expect marginal coverage to hold and at "
    "least one task to fall below it."
)


@dataclass(frozen=True)
class ConformalRow:
    """One span, reduced to what conformal prediction needs from it."""

    response_id: str
    task: str
    probabilities: Tuple[float, ...]
    gold_index: Optional[int]


def normalise_scores(scores: Sequence[float]) -> Tuple[float, ...]:
    """Turn one context's raw scores into a distribution over its own sentences.

    Plain normalisation, not softmax: the backends already return bounded
    non-negative scores, and a softmax temperature would be one more knob with
    no principled setting. Negative scores (cosine similarity can be negative)
    are shifted to zero first. A context whose scores are all zero -- which
    happens when the pre-filter kept nothing informative -- becomes uniform,
    which is the correct statement that the model has no opinion.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        return ()
    values = values - min(0.0, float(values.min()))
    total = float(values.sum())
    if total <= 0.0:
        return tuple([1.0 / values.size] * values.size)
    return tuple(float(v) for v in values / total)


def nonconformity(row: ConformalRow) -> Optional[float]:
    """LAC: 1 - p(gold sentence) inside this row's own context."""
    if row.gold_index is None or not row.probabilities:
        return None
    if not 0 <= row.gold_index < len(row.probabilities):
        return None
    return 1.0 - row.probabilities[row.gold_index]


def prediction_set(row: ConformalRow, threshold: float) -> Tuple[int, ...]:
    """Every sentence whose non-conformity clears the calibrated bar."""
    if math.isinf(threshold):
        return tuple(range(len(row.probabilities)))
    return tuple(
        index
        for index, p in enumerate(row.probabilities)
        if (1.0 - p) <= threshold
    )


def fit(rows: Sequence[ConformalRow], alpha: float) -> float:
    """The LAC threshold from a calibration split. +inf means "too small for this alpha"."""
    scores = [s for s in (nonconformity(r) for r in rows) if s is not None]
    if not scores:
        return float("inf")
    return conformal_quantile(scores, alpha)


def evaluate(
    calib: Sequence[ConformalRow],
    test: Sequence[ConformalRow],
    alpha: float,
    with_intervals: bool = True,
) -> Dict[str, Any]:
    """Coverage and the decisiveness it cost, at one alpha.

    Coverage without a set-size column is free -- returning every sentence
    always covers -- so the two are always reported together.
    """
    threshold = fit(calib, alpha)
    scored = [r for r in test if r.gold_index is not None]
    if not scored:
        return {"alpha": alpha, "n_test": 0}

    sets = [prediction_set(r, threshold) for r in scored]
    covered = np.asarray(
        [1.0 if r.gold_index in s else 0.0 for r, s in zip(scored, sets)]
    )
    sizes = np.asarray([len(s) for s in sets], dtype=np.float64)
    pool = np.asarray([len(r.probabilities) for r in scored], dtype=np.float64)

    n_calibration = sum(1 for r in calib if nonconformity(r) is not None)
    band = coverage_tolerance(alpha, n_calibration, len(scored))
    empirical = float(covered.mean())

    block: Dict[str, Any] = {
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "empirical_coverage": empirical,
        "threshold": threshold,
        "n_calibration": n_calibration,
        "n_test": len(scored),
        "band": band,
        "inside_band": bool(empirical >= (1.0 - alpha) - band),
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float(np.mean(sizes == 1)),
        "empty_set_rate": float(np.mean(sizes == 0)),
        "full_set_rate": float(np.mean(sizes >= pool)),
        "mean_pool_size": float(pool.mean()),
        "compression": float(np.mean(sizes / np.maximum(pool, 1.0))),
    }
    if with_intervals:
        response_ids = [r.response_id for r in scored]
        block["coverage_ci"] = bootstrap_by_response(
            response_ids, lambda idx: float(covered[idx].mean())
        )
        block["set_size_ci"] = bootstrap_by_response(
            response_ids, lambda idx: float(sizes[idx].mean())
        )
    return block


def coverage_table(
    calib: Sequence[ConformalRow],
    test: Sequence[ConformalRow],
    alphas: Sequence[float] = ALPHAS,
) -> List[Dict[str, Any]]:
    return [evaluate(calib, test, alpha) for alpha in alphas]


# --------------------------------------------------------------------------
# 6b -- marginal coverage against per-task coverage
# --------------------------------------------------------------------------


def per_task(
    calib: Sequence[ConformalRow],
    test: Sequence[ConformalRow],
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """What marginal coverage hides, and the Mondrian threshold that fixes it.

    Two columns per task: the coverage a single global threshold delivers there,
    and the coverage a threshold calibrated on that task alone delivers. The
    first is what "90% coverage" actually means for a user working on one task.
    """
    global_threshold = fit(calib, alpha)
    tasks = sorted({r.task for r in test})
    out: Dict[str, Any] = {"alpha": alpha, "global_threshold": global_threshold, "tasks": {}}

    for task in tasks:
        calib_task = [r for r in calib if r.task == task]
        test_task = [r for r in test if r.task == task and r.gold_index is not None]
        if not test_task:
            continue

        marginal_sets = [prediction_set(r, global_threshold) for r in test_task]
        marginal_cov = float(
            np.mean([1.0 if r.gold_index in s else 0.0 for r, s in zip(test_task, marginal_sets)])
        )
        marginal_size = float(np.mean([len(s) for s in marginal_sets]))

        conditional = evaluate(calib_task, test_task, alpha, with_intervals=False)
        n_calibration = sum(1 for r in calib_task if nonconformity(r) is not None)
        band = coverage_tolerance(alpha, max(n_calibration, 1), len(test_task))

        out["tasks"][task] = {
            "n_calibration": n_calibration,
            "n_test": len(test_task),
            "marginal_threshold_coverage": marginal_cov,
            "marginal_threshold_mean_set_size": marginal_size,
            "marginal_inside_band": bool(marginal_cov >= (1.0 - alpha) - band),
            "band": band,
            # A group with a handful of calibration spans has a 3-sigma band as
            # wide as the miscoverage level it is testing, at which point
            # "inside the band" says only that there is too little data to tell.
            # Flagged rather than left to look like a pass.
            "band_is_uninformative": bool(band >= alpha),
            "min_calibration_size_for_alpha": int(math.ceil(1.0 / alpha) - 1),
            "group_conditional": {
                key: conditional.get(key)
                for key in (
                    "threshold",
                    "empirical_coverage",
                    "mean_set_size",
                    "singleton_rate",
                    "inside_band",
                )
            },
        }
    out["prediction"] = PREDICTION_PER_TASK
    under = [t for t, b in out["tasks"].items() if not b["marginal_inside_band"]]
    uninformative = [t for t, b in out["tasks"].items() if b["band_is_uninformative"]]
    out["tasks_under_covered_by_marginal"] = under
    out["tasks_whose_band_is_uninformative"] = uninformative
    verdict = (
        f"Marginal coverage leaves {under} below the band; the per-task "
        "threshold is the repair."
        if under
        else (
            "No task fell below its band under the single global threshold. The "
            "prediction written down before the run said at least one would, "
            "and the measurement did not support it. Reported as measured."
        )
    )
    if uninformative:
        verdict += (
            f" Read {uninformative} with care: their calibration slices are small "
            "enough that the 3-sigma band is as wide as alpha itself, so "
            "'inside the band' there is not evidence of coverage holding, only "
            "of there being too little data to say."
        )
    out["verdict"] = verdict
    return out


# --------------------------------------------------------------------------
# 6c -- the conformal-risk-control cross-check
# --------------------------------------------------------------------------


def risk_control_cross_check(
    calib: Sequence[ConformalRow],
    test: Sequence[ConformalRow],
    alpha: float = 0.10,
    grid: Sequence[float] = RISK_GRID,
) -> Dict[str, Any]:
    """The same statement in the formalism that never assumes a fixed label space.

    The loss is "the gold sentence is not in the returned set", averaged within a
    response, and the threshold is the *inclusion* cut on the normalised score.
    A larger cut returns fewer sentences and so cannot lower the loss, which is
    the monotonicity conformal risk control requires. C2's `choose_threshold`
    does the arithmetic unchanged -- it is completely generic in its losses.
    """
    grid = np.asarray(list(grid), dtype=np.float64)
    calib_scored = [r for r in calib if r.gold_index is not None]
    if not calib_scored:
        return {"alpha": alpha, "feasible": False, "reason": "no calibration rows"}

    by_response: Dict[str, List[ConformalRow]] = {}
    for row in calib_scored:
        by_response.setdefault(row.response_id, []).append(row)

    losses = np.zeros((len(by_response), grid.size), dtype=np.float64)
    for position, rows in enumerate(by_response.values()):
        for column, cut in enumerate(grid):
            miss = [
                0.0 if row.probabilities[row.gold_index] >= cut else 1.0
                for row in rows
            ]
            losses[position, column] = float(np.mean(miss))

    chosen = choose_threshold(losses, grid, alpha)
    result: Dict[str, Any] = {"alpha": alpha, "calibration": chosen}

    if chosen.get("feasible"):
        cut = float(chosen["threshold"])
        test_scored = [r for r in test if r.gold_index is not None]
        sets = [
            tuple(i for i, p in enumerate(r.probabilities) if p >= cut)
            for r in test_scored
        ]
        covered = np.asarray(
            [1.0 if r.gold_index in s else 0.0 for r, s in zip(test_scored, sets)]
        )
        sizes = np.asarray([len(s) for s in sets], dtype=np.float64)
        result["test"] = {
            "n_test": len(test_scored),
            "empirical_coverage": float(covered.mean()) if covered.size else float("nan"),
            "mean_set_size": float(sizes.mean()) if sizes.size else float("nan"),
            "singleton_rate": float(np.mean(sizes == 1)) if sizes.size else float("nan"),
            "empty_set_rate": float(np.mean(sizes == 0)) if sizes.size else float("nan"),
        }
    return result


def compare_formalisms(lac: Dict[str, Any], crc: Dict[str, Any]) -> Dict[str, Any]:
    """Do LAC and conformal risk control agree? Either answer is reportable."""
    if not crc.get("test"):
        return {
            "agree": None,
            "note": (
                "Conformal risk control found no feasible threshold at this "
                "alpha, which is a real answer rather than a failure: it says "
                "the level is unreachable with this scorer and this much "
                "calibration data."
            ),
            "reason": crc.get("calibration", {}).get("reason"),
        }
    delta_coverage = crc["test"]["empirical_coverage"] - lac["empirical_coverage"]
    delta_size = crc["test"]["mean_set_size"] - lac["mean_set_size"]
    lac_cut = 1.0 - lac.get("threshold", float("nan"))
    crc_cut = crc.get("calibration", {}).get("threshold")
    return {
        "lac_inclusion_cut": lac_cut,
        "crc_inclusion_cut": crc_cut,
        "cuts_coincide": (
            bool(abs(lac_cut - float(crc_cut)) <= 0.01)
            if crc_cut is not None and lac_cut == lac_cut
            else None
        ),
        "lac_coverage": lac["empirical_coverage"],
        "crc_coverage": crc["test"]["empirical_coverage"],
        "delta_coverage": delta_coverage,
        "lac_mean_set_size": lac["mean_set_size"],
        "crc_mean_set_size": crc["test"]["mean_set_size"],
        "delta_mean_set_size": delta_size,
        "agree": bool(abs(delta_coverage) <= lac.get("band", 0.05)),
        "note": (
            "Both formalisms return the same FAMILY of rule -- include every "
            "sentence whose normalised score clears a cut -- and differ only in "
            "how the cut is chosen: LAC takes the ceil((n+1)(1-alpha)) quantile "
            "of the gold sentence's non-conformity, conformal risk control takes "
            "the largest cut whose finite-sample risk bound still clears alpha. "
            "So this is a cross-check on the CALIBRATION step, not on the "
            "prediction rule, and it is worth stating that way rather than "
            "claiming more. What it establishes is that the coverage statement "
            "does not depend on which of the two the examiner prefers -- which "
            "matters here because no named theorem covers LAC over "
            "instance-dependent label spaces, while conformal risk control never "
            "assumes a shared label space at all. `cuts_coincide` says whether "
            "the two landed on the same cut to the grid resolution."
        ),
    }


# --------------------------------------------------------------------------
# 6d -- what a set of two actually means
# --------------------------------------------------------------------------


def set_shape(
    calib: Sequence[ConformalRow],
    test: Sequence[ConformalRow],
    alpha: float = 0.10,
) -> Dict[str, Any]:
    """When the set has two sentences, are they adjacent or far apart?

    Adjacent means the evidence straddles a sentence boundary -- a segmentation
    artefact, and one Layer 1c already bounded. Far apart means the context says
    it twice, or the model cannot tell. Those are different findings and the
    distinction costs nothing to compute.
    """
    threshold = fit(calib, alpha)
    scored = [r for r in test if r.gold_index is not None]
    pairs = []
    distances: List[int] = []
    for row in scored:
        candidate = prediction_set(row, threshold)
        if len(candidate) == 2:
            pairs.append(candidate)
            distances.append(abs(candidate[1] - candidate[0]))
    adjacent = sum(1 for d in distances if d == 1)
    sizes = [len(prediction_set(r, threshold)) for r in scored]
    return {
        "alpha": alpha,
        "n_test": len(scored),
        "n_size_two": len(pairs),
        "share_size_two": len(pairs) / len(scored) if scored else 0.0,
        "adjacent": adjacent,
        "share_adjacent_of_pairs": adjacent / len(pairs) if pairs else float("nan"),
        "mean_index_distance": float(np.mean(distances)) if distances else float("nan"),
        "size_histogram": {
            str(size): sizes.count(size) for size in sorted(set(sizes))
        },
    }
