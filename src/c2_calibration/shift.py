"""Does the conformal guarantee survive a change of domain, and can it be repaired?

This is C2's third claim and the one with real teeth, because it is the question
a deployment actually turns on. Split conformal promises coverage under one
precondition: the calibration data and the data seen at run time are
exchangeable. Move to another corpus and that precondition is not weakened, it
is gone, and with it the promise. Whether coverage then actually falls, by how
much, and whether reweighting brings it back, is a measurement nobody has taken
at span level.

**The story boundary matters here and must not be crossed.** C1 owns "the
detector does not transfer" -- its example F1 on RAGBench falls below the trivial
baseline. C2 owns something different: "the *promise* does not survive the shift,
here is the diagnosis, and here is what does and does not repair it." A detector
can transfer badly and still be conformally covered, because coverage is a
statement about prediction sets and not about accuracy. The two claims are
independent and are made from different tables.

Four steps, in the order they have to be taken.

**1. Break.** Calibrate on RAGTruth, test on RAGBench, and report what happens.
That is `run_c2 --allow-violations`, not this file. A violation there is the
result; the run records it and stamps the file VOID rather than exiting.

**2. Diagnose which shift.** Three numbers, and no cause the numbers do not
support:

    label shift        P(y) moves, P(x|y) does not. The candidate here: the
                       span positive rate on RAGTruth against RAGBench.
    covariate shift    P(x) moves, P(y|x) does not. Measured by asking a
                       classifier to tell a RAGTruth span from a RAGBench span
                       using only the features the conformal layer sees. If it
                       cannot, there is no covariate shift to correct.
    concept shift      P(y|x) itself moves. Neither reweighting below repairs
                       this, and if the evidence points here the honest
                       conclusion is "recalibrate in-domain".

**3. Repair, two ways, both from the literature and both on CPU.**

    label-shift conformal      Podkopaev & Ramdas, UAI 2021, arXiv:2103.03323.
                               Estimate the target class priors from unlabeled
                               target data, weight each calibration point by
                               pi_target(y_i) / pi_source(y_i), take a weighted
                               quantile. Because the weight depends on the
                               label, the threshold is computed once per
                               candidate label, giving two thresholds instead
                               of one.
    covariate-shift conformal  Tibshirani, Barber, Candes, Ramdas, NeurIPS 2019,
                               arXiv:1904.06019. Weight each calibration point
                               by the likelihood ratio p_target(x)/p_source(x),
                               estimated as the odds of a domain classifier, and
                               take a weighted quantile that also depends on the
                               test point's own weight -- so here the threshold
                               is computed once per test point.

**4. Report honestly.** If neither repair restores coverage, the deployment rule
is "recalibrate on your own data before you trust the dial", and that is the
finding. Do not go looking for a third repair. A weight estimate that lands on
the edge of its clipping range is not an estimate, and this module says so in
the JSON rather than quietly using it.

Everything here runs on the CPU over probability dumps that already exist.

Run:

    python -m src.c2_calibration.shift \\
        --source results/c1/calib/probabilities.jsonl \\
        --target results/ood/ragbench-probs/probabilities.jsonl \\
        --out-dir results/c2/ragbench
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.c2_calibration.calibration import Calibrator, to_logit
from src.c2_calibration.clustered import (
    _nonconformity,
    fit_calibrator,
    quantile_from_sorted,
)
from src.c2_calibration.conformal import coverage_tolerance
from src.c2_calibration.run_c2 import read_probability_file, span_units

SEED = 42
ALPHAS: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
TARGET_ESTIMATION_FRACTION = 0.5
# Importance weights are ratios of two estimated densities and both estimates are
# noisy in the tails, so an unclipped ratio can reach thousands and hand a single
# calibration span the entire quantile. Clipping is standard practice; what is
# not standard, and is done here, is reporting how much of the mass was clipped,
# because a weight distribution that is mostly at the clip is not an estimate.
WEIGHT_CLIP = 20.0


# --------------------------------------------------------------------------
# Spans, with the features a domain classifier is allowed to look at
# --------------------------------------------------------------------------

FEATURE_NAMES = (
    "logit_mean_prob",
    "logit_max_prob",
    "logit_min_prob",
    "log_span_chars",
    "log_span_tokens",
    "log_answer_tokens",
    "log_spans_in_response",
)


@dataclass
class SpanTable:
    """Every predicted span in a corpus, flat, with what it needs to be reweighted."""

    scores: np.ndarray
    labels: np.ndarray
    groups: List[str]
    features: np.ndarray
    response_of_row: np.ndarray

    @property
    def n(self) -> int:
        return int(self.scores.size)

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean()) if self.n else 0.0


def build_span_table(
    records: Sequence[Dict[str, Any]],
    score_key: str = "mean_prob",
    group_key: str = "task_type",
) -> SpanTable:
    """Flat span rows plus a small feature vector per span.

    Scores and labels come from run_c2's own `span_units`, so this table is the
    same population the headline coverage numbers were computed on. Only the
    feature matrix is new.

    **What the features may contain.** Only quantities the conformal layer
    actually has at run time: the span's three aggregated probabilities, its
    size, and the size of the answer it came from. The context length would be
    a natural fourth -- cuad and techqa lose most of their context to
    truncation, so it is very likely a real axis of the shift -- but the
    probability dump does not carry the context, only the answer, and inventing
    a proxy for it would make the domain classifier's AUC unreadable. Its
    absence is a limitation of this diagnosis and is stated as one.

    Probabilities go in as logits. A domain classifier is a logistic regression,
    so a feature that is already on the log-odds scale is the one it can use
    linearly; feeding raw probabilities would make a genuine shift in the tails
    look like no shift at all.
    """
    scores, labels, groups = span_units(records, score_key, group_key)

    rows: List[List[float]] = []
    response_of_row: List[int] = []
    for response_index, record in enumerate(records):
        spans = record.get("pred_spans", [])
        n_answer_tokens = len(record.get("token_probs", []))
        for span in spans:
            rows.append(
                [
                    float(to_logit([span["mean_prob"]])[0]),
                    float(to_logit([span["max_prob"]])[0]),
                    float(to_logit([span["min_prob"]])[0]),
                    math.log1p(max(0, int(span["end"]) - int(span["start"]))),
                    math.log1p(max(0, int(span.get("n_tokens", 0)))),
                    math.log1p(n_answer_tokens),
                    math.log1p(len(spans)),
                ]
            )
            response_of_row.append(response_index)

    return SpanTable(
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
        groups=list(groups),
        features=np.asarray(rows, dtype=np.float64).reshape(len(rows), len(FEATURE_NAMES)),
        response_of_row=np.asarray(response_of_row, dtype=np.int64),
    )


def split_target(
    records: Sequence[Dict[str, Any]],
    fraction: float = TARGET_ESTIMATION_FRACTION,
    seed: int = SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stratified split of the target corpus, by question rather than by response.

    Two properties, and the second one was found by measurement rather than
    assumed.

    **Stratified by subset**, the way `ood_operating_point.split_by_subset`
    carves its halves: grouped by subset, ordered deterministically, shuffled
    with the same seed. The twelve subsets differ by an order of magnitude in
    hallucination rate, so an unstratified draw could hand the estimation half a
    base rate the evaluation half does not share -- and estimating that base
    rate is the entire point of the label-shift repair.

    **Grouped by question id**, which is the part `split_by_subset` does not do
    and this corpus needs. RAGBench record ids are not unique: 4,355 of its
    7,446 questions carry *two* responses, one from gpt-3.5-turbo-0125 and one
    from claude-3-haiku-20240307, sharing a question and a context and differing
    only in which model wrote the answer. Splitting by response would put one
    model's answer to a question in the half the weights are estimated on and
    the other model's answer to the same question in the half they are judged
    on. The two halves would then share information, the evaluation half would
    flatter the repair, and nothing in the output would say so.

    RAGTruth has no such structure -- all 2,700 test ids are distinct -- which
    is why this only shows up on the target side.

    Everything the repairs estimate is estimated on the first half. Every number
    they are judged by is measured on the second. The estimation half's labels
    are never read: BBSE and the domain classifier both take unlabeled target
    data, which is what makes the repair deployable at all.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")

    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[str(record.get("subset", "unknown"))][str(record.get("id"))].append(
            record
        )

    rng = random.Random(seed)
    estimation: List[Dict[str, Any]] = []
    evaluation: List[Dict[str, Any]] = []
    for subset in sorted(grouped):
        questions = sorted(grouped[subset])
        rng.shuffle(questions)
        cut = int(round(fraction * len(questions)))
        for question in questions[:cut]:
            estimation.extend(grouped[subset][question])
        for question in questions[cut:]:
            evaluation.extend(grouped[subset][question])
    return estimation, evaluation


# --------------------------------------------------------------------------
# C3 -- which shift is it?
# --------------------------------------------------------------------------


def _summary(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {"n": 0}
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


def base_rate_diagnosis(source: SpanTable, target: SpanTable) -> Dict[str, Any]:
    """Has P(y) moved, and by how much, overall and per target group.

    This is the label-shift candidate. On its own a moved base rate is not proof
    of label shift -- covariate shift moves the marginal base rate too -- so it
    is reported beside the conditional score distributions below, which is what
    separates the two.
    """
    per_group: Dict[str, Dict[str, float]] = {}
    groups = np.asarray(target.groups)
    for group in sorted(set(target.groups)):
        mask = groups == group
        per_group[group] = {
            "n_spans": int(mask.sum()),
            "positive_rate": float(target.labels[mask].mean()) if mask.any() else 0.0,
        }
    return {
        "source_span_positive_rate": source.positive_rate,
        "target_span_positive_rate": target.positive_rate,
        "absolute_change": target.positive_rate - source.positive_rate,
        "ratio": (
            target.positive_rate / source.positive_rate
            if source.positive_rate > 0
            else float("nan")
        ),
        "source_n_spans": source.n,
        "target_n_spans": target.n,
        "target_per_group": per_group,
    }


def conditional_score_diagnosis(source: SpanTable, target: SpanTable) -> Dict[str, Any]:
    """Does P(score | y) move between corpora? Pure label shift says it must not.

    For each class separately, summarise the raw detector score on each corpus
    and run a two-sample Kolmogorov-Smirnov test on the two distributions. The
    KS statistic is the largest vertical gap between the two empirical CDFs, so
    it reads directly as "how far apart are these two score distributions", on a
    0-to-1 scale.

    Read it like this. A small statistic in both classes, with a base rate that
    has clearly moved, is the signature of label shift and the label-shift
    repair is the right tool. A large statistic means P(x|y) moved too, so the
    label-shift assumption is violated and its repair cannot be expected to
    work -- which is a prediction this module makes before running the repair,
    and one the repair's own numbers then confirm or refute.

    The p-value is reported but should not be read as a decision. With tens of
    thousands of spans, any difference at all is significant; the statistic is
    the size of the difference and that is the quantity of interest.
    """
    from scipy.stats import ks_2samp

    out: Dict[str, Any] = {}
    for label in (0, 1):
        source_scores = source.scores[source.labels == label]
        target_scores = target.scores[target.labels == label]
        block: Dict[str, Any] = {
            "source": _summary(source_scores),
            "target": _summary(target_scores),
        }
        if source_scores.size and target_scores.size:
            statistic, p_value = ks_2samp(source_scores, target_scores)
            block["ks_statistic"] = float(statistic)
            block["ks_p_value"] = float(p_value)
            block["mean_shift"] = float(target_scores.mean() - source_scores.mean())
        out[f"y={label}"] = block
    return out


@dataclass
class DomainClassifier:
    """A fitted source-vs-target classifier, and the weights it implies."""

    model: Any
    n_source: int
    n_target: int
    held_out_auc: float
    coefficients: Dict[str, float]
    intercept: float

    def probability_target(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(features)[:, 1], dtype=np.float64)

    def weights(self, features: np.ndarray) -> np.ndarray:
        """p_target(x) / p_source(x), from the classifier's odds.

        A classifier trained on a pooled sample estimates
        P(target | x) = p_t(x) n_t / (p_t(x) n_t + p_s(x) n_s), so the odds
        d/(1-d) equal (p_t/p_s)(n_t/n_s) and the class-size ratio has to be
        divided back out. Skipping that step is a common and silent error: it
        scales every weight by the same constant, which a *self-normalised*
        weighted quantile would absorb -- but the conformal quantile is not
        self-normalised, it carries an extra point mass for the test point, so
        the constant does not cancel and the threshold comes out wrong.
        """
        probability = np.clip(self.probability_target(features), 1e-6, 1 - 1e-6)
        odds = probability / (1.0 - probability)
        return odds * (self.n_source / max(self.n_target, 1))


def fit_domain_classifier(
    source: SpanTable, target: SpanTable, seed: int = SEED
) -> DomainClassifier:
    """Can anything tell a source span from a target span, using run-time features only?

    The AUC is the diagnosis. Near 0.5 means the two corpora look identical
    through the conformal layer's eyes, so there is no covariate shift for a
    reweighting to correct and a covariate-shift repair will do nothing. Well
    above 0.5 means the feature distribution has genuinely moved and the
    likelihood ratio is worth estimating.

    The AUC is measured on a held-out third of the pooled data, never on the
    data the classifier was fitted to. The weights are then taken from a refit
    on everything, which is standard for importance weighting and is noted here
    so the two numbers are not confused: the AUC is honest, the weights are
    fitted on all available data.

    Features are standardised first. Logistic regression with an L2 penalty is
    not scale-free, and the raw features here range from log-odds near -14 to a
    log token count near 3, so an unstandardised fit would penalise them
    unevenly for no reason.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.vstack([source.features, target.features])
    domain = np.concatenate(
        [np.zeros(source.n, dtype=int), np.ones(target.n, dtype=int)]
    )

    def make_model() -> Any:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("lr", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        )

    train_x, test_x, train_y, test_y = train_test_split(
        features, domain, test_size=0.33, random_state=seed, stratify=domain
    )
    held_out = make_model().fit(train_x, train_y)
    auc = float(roc_auc_score(test_y, held_out.predict_proba(test_x)[:, 1]))

    full = make_model().fit(features, domain)
    coefficients = full.named_steps["lr"].coef_[0]
    return DomainClassifier(
        model=full,
        n_source=source.n,
        n_target=target.n,
        held_out_auc=auc,
        coefficients={
            name: float(value) for name, value in zip(FEATURE_NAMES, coefficients)
        },
        intercept=float(full.named_steps["lr"].intercept_[0]),
    )


# --------------------------------------------------------------------------
# Weighted conformal machinery
# --------------------------------------------------------------------------


def weighted_thresholds(
    sorted_scores: np.ndarray,
    cumulative_weights: np.ndarray,
    total_weight: float,
    extra_weights: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """The weighted conformal quantile, once per test point, vectorised.

    Unweighted split conformal takes the ceil((n+1)(1-alpha))-th smallest
    calibration score. The `n + 1` is a point mass for the test point itself,
    which under exchangeability is worth exactly as much as any calibration
    point. Weighted conformal keeps that structure and replaces the equal masses
    with unequal ones: calibration point i carries w_i and the test point
    carries its own w(x), so the threshold is

        inf { s : sum_i w_i * 1{V_i <= s} >= (1 - alpha) * (sum_i w_i + w(x)) }

    and it depends on the test point. That is why this returns an array rather
    than a scalar, and why a covariate-shift-corrected run has a different
    threshold for every span it scores.

    When no calibration score is large enough the answer is +inf, meaning every
    label is in the set: the method has nothing useful to say at this alpha with
    these weights, which is the correct output and not a number to paper over.
    """
    needed = (1.0 - alpha) * (total_weight + np.asarray(extra_weights, dtype=np.float64))
    position = np.searchsorted(cumulative_weights, needed, side="left")
    inside = position < sorted_scores.size
    out = np.full(needed.shape, float("inf"), dtype=np.float64)
    out[inside] = sorted_scores[position[inside]]
    return out


def summarise_decisions(
    keep_zero: np.ndarray,
    keep_one: np.ndarray,
    labels: np.ndarray,
    groups: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Coverage and decision rates from two membership masks, overall and per group.

    Written against masks rather than a threshold because the three methods
    compared here produce thresholds of three different shapes -- one scalar,
    two scalars, one per test point -- and the scoring must not differ between
    them by so much as a comparison operator.
    """
    covered = np.where(labels == 1, keep_one, keep_zero)
    neither = ~keep_zero & ~keep_one
    both = keep_zero & keep_one
    flag = keep_one & ~keep_zero

    def block(mask: np.ndarray) -> Dict[str, float]:
        size = max(int(mask.sum()), 1)
        return {
            "n": int(mask.sum()),
            "empirical_coverage": float(covered[mask].sum()) / size,
            "abstention_rate": float((both | neither)[mask].sum()) / size,
            "empty_set_rate": float(neither[mask].sum()) / size,
            "flag_rate": float(flag[mask].sum()) / size,
            "positive_rate": float(labels[mask].mean()) if mask.any() else 0.0,
        }

    everything = np.ones(labels.shape, dtype=bool)
    out: Dict[str, Any] = {"overall": block(everything), "n_test": int(labels.size)}
    if groups is not None:
        array = np.asarray(groups)
        out["per_group"] = {
            group: block(array == group) for group in sorted(set(groups))
        }
    return out


def unweighted_transfer(
    source_nonconformity_sorted: np.ndarray,
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    groups: Sequence[str],
    alphas: Sequence[float] = ALPHAS,
) -> List[Dict[str, Any]]:
    """The unrepaired baseline: the source threshold applied straight to the target.

    This is the run that is expected to break, reproduced here rather than read
    out of the run_c2 output so that all three methods are scored by the same
    function on the same evaluation half. The run_c2 VOID file remains the
    headline record of the break; this is its like-for-like comparison row.
    """
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        threshold = quantile_from_sorted(source_nonconformity_sorted, alpha)
        keep_one = (1.0 - target_scores) <= threshold
        keep_zero = target_scores <= threshold
        rows.append(
            {
                "alpha": alpha,
                "target_coverage": 1.0 - alpha,
                "threshold": threshold,
                **summarise_decisions(keep_zero, keep_one, target_labels, groups),
            }
        )
    return rows


# --------------------------------------------------------------------------
# C4a -- label-shift conformal (Podkopaev & Ramdas, arXiv:2103.03323)
# --------------------------------------------------------------------------


def estimate_target_priors(
    source_scores: np.ndarray,
    source_labels: np.ndarray,
    target_scores: np.ndarray,
    target_labels_for_reference: np.ndarray,
    decision_threshold: float = 0.5,
    score_name: str = "calibrated span score",
) -> Dict[str, Any]:
    """Black-box shift estimation of the target class priors, from unlabeled data.

    **Which score the predictor thresholds is not a free choice here, and the
    reason is worth a line in the report.** The obvious predictor is the
    detector's own decision, raw score >= 0.5. At span level that predictor is
    *constant*: a span only exists because argmax called its tokens
    hallucinated, so every predicted span already scores above 0.5 and the
    predictor says 1 to all of them. Its confusion matrix has a zero row, it is
    singular, and BBSE has nothing to invert. Measured on this repository, the
    condition number came back as infinity.

    So the predictor thresholds the **calibrated** score at 0.5 instead. That is
    not a dial being turned to make the estimate behave -- the threshold is
    still 0.5, and Platt's intercept is what moves spans to either side of it.
    It is the difference between an estimator that is computable and one that is
    not, and the degenerate case above is left detectable rather than assumed
    away: if the caller passes raw scores, this function reports the singular
    matrix rather than returning a number.

    The trick is that a fixed predictor's *output* distribution is observable on
    the target even though the labels are not. Under label shift P(pred | y) is
    the same on both corpora, so

        P_target(pred = i) = sum_j P(pred = i | y = j) * pi_target(j)
                           = sum_j C[i][j] * w_j

    with C[i][j] = P_source(pred = i, y = j) the source joint, which is
    observable because source labels exist, and w_j = pi_target(j)/pi_source(j)
    the importance weights we want. Two labels, so this is a 2x2 solve.

    Three ways this can fail, all of them reported rather than absorbed:

      * C is near-singular -- the predictor barely separates the classes, so
        the target's predicted-label distribution carries almost no information
        about its true label distribution. The condition number is returned.
      * a solved weight is negative, which is not a possible ratio of priors.
      * the implied prior falls outside (0, 1).

    Any of those sets `usable` to False. A repair run on an unusable estimate is
    a number with no meaning, and the caller is expected to report the estimate
    and stop rather than clip its way to a plausible answer.

    The true target prior is also returned. It is used for nothing except
    reporting how good the unlabeled estimate was -- a deployment would not have
    it, and no repair below reads it.
    """
    source_predicted = (source_scores >= decision_threshold).astype(int)
    target_predicted = (target_scores >= decision_threshold).astype(int)

    joint = np.zeros((2, 2), dtype=np.float64)
    for i in (0, 1):
        for j in (0, 1):
            joint[i, j] = float(((source_predicted == i) & (source_labels == j)).mean())
    observed = np.array(
        [float((target_predicted == i).mean()) for i in (0, 1)], dtype=np.float64
    )

    condition = float(np.linalg.cond(joint))
    source_positive_rate = float(source_labels.mean())
    source_prior = np.array(
        [1.0 - source_positive_rate, source_positive_rate], dtype=np.float64
    )
    target_positive_rate = float(target_labels_for_reference.mean())

    problems: List[str] = []
    predicted_rate = float(source_predicted.mean())
    if predicted_rate in (0.0, 1.0):
        problems.append(
            f"the predictor is constant on the source split (it says "
            f"{int(predicted_rate)} to every span), so its confusion matrix has "
            "a zero row and carries no information about the target's label "
            "distribution"
        )
    try:
        weights = np.linalg.solve(joint, observed)
    except np.linalg.LinAlgError:
        weights = np.array([float("nan"), float("nan")])
        problems.append("the source joint matrix is singular and cannot be inverted")

    estimated_prior = weights * source_prior
    if np.any(~np.isfinite(weights)):
        problems.append("the solved weights are not finite")
    else:
        if np.any(weights < 0):
            problems.append(
                f"a solved weight is negative ({weights.tolist()}), which is not a "
                "possible ratio of two priors"
            )
        if not (0.0 < estimated_prior[1] < 1.0):
            problems.append(
                f"the implied target positive rate {estimated_prior[1]:.4f} is "
                "outside (0, 1)"
            )
    if condition > 1e4:
        problems.append(
            f"the source joint matrix is badly conditioned (cond {condition:.1f}); "
            "the predictor separates the classes too weakly for this estimate to "
            "carry information"
        )

    return {
        "method": "BBSE (black-box shift estimation), confusion-matrix form",
        "predictor": f"{score_name} >= {decision_threshold}",
        "decision_threshold": decision_threshold,
        "source_predicted_positive_rate": predicted_rate,
        "source_joint_pred_by_true": joint.tolist(),
        "condition_number": condition,
        "target_predicted_distribution": observed.tolist(),
        "source_prior": source_prior.tolist(),
        "estimated_weights": weights.tolist(),
        "estimated_target_prior": estimated_prior.tolist(),
        "true_target_prior_for_reference_only": [
            1.0 - target_positive_rate,
            target_positive_rate,
        ],
        "estimation_error_on_positive_rate": float(
            estimated_prior[1] - target_positive_rate
        ),
        "usable": not problems,
        "problems": problems,
    }


def label_shift_conformal(
    source_nonconformity: np.ndarray,
    source_labels: np.ndarray,
    label_weights: Sequence[float],
    target_scores: np.ndarray,
    target_labels: np.ndarray,
    groups: Sequence[str],
    alphas: Sequence[float] = ALPHAS,
) -> List[Dict[str, Any]]:
    """Weighted split conformal where the weight depends on the label, not on x.

    Each calibration span is weighted by pi_target(y_i) / pi_source(y_i). The
    test point's own weight is needed too, and its label is exactly what is
    unknown -- so the threshold is computed once per *candidate* label, using
    that candidate's weight for the extra point mass, and label y is admitted to
    the set when its own non-conformity clears its own threshold. Two thresholds
    where unweighted LAC has one.

    This is why the abstention rate can move in either direction after the
    repair: the two thresholds separate, and how far they separate depends on
    how much the priors moved.
    """
    weights = np.asarray(
        [label_weights[int(y)] for y in source_labels], dtype=np.float64
    )
    order = np.argsort(source_nonconformity)
    sorted_scores = source_nonconformity[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    total = float(cumulative[-1]) if cumulative.size else 0.0

    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        thresholds = weighted_thresholds(
            sorted_scores,
            cumulative,
            total,
            np.asarray([label_weights[0], label_weights[1]], dtype=np.float64),
            alpha,
        )
        threshold_zero, threshold_one = float(thresholds[0]), float(thresholds[1])
        keep_zero = target_scores <= threshold_zero
        keep_one = (1.0 - target_scores) <= threshold_one
        rows.append(
            {
                "alpha": alpha,
                "target_coverage": 1.0 - alpha,
                "threshold_label_0": threshold_zero,
                "threshold_label_1": threshold_one,
                **summarise_decisions(keep_zero, keep_one, target_labels, groups),
            }
        )
    return rows


# --------------------------------------------------------------------------
# C4b -- covariate-shift conformal (Tibshirani et al., arXiv:1904.06019)
# --------------------------------------------------------------------------


def clip_weights(weights: np.ndarray, limit: float = WEIGHT_CLIP) -> Dict[str, Any]:
    """Clip a likelihood ratio into [1/limit, limit] and report what that cost.

    C1's D1 lesson, restated for weights: an estimate that lands on the edge of
    its range is not an estimate. If a large share of the calibration mass sits
    at the clip, the weighted quantile is being decided by the clipping constant
    rather than by the data, and the repair's result says more about `limit`
    than about the shift. That share is returned so it can be printed beside the
    repaired coverage instead of discovered at the viva.

    The effective sample size, (sum w)^2 / sum w^2, is the other half of the
    picture: it is how many equally-weighted calibration points the weighted set
    is worth. If it collapses from sixteen hundred to a few dozen, the repaired
    threshold is noisy no matter how well the weights are estimated.
    """
    clipped = np.clip(weights, 1.0 / limit, limit)
    at_ceiling = float((weights > limit).mean())
    at_floor = float((weights < 1.0 / limit).mean())
    total = float(clipped.sum())
    effective = float(total**2 / np.sum(clipped**2)) if clipped.size else 0.0
    return {
        "weights": clipped,
        "limit": limit,
        "fraction_at_ceiling": at_ceiling,
        "fraction_at_floor": at_floor,
        "fraction_clipped": at_ceiling + at_floor,
        "mean": float(clipped.mean()) if clipped.size else 0.0,
        "max": float(clipped.max()) if clipped.size else 0.0,
        "min": float(clipped.min()) if clipped.size else 0.0,
        "effective_sample_size": effective,
        "n": int(clipped.size),
    }


def covariate_shift_conformal(
    source_nonconformity: np.ndarray,
    source_weights: np.ndarray,
    target_scores: np.ndarray,
    target_weights: np.ndarray,
    target_labels: np.ndarray,
    groups: Sequence[str],
    alphas: Sequence[float] = ALPHAS,
) -> List[Dict[str, Any]]:
    """Weighted split conformal with a per-test-point threshold.

    The weight here depends on x and not on y, so unlike the label-shift case
    one threshold serves both candidate labels -- but it is a different
    threshold for every test span, because the test point's own weight enters
    the normalisation. A span that looks very unlike the calibration corpus
    carries a large weight, which pushes its own threshold up and makes its
    prediction set wider. That is the mechanism by which this repair is supposed
    to buy back coverage, and it is also why it costs abstention.
    """
    order = np.argsort(source_nonconformity)
    sorted_scores = source_nonconformity[order]
    cumulative = np.cumsum(source_weights[order])
    total = float(cumulative[-1]) if cumulative.size else 0.0

    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        thresholds = weighted_thresholds(
            sorted_scores, cumulative, total, target_weights, alpha
        )
        keep_zero = target_scores <= thresholds
        keep_one = (1.0 - target_scores) <= thresholds
        finite = thresholds[np.isfinite(thresholds)]
        rows.append(
            {
                "alpha": alpha,
                "target_coverage": 1.0 - alpha,
                "threshold_mean": float(finite.mean()) if finite.size else float("inf"),
                "threshold_min": float(finite.min()) if finite.size else float("inf"),
                "threshold_max": float(finite.max()) if finite.size else float("inf"),
                "fraction_infinite_threshold": float(
                    (~np.isfinite(thresholds)).mean()
                ),
                **summarise_decisions(keep_zero, keep_one, target_labels, groups),
            }
        )
    return rows


# --------------------------------------------------------------------------
# The study
# --------------------------------------------------------------------------


def _question_structure(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """How many responses share a question, and which models wrote them.

    Recorded in the diagnosis because it is the justification for splitting by
    question, and because a reader who does not know RAGBench pairs two model
    answers per question would otherwise read the response count as a sample
    size it is not.
    """
    per_question: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for record in records:
        per_question[
            (str(record.get("subset", "unknown")), str(record.get("id")))
        ].append(str(record.get("model")))
    sizes = [len(models) for models in per_question.values()]
    models: Dict[str, int] = defaultdict(int)
    for record in records:
        models[str(record.get("model"))] += 1
    return {
        "n_responses": len(records),
        "n_questions": len(per_question),
        "responses_per_question_histogram": {
            str(size): sizes.count(size) for size in sorted(set(sizes))
        },
        "responses_by_model": dict(sorted(models.items(), key=lambda kv: -kv[1])),
        "note": (
            "a question with more than one response has one context and one "
            "question answered by several models. The split groups by question "
            "so both responses land on the same side."
        ),
    }


def meets_target(row: Dict[str, Any], band: float) -> bool:
    """Is this row's shortfall small enough to be sampling noise? Same rule as check_coverage."""
    shortfall = row["target_coverage"] - row["overall"]["empirical_coverage"]
    return shortfall <= max(band, 0.005)


def run_shift_study(
    source_records: Sequence[Dict[str, Any]],
    target_records: Sequence[Dict[str, Any]],
    method: str = "platt",
    score_key: str = "mean_prob",
    alphas: Sequence[float] = ALPHAS,
    seed: int = SEED,
    fraction: float = TARGET_ESTIMATION_FRACTION,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Diagnose the shift, then try both repairs. Returns (diagnosis, repairs).

    The calibrator is the one the in-distribution run selected and is refitted on
    the source split only. That is not a shortcut: it is the deployment
    scenario. A team ships the calibrator it fitted at home, and the question
    this study asks is what that shipped artifact does when the data changes.
    Re-selecting the calibrator against target ECE would answer a question
    nobody can answer at deployment time, because it needs target labels.
    """
    source = build_span_table(source_records, score_key, "task_type")
    estimation_records, evaluation_records = split_target(target_records, fraction, seed)
    target_all = build_span_table(target_records, score_key, "subset")
    target_estimation = build_span_table(estimation_records, score_key, "subset")
    target_evaluation = build_span_table(evaluation_records, score_key, "subset")

    calibrator: Calibrator = fit_calibrator(method, source.scores, source.labels)
    source_calibrated = np.asarray(calibrator.transform(source.scores))
    estimation_calibrated = np.asarray(calibrator.transform(target_estimation.scores))
    evaluation_calibrated = np.asarray(calibrator.transform(target_evaluation.scores))
    source_nonconformity = _nonconformity(source_calibrated, source.labels)
    sorted_nonconformity = np.sort(source_nonconformity)

    classifier = fit_domain_classifier(source, target_estimation, seed)
    priors = estimate_target_priors(
        source_calibrated,
        source.labels,
        estimation_calibrated,
        target_estimation.labels,
        score_name="calibrated span score",
    )
    # The same estimate attempted on the raw score, kept because it fails in an
    # informative way: every predicted span already scores above 0.5 by
    # construction, so the raw-score predictor is constant, its confusion matrix
    # is singular, and BBSE has nothing to invert. Recording the failure is what
    # justifies using the calibrated score, rather than it looking like a choice
    # made to get a nicer answer.
    priors_raw = estimate_target_priors(
        source.scores,
        source.labels,
        target_estimation.scores,
        target_estimation.labels,
        score_name="raw span score",
    )

    diagnosis: Dict[str, Any] = {
        "seed": seed,
        "method": method,
        "score_key": score_key,
        "n_source_responses": len(source_records),
        "n_target_responses": len(target_records),
        "n_target_estimation_responses": len(estimation_records),
        "n_target_evaluation_responses": len(evaluation_records),
        "target_question_structure": _question_structure(target_records),
        "base_rate": base_rate_diagnosis(source, target_all),
        "conditional_scores": conditional_score_diagnosis(source, target_all),
        "domain_classifier": {
            "features": list(FEATURE_NAMES),
            "held_out_auc": classifier.held_out_auc,
            "coefficients": classifier.coefficients,
            "intercept": classifier.intercept,
            "n_source_spans": classifier.n_source,
            "n_target_spans": classifier.n_target,
            "note": (
                "AUC measured on a held-out third of the pooled spans; the "
                "coefficients come from a refit on all of them, which is what "
                "the importance weights are taken from. Context length is not "
                "among the features because the probability dump does not carry "
                "the context."
            ),
        },
        "label_shift_estimate": priors,
        "label_shift_estimate_on_raw_score": priors_raw,
    }

    baseline = unweighted_transfer(
        sorted_nonconformity,
        evaluation_calibrated,
        target_evaluation.labels,
        target_evaluation.groups,
        alphas,
    )

    repairs: Dict[str, Any] = {
        "seed": seed,
        "method": method,
        "n_calibration_spans": source.n,
        "n_evaluation_spans": target_evaluation.n,
        "source_positive_rate": source.positive_rate,
        "evaluation_positive_rate": target_evaluation.positive_rate,
        "unrepaired": baseline,
    }

    # Label shift, twice: once from the unlabeled estimate a deployment could
    # actually make, and once from the true target prior. The gap between them
    # separates "the estimate was bad" from "the label-shift assumption is
    # wrong", which no single run can tell apart.
    estimated_weights = priors["estimated_weights"]
    if priors["usable"]:
        repairs["label_shift_estimated"] = label_shift_conformal(
            source_nonconformity,
            source.labels,
            estimated_weights,
            evaluation_calibrated,
            target_evaluation.labels,
            target_evaluation.groups,
            alphas,
        )
    else:
        repairs["label_shift_estimated"] = None
        repairs["label_shift_estimated_skipped_because"] = priors["problems"]

    source_prior = np.asarray(priors["source_prior"], dtype=np.float64)
    true_prior = np.asarray(
        [
            1.0 - target_evaluation.positive_rate,
            target_evaluation.positive_rate,
        ],
        dtype=np.float64,
    )
    oracle_weights = (true_prior / np.maximum(source_prior, 1e-12)).tolist()
    repairs["label_shift_oracle"] = label_shift_conformal(
        source_nonconformity,
        source.labels,
        oracle_weights,
        evaluation_calibrated,
        target_evaluation.labels,
        target_evaluation.groups,
        alphas,
    )
    repairs["label_shift_oracle_weights"] = oracle_weights
    repairs["label_shift_oracle_note"] = (
        "uses the evaluation half's true positive rate, which a deployment "
        "cannot know. Reported only to separate a bad prior estimate from a "
        "false label-shift assumption. Never quote it as a result."
    )

    source_weight_report = clip_weights(classifier.weights(source.features))
    target_weight_report = clip_weights(
        classifier.weights(target_evaluation.features)
    )
    repairs["covariate_shift"] = covariate_shift_conformal(
        source_nonconformity,
        source_weight_report["weights"],
        evaluation_calibrated,
        target_weight_report["weights"],
        target_evaluation.labels,
        target_evaluation.groups,
        alphas,
    )
    repairs["covariate_shift_weights"] = {
        "calibration": {
            key: value
            for key, value in source_weight_report.items()
            if key != "weights"
        },
        "evaluation": {
            key: value
            for key, value in target_weight_report.items()
            if key != "weights"
        },
    }

    verdict: List[Dict[str, Any]] = []
    for index, alpha in enumerate(alphas):
        band = coverage_tolerance(alpha, source.n, target_evaluation.n)
        entry: Dict[str, Any] = {"alpha": alpha, "band": band}
        for name in (
            "unrepaired",
            "label_shift_estimated",
            "label_shift_oracle",
            "covariate_shift",
        ):
            rows = repairs.get(name)
            if not rows:
                entry[name] = None
                continue
            row = rows[index]
            entry[name] = {
                "coverage": row["overall"]["empirical_coverage"],
                "shortfall": row["target_coverage"]
                - row["overall"]["empirical_coverage"],
                "abstention_rate": row["overall"]["abstention_rate"],
                "meets_target": meets_target(row, band),
            }
        verdict.append(entry)
    repairs["verdict"] = verdict

    return diagnosis, repairs


# --------------------------------------------------------------------------
# Printing and CLI
# --------------------------------------------------------------------------


def format_diagnosis(diagnosis: Dict[str, Any]) -> str:
    base = diagnosis["base_rate"]
    classifier = diagnosis["domain_classifier"]
    priors = diagnosis["label_shift_estimate"]
    lines = [
        "WHICH SHIFT IS IT?",
        "",
        "1. base rate  P(y)",
        f"   source spans {base['source_n_spans']:,} positive "
        f"{base['source_span_positive_rate']:.4f}",
        f"   target spans {base['target_n_spans']:,} positive "
        f"{base['target_span_positive_rate']:.4f}   "
        f"change {base['absolute_change']:+.4f}  ratio {base['ratio']:.3f}",
        "",
        "2. conditional score distributions  P(score | y)",
        "   pure label shift requires these to be UNCHANGED",
    ]
    for label in (0, 1):
        block = diagnosis["conditional_scores"].get(f"y={label}", {})
        if "ks_statistic" not in block:
            lines.append(f"   y={label}: one corpus has no spans of this class")
            continue
        lines.append(
            f"   y={label}: source mean {block['source']['mean']:.4f} "
            f"(n {block['source']['n']:,})  target mean {block['target']['mean']:.4f} "
            f"(n {block['target']['n']:,})  shift {block['mean_shift']:+.4f}  "
            f"KS {block['ks_statistic']:.4f}"
        )
    lines += [
        "",
        "3. covariate shift  P(x)",
        f"   domain classifier held-out AUC {classifier['held_out_auc']:.4f}  "
        "(0.5 = the two corpora are indistinguishable through the features)",
        "   coefficients, standardised: "
        + ", ".join(
            f"{name} {value:+.3f}" for name, value in classifier["coefficients"].items()
        ),
        "",
        "4. label-shift prior estimate, from UNLABELLED target data",
        f"   predictor              {priors['predictor']}, which calls "
        f"{priors['source_predicted_positive_rate']:.4f} of source spans positive",
        f"   source prior           {priors['source_prior'][1]:.4f} positive",
        f"   BBSE estimate          {priors['estimated_target_prior'][1]:.4f} positive",
        f"   true target prior      "
        f"{priors['true_target_prior_for_reference_only'][1]:.4f} "
        "positive   (reference only, a deployment cannot see this)",
        f"   estimation error       {priors['estimation_error_on_positive_rate']:+.4f}",
        f"   matrix condition       {priors['condition_number']:.1f}",
        f"   usable                 {'yes' if priors['usable'] else 'NO'}",
    ]
    for problem in priors["problems"]:
        lines.append(f"     - {problem}")

    raw = diagnosis.get("label_shift_estimate_on_raw_score")
    if raw is not None:
        lines += [
            "",
            f"   the same estimate on the RAW score: usable "
            f"{'yes' if raw['usable'] else 'NO'}, predictor calls "
            f"{raw['source_predicted_positive_rate']:.4f} of source spans positive",
            "   (a predicted span exists because argmax already called it "
            "hallucinated, so",
            "    the raw-score predictor is constant and BBSE has nothing to "
            "invert)",
        ]
    return "\n".join(lines)


def format_repairs(repairs: Dict[str, Any]) -> str:
    lines = [
        "",
        "DOES REWEIGHTING BRING COVERAGE BACK?",
        f"  calibration {repairs['n_calibration_spans']:,} source spans "
        f"(positive {repairs['source_positive_rate']:.4f})   "
        f"evaluation {repairs['n_evaluation_spans']:,} target spans "
        f"(positive {repairs['evaluation_positive_rate']:.4f})",
        "",
        f"  {'alpha':<7}{'target':<9}{'method':<26}{'coverage':<11}"
        f"{'shortfall':<12}{'abstain':<10}{'in band'}",
        "  " + "-" * 88,
    ]
    labels = {
        "unrepaired": "unrepaired (VOID)",
        "label_shift_estimated": "label shift, estimated",
        "label_shift_oracle": "label shift, oracle prior",
        "covariate_shift": "covariate shift",
    }
    for entry in repairs["verdict"]:
        first = True
        for key, label in labels.items():
            row = entry.get(key)
            alpha_text = f"  {entry['alpha']:<5.2f}  {1 - entry['alpha']:<7.3f}" if first else " " * 16
            first = False
            if row is None:
                lines.append(f"{alpha_text}{label:<26}not run")
                continue
            lines.append(
                f"{alpha_text}{label:<26}{row['coverage']:<11.4f}"
                f"{row['shortfall']:<+12.4f}{row['abstention_rate']:<10.4f}"
                f"{'yes' if row['meets_target'] else 'NO'}"
            )
        lines.append("")
    weights = repairs["covariate_shift_weights"]["calibration"]
    lines.append(
        f"  covariate weights on the calibration set: mean {weights['mean']:.3f}, "
        f"max {weights['max']:.3f}, {weights['fraction_clipped']:.1%} at the clip, "
        f"effective sample size {weights['effective_sample_size']:.0f} "
        f"of {weights['n']:,}"
    )
    lines.append(
        "  A large clipped fraction or a collapsed effective sample size means the\n"
        "  repaired threshold is being decided by the clipping constant, not by the\n"
        "  data. Read those two numbers before reading the coverage column."
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose and try to repair the conformal guarantee under shift."
    )
    parser.add_argument(
        "--source",
        default="results/c1/calib/probabilities.jsonl",
        help="in-domain calibration dump the shipped threshold was fitted on",
    )
    parser.add_argument(
        "--target",
        default="results/ood/ragbench-probs/probabilities.jsonl",
        help="out-of-domain dump, with a `subset` field on every record",
    )
    parser.add_argument("--out-dir", default="results/c2/ragbench")
    parser.add_argument(
        "--method",
        default="platt",
        help=(
            "calibrator to carry over from the in-domain run. Not re-selected "
            "here: selecting against target ECE would need target labels, which "
            "a deployment does not have"
        ),
    )
    parser.add_argument(
        "--score-key", default="mean_prob", choices=["mean_prob", "max_prob", "min_prob"]
    )
    parser.add_argument("--alphas", default="0.05,0.10,0.15,0.20,0.30,0.40")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--estimation-fraction", type=float, default=TARGET_ESTIMATION_FRACTION
    )
    args = parser.parse_args(argv)

    alphas = [float(a) for a in args.alphas.split(",")]
    source_records = read_probability_file(args.source)
    target_records = read_probability_file(args.target)
    print(
        f"source {len(source_records):,} responses | "
        f"target {len(target_records):,} responses"
    )
    if not any("subset" in record for record in target_records):
        print(
            "warning: no record in the target dump carries a `subset` field, so "
            "per-subset coverage will collapse into one group. Re-run "
            "evaluate_ood with --dump-probs to get it."
        )

    diagnosis, repairs = run_shift_study(
        source_records,
        target_records,
        method=args.method,
        score_key=args.score_key,
        alphas=alphas,
        seed=args.seed,
        fraction=args.estimation_fraction,
    )

    print()
    print(format_diagnosis(diagnosis))
    print(format_repairs(repairs))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("shift_diagnosis.json", diagnosis),
        ("repair.json", repairs),
    ):
        path = out_dir / name
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
