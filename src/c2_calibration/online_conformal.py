"""Online (adaptive) conformal prediction: repair coverage under domain shift
with a human in the loop.

Where this sits. Split conformal (`conformal.py`) carries its coverage guarantee
only while the test data is exchangeable with the calibration split. `shift.py`
measured what happens when it is not: on RAGBench the empirical span-level
coverage at alpha = 0.10 falls from 0.8808 in domain to 0.7372, a seven-band
miss, and both static repairs -- label-shift reweighting and covariate-shift
reweighting -- failed to close it. The conclusion recorded there was an
instruction addressed to a person: recalibrate on your own data before you trust
the dial.

This module turns that instruction into a mechanism. Adaptive Conformal Inference
(ACI; Gibbs & Candes, "Adaptive Conformal Inference Under Distribution Shift",
NeurIPS 2021, arXiv:2106.00170 -- identifier confirmed 2026-10-01) keeps a single
running miscoverage level alpha_t. After
each answer the system shows a reviewer, the reviewer says whether the span was
really unsupported, and that one bit of feedback updates the level:

        alpha_{t+1} = alpha_t + gamma * (alpha_target - err_t)

err_t is 1 when the true label fell outside the prediction set and 0 when it was
covered. That update is the entire method. It uses no model of the shift, no
density ratio, and no assumption that the new domain resembles the old one --
which is exactly the assumption split conformal lost. In return it gives a
LONG-RUN guarantee: the realized miscoverage averaged over the stream converges
to alpha_target for any sequence, adversarial ones included. It does not promise
coverage on any single step, and it pays for the recovered coverage honestly, in
wider prediction sets and a higher abstention rate.

The story boundary with the shift study is kept. `shift.py` owns "the guarantee
does not survive the shift, and exchangeability is what breaks." This module owns
"a feedback loop restores long-run coverage without exchangeability, and here is
what it costs." Neither claims the other's ground.

The human is simulated by the gold RAGBench label. This is the standard way ACI
is evaluated -- the feedback signal is the revealed truth -- and it is a
simulation of a review loop, not a measurement of one with real reviewers. Every
result this module writes is labelled that way, and the honest limitation is
stated once here so it is not mistaken later: full ACI assumes feedback on every
step. A deployment reviews only the spans it shows a human, so the review-only
variant computed alongside is the realistic one, and it is weaker.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.c2_calibration.calibration import calibrator_from_dict
from src.c2_calibration.conformal import conformal_quantile
from src.common.schema import ConformalDecision

SEED = 42
ALPHA_TARGET = 0.10
# Learning rate for the alpha_t update. Small enough that a single review cannot
# swing the dial, large enough to track a shift over a few hundred spans. The
# runner sweeps {0.01, 0.02, 0.05} and reports the sensitivity rather than
# hiding it behind one hand-picked value.
DEFAULT_GAMMA = 0.02
GAMMA_SWEEP = (0.01, 0.02, 0.05)


def nonconformity(prob: float, label: int) -> float:
    """LAC non-conformity: 1 - p(true label). Matches SplitConformal.fit."""
    return float(1.0 - prob) if label == 1 else float(prob)


def _effective_threshold(sorted_nonconformity: np.ndarray, alpha_t: float) -> float:
    """The calibration quantile at the running level, with the two edges handled.

    ACI lets alpha_t leave [0, 1]. At or below 0 the method wants to cover every
    label, so the threshold is +inf; at or above 1 it wants to cover none, so the
    threshold is -inf. Between, it is the ordinary finite-sample conformal
    quantile computed on the fixed calibration scores -- the same quantile the
    static method uses, only at a level that moves.
    """
    if alpha_t <= 0.0:
        return float("inf")
    if alpha_t >= 1.0:
        return float("-inf")
    return conformal_quantile(sorted_nonconformity, alpha_t)


def _prediction_set(prob: float, threshold: float) -> Tuple[int, ...]:
    keep_zero = prob <= threshold
    keep_one = (1.0 - prob) <= threshold
    return tuple(label for label, keep in ((0, keep_zero), (1, keep_one)) if keep)


def _decision(candidate: Tuple[int, ...]) -> ConformalDecision:
    if candidate == (1,):
        return ConformalDecision.FLAG
    if candidate == (0,):
        return ConformalDecision.PASS
    return ConformalDecision.ABSTAIN


@dataclass
class AdaptiveConformal:
    """One running conformal predictor whose miscoverage level tracks feedback.

    `calibration_nonconformity` is the fixed set of calibration scores the
    threshold is read off, already on the calibrated scale (the served artifact
    stores exactly this array). `alpha_target` is the coverage the loop steers
    towards; `gamma` is how hard each review pushes the dial.
    """

    calibration_nonconformity: np.ndarray
    alpha_target: float = ALPHA_TARGET
    gamma: float = DEFAULT_GAMMA
    review_only: bool = False
    alpha_t: float = field(init=False)
    _sorted: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._sorted = np.sort(np.asarray(self.calibration_nonconformity, dtype=np.float64))
        if self._sorted.size == 0:
            raise ValueError("cannot run ACI on an empty calibration set")
        self.alpha_t = float(self.alpha_target)

    def step(self, prob: float, label: int) -> Dict[str, Any]:
        """Predict for one span at the current level, then fold in the feedback.

        Returns the decision taken *before* the update, so a caller reading the
        record sees what the reviewer actually saw.
        """
        threshold = _effective_threshold(self._sorted, self.alpha_t)
        candidate = _prediction_set(prob, threshold)
        decision = _decision(candidate)
        covered = label in candidate
        err = 0.0 if covered else 1.0

        alpha_before = self.alpha_t
        # review_only: the dial only learns from spans a human would actually be
        # shown (FLAG or ABSTAIN). A PASS is never reviewed, so it carries no
        # feedback. This is the realistic loop and it is weaker than full ACI.
        reviewed = (not self.review_only) or decision is not ConformalDecision.PASS
        if reviewed:
            self.alpha_t = float(
                min(1.0, max(0.0, self.alpha_t + self.gamma * (self.alpha_target - err)))
            )

        return {
            "prob": float(prob),
            "label": int(label),
            "alpha_t": float(alpha_before),
            "threshold": float(threshold) if math.isfinite(threshold) else None,
            "set": list(candidate),
            "set_size": len(candidate),
            "decision": decision.value,
            "covered": bool(covered),
            "reviewed": bool(reviewed),
        }


def static_threshold(calibration_nonconformity: Sequence[float], alpha: float) -> float:
    return conformal_quantile(calibration_nonconformity, alpha)


def _rates(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n = len(records)
    if n == 0:
        return {"coverage": 0.0, "flag_rate": 0.0, "abstain_rate": 0.0, "mean_set_size": 0.0}
    covered = sum(1 for r in records if r["covered"])
    flag = sum(1 for r in records if r["decision"] == "flag")
    abstain = sum(1 for r in records if r["decision"] == "abstain")
    size = sum(r["set_size"] for r in records)
    return {
        "coverage": covered / n,
        "flag_rate": flag / n,
        "abstain_rate": abstain / n,
        "mean_set_size": size / n,
    }


def run_stream(
    calibration_nonconformity: Sequence[float],
    scores: Sequence[float],
    labels: Sequence[int],
    alpha_target: float = ALPHA_TARGET,
    gamma: float = DEFAULT_GAMMA,
    shuffle: bool = True,
    seed: int = SEED,
    review_only: bool = False,
) -> Dict[str, Any]:
    """Run the adaptive loop over a stream and compare it to the static method.

    The static baseline fixes the threshold at `alpha_target` on the same
    calibration scores -- it is exactly what the deployed artifact does today,
    and on a shifted stream it is what under-covers. The adaptive loop sees the
    same spans in the same order and is judged on the same coverage.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must be the same length")
    n = scores.size
    if n == 0:
        raise ValueError("empty stream")

    order = np.arange(n)
    if shuffle:
        rng = random.Random(seed)
        order = np.array(rng.sample(range(n), n))

    aci = AdaptiveConformal(
        np.asarray(calibration_nonconformity, dtype=np.float64),
        alpha_target=alpha_target,
        gamma=gamma,
        review_only=review_only,
    )
    static_q = static_threshold(calibration_nonconformity, alpha_target)

    steps: List[Dict[str, Any]] = []
    static_records: List[Dict[str, Any]] = []
    cum_cov_aci = 0
    cum_cov_static = 0
    rolling_aci: List[float] = []
    rolling_static: List[float] = []
    alpha_path: List[float] = []
    reviews = 0

    for t, idx in enumerate(order, start=1):
        p = float(scores[idx])
        y = int(labels[idx])

        record = aci.step(p, y)
        steps.append(record)
        alpha_path.append(record["alpha_t"])
        if record["reviewed"]:
            reviews += 1
        cum_cov_aci += 1 if record["covered"] else 0
        rolling_aci.append(cum_cov_aci / t)

        s_set = _prediction_set(p, static_q)
        s_covered = y in s_set
        static_records.append(
            {"covered": s_covered, "decision": _decision(s_set).value, "set_size": len(s_set)}
        )
        cum_cov_static += 1 if s_covered else 0
        rolling_static.append(cum_cov_static / t)

    aci_rates = _rates(steps)
    static_rates = _rates(static_records)
    target = 1.0 - alpha_target

    return {
        "alpha_target": alpha_target,
        "target_coverage": target,
        "gamma": gamma,
        "n_stream": int(n),
        "shuffle": shuffle,
        "seed": seed,
        "review_only": review_only,
        "static_threshold": float(static_q),
        "final_alpha_t": float(aci.alpha_t),
        "reviews": int(reviews),
        "review_fraction": reviews / n,
        "adaptive": aci_rates,
        "static": static_rates,
        # Long-run coverage is the headline number: the average over the whole
        # stream, which is what ACI's guarantee is about. Per-step coverage is
        # not claimed.
        "long_run_coverage_adaptive": aci_rates["coverage"],
        "long_run_coverage_static": static_rates["coverage"],
        "meets_target_adaptive": bool(target - aci_rates["coverage"] <= 0.01),
        "meets_target_static": bool(target - static_rates["coverage"] <= 0.01),
        "rolling_coverage_adaptive": rolling_aci,
        "rolling_coverage_static": rolling_static,
        "alpha_path": alpha_path,
    }


# --------------------------------------------------------------------------
# Loading the calibration scores and the stream from artefacts on disk
# --------------------------------------------------------------------------


def calibration_nonconformity_from_artifact(path: Path | str) -> Tuple[List[float], Dict[str, Any]]:
    """The calibrated non-conformity array the served artifact already carries.

    Reusing it rather than recomputing guarantees the online loop starts from the
    exact calibration the backend deploys, and it needs no access to the raw
    RAGTruth calibration dump.
    """
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if "nonconformity_sorted" not in artifact:
        raise ValueError(f"{path} is not a C2 serving artifact (no nonconformity_sorted)")
    return list(artifact["nonconformity_sorted"]), artifact


def stream_from_probability_file(
    path: Path | str,
    calibrator_payload: Dict[str, Any],
    score_key: str = "mean_prob",
    group_key: str = "subset",
) -> Tuple[List[float], List[int], List[str]]:
    """One row per predicted span, calibrated with the deployed calibrator.

    The calibration is applied here for the same reason the static pipeline
    applies it: so the score the loop thresholds is on the calibrated scale the
    artifact was built on. LAC is invariant to a monotone recalibration, so this
    does not change coverage; it keeps the two paths honest with each other.
    """
    calibrator = calibrator_from_dict(calibrator_payload)
    scores: List[float] = []
    labels: List[int] = []
    groups: List[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            group = str(record.get(group_key, "unknown"))
            for span in record.get("pred_spans", []):
                scores.append(float(span[score_key]))
                labels.append(int(bool(span["is_hallucinated"])))
                groups.append(group)
    calibrated = list(calibrator.transform(scores)) if scores else []
    return calibrated, labels, groups


def _downsample(values: Sequence[float], k: int = 300) -> List[Dict[str, float]]:
    """Thin a long per-step series to about k points for the interface."""
    n = len(values)
    if n == 0:
        return []
    if n <= k:
        return [{"t": i + 1, "v": float(v)} for i, v in enumerate(values)]
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return [{"t": int(i) + 1, "v": float(values[i])} for i in idx]


def build_timeline(result: Dict[str, Any], k: int = 300) -> Dict[str, Any]:
    """The compact series the frontend animates. No arithmetic the report needs."""
    return {
        "n_stream": result["n_stream"],
        "alpha_target": result["alpha_target"],
        "target_coverage": result["target_coverage"],
        "gamma": result["gamma"],
        "final_alpha_t": result["final_alpha_t"],
        "long_run_coverage_adaptive": result["long_run_coverage_adaptive"],
        "long_run_coverage_static": result["long_run_coverage_static"],
        "adaptive": result["adaptive"],
        "static": result["static"],
        "review_fraction": result["review_fraction"],
        "coverage_adaptive": _downsample(result["rolling_coverage_adaptive"], k),
        "coverage_static": _downsample(result["rolling_coverage_static"], k),
        "alpha_path": _downsample(result["alpha_path"], k),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adaptive (online) conformal: repair coverage under shift with feedback."
    )
    parser.add_argument(
        "--artifact",
        default="results/c2/c1/c2_artifact.json",
        help="C2 serving artifact carrying the calibrator and calibration scores",
    )
    parser.add_argument(
        "--stream",
        default="results/ood/ragbench-probs/probabilities.jsonl",
        help="shifted probability dump to stream (RAGBench)",
    )
    parser.add_argument("--alpha", type=float, default=ALPHA_TARGET)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--score-key", default="mean_prob")
    parser.add_argument("--group-key", default="subset")
    parser.add_argument("--out-dir", default="results/c2/online")
    parser.add_argument(
        "--ui-out",
        default=None,
        help="optional path to also write the compact timeline the frontend reads",
    )
    args = parser.parse_args()

    calib_nonconf, artifact = calibration_nonconformity_from_artifact(args.artifact)
    scores, labels, groups = stream_from_probability_file(
        args.stream, artifact["calibrator"], args.score_key, args.group_key
    )
    if not scores:
        raise SystemExit("the stream produced no predicted spans; nothing to run")

    print(
        f"calibration scores {len(calib_nonconf):,} | stream spans {len(scores):,} "
        f"(positive rate {sum(labels) / len(labels):.4f})"
    )

    # Headline run: shuffled stream, full feedback, default gamma.
    headline = run_stream(
        calib_nonconf, scores, labels, args.alpha, args.gamma, shuffle=True, seed=SEED
    )
    print(
        f"\nstatic coverage {headline['long_run_coverage_static']:.4f} "
        f"(target {headline['target_coverage']:.2f})  ->  "
        f"adaptive coverage {headline['long_run_coverage_adaptive']:.4f} "
        f"[gamma={args.gamma}, final alpha_t={headline['final_alpha_t']:.4f}]"
    )
    print(
        f"cost of the repair: abstention {headline['static']['abstain_rate']:.3f} -> "
        f"{headline['adaptive']['abstain_rate']:.3f}, "
        f"mean|C| {headline['static']['mean_set_size']:.3f} -> "
        f"{headline['adaptive']['mean_set_size']:.3f}"
    )

    # Gamma sensitivity, so no single learning rate is taken on trust.
    gamma_rows = []
    for g in GAMMA_SWEEP:
        r = run_stream(calib_nonconf, scores, labels, args.alpha, g, shuffle=True, seed=SEED)
        gamma_rows.append(
            {
                "gamma": g,
                "coverage": r["long_run_coverage_adaptive"],
                "abstain_rate": r["adaptive"]["abstain_rate"],
                "final_alpha_t": r["final_alpha_t"],
            }
        )

    # Review-only feedback: the realistic, weaker loop.
    review_only = run_stream(
        calib_nonconf, scores, labels, args.alpha, args.gamma,
        shuffle=True, seed=SEED, review_only=True,
    )

    # Natural (subset-grouped) order: a harder test where the domain changes in
    # blocks and the loop has to re-adapt at each boundary.
    natural = run_stream(
        calib_nonconf, scores, labels, args.alpha, args.gamma, shuffle=False, seed=SEED
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full = {
        "detector": artifact.get("detector"),
        "calibrator": artifact.get("calibrator"),
        "stream_file": str(args.stream),
        "n_calibration": len(calib_nonconf),
        "note": (
            "The reviewer is simulated by the gold RAGBench label, the standard "
            "ACI evaluation. This is a simulation of a review loop, not a "
            "measurement of one with real reviewers."
        ),
        "headline": headline,
        "gamma_sweep": gamma_rows,
        "review_only": review_only,
        "natural_order": natural,
    }
    results_path = out_dir / "aci_results.json"
    results_path.write_text(json.dumps(full, indent=2), encoding="utf-8")
    print(f"\nwrote {results_path}")

    timeline = {
        "note": full["note"],
        "detector": artifact.get("detector"),
        "headline": build_timeline(headline),
        "review_only": build_timeline(review_only),
        "natural_order": build_timeline(natural),
        "gamma_sweep": gamma_rows,
    }
    ui_path = Path(args.ui_out) if args.ui_out else out_dir / "aci_timeline.json"
    ui_path.parent.mkdir(parents=True, exist_ok=True)
    ui_path.write_text(json.dumps(timeline), encoding="utf-8")
    print(f"wrote {ui_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
