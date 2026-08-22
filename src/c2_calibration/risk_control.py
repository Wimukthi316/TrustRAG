"""Conformal risk control: a promise a product manager can act on.

Split conformal gives coverage -- "the true label is in the returned set at
least 1 - alpha of the time". That is a statistician's promise. It says nothing
directly about the thing a person deploying a hallucination detector actually
worries about, which is: **how much of the hallucinated text will this thing let
through?**

Conformal risk control (Angelopoulos, Bates, Fisch, Lei, Schuster, ICLR 2024,
arXiv:2208.02814) generalises the coverage guarantee to any loss that shrinks as
the prediction set grows. Coverage is the special case where the loss is "did we
miss the true label". Here the loss is the **false-negative rate over
hallucinated tokens**: of the tokens a human annotator marked as hallucinated,
what fraction did the system fail to flag. Bounding its expectation gives a
promise in the product's own language:

    at most alpha of hallucinated text goes unflagged, on average over responses

**The construction.** Take a family of decision rules indexed by a threshold t:
flag every token whose P(hallucinated) is at least t. Lower t flags more, so the
per-response false-negative rate L_i(t) is non-decreasing in t. Compute the mean
loss on the calibration split, R(t), and choose

    t_hat = the largest t with   ( n * R(t) + B ) / ( n + 1 )  <=  alpha

where n is the number of calibration responses and B = 1 bounds the loss. The
`+ B` and the `n + 1` are the same finite-sample correction that appears in the
conformal quantile, and they are what makes this hold at the n we actually have
rather than asymptotically. Then E[L_test(t_hat)] <= alpha, over the draw of the
calibration set, with no assumption about the detector.

**The exchangeable unit is the response, and here that falls out for free.** The
loss is defined per response and averaged over responses, so unlike the pooled
token analysis this one never treats two tokens from one answer as two
independent observations. After Block B measured that the pooled token band is
optimistic by about six times, that is worth noticing rather than glossing.

**Two conventions that have to be stated, not assumed.**

1. A response with no hallucinated tokens has no false-negative rate -- there is
   nothing to miss. Counting it as a loss of zero is defensible and is what
   dilutes the average toward zero; excluding it is also defensible and gives
   the stricter reading. Both are computed and reported. The **excluded**
   version is the headline, because a promise about "hallucinated text" should
   be measured on responses that contain some.
2. The threshold is chosen on a grid. A t_hat sitting on the edge of that grid
   is not a chosen threshold, it is the grid running out, and C1's D1 sweep
   already made that mistake once. This module reports the grid edges and
   whether t_hat is on one.

**What a negative result looks like here, and it is the likely one.** Driving
the false-negative rate down means flagging aggressively, and the detector's
scores on hallucinated tokens are not high. If t_hat comes out near zero, the
rule flags nearly every token, the guarantee is satisfied by a system that
highlights the whole answer, and it is useless. So the flag rate is printed
beside every risk, along with the flag-everything baseline, and there is a test
that refuses to let a rule flagging over 90% of tokens be written up as a win.

Run:

    python -m src.c2_calibration.risk_control \\
        --calib results/c1/calib/probabilities.jsonl \\
        --test  results/c1/test/probabilities.jsonl \\
        --out   results/c2/c1/c2_risk_control.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from src.c2_calibration.run_c2 import read_probability_file

SEED = 42
ALPHAS: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
GRID_POINTS = 2001
LOSS_BOUND = 1.0


def gold_token_mask(record: Dict[str, Any]) -> np.ndarray:
    """Which answer tokens overlap a gold hallucinated span.

    Recomputed from `gold_spans` and `answer_offsets` rather than stored, the
    same way `run_c2.token_units` does it, so the two never disagree about what
    a positive token is.
    """
    gold = [(g["start"], g["end"]) for g in record.get("gold_spans", [])]
    offsets = record["answer_offsets"]
    if not gold:
        return np.zeros(len(offsets), dtype=bool)
    starts = np.asarray([o[0] for o in offsets])
    ends = np.asarray([o[1] for o in offsets])
    mask = np.zeros(len(offsets), dtype=bool)
    for gold_start, gold_end in gold:
        mask |= (starts < gold_end) & (gold_start < ends)
    return mask


def positive_probabilities(
    records: Sequence[Dict[str, Any]],
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Per response: the scores of its hallucinated tokens, sorted.

    Returns (sorted score array per response, token count per response, total
    token count per response). Sorting once here is what lets the whole
    threshold grid be evaluated with a searchsorted instead of a scan.
    """
    per_response: List[np.ndarray] = []
    n_positive: List[int] = []
    n_tokens: List[int] = []
    for record in records:
        probs = np.asarray(record["token_probs"], dtype=np.float64)
        mask = gold_token_mask(record)
        if mask.size != probs.size:
            raise ValueError(
                f"record {record.get('id')!r}: {probs.size} token probabilities "
                f"but {mask.size} offsets"
            )
        per_response.append(np.sort(probs[mask]))
        n_positive.append(int(mask.sum()))
        n_tokens.append(int(probs.size))
    return (
        per_response,
        np.asarray(n_positive, dtype=np.int64),
        np.asarray(n_tokens, dtype=np.int64),
    )


def false_negative_matrix(
    sorted_positives: Sequence[np.ndarray], thresholds: np.ndarray
) -> np.ndarray:
    """L[i, j] = fraction of response i's hallucinated tokens missed at thresholds[j].

    A token is flagged when its score is at or above the threshold, so it is
    missed when its score is strictly below. `searchsorted` on the sorted score
    array counts exactly those, for every threshold at once.

    Responses with no hallucinated tokens get a row of zeros. Whether those rows
    belong in the average is a decision the caller makes, not this function.
    """
    losses = np.zeros((len(sorted_positives), thresholds.size), dtype=np.float64)
    for index, scores in enumerate(sorted_positives):
        if scores.size == 0:
            continue
        missed = np.searchsorted(scores, thresholds, side="left")
        losses[index] = missed / scores.size
    return losses


def flag_rates(
    records: Sequence[Dict[str, Any]], thresholds: np.ndarray
) -> Dict[str, np.ndarray]:
    """What flagging at each threshold costs: tokens flagged, responses touched.

    The risk column is meaningless without this one. A rule that satisfies any
    false-negative bound by flagging the entire answer is not a detector, and
    the only thing that makes that visible is the flag rate printed beside it.
    """
    flagged = np.zeros(thresholds.size, dtype=np.float64)
    total = 0
    responses_touched = np.zeros(thresholds.size, dtype=np.float64)
    for record in records:
        probs = np.asarray(record["token_probs"], dtype=np.float64)
        total += probs.size
        if probs.size == 0:
            continue
        ordered = np.sort(probs)
        below = np.searchsorted(ordered, thresholds, side="left")
        flagged += ordered.size - below
        responses_touched += (ordered.size - below) > 0
    return {
        "token_flag_rate": flagged / max(total, 1),
        "response_flag_rate": responses_touched / max(len(records), 1),
        "n_tokens": total,
    }


def choose_threshold(
    losses: np.ndarray,
    thresholds: np.ndarray,
    alpha: float,
    bound: float = LOSS_BOUND,
) -> Dict[str, Any]:
    """The largest threshold whose finite-sample risk bound still clears alpha.

    R(t) is the mean per-response loss on the calibration split. The bound is

        ( n * R(t) + B ) / ( n + 1 )  <=  alpha

    and t_hat is the largest grid point satisfying it. Larger t means fewer
    tokens flagged, so taking the largest is taking the cheapest rule that still
    carries the promise.

    Three outcomes, and all three are reported rather than collapsed:

      * a threshold strictly inside the grid -- the normal case;
      * the smallest grid point, meaning even flagging everything at that
        threshold only just satisfies the bound, and the answer is being decided
        by where the grid stops rather than by the data;
      * no grid point at all, meaning this alpha is unreachable with this
        detector and this much calibration data. That is a real answer and it is
        returned as such, not as a threshold of zero.
    """
    n = losses.shape[0]
    if n == 0:
        raise ValueError("cannot choose a threshold from an empty calibration set")
    risk = losses.mean(axis=0)
    corrected = (n * risk + bound) / (n + 1)
    feasible = corrected <= alpha

    if not feasible.any():
        return {
            "alpha": alpha,
            "n_calibration_responses": n,
            "threshold": None,
            "feasible": False,
            "reason": (
                f"no threshold on the grid satisfies the bound at alpha={alpha}. "
                f"The smallest corrected risk available is {corrected.min():.4f}, "
                f"and with n={n} the correction alone contributes "
                f"{bound / (n + 1):.4f}"
            ),
            "smallest_corrected_risk": float(corrected.min()),
            "correction_floor": float(bound / (n + 1)),
        }

    index = int(np.max(np.flatnonzero(feasible)))
    on_edge = index in (0, thresholds.size - 1)
    return {
        "alpha": alpha,
        "n_calibration_responses": n,
        "threshold": float(thresholds[index]),
        "feasible": True,
        "calibration_risk": float(risk[index]),
        "calibration_risk_corrected": float(corrected[index]),
        "correction_floor": float(bound / (n + 1)),
        "grid_index": index,
        "grid_size": int(thresholds.size),
        "on_grid_edge": bool(on_edge),
        "grid_edge_warning": (
            "t_hat sits on the edge of the search grid, so it is where the grid "
            "stopped and not where the data pointed. Widen the grid or report "
            "the level as unreached."
            if on_edge
            else ""
        ),
    }


def evaluate_threshold(
    records: Sequence[Dict[str, Any]], threshold: float
) -> Dict[str, float]:
    """Empirical false-negative rate and flag cost of one threshold, on a test set."""
    sorted_positives, n_positive, _ = positive_probabilities(records)
    grid = np.asarray([threshold], dtype=np.float64)
    losses = false_negative_matrix(sorted_positives, grid)[:, 0]
    has_positives = n_positive > 0
    cost = flag_rates(records, grid)
    return {
        "threshold": float(threshold),
        "fnr_over_responses_with_hallucinations": (
            float(losses[has_positives].mean()) if has_positives.any() else float("nan")
        ),
        "fnr_over_all_responses": float(losses.mean()),
        "n_responses": int(len(records)),
        "n_responses_with_hallucinations": int(has_positives.sum()),
        "token_flag_rate": float(cost["token_flag_rate"][0]),
        "response_flag_rate": float(cost["response_flag_rate"][0]),
        "n_tokens": int(cost["n_tokens"]),
    }


def run_risk_control(
    calib_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    alphas: Sequence[float] = ALPHAS,
    grid_points: int = GRID_POINTS,
    restrict_to_responses_with_hallucinations: bool = True,
) -> Dict[str, Any]:
    """Choose a threshold per alpha on calibration, apply each once to test."""
    thresholds = np.linspace(0.0, 1.0, grid_points)

    sorted_positives, n_positive, _ = positive_probabilities(calib_records)
    all_losses = false_negative_matrix(sorted_positives, thresholds)
    keep = (
        n_positive > 0
        if restrict_to_responses_with_hallucinations
        else np.ones(len(calib_records), dtype=bool)
    )
    losses = all_losses[keep]

    trivial = evaluate_threshold(test_records, 0.0)

    # The bound is a promise about the risk that was averaged on calibration, so
    # it has to be judged against the same average on test. Calibrating on the
    # undiluted loss and then checking the diluted one -- or the reverse -- makes
    # a correct run look like a violation, which is a mistake worth being
    # explicit about rather than quietly getting right.
    judged_by = (
        "fnr_over_responses_with_hallucinations"
        if restrict_to_responses_with_hallucinations
        else "fnr_over_all_responses"
    )

    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        chosen = choose_threshold(losses, thresholds, alpha)
        row: Dict[str, Any] = {"chosen": chosen}
        if chosen["feasible"]:
            row["test"] = evaluate_threshold(test_records, chosen["threshold"])
            row["judged_by"] = judged_by
            row["test_risk"] = row["test"][judged_by]
            row["bound_held"] = bool(row["test_risk"] <= alpha)
        else:
            row["test"] = None
            row["judged_by"] = judged_by
            row["test_risk"] = None
            row["bound_held"] = None
        rows.append(row)

    return {
        "loss": "per-response false-negative rate over gold hallucinated tokens",
        "convention": (
            "responses with no hallucinated tokens are excluded from the "
            "calibration average"
            if restrict_to_responses_with_hallucinations
            else "responses with no hallucinated tokens count as a loss of zero"
        ),
        "grid": {"points": grid_points, "low": 0.0, "high": 1.0},
        "n_calibration_responses_used": int(losses.shape[0]),
        "n_calibration_responses_total": len(calib_records),
        "n_test_responses": len(test_records),
        "judged_by": judged_by,
        "flag_everything_baseline": trivial,
        "rows": rows,
    }


def format_risk_control(result: Dict[str, Any]) -> str:
    baseline = result["flag_everything_baseline"]
    lines = [
        "CONFORMAL RISK CONTROL -- bounding missed hallucinated tokens",
        f"  loss: {result['loss']}",
        f"  {result['convention']}",
        f"  calibration responses used {result['n_calibration_responses_used']:,} "
        f"of {result['n_calibration_responses_total']:,}   "
        f"test responses {result['n_test_responses']:,}",
        "",
        "  flag-everything baseline (threshold 0.0): "
        f"FNR {baseline['fnr_over_responses_with_hallucinations']:.4f}, "
        f"tokens flagged {baseline['token_flag_rate']:.4f}",
        "  Any rule whose flag rate approaches that one has stopped being a "
        "detector.",
        "",
        f"  judged on test by: {result['judged_by']}",
        "",
        f"  {'alpha':<7}{'t_hat':<9}{'calib risk':<12}{'test risk':<11}"
        f"{'tokens flagged':<16}{'responses hit':<15}{'held'}",
        "  " + "-" * 78,
    ]
    for row in result["rows"]:
        chosen = row["chosen"]
        if not chosen["feasible"]:
            lines.append(
                f"  {chosen['alpha']:<7.2f}{'none':<9}"
                f"unreachable: {chosen['reason']}"
            )
            continue
        test = row["test"]
        lines.append(
            f"  {chosen['alpha']:<7.2f}{chosen['threshold']:<9.4f}"
            f"{chosen['calibration_risk']:<12.4f}"
            f"{row['test_risk']:<11.4f}"
            f"{test['token_flag_rate']:<16.4f}"
            f"{test['response_flag_rate']:<15.4f}"
            f"{'yes' if row['bound_held'] else 'NO'}"
        )
        if chosen["on_grid_edge"]:
            lines.append(f"           {chosen['grid_edge_warning']}")
    lines.append(
        "\n  'held' compares the test FNR against alpha. The guarantee is in "
        "expectation\n  over calibration draws, so a single run landing just "
        "above alpha is not a\n  violation on its own -- the same marginal "
        "caveat as coverage."
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Conformal risk control over token-level false negatives."
    )
    parser.add_argument("--calib", default="results/c1/calib/probabilities.jsonl")
    parser.add_argument("--test", default="results/c1/test/probabilities.jsonl")
    parser.add_argument("--out", default="results/c2/c1/c2_risk_control.json")
    parser.add_argument("--alphas", default="0.05,0.10,0.15,0.20,0.30,0.40")
    parser.add_argument("--grid-points", type=int, default=GRID_POINTS)
    parser.add_argument(
        "--include-clean-responses",
        action="store_true",
        help=(
            "count responses with no hallucinated tokens as a loss of zero "
            "instead of excluding them. Dilutes the risk toward zero; reported "
            "for completeness, not as the headline"
        ),
    )
    args = parser.parse_args(argv)

    alphas = [float(a) for a in args.alphas.split(",")]
    calib = read_probability_file(args.calib)
    test = read_probability_file(args.test)
    print(f"calibration {len(calib):,} responses | test {len(test):,} responses")

    result = run_risk_control(
        calib,
        test,
        alphas,
        args.grid_points,
        restrict_to_responses_with_hallucinations=not args.include_clean_responses,
    )
    result["calib_file"] = args.calib
    result["test_file"] = args.test
    print()
    print(format_risk_control(result))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
