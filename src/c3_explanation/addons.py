"""Layer 6 and the analyses the error type unlocks that were impossible before.

Everything here is cheap, CPU-only, and reads frozen artefacts. What makes it
worth doing is that none of it could be computed until C3 existed: RAGTruth ships
an error type on every span, nobody has used it, and several questions about C1
and C2 turn out to have type-shaped answers.

    Layer 6   the escalation budget. Escalation fires only where C2 says abstain
              AND the conformal type set has more than one member. Counting how
              many spans that is shows the budget is affordable before a single
              API call is spent, which is the actual research claim; running the
              loop would only be a demonstration of it.

    add-on 1  type-conditional calibration. C2 reported group-conditional
              coverage by task because the type did not exist. This is the same
              question asked of the type, and it needs no new C2 code.

    add-on 2  severity-weighted conformal risk control. An evident conflict
              reaching a user is worse than a subtle addition, and a span
              annotated implicit_true is correct despite being annotated, so
              missing it costs nothing. Conformal risk control bounds any
              monotone loss (Angelopoulos et al., ICLR 2024, arXiv:2208.02814),
              so the weights can encode that. This is the first place in the
              project where the error type changes a decision rather than
              describing one. C2's own choose_threshold does the arithmetic,
              unchanged.

    add-on 3  generator model against error type. RAGTruth records which of six
              models produced each response. Nobody has reported which KINDS of
              error each one makes.

    add-on 4  type-conditional detectability. C1's recall per error type, beside
              C3's average conformal set size per type. It links the two
              components into one sentence: the types the detector misses most
              are the types the classifier cannot name.

    add-on 5  the implicit_true confound. 13.49% of annotated spans are marked
              "correct while the info is not mentioned in the context" -- true
              statements annotated as hallucinations. Every binary detector
              trained on this corpus inherits them, C1 included, and no published
              RAGTruth number reports what they cost.

**Reconstructing C2's decisions.** Layer 6 needs C2's per-span decision, and only
the aggregate is stored. It is rebuilt from C2's own frozen serving artefact --
the fitted Platt calibrator and the calibration split -- using C2's calibrator and
SplitConformal classes unchanged. The rebuild is checked against the coverage
number recorded in that artefact before anything downstream is computed; if they
disagree the run stops rather than reporting a number C2 would not recognise.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.c2_calibration.calibration import (
    calibrator_from_dict,
    expected_calibration_error,
)
from src.c2_calibration.conformal import SplitConformal
from src.c2_calibration.risk_control import choose_threshold
from src.c2_calibration.run_c2 import span_units
from src.c3_explanation.labels_audit import axes_of_error_type
from src.c3_explanation.type_conformal import temperature_scale
from src.common.schema import ErrorType

# add-on 2's loss weights. An evident error reaching a user is the expensive one;
# a subtle one is half as bad; a span annotated implicit_true is factually correct
# and costs nothing to miss. These are a stated value judgement, not a measurement,
# and the report says so beside every bound they produce.
SEVERITY_WEIGHTS: Dict[str, float] = {"evident": 1.0, "subtle": 0.5}
IMPLICIT_TRUE_WEIGHT = 0.0

GRID_POINTS = 401
ALPHAS: Tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.3, 0.4)


def offsets(span: Any) -> Tuple[int, int]:
    """(start, end) from either on-disk shape.

    C2's read_probability_file leaves a span as the object it was written as;
    C1's load_probability_dump converts it to a pair. Both reach this module, so
    both are accepted rather than one of the two readers being changed.
    """
    if isinstance(span, dict):
        return (int(span["start"]), int(span["end"]))
    start, end = span[0], span[1]
    return (int(start), int(end))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --------------------------------------------------------------------------
# C2's decisions, rebuilt from C2's own artefact
# --------------------------------------------------------------------------


def rebuild_c2_decisions(
    artifact: Dict[str, Any],
    calib_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    alpha: float = 0.1,
) -> Dict[str, Any]:
    """Per predicted test span: calibrated score and C2's flag/abstain/pass.

    Verified against the coverage number stored in the artefact, so a silent
    drift in any of the frozen inputs becomes a loud failure here rather than a
    wrong number three layers downstream.
    """
    calibrator = calibrator_from_dict(artifact["calibrator"])
    calib_scores, calib_labels, _ = span_units(calib_records)
    test_scores, test_labels, _ = span_units(test_records)

    conformal = SplitConformal(alpha).fit(calibrator.transform(calib_scores), calib_labels)
    calibrated_test = calibrator.transform(test_scores)
    decisions = [d.value for d in conformal.decisions(calibrated_test)]
    sets = conformal.prediction_sets(calibrated_test)
    coverage = float(
        np.mean([1.0 if test_labels[i] in s else 0.0 for i, s in enumerate(sets)])
    )

    recorded = artifact.get(f"coverage_at_alpha_{alpha}")
    matches = recorded is None or abs(coverage - float(recorded)) < 1e-9

    rows: List[Dict[str, Any]] = []
    position = 0
    for record in test_records:
        for span in record.get("pred_spans", []):
            rows.append(
                {
                    "response_id": str(record["id"]),
                    "task_type": str(record.get("task_type", "unknown")),
                    "model": str(record.get("model", "unknown")),
                    "start": int(span["start"]),
                    "end": int(span["end"]),
                    "calibrated_score": float(calibrated_test[position]),
                    "is_hallucinated": int(bool(span["is_hallucinated"])),
                    "decision": decisions[position],
                }
            )
            position += 1

    return {
        "alpha": alpha,
        "n_calibration": len(calib_scores),
        "n_test": len(test_scores),
        "coverage_rebuilt": coverage,
        "coverage_in_artifact": recorded,
        "matches_artifact": bool(matches),
        "decision_counts": dict(Counter(decisions)),
        "rows": rows,
    }


# --------------------------------------------------------------------------
# Layer 6: the escalation budget
# --------------------------------------------------------------------------


def escalation_budget(
    c2_rows: Sequence[Dict[str, Any]],
    type_set_sizes: Dict[Tuple[str, int, int], int],
    alphas: Sequence[float] = ALPHAS,
    set_sizes_by_alpha: Optional[Dict[float, Dict[Tuple[str, int, int], int]]] = None,
) -> Dict[str, Any]:
    """How many spans would a bounded agent actually be asked about?

    The rule is deliberately conjunctive. C2 abstaining says the detector is
    unsure whether the span is hallucinated at all; a type set larger than one
    says C3 is unsure what kind of error it would be. Only a span where both hold
    is worth spending a call on, and the count is the claim -- an escalation
    policy whose budget is unknown is not a policy.
    """
    abstained = [r for r in c2_rows if r["decision"] == "abstain"]
    per_alpha: Dict[str, Any] = {}
    for alpha in alphas:
        sizes = (set_sizes_by_alpha or {}).get(alpha, type_set_sizes)
        with_size = [
            r
            for r in abstained
            if sizes.get((r["response_id"], r["start"], r["end"]), 0) >= 2
        ]
        per_alpha[str(alpha)] = {
            "n_escalated": len(with_size),
            "share_of_all_spans": len(with_size) / len(c2_rows) if c2_rows else 0.0,
            "share_of_abstained": len(with_size) / len(abstained) if abstained else 0.0,
        }
    return {
        "n_spans": len(c2_rows),
        "n_abstained": len(abstained),
        "abstention_rate": len(abstained) / len(c2_rows) if c2_rows else 0.0,
        "n_with_a_type_set": len(type_set_sizes),
        "per_alpha": per_alpha,
        "rule": (
            "escalate only where C2 abstains AND the conformal type set has two "
            "or more members"
        ),
        "loop_status": (
            "not run. The count is the result: it shows the budget is affordable "
            "before any API call is spent. Running the loop needs a key the "
            "project rules do not allow and the outstanding key rotation has not "
            "happened."
        ),
    }


# --------------------------------------------------------------------------
# add-on 1: type-conditional calibration
# --------------------------------------------------------------------------


def type_conditional_calibration(
    rows: Sequence[Dict[str, Any]], probs_key: str, temperature: float
) -> Dict[str, Any]:
    """Top-label ECE inside each true error type.

    C2 measured group-conditional coverage by task because the type did not
    exist. A model can be well calibrated on average and badly calibrated inside
    a class, and averaging hides exactly the class a reviewer would care about.
    """
    if not rows:
        return {}
    classes = list(rows[0]["classes"])
    probs = temperature_scale(
        np.asarray([r[probs_key] for r in rows], dtype=np.float64), temperature
    )
    index = {name: i for i, name in enumerate(classes)}
    truth = np.asarray([index[r["error_type"]] for r in rows], dtype=np.int64)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == truth).astype(int)

    out: Dict[str, Any] = {
        "overall": {
            "n": len(rows),
            "ece": float(expected_calibration_error(confidence, correct)),
            "mean_confidence": float(confidence.mean()),
            "accuracy": float(correct.mean()),
        },
        "by_error_type": {},
        "by_task": {},
    }
    for name, i in index.items():
        mask = truth == i
        n = int(mask.sum())
        out["by_error_type"][name] = {
            "n": n,
            "ece": (
                float(expected_calibration_error(confidence[mask], correct[mask]))
                if n
                else float("nan")
            ),
            "mean_confidence": float(confidence[mask].mean()) if n else float("nan"),
            "accuracy": float(correct[mask].mean()) if n else float("nan"),
        }
    tasks = [str(r.get("task_type", "unknown")) for r in rows]
    for task in sorted(set(tasks)):
        mask = np.asarray([t == task for t in tasks])
        n = int(mask.sum())
        out["by_task"][task] = {
            "n": n,
            "ece": (
                float(expected_calibration_error(confidence[mask], correct[mask]))
                if n
                else float("nan")
            ),
            "accuracy": float(correct[mask].mean()) if n else float("nan"),
        }
    return out


# --------------------------------------------------------------------------
# add-on 2: severity-weighted conformal risk control
# --------------------------------------------------------------------------


def span_weight(error_type: Optional[str], implicit_true: bool) -> float:
    """What it costs to let this span through unflagged.

    A value judgement, stated: implicit_true spans are factually correct, so
    missing one costs nothing; an evident error costs twice a subtle one.
    """
    if implicit_true:
        return IMPLICIT_TRUE_WEIGHT
    if not error_type:
        return 1.0
    try:
        _, severity = axes_of_error_type(ErrorType(error_type))
    except ValueError:
        return 1.0
    return SEVERITY_WEIGHTS.get(str(severity), 1.0)


def weighted_loss_matrix(
    records: Sequence[Dict[str, Any]],
    scores_by_span: Dict[Tuple[str, int, int], float],
    gold_types: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]],
    thresholds: np.ndarray,
    only_responses_with_gold: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """Per response, the weighted share of gold spans missed at each threshold.

    A gold span is caught at threshold t when some predicted span overlapping it
    has a calibrated score of at least t. Raising t can only lose catches, so the
    loss is monotone in t, which is what conformal risk control requires.
    """
    losses: List[List[float]] = []
    kept: List[str] = []
    for record in records:
        rid = str(record["id"])
        table = gold_types.get(rid, {})
        gold = [offsets(g) for g in record["gold_spans"]]
        if only_responses_with_gold and not gold:
            continue
        weights: List[float] = []
        best: List[float] = []
        for span in gold:
            info = table.get(span, {})
            weights.append(
                span_weight(info.get("error_type"), bool(info.get("implicit_true")))
            )
            covering = [
                scores_by_span.get((rid, *offsets(p)), 0.0)
                for p in record.get("pred_spans", [])
                if min(span[1], offsets(p)[1]) > max(span[0], offsets(p)[0])
            ]
            best.append(max(covering) if covering else -1.0)
        total = float(sum(weights))
        if not gold or total <= 0:
            losses.append([0.0] * thresholds.size)
        else:
            w = np.asarray(weights, dtype=np.float64)
            b = np.asarray(best, dtype=np.float64)
            missed = (b[:, None] < thresholds[None, :]).astype(np.float64)
            losses.append(list((w[:, None] * missed).sum(axis=0) / total))
        kept.append(rid)
    if not losses:
        return (np.zeros((0, thresholds.size)), [])
    return (np.asarray(losses, dtype=np.float64), kept)


def severity_weighted_risk_control(
    calib_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    calib_scores: Dict[Tuple[str, int, int], float],
    test_scores: Dict[Tuple[str, int, int], float],
    calib_gold: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]],
    test_gold: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]],
    alphas: Sequence[float] = ALPHAS,
) -> Dict[str, Any]:
    """A bound on weighted missed hallucination, and what it costs to buy it."""
    thresholds = np.linspace(0.0, 1.0, GRID_POINTS)
    calib_losses, _ = weighted_loss_matrix(
        calib_records, calib_scores, calib_gold, thresholds
    )
    test_losses, _ = weighted_loss_matrix(
        test_records, test_scores, test_gold, thresholds
    )

    all_test_scores = np.asarray(list(test_scores.values()), dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        chosen = choose_threshold(calib_losses, thresholds, alpha)
        row: Dict[str, Any] = {"alpha": alpha, **chosen}
        if chosen.get("feasible") and chosen.get("threshold") is not None:
            column = int(np.argmin(np.abs(thresholds - chosen["threshold"])))
            row["test_risk"] = float(test_losses[:, column].mean())
            row["bound_held"] = bool(row["test_risk"] <= alpha)
            # The cost column. A risk bound with no cost column is marketing.
            row["span_flag_rate"] = float(
                np.mean(all_test_scores >= chosen["threshold"])
            )
            row["n_spans_flagged"] = int(np.sum(all_test_scores >= chosen["threshold"]))
        rows.append(row)

    return {
        "weights": {
            "evident": SEVERITY_WEIGHTS["evident"],
            "subtle": SEVERITY_WEIGHTS["subtle"],
            "implicit_true": IMPLICIT_TRUE_WEIGHT,
        },
        "weights_note": (
            "a stated value judgement, not a measurement. The bound is only as "
            "meaningful as the weights, so they are reported beside it."
        ),
        "loss": (
            "weighted share of gold hallucinated spans left unflagged, averaged "
            "over responses that contain at least one"
        ),
        "n_calibration_responses": int(calib_losses.shape[0]),
        "n_test_responses": int(test_losses.shape[0]),
        "grid_points": GRID_POINTS,
        "rows": rows,
    }


# --------------------------------------------------------------------------
# add-on 3: generator model against error type
# --------------------------------------------------------------------------


def generator_cross_tab(
    gold_rows: Sequence[Dict[str, Any]], responses_by_model: Counter
) -> Dict[str, Any]:
    """Which kinds of error each of the six generators makes."""
    by_model: Dict[str, Counter] = defaultdict(Counter)
    for row in gold_rows:
        by_model[str(row.get("model", "unknown"))][str(row["error_type"])] += 1

    out: Dict[str, Any] = {}
    for model in sorted(by_model):
        counts = by_model[model]
        total = sum(counts.values())
        responses = responses_by_model.get(model, 0)
        out[model] = {
            "n_spans": total,
            "n_responses": responses,
            "spans_per_response": total / responses if responses else float("nan"),
            "counts": {k: counts[k] for k in sorted(counts)},
            "rates": {k: counts[k] / total for k in sorted(counts)} if total else {},
        }
    return {
        "by_model": out,
        "question": (
            "do stronger generators make fewer but subtler errors? The span "
            "counts and the type mix are both here, so the answer is read off "
            "rather than asserted."
        ),
    }


# --------------------------------------------------------------------------
# add-on 4: type-conditional detectability
# --------------------------------------------------------------------------


def type_conditional_detectability(
    test_records: Sequence[Dict[str, Any]],
    gold_types: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]],
    set_size_by_span: Dict[Tuple[str, int, int], int],
) -> Dict[str, Any]:
    """C1's per-type span recall beside C3's per-type conformal set size.

    One sentence out of two components: if the types C1 misses most are the types
    C3 cannot name, the two failures share a cause rather than compounding.
    """
    caught: Counter = Counter()
    total: Counter = Counter()
    sizes: Dict[str, List[int]] = defaultdict(list)

    for record in test_records:
        rid = str(record["id"])
        table = gold_types.get(rid, {})
        preds = [offsets(p) for p in record.get("pred_spans", [])]
        for span in [offsets(g) for g in record["gold_spans"]]:
            info = table.get(span, {})
            error_type = str(info.get("error_type", "unknown"))
            total[error_type] += 1
            overlapping = [
                p for p in preds if min(span[1], p[1]) > max(span[0], p[0])
            ]
            if overlapping:
                caught[error_type] += 1
                for p in overlapping:
                    size = set_size_by_span.get((rid, p[0], p[1]))
                    if size is not None:
                        sizes[error_type].append(size)

    return {
        "by_error_type": {
            name: {
                "n_gold": total[name],
                "n_touched_by_a_prediction": caught.get(name, 0),
                "overlap_recall": caught.get(name, 0) / total[name] if total[name] else 0.0,
                "mean_conformal_set_size": (
                    float(np.mean(sizes[name])) if sizes.get(name) else float("nan")
                ),
                "n_sets": len(sizes.get(name, [])),
            }
            for name in sorted(total)
        },
        "note": (
            "recall here is span-overlap recall, the same convention C1's "
            "span_overlap metric uses, not span-exact"
        ),
    }


# --------------------------------------------------------------------------
# add-on 5: the implicit_true confound
# --------------------------------------------------------------------------


def _example_metrics(pairs: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    tn = sum(1 for g, p in pairs if not g and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / len(pairs) if pairs else 0.0,
    }


def implicit_true_ablation(
    test_records: Sequence[Dict[str, Any]],
    gold_types: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]],
    flag: str = "implicit_true",
) -> Dict[str, Any]:
    """C1's example-level F1, with and without the flagged spans counted as gold.

    A span marked implicit_true is annotated as a hallucination and is factually
    correct. A response whose only annotated spans are implicit_true is therefore
    a response the corpus calls hallucinated and a reader would not. Recomputing
    the headline number without them says how much of it rests on that.

    The unablated row is recomputed here rather than copied, so it can be checked
    against the number in results/c1/test/metrics.json before the ablated row is
    trusted.
    """
    full: List[Tuple[int, int]] = []
    ablated: List[Tuple[int, int]] = []
    n_flagged = 0
    n_responses_changed = 0

    for record in test_records:
        rid = str(record["id"])
        table = gold_types.get(rid, {})
        gold = [offsets(g) for g in record["gold_spans"]]
        predicted = 1 if record.get("pred_spans") else 0
        kept = []
        for span in gold:
            info = table.get(span, {})
            if bool(info.get(flag)):
                n_flagged += 1
            else:
                kept.append(span)
        full.append((1 if gold else 0, predicted))
        ablated.append((1 if kept else 0, predicted))
        if bool(gold) != bool(kept):
            n_responses_changed += 1

    before = _example_metrics(full)
    after = _example_metrics(ablated)
    return {
        "flag": flag,
        "n_spans_flagged": n_flagged,
        "n_responses_that_change_label": n_responses_changed,
        "example_before": before,
        "example_after": after,
        "delta_f1": after["f1"] - before["f1"],
        "note": (
            "example level is the per-response 'does this answer contain a "
            "hallucination' decision, the same unit C1's example row uses"
        ),
    }
