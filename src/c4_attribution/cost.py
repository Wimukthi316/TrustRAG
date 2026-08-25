"""Layer 10 -- the review-cost curve. The number a buyer actually asks for.

C2 printed the flag rate beside every risk bound. C3 printed the escalation
budget -- 37.4% of spans at alpha 0.10 -- and called it unaffordable, which was
the result. C4's version is more concrete: **how much text does a human have to
read?**

At each alpha, three policies are priced on the same rows:

    return everything    the reviewer opens the whole context
    return one guess     the reviewer opens one sentence and trusts it
    conformal set        the reviewer opens k sentences and has a coverage
                         statement about them

"At 90% coverage a reviewer reads 2 of 16 sentences instead of all 16" is a
measured reduction in reading, and it is the single most useful sentence this
component can hand a commercialisation criterion. If the number turns out bad,
it goes in the same table at the same size.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from src.c4_attribution.evidence_conformal import (
    ALPHAS,
    ConformalRow,
    fit,
    prediction_set,
)


def curve(
    calib: Sequence[ConformalRow],
    test: Sequence[ConformalRow],
    alphas: Sequence[float] = ALPHAS,
) -> List[Dict[str, Any]]:
    """Sentences read per span and per response, against the two naive policies."""
    scored = [r for r in test if r.gold_index is not None]
    if not scored:
        return []

    pools = np.asarray([len(r.probabilities) for r in scored], dtype=np.float64)
    by_response: Dict[str, List[int]] = {}
    for position, row in enumerate(scored):
        by_response.setdefault(row.response_id, []).append(position)

    rows: List[Dict[str, Any]] = []
    for alpha in alphas:
        threshold = fit(calib, alpha)
        sets = [prediction_set(r, threshold) for r in scored]
        sizes = np.asarray([len(s) for s in sets], dtype=np.float64)
        covered = np.asarray(
            [1.0 if r.gold_index in s else 0.0 for r, s in zip(scored, sets)]
        )

        # Per response, the union of the sentences every flagged span points at:
        # a reviewer opens each sentence once, not once per span.
        per_response_union = []
        per_response_pool = []
        for positions in by_response.values():
            union = set()
            for position in positions:
                union |= set(sets[position])
            per_response_union.append(len(union))
            per_response_pool.append(len(scored[positions[0]].probabilities))
        union = np.asarray(per_response_union, dtype=np.float64)
        pool_per_response = np.asarray(per_response_pool, dtype=np.float64)

        rows.append(
            {
                "alpha": alpha,
                "empirical_coverage": float(covered.mean()),
                "mean_sentences_per_span": float(sizes.mean()),
                "mean_pool_per_span": float(pools.mean()),
                "compression_per_span": float(np.mean(sizes / np.maximum(pools, 1.0))),
                "reduction_in_reading_per_span": float(
                    1.0 - np.mean(sizes / np.maximum(pools, 1.0))
                ),
                "mean_sentences_per_response": float(union.mean()),
                "mean_pool_per_response": float(pool_per_response.mean()),
                "compression_per_response": float(
                    np.mean(union / np.maximum(pool_per_response, 1.0))
                ),
                "policy_return_everything_sentences_per_span": float(pools.mean()),
                "policy_one_guess_sentences_per_span": 1.0,
                "policy_one_guess_coverage": float(
                    np.mean(
                        [
                            1.0 if s and r.gold_index == s[0] else 0.0
                            for r, s in zip(scored, sets)
                        ]
                    )
                ),
            }
        )
    return rows


def headline(rows: Sequence[Dict[str, Any]], alpha: float = 0.10) -> Dict[str, Any]:
    """The one sentence this layer exists to produce, with its own numbers attached."""
    match = next((r for r in rows if abs(r["alpha"] - alpha) < 1e-9), None)
    if match is None:
        return {"alpha": alpha, "available": False}
    return {
        "alpha": alpha,
        "available": True,
        "coverage": match["empirical_coverage"],
        "sentences_read": match["mean_sentences_per_span"],
        "sentences_available": match["mean_pool_per_span"],
        "reduction_in_reading": match["reduction_in_reading_per_span"],
        "sentence": (
            f"At {(1 - alpha) * 100:.0f}% target coverage a reviewer reads "
            f"{match['mean_sentences_per_span']:.1f} of "
            f"{match['mean_pool_per_span']:.1f} context sentences per flagged "
            f"span, a {match['reduction_in_reading_per_span'] * 100:.1f}% "
            f"reduction, at measured coverage "
            f"{match['empirical_coverage']:.4f}."
        ),
    }
