"""Layer 12 -- the serving artefact, and the parity check that keeps the demo honest.

C4 ships as a JSON artefact rather than a pickle, for the same reason C2 and C3
do: a JSON file can be read, diffed and checked by a test, and a pickle cannot.
The artefact carries everything the backend needs and nothing it does not --
which backend produced it, the abstention threshold, and the LAC threshold at
each alpha.

**What ships is the lexical attributor, and that is a decision with a reason.**
The serving process already holds ModernBERT for C1. Loading a second
cross-encoder for every request would add hundreds of milliseconds and hundreds
of megabytes to a demo that runs on a 6 GB laptop, and the attribution is an
addition to the product rather than the product. So the offline study measures
four backends and the shipped path runs the one that costs nothing, with the
artefact recording which is which so the report and the demo cannot quietly
disagree.

**Three serving rules, inherited and not negotiable.**

1. `model_version` does **not** gain a `+c4` suffix. It names what produced the
   span scores and C4 produces none. Appending broke four pinned assertions in
   `tests/test_serving.py` when C3 tried it.
2. The evidence string may never contain "coverage", "guarantee" or a percent
   sign. A per-span string is not the place a distribution-free guarantee gets
   claimed, and `tests/test_c4_serving.py` asserts it.
3. When `guarantee_applies` is False the evidence still renders, and the
   conformal set may not be presented as a guaranteed quantity. The spans and
   their evidence are the detection; what is withdrawn is the promise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ARTIFACT = Path("results/c4/c4_artifact.json")
SCHEMA_VERSION = "1.0.0"

# Words a per-span evidence string may never contain. Rule 2 above.
FORBIDDEN_IN_EVIDENCE: Tuple[str, ...] = ("coverage", "guarantee", "%")


def build_artifact(
    backend: str,
    abstention_threshold: Optional[float],
    lac_thresholds: Dict[str, float],
    offline_backend: str,
    notes: Sequence[str] = (),
) -> Dict[str, Any]:
    """Everything the serving path needs, in a form a test can check."""
    return {
        "schema_version": SCHEMA_VERSION,
        "component": "c4_attribution",
        "served_backend": backend,
        "offline_best_backend": offline_backend,
        "abstention_threshold": abstention_threshold,
        "lac_thresholds": {str(k): v for k, v in lac_thresholds.items()},
        "notes": list(notes),
    }


def write_artifact(payload: Dict[str, Any], path: Path = ARTIFACT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_artifact(path: Path | str = ARTIFACT) -> Optional[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def evidence_is_safe(text: str) -> bool:
    """Rule 2, as a function so the backend and the test call the same one."""
    lowered = (text or "").lower()
    return not any(word in lowered for word in FORBIDDEN_IN_EVIDENCE)


class C4Attributor:
    """The shipped attributor: lexical scoring, an abstention threshold, a set.

    Deliberately dependency-free. It imports `src.c4_attribution.lexical`, which
    imports numpy and nothing else, so adding C4 to the serving path costs one
    already-loaded library and no model download.
    """

    def __init__(self, artifact: Optional[Dict[str, Any]] = None) -> None:
        self.artifact = artifact or {}
        self.threshold = self.artifact.get("abstention_threshold")
        self.lac = {
            float(k): float(v)
            for k, v in (self.artifact.get("lac_thresholds") or {}).items()
        }

    @classmethod
    def load(cls, path: Path | str = ARTIFACT) -> "C4Attributor":
        return cls(load_artifact(path))

    @property
    def available(self) -> bool:
        return bool(self.artifact)

    def scores(self, span_text: str, context_sentences: Sequence[str]) -> List[float]:
        """The offline scorer, called by the identical code path.

        `tests/test_c4_serving.py` asserts this equals the offline
        `lexical.idf_overlap_scores` to 1e-9 on the same input, so a demo cannot
        drift from the report.
        """
        from src.c4_attribution.lexical import idf_overlap_scores

        return idf_overlap_scores(span_text, list(context_sentences))

    def evidence_set(self, scores: Sequence[float], alpha: float) -> Tuple[int, ...]:
        """The conformal set at the nearest calibrated alpha, or an empty tuple.

        Returned for display only. The caller decides whether it may be
        described as guaranteed, because only the caller knows whether the
        run-time distribution check passed.
        """
        from src.c4_attribution.evidence_conformal import normalise_scores

        if not self.lac or not scores:
            return ()
        nearest = min(self.lac, key=lambda a: abs(a - alpha))
        threshold = self.lac[nearest]
        probabilities = normalise_scores(scores)
        return tuple(
            index
            for index, p in enumerate(probabilities)
            if (1.0 - p) <= threshold
        )

    def annotate(
        self,
        span_text: str,
        context_sentences: Sequence[str],
        alpha: float = 0.1,
    ) -> Dict[str, Any]:
        """C4's three contract fields for one span, or all three None.

        Abstaining returns None for all three, which the frontend's truthiness
        guard already renders as nothing. That empty state is the correct
        behaviour and it is the thing worth clicking on in a demo: most systems
        cannot say "there is no supporting sentence here".
        """
        empty = {
            "evidence_sentence": None,
            "evidence_index": None,
            "entailment_score": None,
            "evidence_set": [],
            "abstained": True,
        }
        sentences = list(context_sentences)
        # No artefact means no calibrated threshold, so there is nothing to
        # abstain against and no operating point to stand behind. The layer
        # returns nothing and the interface renders what it rendered before C4
        # existed, which is the same contract C2 and C3 keep.
        if not self.available or not sentences or not span_text:
            return empty

        scores = self.scores(span_text, sentences)
        best = max(range(len(scores)), key=lambda i: (scores[i], -i))
        confidence = float(scores[best])

        if self.threshold is not None and confidence < float(self.threshold):
            return {
                "evidence_sentence": None,
                "evidence_index": None,
                "entailment_score": None,
                "evidence_set": [],
                "abstained": True,
            }

        return {
            "evidence_sentence": sentences[best],
            "evidence_index": best,
            # Defined once, in the contract's own terms: the model's confidence
            # that this sentence is the one a human would cite for this span.
            # Not P(entailment). The field name is inherited from the frozen
            # schema and renaming a frozen contract costs more than a docstring.
            "entailment_score": round(min(max(confidence, 0.0), 1.0), 4),
            "evidence_set": list(self.evidence_set(scores, alpha)),
            "abstained": False,
        }
