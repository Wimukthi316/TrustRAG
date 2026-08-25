"""The serving side of C3: name the error, say why, and know when not to commit.

C3 attaches three fields the contract already declares and the interface already
has a place for: `Span.error_type`, `Span.explanation` and `Span.escalated`.

**Why the logistic model and not the transformer.** The type model shipped here
is the CPU feature baseline, stored as plain JSON coefficients by
src/c3_explanation/type_baseline.py. Scoring a span is a standardisation and a
dot product, so the API needs no scikit-learn, no torch beyond what C1 already
loads, and no second model in memory. The transformer type head is the research
comparison; if it is trained, its probability dump feeds the same offline layers
and this file is the one place that would change.

**What it refuses to do.** The explanation never states a coverage number. When
`AnalysisResult.guarantee_applies` is False the type and the explanation are
still shown -- the detection is not in doubt, only the promise about it -- and
nothing here prints a percentage next to them. That rule is C2's, in section 14.2
of the handoff, and C3 inherits it rather than reinterpreting it.

**Degrading, not failing.** Every artefact is optional. With no type model on
disk the layer reports itself unavailable and the spans come back exactly as C2
left them, which is what the demo did before C3 existed.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.c3_explanation.explanations import explain
from src.common.schema import ConformalDecision, ErrorType

DEFAULT_MODEL_PATH = Path("results/c3/type/baseline_model.json")
DEFAULT_CONFORMAL_PATH = Path("results/c3/type/c3_conformal.json")


def _softmax(scores: Sequence[float]) -> List[float]:
    top = max(scores)
    exponentials = [math.exp(s - top) for s in scores]
    total = sum(exponentials)
    return [e / total for e in exponentials]


class C3Typer:
    """Types a span, wraps it in a conformal set, and writes the explanation."""

    def __init__(
        self,
        model: Dict[str, Any],
        classes: Sequence[str],
        temperature: float = 1.0,
        thresholds: Optional[Dict[float, float]] = None,
    ) -> None:
        self.model = model
        self.classes = list(classes)
        self.temperature = float(temperature) if temperature and temperature > 0 else 1.0
        self.thresholds = dict(thresholds or {})

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        conformal_path: Path = DEFAULT_CONFORMAL_PATH,
    ) -> Optional["C3Typer"]:
        """Return None rather than raising when C3 has not been run yet."""
        model_path = Path(model_path)
        if not model_path.exists():
            return None
        try:
            stored = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if "flat" not in stored:
            return None

        classes: List[str] = list(stored["flat"]["classes"])
        temperature = 1.0
        thresholds: Dict[float, float] = {}
        conformal_path = Path(conformal_path)
        if conformal_path.exists():
            try:
                report = json.loads(conformal_path.read_text(encoding="utf-8"))
                temperature = float(report.get("temperature", 1.0))
                classes = list(report.get("classes", classes))
                for row in report.get("coverage", []):
                    threshold = row.get("threshold")
                    if threshold is not None and math.isfinite(float(threshold)):
                        thresholds[float(row["alpha"])] = float(threshold)
            except (OSError, ValueError, TypeError):
                thresholds = {}
        return cls(stored, classes, temperature, thresholds)

    @property
    def available(self) -> bool:
        return bool(self.model)

    @property
    def has_conformal(self) -> bool:
        return bool(self.thresholds)

    # -- scoring ---------------------------------------------------------

    def _probabilities(self, features: Dict[str, float]) -> List[float]:
        flat = self.model["flat"]
        vector = [float(features[name]) for name in flat["feature_names"]]
        standardised = [
            (value - mean) / scale
            for value, mean, scale in zip(vector, flat["mean"], flat["scale"])
        ]
        logits = [
            sum(v * w for v, w in zip(standardised, row)) + bias
            for row, bias in zip(flat["coef"], flat["intercept"])
        ]
        probabilities = _softmax(logits)
        ordered = dict(zip(flat["classes"], probabilities))
        raw = [ordered.get(name, 0.0) for name in self.classes]
        return self._temper(raw)

    def _temper(self, probabilities: Sequence[float]) -> List[float]:
        if self.temperature == 1.0:
            return list(probabilities)
        logits = [math.log(max(p, 1e-12)) / self.temperature for p in probabilities]
        return _softmax(logits)

    def _nearest_alpha(self, alpha: float) -> Optional[float]:
        """The calibrated alpha closest to the one requested.

        The thresholds come from an offline run over a fixed grid, so a request
        for an alpha that was never calibrated is answered with the nearest one
        that was -- and `type_set_alpha` on the result says which, so the
        interface is never showing a set at an alpha nobody measured.
        """
        if not self.thresholds:
            return None
        return min(self.thresholds, key=lambda a: abs(a - float(alpha)))

    def type_set(
        self, probabilities: Sequence[float], alpha: float
    ) -> Tuple[List[str], Optional[float]]:
        used = self._nearest_alpha(alpha)
        if used is None:
            return ([], None)
        threshold = self.thresholds[used]
        members = [
            name
            for name, p in zip(self.classes, probabilities)
            if (1.0 - p) <= threshold
        ]
        return (members, used)

    def annotate(
        self,
        features: Dict[str, float],
        span_text: str,
        context_sentences: Sequence[str],
        alpha: float,
        decision: Optional[ConformalDecision] = None,
    ) -> Dict[str, Any]:
        """Everything C3 knows about one span."""
        probabilities = self._probabilities(features)
        best = max(range(len(probabilities)), key=lambda i: probabilities[i])
        predicted = self.classes[best]
        members, used_alpha = self.type_set(probabilities, alpha)

        generated = explain(
            span_text,
            context_sentences,
            error_type=predicted,
            type_set=members or None,
        )
        # Escalation is conjunctive on purpose: C2 unsure whether the span is
        # hallucinated at all, AND C3 unsure what kind of error it would be.
        # The offline count of how often that fires is C3-9.
        escalated = bool(
            decision is ConformalDecision.ABSTAIN and len(members) >= 2
        )
        try:
            error_type = ErrorType(predicted)
        except ValueError:
            error_type = ErrorType.UNKNOWN
        return {
            "error_type": error_type,
            "type_probabilities": {
                name: round(float(p), 4)
                for name, p in zip(self.classes, probabilities)
            },
            "type_set": members,
            "type_set_alpha": used_alpha,
            "explanation": generated["explanation"],
            "quoted_sentence_index": generated["quoted_sentence_index"],
            "escalated": escalated,
        }
