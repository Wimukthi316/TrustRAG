"""Natural Language Inference (NLI) Cross-Encoder engine for Component 4.

Evaluates deductive premise-hypothesis pairs between context sentences (premise)
and generated answer spans (hypothesis) to compute 3-class probability distributions:
    [p_entailment, p_neutral, p_contradiction]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
FALLBACK_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"


@dataclass(frozen=True)
class NLIPrediction:
    """NLI probabilities for a single (premise, hypothesis) pair."""

    premise: str
    hypothesis: str
    entailment: float
    neutral: float
    contradiction: float
    predicted_label: str

    def to_dict(self) -> Dict[str, Union[str, float]]:
        return {
            "premise": self.premise,
            "hypothesis": self.hypothesis,
            "entailment": self.entailment,
            "neutral": self.neutral,
            "contradiction": self.contradiction,
            "predicted_label": self.predicted_label,
        }


class NLIScorer:
    """Evaluates Natural Language Inference using cross-encoder transformer checkpoints."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        device: Optional[str] = None,
        lazy_load: bool = False,
    ) -> None:
        """Initializes the NLI cross-encoder scorer.

        Args:
            model_name: Hugging Face model repository ID.
            device: 'cuda', 'cpu', or None (auto-detect).
            lazy_load: If True, delays loading weights until the first prediction call.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._label_mapping: Dict[int, str] = {}

        if not lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        """Loads tokenizer and classification model weights."""
        if self._model is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.device)
        self._model.eval()

        # Build label index mapping from model config
        id2label = getattr(self._model.config, "id2label", None)
        if id2label:
            for idx, label_str in id2label.items():
                lbl = str(label_str).lower()
                if "entail" in lbl:
                    self._label_mapping[int(idx)] = "entailment"
                elif "contra" in lbl:
                    self._label_mapping[int(idx)] = "contradiction"
                else:
                    self._label_mapping[int(idx)] = "neutral"
        else:
            # Standard MNLI default mapping
            self._label_mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def predict_pairs(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
    ) -> List[NLIPrediction]:
        """Computes NLI probabilities for a batch of (premise, hypothesis) text pairs.

        Args:
            pairs: List of (premise, hypothesis) tuples.
            batch_size: Number of pairs to process per inference batch.

        Returns:
            List of NLIPrediction objects in matching order.
        """
        if not pairs:
            return []

        self._load_model()
        results: List[NLIPrediction] = []

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            premises = [p[0] for p in batch]
            hypotheses = [p[1] for p in batch]

            inputs = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for j, (premise, hypothesis) in enumerate(batch):
                prob_vec = probs[j]
                
                # Extract probabilities according to mapped label positions
                entail_prob = 0.0
                neutral_prob = 0.0
                contra_prob = 0.0

                for idx, lbl in self._label_mapping.items():
                    if idx < len(prob_vec):
                        p_val = float(prob_vec[idx])
                        if lbl == "entailment":
                            entail_prob = p_val
                        elif lbl == "contradiction":
                            contra_prob = p_val
                        else:
                            neutral_prob = p_val

                # Determine argmax label
                max_prob = max(entail_prob, neutral_prob, contra_prob)
                if max_prob == entail_prob:
                    best_label = "entailment"
                elif max_prob == contra_prob:
                    best_label = "contradiction"
                else:
                    best_label = "neutral"

                results.append(
                    NLIPrediction(
                        premise=premise,
                        hypothesis=hypothesis,
                        entailment=entail_prob,
                        neutral=neutral_prob,
                        contradiction=contra_prob,
                        predicted_label=best_label,
                    )
                )

        return results
