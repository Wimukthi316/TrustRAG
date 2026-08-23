"""Unit tests for Component 4 NLI inference and evidence scoring."""

import unittest
from typing import List, Tuple

from src.c4_attribution.alignment import SemanticAligner
from src.c4_attribution.embeddings import EmbeddingEncoder
from src.c4_attribution.nli import NLIPrediction, NLIScorer
from src.c4_attribution.scoring import EvidenceScorer, SpanEvidenceScore
from src.common.schema import Span


class DummyEncoder(EmbeddingEncoder):
    """Deterministic dummy encoder for fast offline tests."""

    def __init__(self) -> None:
        self.device = "cpu"
        self._st_model = None
        self._hf_model = None
        self._hf_tokenizer = None

    def encode(self, texts, batch_size=32, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        import numpy as np

        vecs = []
        for t in texts:
            v = np.zeros(8, dtype=np.float32)
            for ch in t.lower():
                v[ord(ch) % 8] += 1.0
            norm = np.linalg.norm(v)
            if normalize_embeddings and norm > 0:
                v = v / norm
            vecs.append(v)
        return np.vstack(vecs).astype(np.float32)


class DummyNLIScorer(NLIScorer):
    """Deterministic dummy NLI scorer for fast offline tests."""

    def __init__(self) -> None:
        self.device = "cpu"
        self._model = None
        self._tokenizer = None
        self._label_mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def predict_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[NLIPrediction]:
        results = []
        for premise, hypothesis in pairs:
            p_lower = premise.lower()
            h_lower = hypothesis.lower()

            if "contradict" in h_lower or "not" in h_lower or "1995" in h_lower:
                entail = 0.05
                neutral = 0.15
                contra = 0.80
                lbl = "contradiction"
            elif "paris" in p_lower and "paris" in h_lower:
                entail = 0.92
                neutral = 0.05
                contra = 0.03
                lbl = "entailment"
            elif "1889" in p_lower and "1889" in h_lower:
                entail = 0.95
                neutral = 0.03
                contra = 0.02
                lbl = "entailment"
            else:
                entail = 0.20
                neutral = 0.70
                contra = 0.10
                lbl = "neutral"

            results.append(
                NLIPrediction(
                    premise=premise,
                    hypothesis=hypothesis,
                    entailment=entail,
                    neutral=neutral,
                    contradiction=contra,
                    predicted_label=lbl,
                )
            )
        return results


class TestC4NLIScoring(unittest.TestCase):
    def setUp(self):
        self.encoder = DummyEncoder()
        self.aligner = SemanticAligner(encoder=self.encoder, similarity_threshold=0.30)
        self.nli_scorer = DummyNLIScorer()
        self.evidence_scorer = EvidenceScorer(
            aligner=self.aligner,
            nli_scorer=self.nli_scorer,
            top_k_candidates=3,
        )

    def test_nli_prediction_probabilities(self):
        preds = self.nli_scorer.predict_pairs([("Paris is in France", "Paris is the capital")])
        self.assertEqual(len(preds), 1)
        p = preds[0]
        self.assertAlmostEqual(p.entailment + p.neutral + p.contradiction, 1.0, places=4)
        self.assertEqual(p.predicted_label, "entailment")

    def test_evidence_scorer_supporting_evidence(self):
        context = "The Eiffel Tower was completed in 1889. It is situated in Paris. Tokyo is in Japan."
        spans = [
            Span(start=0, end=17, text="completed in 1889"),
            Span(start=18, end=35, text="situated in Paris"),
        ]

        scores = self.evidence_scorer.score_spans(context, spans)
        self.assertEqual(len(scores), 2)

        # First span should match sentence 0 ("completed in 1889")
        self.assertEqual(scores[0].evidence_index, 0)
        self.assertIn("1889", scores[0].best_supporting_sentence)
        self.assertGreater(scores[0].max_entailment, 0.90)

        # Second span should match sentence 1 ("situated in Paris")
        self.assertEqual(scores[1].evidence_index, 1)
        self.assertIn("Paris", scores[1].best_supporting_sentence)
        self.assertGreater(scores[1].max_entailment, 0.90)

    def test_evidence_scorer_contradiction(self):
        context = "The tower was built in 1889. It stands in Paris."
        spans = [Span(start=0, end=20, text="It was built in 1995")]

        scores = self.evidence_scorer.score_spans(context, spans)
        self.assertEqual(len(scores), 1)
        self.assertGreater(scores[0].max_contradiction, 0.70)

    def test_enrich_schema_spans(self):
        context = "The Eiffel Tower is in Paris. It was finished in 1889."
        spans = [
            Span(start=0, end=28, text="The Eiffel Tower is in Paris"),
            Span(start=29, end=54, text="It was finished in 1889"),
        ]

        enriched = self.evidence_scorer.enrich_spans(context, spans)
        self.assertEqual(len(enriched), 2)

        for s in enriched:
            self.assertIsNotNone(s.evidence_sentence)
            self.assertIsNotNone(s.evidence_index)
            self.assertIsNotNone(s.entailment_score)
            self.assertGreater(s.entailment_score, 0.90)

    def test_empty_context_scoring(self):
        spans = [Span(start=0, end=4, text="test")]
        scores = self.evidence_scorer.score_spans("", spans)
        self.assertEqual(len(scores), 1)
        self.assertIsNone(scores[0].best_supporting_sentence)
        self.assertEqual(scores[0].max_entailment, 0.0)


if __name__ == "__main__":
    unittest.main()
