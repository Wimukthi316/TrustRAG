"""Unit tests for Component 4 SHOA diagnostic engine and disagreement analysis."""

import unittest
from typing import List, Tuple

from src.c4_attribution.alignment import SemanticAligner
from src.c4_attribution.disagreement import DisagreementAnalyzer
from src.c4_attribution.embeddings import EmbeddingEncoder
from src.c4_attribution.nli import NLIPrediction, NLIScorer
from src.c4_attribution.scoring import EvidenceScorer
from src.c4_attribution.shoa import OriginType, PipelineBlame, SHOAEngine
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


class MockDiagnosticNLIScorer(NLIScorer):
    """Deterministic mock NLI scorer for diagnostic testing."""

    def __init__(self) -> None:
        self.device = "cpu"
        self._model = None
        self._tokenizer = None
        self._label_mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}

    def predict_pairs(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> List[NLIPrediction]:
        results = []
        for premise, hypothesis in pairs:
            p_low = premise.lower()
            h_low = hypothesis.lower()

            if "1995" in h_low and "1889" in p_low:
                # Contradiction
                entail = 0.02
                neutral = 0.08
                contra = 0.90
                lbl = "contradiction"
            elif "paris" in p_low and "paris" in h_low:
                # Entailment
                entail = 0.95
                neutral = 0.03
                contra = 0.02
                lbl = "entailment"
            else:
                # Neutral
                entail = 0.15
                neutral = 0.75
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


class TestC4SHOAEngine(unittest.TestCase):
    def setUp(self):
        self.encoder = DummyEncoder()
        self.aligner = SemanticAligner(encoder=self.encoder, similarity_threshold=0.30)
        self.nli_scorer = MockDiagnosticNLIScorer()
        self.evidence_scorer = EvidenceScorer(
            aligner=self.aligner,
            nli_scorer=self.nli_scorer,
            top_k_candidates=3,
        )
        self.shoa = SHOAEngine(
            theta_sim=0.30,
            theta_entail=0.70,
            theta_contra=0.60,
            evidence_scorer=self.evidence_scorer,
        )

    def test_shoa_type_c_contextual_conflict(self):
        context = "The landmark was constructed in 1889."
        spans = [Span(start=0, end=21, text="It was built in 1995")]

        diagnoses = self.shoa.diagnose_spans(context, spans)
        self.assertEqual(len(diagnoses), 1)
        d = diagnoses[0]

        self.assertTrue(d.is_hallucination)
        self.assertEqual(d.origin_type, OriginType.TYPE_C)
        self.assertEqual(d.pipeline_blame, PipelineBlame.GENERATOR)
        self.assertFalse(d.is_detector_disagreement)
        self.assertIn("Generator", d.actionable_remediation)
        self.assertIn("1889", d.evidence_sentence)

    def test_shoa_type_s_sufficiency_failure(self):
        # Empty / completely unrelated context
        context = "Z"  # very short unrelated string
        spans = [Span(start=0, end=16, text="The Eiffel Tower")]

        # Configure high similarity threshold to guarantee sufficiency trigger
        strict_shoa = SHOAEngine(
            theta_sim=0.99,
            evidence_scorer=self.evidence_scorer,
        )
        diagnoses = strict_shoa.diagnose_spans(context, spans)
        self.assertEqual(len(diagnoses), 1)
        d = diagnoses[0]

        self.assertTrue(d.is_hallucination)
        self.assertEqual(d.origin_type, OriginType.TYPE_S)
        self.assertEqual(d.pipeline_blame, PipelineBlame.RETRIEVER)
        self.assertIn("Retriever", d.actionable_remediation)

    def test_shoa_type_p_parametric_injection(self):
        context = "The Eiffel Tower stands in the city of Paris."
        spans = [Span(start=0, end=23, text="designed by Eiffel Co.")]

        diagnoses = self.shoa.diagnose_spans(context, spans)
        self.assertEqual(len(diagnoses), 1)
        d = diagnoses[0]

        self.assertTrue(d.is_hallucination)
        self.assertEqual(d.origin_type, OriginType.TYPE_P)
        self.assertEqual(d.pipeline_blame, PipelineBlame.PARAMETRIC_GUARDRAIL)
        self.assertIn("Guardrails", d.actionable_remediation)

    def test_shoa_detector_disagreement(self):
        context = "The Eiffel Tower is situated in Paris."
        spans = [Span(start=0, end=28, text="The Eiffel Tower is in Paris")]

        diagnoses = self.shoa.diagnose_spans(context, spans)
        self.assertEqual(len(diagnoses), 1)
        d = diagnoses[0]

        self.assertFalse(d.is_hallucination)
        self.assertIsNone(d.origin_type)
        self.assertTrue(d.is_detector_disagreement)
        self.assertEqual(d.pipeline_blame, PipelineBlame.DETECTOR_MISMATCH)

    def test_disagreement_analyzer(self):
        context = "The tower is in Paris. It was built in 1889."
        spans = [
            Span(start=0, end=14, text="tower in Paris"),  # Disagreement (entailed)
            Span(start=15, end=30, text="built in 1995"),   # TYPE-C (contradiction)
            Span(start=31, end=45, text="made of gold"),     # TYPE-P (parametric)
        ]

        diagnoses = self.shoa.diagnose_spans(context, spans)
        report = DisagreementAnalyzer.analyze(diagnoses)

        self.assertEqual(report.total_evaluated_spans, 3)
        self.assertEqual(report.disagreement_count, 1)
        self.assertAlmostEqual(report.disagreement_rate, 1 / 3, places=2)
        self.assertEqual(report.type_c_count, 1)
        self.assertEqual(report.type_p_count, 1)
        self.assertEqual(report.type_s_count, 0)

        # Check serialization
        d_dict = report.to_dict()
        self.assertIn("disagreement_rate_pct", d_dict)
        self.assertIn("breakdown", d_dict)


if __name__ == "__main__":
    unittest.main()
