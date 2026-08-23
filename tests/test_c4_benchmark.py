"""Unit tests for Component 4 RAGTruth benchmark evaluation and cross-LLM analysis."""

import unittest
from typing import List, Tuple

from src.c4_attribution.alignment import SemanticAligner
from src.c4_attribution.benchmark import (
    RAGTruthEvaluator,
    RAGTruthRecord,
    SpanBenchmarkResult,
)
from src.c4_attribution.cross_llm import CrossLLMAnalyzer
from src.c4_attribution.embeddings import EmbeddingEncoder
from src.c4_attribution.nli import NLIPrediction, NLIScorer
from src.c4_attribution.scoring import EvidenceScorer
from src.c4_attribution.shoa import OriginType, SHOAEngine


class DummyEncoder(EmbeddingEncoder):
    """Deterministic dummy encoder for fast offline testing."""

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


class MockBenchmarkNLIScorer(NLIScorer):
    """Mock NLI scorer returning controlled predictions for benchmark verification."""

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

            if "conflict" in h_low or "1995" in h_low:
                entail = 0.02
                neutral = 0.08
                contra = 0.90
                lbl = "contradiction"
            elif "entail" in h_low or "paris" in h_low:
                entail = 0.95
                neutral = 0.03
                contra = 0.02
                lbl = "entailment"
            else:
                entail = 0.10
                neutral = 0.80
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


class TestC4BenchmarkEvaluation(unittest.TestCase):
    def setUp(self):
        encoder = DummyEncoder()
        aligner = SemanticAligner(encoder=encoder, similarity_threshold=0.30)
        nli = MockBenchmarkNLIScorer()
        scorer = EvidenceScorer(aligner=aligner, nli_scorer=nli, top_k_candidates=3)
        self.shoa = SHOAEngine(theta_sim=0.30, theta_entail=0.70, theta_contra=0.60, evidence_scorer=scorer)
        self.evaluator = RAGTruthEvaluator(shoa_engine=self.shoa)

    def test_evaluate_single_record(self):
        rec = RAGTruthRecord(
            source_id="src_001",
            task_type="qa",
            source_context="The Eiffel Tower is situated in Paris. It was finished in 1889.",
            response_text="The Eiffel Tower is in Paris and was completed in 1995.",
            model_name="gpt-4-0613",
            human_spans=[
                {"start": 0, "end": 28, "label": "Evident Conflict", "text": "The Eiffel Tower is in Paris"},
                {"start": 37, "end": 54, "label": "Evident Baseless Info", "text": "completed in 1995"},
            ],
        )

        results = self.evaluator.evaluate_record(rec)
        self.assertEqual(len(results), 2)

        # First span matches context -> Disagreement
        self.assertTrue(results[0].diagnosis.is_detector_disagreement)
        self.assertEqual(results[0].model_name, "gpt-4-0613")

        # Second span contradicts 1889 -> TYPE-C
        self.assertEqual(results[1].diagnosis.origin_type, OriginType.TYPE_C)

    def test_baseless_decomposition_calculation(self):
        rec1 = RAGTruthRecord(
            source_id="rec_1",
            task_type="qa",
            source_context="Short context about science.",
            response_text="Quantum theory and astrophysics.",
            model_name="gpt-4-0613",
            human_spans=[
                {"start": 0, "end": 14, "label": "Evident Baseless Info", "text": "Quantum theory"},  # TYPE-P
            ],
        )

        results = self.evaluator.evaluate_dataset([rec1])
        decomp = RAGTruthEvaluator.compute_baseless_decomposition(results)

        self.assertEqual(decomp["total_baseless_spans"], 1)
        self.assertIn("type_p_parametric_injection_pct", decomp)
        self.assertIn("type_s_retriever_omission_pct", decomp)

    def test_cross_llm_profiles(self):
        rec_gpt = RAGTruthRecord(
            source_id="rec_gpt",
            task_type="qa",
            source_context="Information regarding biology.",
            response_text="Biology and molecular structures.",
            model_name="gpt-4-0613",
            human_spans=[{"start": 12, "end": 32, "label": "Subtle Baseless Info", "text": "molecular structures"}],
        )

        rec_llama = RAGTruthRecord(
            source_id="rec_llama",
            task_type="qa",
            source_context="Z",  # Unrelated -> TYPE-S
            response_text="Unrelated generation.",
            model_name="llama-2-7b-chat",
            human_spans=[{"start": 0, "end": 20, "label": "Evident Baseless Info", "text": "Unrelated generation"}],
        )

        # Use strict threshold to trigger TYPE-S on short string
        strict_shoa = SHOAEngine(theta_sim=0.99, evidence_scorer=self.shoa.evidence_scorer)
        evaluator = RAGTruthEvaluator(shoa_engine=strict_shoa)

        results = evaluator.evaluate_dataset([rec_gpt, rec_llama])
        profiles = CrossLLMAnalyzer.compute_profiles(results)

        self.assertIn("gpt-4-0613", profiles)
        self.assertIn("llama-2-7b-chat", profiles)

        gpt_prof = profiles["gpt-4-0613"]
        llama_prof = profiles["llama-2-7b-chat"]

        self.assertEqual(gpt_prof.total_spans, 1)
        self.assertEqual(llama_prof.total_spans, 1)
        self.assertEqual(llama_prof.dominant_failure_mode, "Sufficiency Failure (TYPE-S)")

        # Verify contingency table
        table = CrossLLMAnalyzer.compute_contingency_table(results)
        self.assertEqual(len(table["models"]), 2)
        self.assertEqual(len(table["categories"]), 4)


if __name__ == "__main__":
    unittest.main()
