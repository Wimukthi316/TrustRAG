"""Unit tests for Component 4 dense embeddings and semantic alignment."""

import unittest
import numpy as np
from src.c4_attribution.alignment import SemanticAligner, SpanAlignment
from src.c4_attribution.embeddings import EmbeddingEncoder
from src.common.schema import Span


class DummyEncoder(EmbeddingEncoder):
    """Deterministic dummy encoder for fast offline unit tests."""

    def __init__(self) -> None:
        self.device = "cpu"
        self._st_model = None
        self._hf_model = None
        self._hf_tokenizer = None

    def encode(
        self,
        texts,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            return np.empty((0, 8), dtype=np.float32)

        # Generate deterministic vector based on text character ordinals
        vecs = []
        for t in texts:
            # Simple 8-dim bag-of-chars representation
            v = np.zeros(8, dtype=np.float32)
            for ch in t.lower():
                v[ord(ch) % 8] += 1.0
            norm = np.linalg.norm(v)
            if normalize_embeddings and norm > 0:
                v = v / norm
            vecs.append(v)
        return np.vstack(vecs).astype(np.float32)


class TestC4EmbeddingsAndAlignment(unittest.TestCase):
    def test_dummy_encoder_similarity(self):
        encoder = DummyEncoder()
        queries = ["Paris", "France"]
        corpus = ["Paris is in France.", "Tokyo is in Japan."]

        sim_matrix = encoder.compute_similarity_matrix(queries, corpus)
        self.assertEqual(sim_matrix.shape, (2, 2))
        self.assertTrue(np.all(sim_matrix >= -1.0))
        self.assertTrue(np.all(sim_matrix <= 1.0))

        # Self-similarity should be ~1.0
        self_sim = encoder.compute_similarity_matrix(["Paris"], ["Paris"])[0, 0]
        self.assertAlmostEqual(float(self_sim), 1.0, places=4)

    def test_semantic_aligner_with_schema_spans(self):
        aligner = SemanticAligner(encoder=DummyEncoder(), similarity_threshold=0.30)
        context = "The Eiffel Tower is in Paris. Mount Fuji is in Japan."
        spans = [
            Span(start=0, end=16, text="The Eiffel Tower"),
            Span(start=17, end=27, text="Mount Fuji"),
        ]

        alignments = aligner.align_spans(context, spans, top_k=2)

        self.assertEqual(len(alignments), 2)
        for align in alignments:
            self.assertIsInstance(align, SpanAlignment)
            self.assertIsNotNone(align.top_sentence)
            self.assertIn(align.top_sentence_index, [0, 1])
            self.assertEqual(len(align.candidates), 2)
            self.assertGreaterEqual(align.top_similarity, align.candidates[1].similarity)

    def test_semantic_aligner_empty_context(self):
        aligner = SemanticAligner(encoder=DummyEncoder())
        spans = ["some span"]
        alignments = aligner.align_spans("", spans)

        self.assertEqual(len(alignments), 1)
        self.assertIsNone(alignments[0].top_sentence)
        self.assertEqual(alignments[0].top_similarity, 0.0)
        self.assertFalse(alignments[0].is_topically_related)
        self.assertEqual(alignments[0].candidates, [])

    def test_semantic_aligner_empty_spans(self):
        aligner = SemanticAligner(encoder=DummyEncoder())
        alignments = aligner.align_spans("Some context text.", [])
        self.assertEqual(alignments, [])


if __name__ == "__main__":
    unittest.main()
