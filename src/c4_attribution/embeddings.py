"""Dense sentence embeddings and semantic similarity computation for Component 4.

Supports both `sentence-transformers` and direct Hugging Face `transformers`
mean-pooling pipelines with device-aware batch encoding and cosine similarity.
"""

from __future__ import annotations

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEncoder:
    """Encapsulates dense embedding extraction for text spans and context sentences."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        use_st: bool = True,
    ) -> None:
        """Initializes the encoder.

        Args:
            model_name: Hugging Face model repository ID.
            device: 'cuda', 'cpu', or None (auto-detect).
            use_st: Whether to prefer sentence-transformers if available.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self._st_model = None
        self._hf_model = None
        self._hf_tokenizer = None

        # Attempt to load using sentence-transformers first if requested
        if use_st:
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(model_name, device=self.device)
            except Exception:
                self._st_model = None

        # Fallback to standard Hugging Face transformers
        if self._st_model is None:
            from transformers import AutoModel, AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._hf_model = AutoModel.from_pretrained(model_name).to(self.device)
            self._hf_model.eval()

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encodes a list of texts into normalized embedding vectors.

        Args:
            texts: Single string or list of text strings.
            batch_size: Batch size for tokenization and inference.
            normalize_embeddings: If True, vectors are L2 normalized (unit length).

        Returns:
            2D NumPy array of shape (num_texts, embedding_dim).
        """
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            return np.empty((0, 384), dtype=np.float32)

        if self._st_model is not None:
            embeddings = self._st_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)

        # HF transformers mean pooling fallback
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = self._hf_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                model_output = self._hf_model(**encoded)
                # Mean pooling over attention mask
                token_embeddings = model_output[0]  # First element has hidden states
                input_mask_expanded = (
                    encoded["attention_mask"]
                    .unsqueeze(-1)
                    .expand(token_embeddings.size())
                    .float()
                )
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                pooled = sum_embeddings / sum_mask

                if normalize_embeddings:
                    pooled = F.normalize(pooled, p=2, dim=1)

                all_embeddings.append(pooled.cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def compute_similarity_matrix(
        self,
        query_texts: Union[str, List[str]],
        corpus_texts: Union[str, List[str]],
    ) -> np.ndarray:
        """Computes pairwise cosine similarity between queries and corpus texts.

        Args:
            query_texts: Query string or list of query strings (e.g. answer spans).
            corpus_texts: Corpus string or list of corpus strings (e.g. context sentences).

        Returns:
            2D NumPy array of shape (len(query_texts), len(corpus_texts)) with values in [-1, 1].
        """
        if isinstance(query_texts, str):
            query_texts = [query_texts]
        if isinstance(corpus_texts, str):
            corpus_texts = [corpus_texts]

        if len(query_texts) == 0 or len(corpus_texts) == 0:
            return np.empty((len(query_texts), len(corpus_texts)), dtype=np.float32)

        query_embs = self.encode(query_texts, normalize_embeddings=True)
        corpus_embs = self.encode(corpus_texts, normalize_embeddings=True)

        # Dot product of L2 normalized vectors is cosine similarity
        sim_matrix = np.matmul(query_embs, corpus_embs.T)
        return np.clip(sim_matrix, -1.0, 1.0)
