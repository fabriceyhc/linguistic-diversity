"""Semantic diversity metrics based on distributional semantics.

This module provides metrics for measuring diversity in the semantic content of text
using contextualized and static word embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import faiss  # type: ignore
import numpy as np
import numpy.typing as npt
import torch
from transformers import AutoModel, AutoTokenizer, logging as transformers_logging
from sentence_transformers import SentenceTransformer

from ..metric import MetricConfig, TextDiversity
from ..utils import (
    chunker,
    compute_similarity_matrix_faiss,
    merge_bpe,
    similarity_search_faiss,
)

# Suppress transformers warnings
transformers_logging.set_verbosity_error()


@dataclass
class SemanticConfig(MetricConfig):
    """Configuration for semantic diversity metrics."""

    # Similarity computation
    distance_fn: int = faiss.METRIC_INNER_PRODUCT
    scale_dist: str | None = None
    power_reg: bool = False
    mean_adj: bool = True

    # Feature processing
    remove_stopwords: bool = False
    remove_punct: bool = False
    n_components: int | str | None = None  # PCA dimensions ("auto" or int)

    # Model settings
    model_name: str = "bert-base-uncased"
    batch_size: int = 16
    use_cuda: bool = True


# Model caching to avoid reloading
_MODEL_CACHE: dict[str, Any] = {}


def _get_cached_model(model_name: str, model_class: type) -> Any:
    """Get or load a cached model.

    Args:
        model_name: Name/path of the model.
        model_class: Class to use for loading.

    Returns:
        Loaded model.
    """
    cache_key = f"{model_class.__name__}:{model_name}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = model_class(model_name)
    return _MODEL_CACHE[cache_key]


@lru_cache(maxsize=None)
def _get_stopwords() -> set[str]:
    """Get English stopwords (cached)."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        # Download stopwords if not available
        import nltk
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))


class TokenSemantics(TextDiversity):
    """Token-level semantic diversity using contextualized embeddings.

    This metric computes diversity based on contextualized token embeddings
    from transformer models like BERT. Each token occurrence is treated as
    a separate species.

    Example:
        >>> metric = TokenSemantics()
        >>> corpus = ['one massive earth', 'an enormous globe']
        >>> diversity = metric(corpus)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize token semantic diversity metric.

        Args:
            config: Optional configuration dict.
        """
        super().__init__(config)

        # Load model and tokenizer
        self.model = _get_cached_model(self.config.model_name, AutoModel.from_pretrained)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Special tokens to exclude
        self.undesirable_tokens = {
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
        }

        # Device setup
        self.device = torch.device(
            "cuda" if self.config.use_cuda and torch.cuda.is_available() else "cpu"
        )
        if isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)
            self.model.eval()

    @classmethod
    def _config_class(cls) -> type[SemanticConfig]:
        return SemanticConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        return {
            "model_name": "bert-base-uncased",
            "batch_size": 16,
            "use_cuda": True,
            "distance_fn": faiss.METRIC_Linf,
            "scale_dist": "exp",
            "mean_adj": True,
        }

    @torch.no_grad()
    def _encode_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of inputs.

        Args:
            input_ids: Token IDs (batch_size x seq_len).
            attention_mask: Attention mask (batch_size x seq_len).

        Returns:
            Contextualized embeddings (batch_size x seq_len x hidden_dim).
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use second-to-last layer (often better for semantic tasks)
        return outputs.hidden_states[-2]

    def extract_features(
        self, corpus: list[str]
    ) -> tuple[npt.NDArray[np.float64], list[str]]:
        """Extract token embeddings from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (embeddings, tokens).
        """
        # Tokenize all texts
        inputs = self.tokenizer(
            corpus,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Process in batches
        batches = zip(
            chunker(inputs.input_ids, self.config.batch_size),
            chunker(inputs.attention_mask, self.config.batch_size),
        )

        embeddings_list = []
        for input_ids, attention_mask in batches:
            emb = self._encode_batch(
                input_ids.to(self.device),
                attention_mask.to(self.device),
            )
            embeddings_list.append(emb.cpu())

        # Combine batches
        all_embeddings = torch.cat(embeddings_list)

        # Flatten to (total_tokens x hidden_dim)
        flat_ids = inputs.input_ids.view(-1).numpy()
        flat_embeddings = all_embeddings.view(-1, all_embeddings.shape[-1]).numpy()

        # Filter out special tokens
        valid_mask = ~np.isin(flat_ids, list(self.undesirable_tokens))
        tokens_array = np.array(
            self.tokenizer.convert_ids_to_tokens(flat_ids)
        )[valid_mask]
        embeddings_array = flat_embeddings[valid_mask]

        # Filter stopwords if requested
        if self.config.remove_stopwords:
            stopwords = _get_stopwords()
            keep_mask = ~np.isin(tokens_array, list(stopwords))
            tokens_array = tokens_array[keep_mask]
            embeddings_array = embeddings_array[keep_mask]

        # Filter punctuation if requested
        if self.config.remove_punct:
            punct_chars = set('''!()-[]{};:'",<>./?@#$%^&*_~''')
            keep_mask = ~np.isin(tokens_array, list(punct_chars))
            tokens_array = tokens_array[keep_mask]
            embeddings_array = embeddings_array[keep_mask]

        # Merge BPE tokens if present
        if np.any(np.char.find(tokens_array.astype(str), "##") != -1):
            tokens_array, embeddings_array = merge_bpe(tokens_array, embeddings_array)

        # Optional PCA dimensionality reduction
        if self.config.n_components is not None and len(embeddings_array) > 1:
            from sklearn.decomposition import PCA

            if self.config.n_components == "auto":
                n_comp = min(max(2, len(embeddings_array) // 10), embeddings_array.shape[-1])
            else:
                n_comp = int(self.config.n_components)

            if n_comp > 0 and n_comp < embeddings_array.shape[-1]:
                embeddings_array = PCA(n_components=n_comp).fit_transform(embeddings_array)

        return embeddings_array, tokens_array.tolist()

    def calculate_similarities(
        self, features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate pairwise similarities using FAISS.

        Args:
            features: Token embeddings (n_tokens x dim).

        Returns:
            Similarity matrix (n_tokens x n_tokens).
        """
        Z = compute_similarity_matrix_faiss(
            features,
            distance_metric=self.config.distance_fn,
            postprocess=self.config.scale_dist,
        )

        # Apply power regularization if requested
        if self.config.power_reg:
            Z = np.power(Z, 2)

        # Apply mean adjustment if requested
        if self.config.mean_adj:
            off_diag_mask = ~np.eye(Z.shape[0], dtype=bool)
            mean_sim = Z[off_diag_mask].mean()
            Z[off_diag_mask] -= mean_sim
            Z = np.maximum(Z, 0)  # Clip negative values

        return Z

    def calculate_abundance(self, species: list[str]) -> npt.NDArray[np.float64]:
        """Calculate uniform abundance distribution.

        Args:
            species: List of species (tokens).

        Returns:
            Uniform distribution over species.
        """
        n = len(species)
        return np.full(n, 1.0 / n, dtype=np.float64)


class DocumentSemantics(TextDiversity):
    """Document-level semantic diversity using sentence embeddings.

    This metric computes diversity based on document-level embeddings
    from sentence transformer models, treating each document as a species.

    Example:
        >>> metric = DocumentSemantics()
        >>> corpus = ['one massive earth', 'an enormous globe']
        >>> diversity = metric(corpus)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize document semantic diversity metric.

        Args:
            config: Optional configuration dict.
        """
        super().__init__(config)

        # Device setup
        self.device = torch.device(
            "cuda" if self.config.use_cuda and torch.cuda.is_available() else "cpu"
        )

        # Load sentence transformer model with caching
        cache_key = f"SentenceTransformer:{self.config.model_name}"
        if cache_key not in _MODEL_CACHE:
            _MODEL_CACHE[cache_key] = SentenceTransformer(
                self.config.model_name,
                device=str(self.device)
            )
        self.model = _MODEL_CACHE[cache_key]

    @classmethod
    def _config_class(cls) -> type[SemanticConfig]:
        return SemanticConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        return {
            "model_name": "all-mpnet-base-v2",
            "batch_size": 32,
            "use_cuda": True,
            "distance_fn": faiss.METRIC_INNER_PRODUCT,
            "scale_dist": None,
            "mean_adj": False,
        }

    def extract_features(
        self, corpus: list[str]
    ) -> tuple[npt.NDArray[np.float64], list[str]]:
        """Extract document embeddings from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (embeddings, documents).
        """
        # Encode documents
        embeddings = self.model.encode(
            corpus,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.verbose,
            convert_to_numpy=True,
        )

        return embeddings, corpus

    def calculate_similarities(
        self, features: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Calculate pairwise document similarities.

        Args:
            features: Document embeddings (n_docs x dim).

        Returns:
            Similarity matrix (n_docs x n_docs).
        """
        Z = compute_similarity_matrix_faiss(
            features,
            distance_metric=self.config.distance_fn,
            postprocess=self.config.scale_dist,
        )

        if self.config.power_reg:
            Z = np.power(Z, 2)

        if self.config.mean_adj:
            off_diag_mask = ~np.eye(Z.shape[0], dtype=bool)
            mean_sim = Z[off_diag_mask].mean()
            Z[off_diag_mask] -= mean_sim
            Z = np.maximum(Z, 0)

        return Z

    def calculate_abundance(self, species: list[str]) -> npt.NDArray[np.float64]:
        """Calculate uniform abundance distribution.

        Args:
            species: List of documents.

        Returns:
            Uniform distribution over documents.
        """
        n = len(species)
        return np.full(n, 1.0 / n, dtype=np.float64)

    def calculate_similarity_vector(
        self,
        query_features: npt.NDArray[np.float64],
        corpus_features: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Calculate similarity between query and corpus documents.

        Args:
            query_features: Query document embedding (dim,).
            corpus_features: Corpus document embeddings (n_docs x dim).

        Returns:
            Similarity scores (n_docs,).
        """
        return similarity_search_faiss(
            query_features,
            corpus_features,
            distance_metric=self.config.distance_fn,
            postprocess=self.config.scale_dist,
        )
