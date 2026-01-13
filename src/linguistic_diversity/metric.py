"""Core metric classes for linguistic diversity measurement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt


@dataclass
class MetricConfig:
    """Base configuration for metrics."""

    q: float = 1.0  # Diversity order parameter
    normalize: bool = False  # Normalize diversity by number of species
    verbose: bool = False


class Metric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, config: MetricConfig | dict[str, Any] | None = None) -> None:
        """Initialize metric with configuration.

        Args:
            config: Metric configuration as MetricConfig or dict.
        """
        if config is None:
            config = {}
        if isinstance(config, dict):
            config = self._config_class()(**{**self._default_config(), **config})
        self.config = config

    @classmethod
    def _config_class(cls) -> type[MetricConfig]:
        """Return the configuration class for this metric."""
        return MetricConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        """Return default configuration as dict."""
        return {}

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """Compute the metric."""
        ...


class DiversityMetric(Metric):
    """Base class for diversity metrics."""

    @abstractmethod
    def __call__(self, corpus: list[str]) -> float:
        """Compute diversity for a corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Diversity score.
        """
        ...


class SimilarityMetric(Metric):
    """Base class for similarity metrics."""

    @abstractmethod
    def __call__(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score.
        """
        ...


class TextDiversity(DiversityMetric):
    """Base class for text diversity metrics using Hill numbers.

    This implements the core diversity calculation using similarity-sensitive
    Hill numbers, based on the framework from ecological diversity studies.
    """

    def __call__(self, corpus: list[str]) -> float:
        """Compute diversity for a corpus."""
        return self.diversity(corpus)

    def diversity(self, corpus: list[str]) -> float:
        """Calculate diversity using similarity-sensitive Hill numbers.

        Args:
            corpus: List of text documents.

        Returns:
            Diversity score (effective number of species).
        """
        # Validate inputs
        if not all(isinstance(d, str) and d.strip() for d in corpus):
            if self.config.verbose:
                print(f"Warning: corpus contains invalid inputs, returning 0")
            return 0.0

        # Extract features and species
        features, species = self.extract_features(corpus)

        # If no features, diversity is 0
        if len(features) == 0:
            return 0.0

        # Calculate similarity matrix Z
        Z = self.calculate_similarities(features)

        # Calculate abundance vector p
        p = self.calculate_abundance(species)

        # Calculate diversity
        D = self._calc_diversity(p, Z, self.config.q)

        # Optionally normalize by number of species
        if self.config.normalize:
            D /= len(p)

        return float(D)

    def similarity(self, corpus: list[str]) -> float:
        """Calculate average similarity across corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Average similarity score.
        """
        if not all(isinstance(d, str) and d.strip() for d in corpus):
            if self.config.verbose:
                print(f"Warning: corpus contains invalid inputs, returning 0")
            return 0.0

        # Extract features
        features, _ = self.extract_features(corpus)

        if len(features) == 0:
            return 0.0

        # Get similarity matrix
        Z = self.calculate_similarities(features)

        # Calculate mean similarity
        return float(Z.sum() / (len(Z) ** 2))

    def rank_similarity(
        self,
        query: list[str],
        corpus: list[str],
        top_n: int = 1,
    ) -> tuple[list[str], npt.NDArray[np.float64]]:
        """Rank corpus documents by similarity to query.

        Args:
            query: Query document(s) (typically single element list).
            corpus: Corpus documents to rank.
            top_n: Number of top results to return (-1 for all).

        Returns:
            Tuple of (ranked_documents, scores).
        """
        if top_n == -1:
            top_n = len(corpus)

        # Extract features for query and corpus
        all_feats, all_docs = self.extract_features(query + corpus)
        q_feats, q_corpus = all_feats[0], all_docs[0]
        c_feats, c_corpus = all_feats[1:], all_docs[1:]

        # Handle empty features
        if len(q_feats) == 0 or len(c_feats) == 0:
            return [], np.array([])

        # Calculate similarity vector
        z = self.calculate_similarity_vector(q_feats, c_feats)

        # Rank by similarity (descending)
        rank_idx = np.argsort(z)[::-1]

        ranking = np.array(c_corpus)[rank_idx].tolist()
        scores = z[rank_idx]

        return ranking[:top_n], scores[:top_n]

    @abstractmethod
    def extract_features(
        self, corpus: list[str]
    ) -> tuple[npt.NDArray[Any], list[Any]]:
        """Extract features and species from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (features, species).
        """
        ...

    @abstractmethod
    def calculate_similarities(
        self, features: npt.NDArray[Any]
    ) -> npt.NDArray[np.float64]:
        """Calculate pairwise similarities between features.

        Args:
            features: Feature matrix.

        Returns:
            Similarity matrix Z (n x n).
        """
        ...

    @abstractmethod
    def calculate_abundance(self, species: list[Any]) -> npt.NDArray[np.float64]:
        """Calculate abundance distribution over species.

        Args:
            species: List of species identifiers.

        Returns:
            Abundance vector p (sums to 1).
        """
        ...

    def calculate_similarity_vector(
        self,
        query_features: Any,
        corpus_features: Any,
    ) -> npt.NDArray[np.float64]:
        """Calculate similarity between query and corpus features.

        Args:
            query_features: Query feature(s).
            corpus_features: Corpus features.

        Returns:
            Similarity vector.
        """
        raise NotImplementedError(
            "Ranking requires document-level metrics. "
            "Override this method to support ranking."
        )

    @staticmethod
    def _calc_diversity(
        p: npt.NDArray[np.float64],
        Z: npt.NDArray[np.float64],
        q: float = 1.0,
    ) -> float:
        """Calculate diversity using Hill number formula.

        Args:
            p: Abundance distribution (n-vector, sums to 1).
            Z: Similarity matrix (n x n).
            q: Diversity order parameter.

        Returns:
            Diversity value.
        """
        Zp = Z @ p

        if q == 1:
            # Shannon diversity case: D = 1 / prod(Zp^p)
            # = exp(-sum(p * log(Zp)))
            # Use log-space calculation to avoid numerical underflow
            with np.errstate(divide="ignore", invalid="ignore"):
                # Add small epsilon to avoid log(0)
                log_D = -np.sum(p * np.log(Zp + 1e-10))
                D = np.exp(log_D)
        elif q == float("inf"):
            # Simpson diversity case: D = 1 / max(Zp)
            D = 1.0 / Zp.max()
        else:
            # General case: D = (sum(p * Zp^(q-1)))^(1/(1-q))
            D = np.power((p * np.power(Zp, q - 1)).sum(), 1 / (1 - q))

        return float(D)
