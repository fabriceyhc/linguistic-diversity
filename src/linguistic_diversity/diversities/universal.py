"""Universal linguistic diversity metric combining all dimensions.

This module provides a unified metric that aggregates diversity measurements
across all linguistic dimensions (semantic, syntactic, morphological, phonological)
into a single universal diversity score.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from ..metric import DiversityMetric, MetricConfig


# Fixed ordering of metrics in diversity embeddings
# This ensures consistent vector positions across all calls
DIVERSITY_EMBEDDING_METRICS = [
    "token_semantics",
    "document_semantics",
    "dependency_parse",
    "constituency_parse",
    "pos_sequence",
    "rhythmic",
    "phonemic",
]

# Mapping from metric name to vector index
METRIC_TO_INDEX = {name: idx for idx, name in enumerate(DIVERSITY_EMBEDDING_METRICS)}


class AggregationStrategy(Enum):
    """Strategies for combining diversity scores."""

    WEIGHTED_GEOMETRIC = "weighted_geometric"  # Weighted geometric mean
    WEIGHTED_ARITHMETIC = "weighted_arithmetic"  # Weighted arithmetic mean
    HARMONIC = "harmonic"  # Harmonic mean (sensitive to low scores)
    MINIMUM = "minimum"  # Most conservative: take minimum
    HIERARCHICAL = "hierarchical"  # Geometric within branches, weighted across


@dataclass
class UniversalDiversityConfig(MetricConfig):
    """Configuration for universal diversity metric."""

    # Aggregation strategy
    strategy: str = "hierarchical"

    # Branch-level weights (should sum to 1.0)
    semantic_weight: float = 0.35
    syntactic_weight: float = 0.30
    morphological_weight: float = 0.15
    phonological_weight: float = 0.20

    # Metric-level weights (within branches, for weighted strategies)
    # Semantic branch
    token_semantics_weight: float = 0.6
    document_semantics_weight: float = 0.4

    # Syntactic branch
    dependency_parse_weight: float = 0.6
    constituency_parse_weight: float = 0.4

    # Morphological branch (only one metric)
    pos_sequence_weight: float = 1.0

    # Phonological branch
    rhythmic_weight: float = 0.5
    phonemic_weight: float = 0.5

    # Metric configurations (passed to each metric)
    semantic_config: dict[str, Any] | None = None
    syntactic_config: dict[str, Any] | None = None
    morphological_config: dict[str, Any] | None = None
    phonological_config: dict[str, Any] | None = None

    # Flags to enable/disable branches
    use_semantic: bool = True
    use_syntactic: bool = True
    use_morphological: bool = True
    use_phonological: bool = True

    # Flags to enable/disable specific metrics
    use_token_semantics: bool = True
    use_document_semantics: bool = True
    use_dependency_parse: bool = True
    use_constituency_parse: bool = False  # Optional, requires benepar
    use_pos_sequence: bool = True
    use_rhythmic: bool = True
    use_phonemic: bool = True


class UniversalLinguisticDiversity(DiversityMetric):
    """Universal linguistic diversity metric.

    This metric combines diversity measurements across all linguistic dimensions
    into a unified score that captures the overall linguistic richness of a corpus.

    The metric uses a hierarchical aggregation approach:
    1. Computes diversity for each enabled metric
    2. Aggregates within each linguistic branch (semantic, syntactic, etc.)
    3. Combines across branches using configurable weights

    Example:
        >>> from linguistic_diversity import UniversalLinguisticDiversity
        >>> corpus = ['one massive earth', 'an enormous globe', 'the colossal world']
        >>> metric = UniversalLinguisticDiversity()
        >>> diversity = metric(corpus)
        >>> print(f"Universal diversity: {diversity:.2f}")

    Args:
        config: Configuration dictionary or UniversalDiversityConfig object
    """

    def __init__(
        self, config: UniversalDiversityConfig | dict[str, Any] | None = None
    ) -> None:
        """Initialize universal diversity metric."""
        super().__init__(config)
        self._metrics: dict[str, DiversityMetric] = {}
        self._initialize_metrics()

    @classmethod
    def _config_class(cls) -> type[UniversalDiversityConfig]:
        """Return configuration class."""
        return UniversalDiversityConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        """Return default configuration."""
        return {
            "strategy": "hierarchical",
            "semantic_weight": 0.35,
            "syntactic_weight": 0.30,
            "morphological_weight": 0.15,
            "phonological_weight": 0.20,
        }

    def _initialize_metrics(self) -> None:
        """Lazily initialize metrics as needed."""
        # Import here to avoid circular imports and allow optional dependencies
        from . import (
            DependencyParse,
            DocumentSemantics,
            PartOfSpeechSequence,
            TokenSemantics,
        )

        cfg = self.config

        # Semantic metrics
        if cfg.use_semantic:
            sem_cfg = cfg.semantic_config or {"verbose": False}
            if cfg.use_token_semantics:
                self._metrics["token_semantics"] = TokenSemantics(sem_cfg)
            if cfg.use_document_semantics:
                self._metrics["document_semantics"] = DocumentSemantics(sem_cfg)

        # Syntactic metrics
        if cfg.use_syntactic:
            syn_cfg = cfg.syntactic_config or {"verbose": False}
            if cfg.use_dependency_parse:
                self._metrics["dependency_parse"] = DependencyParse(syn_cfg)
            if cfg.use_constituency_parse:
                try:
                    from . import ConstituencyParse

                    self._metrics["constituency_parse"] = ConstituencyParse(syn_cfg)
                except ImportError:
                    if cfg.verbose:
                        print("Warning: ConstituencyParse requires benepar, skipping")

        # Morphological metrics
        if cfg.use_morphological:
            mor_cfg = cfg.morphological_config or {"verbose": False}
            if cfg.use_pos_sequence:
                self._metrics["pos_sequence"] = PartOfSpeechSequence(mor_cfg)

        # Phonological metrics
        if cfg.use_phonological:
            pho_cfg = cfg.phonological_config or {"verbose": False}
            if cfg.use_rhythmic:
                try:
                    from . import Rhythmic

                    self._metrics["rhythmic"] = Rhythmic(pho_cfg)
                except ImportError:
                    if cfg.verbose:
                        print("Warning: Rhythmic requires cadences, skipping")
            if cfg.use_phonemic:
                try:
                    from . import Phonemic

                    self._metrics["phonemic"] = Phonemic(pho_cfg)
                except ImportError:
                    if cfg.verbose:
                        print("Warning: Phonemic requires g2p_en, skipping")

    def __call__(self, corpus: list[str]) -> float:
        """Compute universal diversity for a corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Universal diversity score combining all enabled dimensions.
        """
        if not corpus or not all(isinstance(d, str) and d.strip() for d in corpus):
            return 0.0

        # Compute diversity for each metric
        scores: dict[str, float] = {}
        for name, metric in self._metrics.items():
            try:
                score = metric(corpus)
                scores[name] = score
                if self.config.verbose:
                    print(f"  {name}: {score:.3f}")
            except Exception as e:
                if self.config.verbose:
                    print(f"  {name}: Error - {e}")
                scores[name] = 0.0

        if not scores:
            return 0.0

        # Aggregate scores based on strategy
        strategy = AggregationStrategy(self.config.strategy)

        if strategy == AggregationStrategy.HIERARCHICAL:
            return self._hierarchical_aggregation(scores)
        elif strategy == AggregationStrategy.WEIGHTED_GEOMETRIC:
            return self._weighted_geometric_mean(scores)
        elif strategy == AggregationStrategy.WEIGHTED_ARITHMETIC:
            return self._weighted_arithmetic_mean(scores)
        elif strategy == AggregationStrategy.HARMONIC:
            return self._harmonic_mean(scores)
        elif strategy == AggregationStrategy.MINIMUM:
            return min(scores.values()) if scores else 0.0
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def _hierarchical_aggregation(self, scores: dict[str, float]) -> float:
        """Hierarchical aggregation: geometric within branches, weighted across.

        This is the recommended default strategy. It:
        1. Takes geometric mean of metrics within each linguistic branch
        2. Weights branches according to their importance
        3. Combines branches using weighted geometric mean

        This approach respects the hierarchical structure of linguistic theory
        while preventing any single low score from dominating.
        """
        cfg = self.config
        branch_scores: dict[str, float] = {}
        branch_weights: dict[str, float] = {}

        # Semantic branch
        if cfg.use_semantic:
            sem_scores = []
            if "token_semantics" in scores:
                sem_scores.append(scores["token_semantics"])
            if "document_semantics" in scores:
                sem_scores.append(scores["document_semantics"])
            if sem_scores:
                branch_scores["semantic"] = float(np.power(np.prod(sem_scores), 1 / len(sem_scores)))
                branch_weights["semantic"] = cfg.semantic_weight

        # Syntactic branch
        if cfg.use_syntactic:
            syn_scores = []
            if "dependency_parse" in scores:
                syn_scores.append(scores["dependency_parse"])
            if "constituency_parse" in scores:
                syn_scores.append(scores["constituency_parse"])
            if syn_scores:
                branch_scores["syntactic"] = float(np.power(np.prod(syn_scores), 1 / len(syn_scores)))
                branch_weights["syntactic"] = cfg.syntactic_weight

        # Morphological branch
        if cfg.use_morphological and "pos_sequence" in scores:
            branch_scores["morphological"] = scores["pos_sequence"]
            branch_weights["morphological"] = cfg.morphological_weight

        # Phonological branch
        if cfg.use_phonological:
            pho_scores = []
            if "rhythmic" in scores:
                pho_scores.append(scores["rhythmic"])
            if "phonemic" in scores:
                pho_scores.append(scores["phonemic"])
            if pho_scores:
                branch_scores["phonological"] = float(np.power(np.prod(pho_scores), 1 / len(pho_scores)))
                branch_weights["phonological"] = cfg.phonological_weight

        if not branch_scores:
            return 0.0

        # Normalize weights
        total_weight = sum(branch_weights.values())
        if total_weight == 0:
            return 0.0
        normalized_weights = {k: v / total_weight for k, v in branch_weights.items()}

        # Weighted geometric mean of branches
        log_sum = sum(
            normalized_weights[branch] * np.log(score + 1e-10)
            for branch, score in branch_scores.items()
        )
        return float(np.exp(log_sum))

    def _weighted_geometric_mean(self, scores: dict[str, float]) -> float:
        """Weighted geometric mean of all metrics."""
        if not scores:
            return 0.0

        cfg = self.config
        weights = {
            "token_semantics": cfg.semantic_weight * cfg.token_semantics_weight,
            "document_semantics": cfg.semantic_weight * cfg.document_semantics_weight,
            "dependency_parse": cfg.syntactic_weight * cfg.dependency_parse_weight,
            "constituency_parse": cfg.syntactic_weight * cfg.constituency_parse_weight,
            "pos_sequence": cfg.morphological_weight,
            "rhythmic": cfg.phonological_weight * cfg.rhythmic_weight,
            "phonemic": cfg.phonological_weight * cfg.phonemic_weight,
        }

        # Filter to only metrics we have
        active_weights = {k: weights[k] for k in scores if k in weights}
        if not active_weights:
            return 0.0

        # Normalize weights
        total_weight = sum(active_weights.values())
        normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

        # Weighted geometric mean
        log_sum = sum(
            normalized_weights[name] * np.log(score + 1e-10)
            for name, score in scores.items()
            if name in normalized_weights
        )
        return float(np.exp(log_sum))

    def _weighted_arithmetic_mean(self, scores: dict[str, float]) -> float:
        """Weighted arithmetic mean of all metrics."""
        if not scores:
            return 0.0

        cfg = self.config
        weights = {
            "token_semantics": cfg.semantic_weight * cfg.token_semantics_weight,
            "document_semantics": cfg.semantic_weight * cfg.document_semantics_weight,
            "dependency_parse": cfg.syntactic_weight * cfg.dependency_parse_weight,
            "constituency_parse": cfg.syntactic_weight * cfg.constituency_parse_weight,
            "pos_sequence": cfg.morphological_weight,
            "rhythmic": cfg.phonological_weight * cfg.rhythmic_weight,
            "phonemic": cfg.phonological_weight * cfg.phonemic_weight,
        }

        # Filter to only metrics we have
        active_weights = {k: weights[k] for k in scores if k in weights}
        if not active_weights:
            return 0.0

        # Normalize weights
        total_weight = sum(active_weights.values())
        normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

        # Weighted arithmetic mean
        return float(
            sum(
                normalized_weights[name] * score
                for name, score in scores.items()
                if name in normalized_weights
            )
        )

    def _harmonic_mean(self, scores: dict[str, float]) -> float:
        """Harmonic mean (very sensitive to low scores)."""
        if not scores:
            return 0.0

        # Filter out zeros to avoid division by zero
        nonzero_scores = [s for s in scores.values() if s > 0]
        if not nonzero_scores:
            return 0.0

        return float(len(nonzero_scores) / sum(1 / s for s in nonzero_scores))

    def get_detailed_scores(self, corpus: list[str]) -> dict[str, Any]:
        """Compute detailed diversity scores across all dimensions.

        Args:
            corpus: List of text documents.

        Returns:
            Dictionary with scores for each metric and aggregated scores.
        """
        if not corpus or not all(isinstance(d, str) and d.strip() for d in corpus):
            return {}

        # Compute individual metric scores
        metric_scores: dict[str, float] = {}
        for name, metric in self._metrics.items():
            try:
                score = metric(corpus)
                metric_scores[name] = score
            except Exception as e:
                if self.config.verbose:
                    print(f"Error computing {name}: {e}")
                metric_scores[name] = 0.0

        # Compute branch-level scores (geometric mean within branch)
        cfg = self.config
        branch_scores: dict[str, float] = {}

        # Semantic
        if cfg.use_semantic:
            sem_scores = [
                s
                for k, s in metric_scores.items()
                if k in ["token_semantics", "document_semantics"]
            ]
            if sem_scores:
                branch_scores["semantic"] = float(np.power(np.prod(sem_scores), 1 / len(sem_scores)))

        # Syntactic
        if cfg.use_syntactic:
            syn_scores = [
                s
                for k, s in metric_scores.items()
                if k in ["dependency_parse", "constituency_parse"]
            ]
            if syn_scores:
                branch_scores["syntactic"] = float(np.power(np.prod(syn_scores), 1 / len(syn_scores)))

        # Morphological
        if cfg.use_morphological and "pos_sequence" in metric_scores:
            branch_scores["morphological"] = metric_scores["pos_sequence"]

        # Phonological
        if cfg.use_phonological:
            pho_scores = [
                s for k, s in metric_scores.items() if k in ["rhythmic", "phonemic"]
            ]
            if pho_scores:
                branch_scores["phonological"] = float(np.power(np.prod(pho_scores), 1 / len(pho_scores)))

        # Compute universal score
        universal_score = self(corpus)

        return {
            "universal": universal_score,
            "branches": branch_scores,
            "metrics": metric_scores,
        }

    def compute_diversity_embedding(
        self,
        corpus: list[str] | None = None,
        precomputed_scores: dict[str, float] | None = None,
        normalize: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Compute a diversity embedding vector for the corpus.

        Unlike the aggregated universal diversity score, this returns a vector
        where each dimension corresponds to a specific diversity metric in a
        fixed order (see DIVERSITY_EMBEDDING_METRICS).

        This embedding can be used for:
        - Submodular optimization to select diverse text sets
        - Clustering corpora by their diversity profiles
        - Comparing diversity profiles across different corpora

        Args:
            corpus: List of text documents. Can be None if precomputed_scores
                contains all needed metrics.
            precomputed_scores: Optional dictionary mapping metric names to
                pre-computed diversity scores. Any metrics not in this dict
                will be computed from corpus. Keys should match metric names:
                'token_semantics', 'document_semantics', 'dependency_parse',
                'constituency_parse', 'pos_sequence', 'rhythmic', 'phonemic'.
            normalize: If True (default), normalize scores to [0, 1] range
                using log transformation and empirical scaling. This makes
                metrics comparable across different scales.

        Returns:
            Numpy array of shape (7,) with diversity scores in fixed positions:
                [0] token_semantics
                [1] document_semantics
                [2] dependency_parse
                [3] constituency_parse
                [4] pos_sequence
                [5] rhythmic
                [6] phonemic

            Disabled or unavailable metrics will have a value of 0.0.

        Example:
            >>> metric = UniversalLinguisticDiversity()
            >>> corpus = ['The quick brown fox.', 'A lazy dog sleeps.']
            >>> # Compute fresh (normalized by default)
            >>> embedding = metric.compute_diversity_embedding(corpus)
            >>> # Or use precomputed scores
            >>> scores = {'token_semantics': 0.85, 'document_semantics': 0.72}
            >>> embedding = metric.compute_diversity_embedding(
            ...     corpus, precomputed_scores=scores
            ... )
        """
        embedding = np.zeros(len(DIVERSITY_EMBEDDING_METRICS), dtype=np.float64)
        precomputed_scores = precomputed_scores or {}

        # First, fill in any precomputed scores
        for name, score in precomputed_scores.items():
            if name in METRIC_TO_INDEX:
                embedding[METRIC_TO_INDEX[name]] = score

        # Check if we need to compute anything
        metrics_to_compute = [
            name
            for name in self._metrics.keys()
            if name in METRIC_TO_INDEX and name not in precomputed_scores
        ]

        # If no corpus and we need to compute, return what we have
        if not metrics_to_compute:
            if normalize:
                embedding = self._normalize_embedding(embedding)
            return embedding

        if corpus is None or not corpus:
            if self.config.verbose:
                print(
                    f"Warning: No corpus provided and missing metrics: {metrics_to_compute}"
                )
            if normalize:
                embedding = self._normalize_embedding(embedding)
            return embedding

        if not all(isinstance(d, str) and d.strip() for d in corpus):
            if self.config.verbose:
                print("Warning: corpus contains invalid inputs")
            if normalize:
                embedding = self._normalize_embedding(embedding)
            return embedding

        # Compute missing metrics from corpus
        for name in metrics_to_compute:
            metric = self._metrics[name]
            try:
                score = metric(corpus)
                embedding[METRIC_TO_INDEX[name]] = score
            except Exception as e:
                if self.config.verbose:
                    print(f"Error computing {name}: {e}")
                embedding[METRIC_TO_INDEX[name]] = 0.0

        if normalize:
            embedding = self._normalize_embedding(embedding)

        return embedding

    @staticmethod
    def _normalize_embedding(
        embedding: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Normalize diversity embedding to [0, 1] range.

        Uses log1p transformation followed by scaling based on empirical
        ranges for each metric type. This handles the fact that different
        diversity metrics have very different scales (e.g., token semantics
        can be 1-100+, while syntactic diversity is typically 1-10).

        Args:
            embedding: Raw diversity embedding of shape (7,).

        Returns:
            Normalized embedding with values in [0, 1].
        """
        # Empirical max values for each metric (after log1p transform)
        # These are approximate upper bounds based on typical corpora
        # log1p(100) ≈ 4.6, log1p(50) ≈ 3.9, log1p(10) ≈ 2.4
        log_max_values = np.array([
            5.0,  # token_semantics: can be very high (Hill number)
            4.0,  # document_semantics: typically lower
            3.0,  # dependency_parse: structural diversity
            3.0,  # constituency_parse: structural diversity
            3.0,  # pos_sequence: morphological patterns
            2.5,  # rhythmic: phonological
            2.5,  # phonemic: phonological
        ])

        # Apply log1p to compress range, then scale to [0, 1]
        normalized = np.log1p(np.maximum(embedding, 0)) / log_max_values

        # Clip to [0, 1] in case values exceed expected range
        return np.clip(normalized, 0.0, 1.0)

    @staticmethod
    def assemble_diversity_embedding(
        scores: dict[str, float],
        normalize: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Assemble a diversity embedding from pre-computed scores.

        This is a convenience method when you have all scores pre-computed
        and just need to assemble them into the standard embedding format.

        Args:
            scores: Dictionary mapping metric names to diversity scores.
            normalize: If True (default), normalize scores to [0, 1] range.

        Returns:
            Numpy array of shape (7,) with scores in standard positions.

        Example:
            >>> scores = {
            ...     'token_semantics': 25.5,
            ...     'document_semantics': 2.7,
            ...     'dependency_parse': 1.5,
            ...     'pos_sequence': 1.8,
            ... }
            >>> embedding = UniversalLinguisticDiversity.assemble_diversity_embedding(scores)
            >>> # Returns normalized values in [0, 1]
        """
        embedding = np.zeros(len(DIVERSITY_EMBEDDING_METRICS), dtype=np.float64)
        for name, score in scores.items():
            if name in METRIC_TO_INDEX:
                embedding[METRIC_TO_INDEX[name]] = score

        if normalize:
            embedding = UniversalLinguisticDiversity._normalize_embedding(embedding)

        return embedding

    def compute_corpus_diversity_embeddings(
        self,
        corpus: list[str] | None = None,
        precomputed_scores: list[dict[str, float]] | npt.NDArray[np.float64] | None = None,
        window_size: int = 1,
        stride: int = 1,
        normalize: bool = True,
        verbose: bool = False,
    ) -> npt.NDArray[np.float64]:
        """Compute diversity embeddings for each document (or window) in corpus.

        For submodular selection, we need per-document embeddings. This method
        computes the diversity "contribution" of each document by measuring
        diversity within a local window around each document.

        Args:
            corpus: List of text documents. Can be None if precomputed_scores
                is provided as a complete matrix.
            precomputed_scores: Optional pre-computed scores. Can be:
                - list[dict]: List of dicts mapping metric names to scores,
                  one dict per document. Missing metrics will be computed.
                - ndarray: Array of shape (n_docs, 7) with scores in standard
                  positions. If provided as ndarray, corpus is not needed.
            window_size: Number of documents to include in each window.
                - window_size=1: Each document's individual characteristics
                - window_size>1: Context-aware diversity (how diverse each doc
                  is relative to its neighbors)
            stride: Step size between windows. Only used when window_size > 1.
            normalize: If True (default), normalize scores to [0, 1] range.
            verbose: Whether to print progress information.

        Returns:
            Numpy array of shape (n_docs, 7) where each row is a diversity
            embedding for one document.

        Example:
            >>> metric = UniversalLinguisticDiversity()
            >>> corpus = ['Text 1', 'Text 2', 'Text 3', 'Text 4']
            >>> # Compute fresh (normalized by default)
            >>> embeddings = metric.compute_corpus_diversity_embeddings(corpus)
            >>> # Or use precomputed
            >>> scores = [{'token_semantics': 25.5}, {'token_semantics': 18.2}, ...]
            >>> embeddings = metric.compute_corpus_diversity_embeddings(
            ...     corpus, precomputed_scores=scores
            ... )
        """
        n_metrics = len(DIVERSITY_EMBEDDING_METRICS)

        # Handle case where precomputed_scores is already a complete matrix
        if isinstance(precomputed_scores, np.ndarray):
            if precomputed_scores.ndim == 2 and precomputed_scores.shape[1] == n_metrics:
                result = precomputed_scores.astype(np.float64)
                if normalize:
                    # Normalize each row
                    for i in range(result.shape[0]):
                        result[i] = self._normalize_embedding(result[i])
                return result
            else:
                raise ValueError(
                    f"precomputed_scores array must have shape (n_docs, {n_metrics}), "
                    f"got {precomputed_scores.shape}"
                )

        # Need corpus for computing embeddings
        if corpus is None or not corpus:
            if precomputed_scores:
                # Assemble from list of dicts
                n_docs = len(precomputed_scores)
                embeddings = np.zeros((n_docs, n_metrics), dtype=np.float64)
                for i, scores in enumerate(precomputed_scores):
                    embeddings[i] = self.assemble_diversity_embedding(
                        scores, normalize=normalize
                    )
                return embeddings
            return np.zeros((0, n_metrics), dtype=np.float64)

        n_docs = len(corpus)
        embeddings = np.zeros((n_docs, n_metrics), dtype=np.float64)

        # Convert precomputed_scores list to easier format if provided
        precomputed_list: list[dict[str, float]] = []
        if precomputed_scores is not None:
            if isinstance(precomputed_scores, list):
                precomputed_list = precomputed_scores
            else:
                precomputed_list = [{} for _ in range(n_docs)]

        # Pad precomputed_list if needed
        while len(precomputed_list) < n_docs:
            precomputed_list.append({})

        if window_size == 1:
            # Per-document embedding: compute features for each doc individually
            if verbose:
                print(f"Computing diversity embeddings for {n_docs} documents...")

            for i, doc in enumerate(corpus):
                if verbose and (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{n_docs} documents")

                if not doc or not doc.strip():
                    continue

                precomputed = precomputed_list[i] if i < len(precomputed_list) else {}

                # If we have all scores precomputed, just assemble
                if precomputed and all(
                    name in precomputed for name in self._metrics.keys()
                    if name in METRIC_TO_INDEX
                ):
                    embeddings[i] = self.assemble_diversity_embedding(
                        precomputed, normalize=normalize
                    )
                else:
                    # Compute with any available precomputed scores
                    embeddings[i] = self._compute_single_doc_embedding(
                        [doc], precomputed_scores=precomputed, normalize=normalize
                    )
        else:
            # Windowed embedding: diversity within local context
            if verbose:
                print(f"Computing windowed embeddings (window={window_size}, stride={stride})...")

            for i in range(0, n_docs, stride):
                window_end = min(i + window_size, n_docs)
                window = corpus[i:window_end]

                if not window:
                    continue

                # For windowed mode, we don't use per-doc precomputed scores
                # (would need to aggregate them meaningfully)
                window_embedding = self.compute_diversity_embedding(
                    window, normalize=normalize
                )

                # Assign embedding to all documents in window
                for j in range(i, window_end):
                    embeddings[j] = window_embedding

                if verbose and (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{n_docs} windows")

        return embeddings

    def _compute_single_doc_embedding(
        self,
        doc_as_list: list[str],
        precomputed_scores: dict[str, float] | None = None,
        normalize: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Compute embedding for a single document.

        For a single document, traditional diversity metrics (which measure
        variation across multiple items) don't directly apply. Instead, we
        extract features that characterize the document's linguistic properties.

        This uses a feature-based approach: for each metric, we extract a scalar
        that represents the document's "richness" in that dimension.

        Args:
            doc_as_list: Single document wrapped in a list.
            precomputed_scores: Optional dict of precomputed metric scores.
                Metrics present in this dict will not be recomputed.
            normalize: If True (default), normalize the embedding to [0, 1].

        Returns:
            Feature vector of shape (7,).
        """
        embedding = np.zeros(len(DIVERSITY_EMBEDDING_METRICS), dtype=np.float64)
        precomputed_scores = precomputed_scores or {}

        # First fill in precomputed values
        for name, score in precomputed_scores.items():
            if name in METRIC_TO_INDEX:
                embedding[METRIC_TO_INDEX[name]] = score

        doc = doc_as_list[0]
        if not doc or not doc.strip():
            if normalize:
                embedding = self._normalize_embedding(embedding)
            return embedding

        # For semantic metrics: use embedding norm or entropy as proxy
        # For structural metrics: use feature counts normalized by length

        for name, metric in self._metrics.items():
            if name not in METRIC_TO_INDEX:
                continue

            # Skip if already precomputed
            if name in precomputed_scores:
                continue

            idx = METRIC_TO_INDEX[name]

            try:
                if name in ["token_semantics", "document_semantics"]:
                    # Semantic richness: compute embedding and use its properties
                    # Higher diversity in a single doc = more diverse vocabulary/concepts
                    embedding[idx] = self._compute_semantic_richness(metric, doc)

                elif name in ["dependency_parse", "constituency_parse"]:
                    # Syntactic complexity: use tree depth/breadth measures
                    embedding[idx] = self._compute_syntactic_complexity(metric, doc)

                elif name == "pos_sequence":
                    # Morphological diversity: POS tag variety
                    embedding[idx] = self._compute_pos_variety(metric, doc)

                elif name in ["rhythmic", "phonemic"]:
                    # Phonological properties
                    embedding[idx] = self._compute_phonological_richness(metric, doc)

            except Exception:
                embedding[idx] = 0.0

        if normalize:
            embedding = self._normalize_embedding(embedding)

        return embedding

    def _compute_semantic_richness(self, metric: DiversityMetric, doc: str) -> float:
        """Compute semantic richness for a single document."""
        # Use a small corpus with just this doc repeated with slight variations
        # or compute based on unique tokens
        words = doc.split()
        if len(words) < 2:
            return 0.0

        # Create pseudo-corpus by splitting doc into chunks
        chunk_size = max(3, len(words) // 3)
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        if len(chunks) < 2:
            # If we can't split, use vocabulary ratio as proxy
            unique_ratio = len(set(words)) / len(words) if words else 0.0
            return unique_ratio

        try:
            return metric(chunks)
        except Exception:
            return len(set(words)) / len(words) if words else 0.0

    def _compute_syntactic_complexity(
        self, metric: DiversityMetric, doc: str
    ) -> float:
        """Compute syntactic complexity for a single document."""
        # Split into sentences and measure diversity across them
        import re

        sentences = re.split(r"[.!?]+", doc)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            # For single sentence, return normalized complexity
            words = doc.split()
            return min(1.0, len(words) / 20.0)  # Normalize by typical sentence length

        try:
            return metric(sentences)
        except Exception:
            return 0.5  # Default middle value

    def _compute_pos_variety(self, metric: DiversityMetric, doc: str) -> float:
        """Compute POS variety for a single document."""
        import re

        sentences = re.split(r"[.!?]+", doc)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            # For single sentence, estimate from word patterns
            words = doc.split()
            if not words:
                return 0.0
            # Use word shape diversity as proxy
            shapes = set()
            for w in words:
                if w.isupper():
                    shapes.add("UPPER")
                elif w[0].isupper() if w else False:
                    shapes.add("Title")
                elif w.islower():
                    shapes.add("lower")
                else:
                    shapes.add("mixed")
            return len(shapes) / 4.0  # Normalize

        try:
            return metric(sentences)
        except Exception:
            return 0.5

    def _compute_phonological_richness(
        self, metric: DiversityMetric, doc: str
    ) -> float:
        """Compute phonological richness for a single document."""
        import re

        sentences = re.split(r"[.!?]+", doc)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 2:
            # Use syllable variety as proxy
            words = doc.split()
            if not words:
                return 0.0
            # Simple syllable count heuristic
            syllable_counts = set()
            for w in words:
                vowels = len(re.findall(r"[aeiouAEIOU]", w))
                syllable_counts.add(min(vowels, 5))  # Cap at 5
            return len(syllable_counts) / 5.0

        try:
            return metric(sentences)
        except Exception:
            return 0.5


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "balanced": {
        "strategy": "hierarchical",
        "semantic_weight": 0.35,
        "syntactic_weight": 0.30,
        "morphological_weight": 0.15,
        "phonological_weight": 0.20,
    },
    "semantic_focus": {
        "strategy": "hierarchical",
        "semantic_weight": 0.60,
        "syntactic_weight": 0.20,
        "morphological_weight": 0.10,
        "phonological_weight": 0.10,
    },
    "structural_focus": {
        "strategy": "hierarchical",
        "semantic_weight": 0.20,
        "syntactic_weight": 0.50,
        "morphological_weight": 0.20,
        "phonological_weight": 0.10,
    },
    "minimal": {
        "strategy": "weighted_geometric",
        "use_constituency_parse": False,
        "use_rhythmic": False,
        "use_phonemic": False,
    },
    "conservative": {
        "strategy": "harmonic",  # Very sensitive to low scores
    },
    "dementia_detector": {
        # Evidence-based weights from DementiaBank evaluation (498 subjects)
        # Based on Cohen's d effect sizes from statistically significant metrics
        #
        # Empirical results:
        #   doc_semantic: d=0.621, p<0.0001 (medium effect, highly significant)
        #   token_semantic: d=0.290, p=0.0013 (small effect, significant)
        #   syntactic_const: d=0.247, p=0.0056 (small effect, significant)
        #   syntactic_dep: d=0.165, p=0.0681 (negligible, not significant)
        #   morphological: d=0.161, p=0.0787 (negligible, not significant)
        #   phonemic: d=0.117, p=0.1911 (negligible, not significant)
        #   rhythmic: d=-0.091, p=0.2984 (negligible, opposite direction)
        #   lexical_ttr: d=0.051, p=0.5666 (negligible, not significant)
        #
        # Strategy V1 (proportional to effect sizes):
        #   Result: d=0.335 (46% decrease from best individual metric)
        #   Conclusion: Adding weaker metrics dilutes strong signal
        #
        # Strategy V2 (quadratic weighting - emphasize strongest signals):
        #   Weight = (Cohen's d)^2 for significant metrics only
        #   doc_semantic: 0.621^2 = 0.386
        #   token_semantic: 0.290^2 = 0.084
        #   syntactic_const: 0.247^2 = 0.061
        #   Total: 0.531
        #   Normalized weights:
        #     doc_semantic: 0.386/0.531 = 72.7%
        #     token_semantic: 0.084/0.531 = 15.8%
        #     syntactic_const: 0.061/0.531 = 11.5%
        #
        #   Branch weights:
        #     Semantic: 72.7% + 15.8% = 88.5%
        #     Syntactic: 11.5%
        #   Within semantic: doc 72.7/(72.7+15.8) = 82.1%, token 17.9%
        "strategy": "weighted_arithmetic",  # Arithmetic for interpretability
        "semantic_weight": 0.885,
        "syntactic_weight": 0.115,
        "morphological_weight": 0.0,
        "phonological_weight": 0.0,
        # Within-branch weights (quadratic emphasis on doc_semantic)
        "token_semantics_weight": 0.179,
        "document_semantics_weight": 0.821,
        "dependency_parse_weight": 0.0,
        "constituency_parse_weight": 1.0,
        # Disable non-significant metrics
        "use_morphological": False,
        "use_phonological": False,
        "use_dependency_parse": False,
    },
    "dementia_detector_pure": {
        # Pure semantic approach: Use ONLY doc_semantic (the strongest metric)
        # This is the optimal single-metric strategy for dementia detection
        # Result: d=0.621 (medium effect, highest discrimination)
        "strategy": "weighted_arithmetic",
        "semantic_weight": 1.0,
        "syntactic_weight": 0.0,
        "morphological_weight": 0.0,
        "phonological_weight": 0.0,
        # Use only document semantics
        "token_semantics_weight": 0.0,
        "document_semantics_weight": 1.0,
        # Disable everything else
        "use_token_semantics": False,
        "use_syntactic": False,
        "use_morphological": False,
        "use_phonological": False,
    },
}


def get_preset_config(preset: str) -> dict[str, Any]:
    """Get a preset configuration by name.

    Available presets:
        - balanced: Equal consideration of all dimensions (default)
        - semantic_focus: Emphasizes semantic diversity
        - structural_focus: Emphasizes syntactic and morphological diversity
        - minimal: Uses only core metrics (no optional dependencies)
        - conservative: Uses harmonic mean (sensitive to low scores)
        - dementia_detector: Evidence-based composite for cognitive impairment detection
                           (quadratic weighting: 88.5% semantic, 11.5% syntactic)
        - dementia_detector_pure: Pure doc_semantic approach (optimal single metric)
                                (100% document-level semantic diversity)

    Args:
        preset: Name of preset configuration.

    Returns:
        Configuration dictionary.

    Raises:
        ValueError: If preset name is unknown.
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(
            f"Unknown preset '{preset}'. Available: {list(PRESET_CONFIGS.keys())}"
        )
    return PRESET_CONFIGS[preset].copy()
