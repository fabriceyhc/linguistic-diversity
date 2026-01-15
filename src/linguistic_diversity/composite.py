"""Composite linguistic diversity metrics using weighted combinations.

This module provides composite metrics that combine multiple diversity measures
using evidence-based weighting strategies.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from .diversities import (
    ConstituencyParse,
    DependencyParse,
    DocumentSemantics,
    PartOfSpeechSequence,
    Phonemic,
    Rhythmic,
    TokenSemantics,
)


class CompositeDiversity:
    """Composite diversity metric combining multiple linguistic levels.

    This metric combines semantic, syntactic, morphological, phonological, and
    lexical diversity using weighted averages. Weights can be determined by:

    1. 'effect_size': Weight by empirical effect sizes (Cohen's d)
    2. 'significance': Weight by statistical significance (inverse p-values)
    3. 'hybrid': Combination of effect size and significance
    4. 'equal': Equal weights (baseline)
    5. 'custom': User-provided weights

    Example:
        >>> # Use effect-size based weighting (recommended for dementia detection)
        >>> metric = CompositeDiversity(strategy='effect_size')
        >>> corpus = ['sentence one', 'sentence two', 'sentence three']
        >>> diversity = metric(corpus)

        >>> # Use custom weights
        >>> weights = {
        ...     'doc_semantic': 0.6,
        ...     'token_semantic': 0.3,
        ...     'syntactic_const': 0.1
        ... }
        >>> metric = CompositeDiversity(strategy='custom', custom_weights=weights)
        >>> diversity = metric(corpus)
    """

    # Empirical evidence from DementiaBank evaluation (498 subjects)
    # Format: metric_name -> (cohen's_d, p_value, direction)
    DEMENTIA_EVIDENCE = {
        'doc_semantic': (0.621, 1.31e-11, 'Control > Dementia'),
        'token_semantic': (0.290, 0.0013, 'Control > Dementia'),
        'syntactic_const': (0.247, 0.0056, 'Control > Dementia'),
        'syntactic_dep': (0.165, 0.0681, 'Control > Dementia'),
        'morphological': (0.161, 0.0787, 'Control > Dementia'),
        'phonemic': (0.117, 0.1911, 'Control > Dementia'),
        'rhythmic': (-0.091, 0.2984, 'Dementia > Control'),  # Inverted
        'lexical_ttr': (0.051, 0.5666, 'Control > Dementia'),
    }

    def __init__(
        self,
        strategy: Literal['effect_size', 'significance', 'hybrid', 'equal', 'custom'] = 'effect_size',
        custom_weights: dict[str, float] | None = None,
        significance_threshold: float = 0.05,
        min_effect_size: float = 0.2,
        normalize_scores: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize composite diversity metric.

        Args:
            strategy: Weighting strategy to use.
            custom_weights: Custom weights for 'custom' strategy.
            significance_threshold: P-value threshold for inclusion (default: 0.05).
            min_effect_size: Minimum Cohen's d for inclusion (default: 0.2 = small).
            normalize_scores: Whether to normalize individual scores before combining.
            config: Configuration passed to individual metrics.
        """
        self.strategy = strategy
        self.custom_weights = custom_weights or {}
        self.significance_threshold = significance_threshold
        self.min_effect_size = min_effect_size
        self.normalize_scores = normalize_scores
        self.config = config or {}

        # Calculate weights based on strategy
        self.weights = self._calculate_weights()

        # Initialize metrics that have non-zero weights
        self.metrics: dict[str, Any] = {}
        self._initialize_metrics()

        # Normalization statistics (computed on first call)
        self.norm_stats: dict[str, tuple[float, float]] = {}

    def _calculate_weights(self) -> dict[str, float]:
        """Calculate metric weights based on strategy.

        Returns:
            Dictionary of metric_name -> weight.
        """
        if self.strategy == 'custom':
            return self.custom_weights.copy()

        elif self.strategy == 'equal':
            # Equal weights for all metrics
            return {name: 1.0 for name in self.DEMENTIA_EVIDENCE.keys()}

        elif self.strategy == 'effect_size':
            # Weight by Cohen's d (only positive effects above threshold)
            weights = {}
            for name, (d, p, direction) in self.DEMENTIA_EVIDENCE.items():
                # Only include metrics with:
                # 1. Positive effect (higher diversity in controls)
                # 2. At least small effect size (|d| >= min_effect_size)
                if d >= self.min_effect_size and 'Control > Dementia' in direction:
                    weights[name] = d
            return weights

        elif self.strategy == 'significance':
            # Weight by inverse p-value (only significant metrics)
            weights = {}
            for name, (d, p, direction) in self.DEMENTIA_EVIDENCE.items():
                if p < self.significance_threshold and 'Control > Dementia' in direction:
                    # Use -log10(p) as weight (so p=0.001 -> weight=3, p=0.05 -> weight=1.3)
                    weights[name] = -np.log10(p)
            return weights

        elif self.strategy == 'hybrid':
            # Combine effect size and significance
            # Weight = d * -log10(p) for significant metrics with small+ effect
            weights = {}
            for name, (d, p, direction) in self.DEMENTIA_EVIDENCE.items():
                if (p < self.significance_threshold and
                    d >= self.min_effect_size and
                    'Control > Dementia' in direction):
                    # Multiplicative: both effect size and significance matter
                    weights[name] = d * (-np.log10(p))
            return weights

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _initialize_metrics(self) -> None:
        """Initialize only the metrics that have non-zero weights."""
        metric_classes = {
            'doc_semantic': DocumentSemantics,
            'token_semantic': TokenSemantics,
            'syntactic_dep': DependencyParse,
            'syntactic_const': ConstituencyParse,
            'morphological': PartOfSpeechSequence,
            'phonemic': Phonemic,
            'rhythmic': Rhythmic,
        }

        for name, weight in self.weights.items():
            if weight > 0 and name in metric_classes:
                # Initialize with user config
                self.metrics[name] = metric_classes[name](self.config.get(name, {}))

    def __call__(self, corpus: list[str]) -> float:
        """Compute composite diversity for a corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Weighted composite diversity score.
        """
        if len(self.metrics) == 0:
            raise ValueError(
                f"No metrics selected with strategy '{self.strategy}'. "
                f"Try adjusting significance_threshold or min_effect_size."
            )

        # Compute diversity for each metric
        scores = {}
        for name, metric in self.metrics.items():
            try:
                score = metric(corpus)
                scores[name] = score
            except Exception as e:
                # Skip metrics that fail
                print(f"Warning: {name} failed with error: {e}")
                continue

        if len(scores) == 0:
            return 0.0

        # Normalize scores if requested
        if self.normalize_scores:
            scores = self._normalize_scores(scores)

        # Compute weighted average
        total_weight = sum(self.weights.get(name, 0) for name in scores.keys())
        if total_weight == 0:
            return 0.0

        composite_score = sum(
            scores[name] * self.weights[name]
            for name in scores.keys()
        ) / total_weight

        return float(composite_score)

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        """Normalize scores to [0, 1] range using expected statistics.

        Args:
            scores: Dictionary of metric_name -> raw_score.

        Returns:
            Dictionary of metric_name -> normalized_score.
        """
        # Expected ranges from DementiaBank data (mean ± 3*std covers ~99.7%)
        expected_ranges = {
            'doc_semantic': (1.556, 7.028),      # Observed range
            'token_semantic': (8.266, 63.446),   # Observed range
            'syntactic_dep': (1.009, 1.083),     # Observed range
            'syntactic_const': (1.000, 1.000),   # Constant
            'morphological': (1.600, 166.504),   # Observed range (with outliers)
            'phonemic': (2.000, 49.000),         # Observed range
            'rhythmic': (1.385, 674.630),        # Observed range (with outliers)
            'lexical_ttr': (0.302, 0.814),       # Observed range
        }

        normalized = {}
        for name, score in scores.items():
            if name in expected_ranges:
                min_val, max_val = expected_ranges[name]
                # Normalize to [0, 1]
                if max_val > min_val:
                    norm_score = (score - min_val) / (max_val - min_val)
                    # Clip to [0, 1] in case of outliers
                    norm_score = np.clip(norm_score, 0.0, 1.0)
                else:
                    norm_score = 0.5  # Constant value
                normalized[name] = norm_score
            else:
                # Unknown metric, keep as-is
                normalized[name] = score

        return normalized

    def get_weights(self) -> dict[str, float]:
        """Get the computed weights for inspection.

        Returns:
            Dictionary of metric_name -> weight.
        """
        return self.weights.copy()

    def get_metric_scores(self, corpus: list[str]) -> dict[str, float]:
        """Get individual metric scores (useful for debugging).

        Args:
            corpus: List of text documents.

        Returns:
            Dictionary of metric_name -> score.
        """
        scores = {}
        for name, metric in self.metrics.items():
            try:
                scores[name] = metric(corpus)
            except Exception as e:
                print(f"Warning: {name} failed: {e}")
                scores[name] = np.nan
        return scores


def get_dementia_detector(
    config: dict[str, Any] | None = None
) -> CompositeDiversity:
    """Get a pre-configured composite metric optimized for dementia detection.

    This uses the 'effect_size' strategy based on DementiaBank empirical evidence,
    which weights metrics by their Cohen's d effect sizes.

    Args:
        config: Optional configuration for individual metrics.

    Returns:
        Configured CompositeDiversity metric.

    Example:
        >>> detector = get_dementia_detector()
        >>> diversity = detector(['The boy is getting a cookie.', 'The woman is washing dishes.'])
    """
    return CompositeDiversity(
        strategy='effect_size',
        min_effect_size=0.2,  # Include small+ effects
        normalize_scores=True,
        config=config or {}
    )
