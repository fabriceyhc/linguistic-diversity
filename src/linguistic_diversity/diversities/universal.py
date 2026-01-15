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

from ..metric import DiversityMetric, MetricConfig


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
