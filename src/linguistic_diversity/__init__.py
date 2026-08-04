"""
Linguistic Diversity - Measure linguistic diversity using similarity-sensitive Hill numbers.

This package provides metrics for quantifying diversity across multiple linguistic dimensions:
- Semantic diversity (token and document level)
- Syntactic diversity (dependency and constituency parse trees)
- Morphological diversity (part-of-speech sequences)
- Phonological diversity (rhythmic and phonemic patterns)

Example:
    >>> from linguistic_diversity import TokenSemantics, DependencyParse
    >>> corpus = ['one massive earth', 'an enormous globe', 'the colossal world']
    >>> metric = TokenSemantics()
    >>> diversity = metric(corpus)
    >>> print(f"Token semantic diversity: {diversity:.2f}")

For diversity-based text selection:
    >>> from linguistic_diversity import UniversalLinguisticDiversity
    >>> from linguistic_diversity.selection import select_diverse_texts
    >>> metric = UniversalLinguisticDiversity()
    >>> embeddings = metric.compute_corpus_diversity_embeddings(corpus)
    >>> result = select_diverse_texts(embeddings, n_select=100)
"""

__version__ = "2.0.0"

from .diversities import (
    # Embedding constants
    DIVERSITY_EMBEDDING_METRICS,
    METRIC_TO_INDEX,
    ConstituencyParse,
    # Syntactic
    DependencyParse,
    DistinctN,
    DocumentSemantics,
    # Morphological
    PartOfSpeechSequence,
    Phonemic,
    # Phonological
    Rhythmic,
    SelfBLEU,
    # Semantic
    TokenSemantics,
    TypeTokenRatio,
    # Universal
    UniversalLinguisticDiversity,
    clear_model_cache,
    get_preset_config,
)
from .ecology import (
    EVENNESS_CLASSES,
    PartitionResult,
    coverage_deficit,
    evenness,
    expected_coverage,
    partition_diversity,
    sample_coverage,
    size_for_coverage,
)
from .metric import DiversityMetric, Metric, ScaledEstimationResult, SimilarityMetric, TextDiversity
from .selection import (
    BalancedCoverageSelector,
    DiversitySelector,
    FacilityLocationSelector,
    MaxMinDiversitySelector,
    SelectionResult,
    select_diverse_texts,
)

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Metric",
    "DiversityMetric",
    "SimilarityMetric",
    "TextDiversity",
    "ScaledEstimationResult",
    # Semantic metrics
    "TokenSemantics",
    "TypeTokenRatio",
    "DistinctN",
    "SelfBLEU",
    "clear_model_cache",
    "DocumentSemantics",
    # Syntactic metrics
    "DependencyParse",
    "ConstituencyParse",
    # Morphological metrics
    "PartOfSpeechSequence",
    # Phonological metrics
    "Rhythmic",
    "Phonemic",
    # Universal metric
    "UniversalLinguisticDiversity",
    "get_preset_config",
    # Embedding constants
    "DIVERSITY_EMBEDDING_METRICS",
    "METRIC_TO_INDEX",
    # Selection
    # Ecology: evenness, sample coverage, alpha/beta/gamma partitioning
    "EVENNESS_CLASSES",
    "PartitionResult",
    "coverage_deficit",
    "evenness",
    "expected_coverage",
    "partition_diversity",
    "sample_coverage",
    "size_for_coverage",
    "SelectionResult",
    "DiversitySelector",
    "FacilityLocationSelector",
    "MaxMinDiversitySelector",
    "BalancedCoverageSelector",
    "select_diverse_texts",
]
