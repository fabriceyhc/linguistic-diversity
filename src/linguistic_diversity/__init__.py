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
"""

__version__ = "1.0.0"

from .metric import DiversityMetric, Metric, ScaledEstimationResult, SimilarityMetric, TextDiversity
from .diversities import (
    # Semantic
    TokenSemantics,
    DocumentSemantics,
    # Syntactic
    DependencyParse,
    ConstituencyParse,
    # Morphological
    PartOfSpeechSequence,
    # Phonological
    Rhythmic,
    Phonemic,
    # Universal
    UniversalLinguisticDiversity,
    get_preset_config,
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
]
