"""Example demonstrating the Universal Linguistic Diversity metric.

This example shows how to use the UniversalLinguisticDiversity metric to
compute a single comprehensive diversity score that combines measurements
across all linguistic dimensions.
"""

from linguistic_diversity import UniversalLinguisticDiversity, get_preset_config


def main() -> None:
    """Demonstrate universal diversity metric with different configurations."""
    print("=" * 80)
    print("Universal Linguistic Diversity - Comprehensive Example")
    print("=" * 80)

    # Example corpora
    corpus1 = [
        "one massive earth",
        "an enormous globe",
        "the colossal world",
    ]
    corpus1_name = "High Paraphrase Corpus"

    corpus2 = [
        "basic human right",
        "you were right",
        "make a right",
    ]
    corpus2_name = "Ambiguous Word Corpus"

    corpus3 = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps above an idle hound.",
        "Swift russet predator vaults past lethargic beast.",
    ]
    corpus3_name = "Literary Paraphrase Corpus"

    # ========================================================================
    # EXAMPLE 1: Default Configuration (Balanced)
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Default Balanced Configuration")
    print("=" * 80)
    print("Uses hierarchical aggregation with balanced weights across all dimensions")
    print()

    metric = UniversalLinguisticDiversity({"verbose": True})

    for corpus, name in [(corpus1, corpus1_name), (corpus2, corpus2_name), (corpus3, corpus3_name)]:
        print(f"\n{name}:")
        print("-" * 80)
        diversity = metric(corpus)
        print(f"Universal Diversity: {diversity:.3f}")

    # ========================================================================
    # EXAMPLE 2: Detailed Scores
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Detailed Score Breakdown")
    print("=" * 80)
    print(f"Analyzing: {corpus3_name}")
    print()

    metric = UniversalLinguisticDiversity()
    detailed = metric.get_detailed_scores(corpus3)

    print(f"Universal Diversity: {detailed['universal']:.3f}")
    print()
    print("Branch-Level Scores:")
    print("-" * 40)
    for branch, score in detailed["branches"].items():
        print(f"  {branch.capitalize():20s}: {score:.3f}")

    print()
    print("Individual Metric Scores:")
    print("-" * 40)
    for metric_name, score in detailed["metrics"].items():
        formatted_name = metric_name.replace("_", " ").title()
        print(f"  {formatted_name:20s}: {score:.3f}")

    # ========================================================================
    # EXAMPLE 3: Preset Configurations
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Comparing Preset Configurations")
    print("=" * 80)
    print(f"Analyzing: {corpus2_name}")
    print()

    presets = ["balanced", "semantic_focus", "structural_focus", "minimal"]

    for preset in presets:
        config = get_preset_config(preset)
        metric = UniversalLinguisticDiversity(config)
        diversity = metric(corpus2)
        print(f"{preset:20s}: {diversity:.3f}")

    # ========================================================================
    # EXAMPLE 4: Custom Configuration
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Custom Configuration")
    print("=" * 80)
    print("Custom weights: Heavy emphasis on semantic and syntactic diversity")
    print()

    custom_config = {
        "strategy": "hierarchical",
        "semantic_weight": 0.50,  # 50% semantic
        "syntactic_weight": 0.35,  # 35% syntactic
        "morphological_weight": 0.10,  # 10% morphological
        "phonological_weight": 0.05,  # 5% phonological
        "verbose": False,
    }

    metric = UniversalLinguisticDiversity(custom_config)

    for corpus, name in [(corpus1, corpus1_name), (corpus2, corpus2_name), (corpus3, corpus3_name)]:
        diversity = metric(corpus)
        print(f"{name:30s}: {diversity:.3f}")

    # ========================================================================
    # EXAMPLE 5: Different Aggregation Strategies
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Comparing Aggregation Strategies")
    print("=" * 80)
    print(f"Analyzing: {corpus1_name}")
    print()

    strategies = [
        "hierarchical",
        "weighted_geometric",
        "weighted_arithmetic",
        "harmonic",
        "minimum",
    ]

    for strategy in strategies:
        config = {"strategy": strategy}
        metric = UniversalLinguisticDiversity(config)
        diversity = metric(corpus1)
        print(f"{strategy:25s}: {diversity:.3f}")

    # ========================================================================
    # EXAMPLE 6: Selective Metrics
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Selective Metrics (Semantic Only)")
    print("=" * 80)
    print("Computing diversity using only semantic metrics")
    print()

    semantic_only_config = {
        "use_semantic": True,
        "use_syntactic": False,
        "use_morphological": False,
        "use_phonological": False,
        "verbose": False,
    }

    metric = UniversalLinguisticDiversity(semantic_only_config)

    for corpus, name in [(corpus1, corpus1_name), (corpus2, corpus2_name), (corpus3, corpus3_name)]:
        diversity = metric(corpus)
        print(f"{name:30s}: {diversity:.3f}")

    # ========================================================================
    # INTERPRETATION GUIDE
    # ========================================================================
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE")
    print("=" * 80)
    print(
        """
The Universal Linguistic Diversity score represents the overall linguistic
richness of a corpus, combining measurements across multiple dimensions:

- **Semantic**: Meaning diversity (similar meanings = lower diversity)
- **Syntactic**: Structural diversity (similar parse trees = lower diversity)
- **Morphological**: POS pattern diversity (similar grammar = lower diversity)
- **Phonological**: Sound diversity (similar rhythms/phonemes = lower diversity)

Score Interpretation:
- Higher scores = More linguistically diverse
- Lower scores = More linguistically uniform
- Score ≈ N means corpus has diversity equivalent to N distinct items

Aggregation Strategies:
- **hierarchical**: Geometric mean within branches, weighted across (default)
- **weighted_geometric**: Multiplicative combination (penalizes low scores)
- **weighted_arithmetic**: Additive combination (average of all metrics)
- **harmonic**: Very conservative (very sensitive to low scores)
- **minimum**: Most conservative (dominated by lowest score)

Use Cases:
- **balanced**: General-purpose analysis
- **semantic_focus**: Content diversity (news, essays, creative writing)
- **structural_focus**: Grammatical complexity (language learning, style)
- **minimal**: Fast computation without optional dependencies
- **custom**: Tailor to specific research questions
"""
    )


if __name__ == "__main__":
    main()
