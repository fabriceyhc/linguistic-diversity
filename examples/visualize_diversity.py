"""Visualization of diversity metrics and internal processes.

This script demonstrates all diversity metrics with visualizations of:
- Similarity matrices
- Diversity scores across different corpora
- Internal representations (embeddings, parse trees, etc.)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

from linguistic_diversity import (
    TokenSemantics,
    DocumentSemantics,
    DependencyParse,
    PartOfSpeechSequence,
)

# Create output directory
output_dir = Path("visualization_output")
output_dir.mkdir(exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def visualize_similarity_matrix(Z, labels, title, filename):
    """Visualize a similarity matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(
        Z,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        square=True,
        cbar_kws={"label": "Similarity"},
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def visualize_diversity_comparison(results_df, title, filename):
    """Visualize diversity scores across different metrics and corpora."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for plotting
    x = np.arange(len(results_df))
    width = 0.35

    # Plot bars
    corpus1_bars = ax.bar(
        x - width/2,
        results_df["High Paraphrase"],
        width,
        label="High Paraphrase Corpus",
        color="#3498db",
        alpha=0.8,
    )
    corpus2_bars = ax.bar(
        x + width/2,
        results_df["Low Diversity"],
        width,
        label="Low Diversity Corpus",
        color="#e74c3c",
        alpha=0.8,
    )

    # Customize
    ax.set_ylabel("Diversity Score", fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["Metric"], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [corpus1_bars, corpus2_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def test_document_semantics():
    """Test and visualize document semantic diversity."""
    print("\n" + "="*70)
    print("DOCUMENT SEMANTIC DIVERSITY")
    print("="*70)

    # Test corpora
    corpus1 = [
        "one massive earth",
        "an enormous globe",
        "the colossal world",
    ]

    corpus2 = [
        "basic human right",
        "you were right",
        "make a right",
    ]

    # Initialize metric
    metric = DocumentSemantics({
        "model_name": "all-MiniLM-L6-v2",
        "use_cuda": False,
        "verbose": False,
    })

    # Extract features and compute similarity matrices
    print("\n1. High Paraphrase Corpus (similar meanings, different words)")
    features1, docs1 = metric.extract_features(corpus1)
    Z1 = metric.calculate_similarities(features1)
    diversity1 = metric(corpus1)
    print(f"   Diversity: {diversity1:.3f}")

    visualize_similarity_matrix(
        Z1,
        [f"Doc {i+1}" for i in range(len(corpus1))],
        "Document Similarity Matrix: High Paraphrase Corpus\n"
        "(Similar meanings → High similarity → Low diversity)",
        "doc_semantic_matrix_corpus1.png",
    )

    print("\n2. Low Diversity Corpus (word 'right' in different contexts)")
    features2, docs2 = metric.extract_features(corpus2)
    Z2 = metric.calculate_similarities(features2)
    diversity2 = metric(corpus2)
    print(f"   Diversity: {diversity2:.3f}")

    visualize_similarity_matrix(
        Z2,
        [f"Doc {i+1}" for i in range(len(corpus2))],
        "Document Similarity Matrix: Low Diversity Corpus\n"
        "(Different meanings → Lower similarity → Higher diversity)",
        "doc_semantic_matrix_corpus2.png",
    )

    return {
        "Metric": "Document Semantics",
        "High Paraphrase": diversity1,
        "Low Diversity": diversity2,
    }


def test_token_semantics():
    """Test and visualize token semantic diversity."""
    print("\n" + "="*70)
    print("TOKEN SEMANTIC DIVERSITY")
    print("="*70)

    corpus1 = [
        "one massive earth",
        "an enormous globe",
        "the colossal world",
    ]

    corpus2 = [
        "basic human right",
        "you were right",
        "make a right",
    ]

    # Initialize metric
    metric = TokenSemantics({
        "model_name": "bert-base-uncased",
        "use_cuda": False,
        "verbose": False,
        "remove_stopwords": True,
    })

    # Corpus 1
    print("\n1. High Paraphrase Corpus")
    features1, tokens1 = metric.extract_features(corpus1)
    Z1 = metric.calculate_similarities(features1)
    diversity1 = metric(corpus1)
    print(f"   Tokens: {tokens1[:10]}... ({len(tokens1)} total)")
    print(f"   Diversity: {diversity1:.3f}")

    # Visualize subset of similarity matrix (first 10 tokens)
    n_display = min(10, len(tokens1))
    visualize_similarity_matrix(
        Z1[:n_display, :n_display],
        tokens1[:n_display],
        f"Token Similarity Matrix: High Paraphrase Corpus (first {n_display} tokens)\n"
        "(Contextualized BERT embeddings)",
        "token_semantic_matrix_corpus1.png",
    )

    # Corpus 2
    print("\n2. Low Diversity Corpus")
    features2, tokens2 = metric.extract_features(corpus2)
    Z2 = metric.calculate_similarities(features2)
    diversity2 = metric(corpus2)
    print(f"   Tokens: {tokens2[:10]}... ({len(tokens2)} total)")
    print(f"   Diversity: {diversity2:.3f}")

    n_display = min(10, len(tokens2))
    visualize_similarity_matrix(
        Z2[:n_display, :n_display],
        tokens2[:n_display],
        f"Token Similarity Matrix: Low Diversity Corpus (first {n_display} tokens)\n"
        "(Note: 'right' appears multiple times with different embeddings)",
        "token_semantic_matrix_corpus2.png",
    )

    return {
        "Metric": "Token Semantics",
        "High Paraphrase": diversity1,
        "Low Diversity": diversity2,
    }


def test_dependency_parse():
    """Test and visualize syntactic diversity (dependency parsing)."""
    print("\n" + "="*70)
    print("SYNTACTIC DIVERSITY (Dependency Parse)")
    print("="*70)

    corpus1 = [
        "one massive earth",
        "an enormous globe",
        "the colossal world",
    ]

    corpus2 = [
        "basic human right",
        "you were right",
        "make a right",
    ]

    # Initialize metric with LDP (fast)
    metric = DependencyParse({
        "similarity_type": "ldp",
        "verbose": False,
    })

    # Corpus 1
    print("\n1. High Paraphrase Corpus")
    features1, docs1 = metric.extract_features(corpus1)
    Z1 = metric.calculate_similarities(features1)
    diversity1 = metric(corpus1)
    print(f"   Diversity: {diversity1:.3f}")

    visualize_similarity_matrix(
        Z1,
        [f"Parse {i+1}" for i in range(len(corpus1))],
        "Dependency Parse Similarity Matrix: High Paraphrase Corpus\n"
        "(Similar syntax: determiner + adjective + noun)",
        "dependency_matrix_corpus1.png",
    )

    # Corpus 2
    print("\n2. Low Diversity Corpus")
    features2, docs2 = metric.extract_features(corpus2)
    Z2 = metric.calculate_similarities(features2)
    diversity2 = metric(corpus2)
    print(f"   Diversity: {diversity2:.3f}")

    visualize_similarity_matrix(
        Z2,
        [f"Parse {i+1}" for i in range(len(corpus2))],
        "Dependency Parse Similarity Matrix: Low Diversity Corpus\n"
        "(Different syntactic structures)",
        "dependency_matrix_corpus2.png",
    )

    return {
        "Metric": "Dependency Parse",
        "High Paraphrase": diversity1,
        "Low Diversity": diversity2,
    }


def test_pos_sequence():
    """Test and visualize morphological diversity (POS sequences)."""
    print("\n" + "="*70)
    print("MORPHOLOGICAL DIVERSITY (POS Sequences)")
    print("="*70)

    corpus1 = [
        "one massive earth",
        "an enormous globe",
        "the colossal world",
    ]

    corpus2 = [
        "basic human right",
        "you were right",
        "make a right",
    ]

    # Initialize metric
    metric = PartOfSpeechSequence({
        "verbose": False,
    })

    # Corpus 1
    print("\n1. High Paraphrase Corpus")
    features1, docs1 = metric.extract_features(corpus1)
    Z1 = metric.calculate_similarities(features1)
    diversity1 = metric(corpus1)
    print(f"   POS sequences: {features1}")
    print(f"   Diversity: {diversity1:.3f}")

    visualize_similarity_matrix(
        Z1,
        [f"POS {i+1}" for i in range(len(corpus1))],
        "POS Sequence Similarity Matrix: High Paraphrase Corpus\n"
        "(All have pattern: DET + ADJ + NOUN)",
        "pos_matrix_corpus1.png",
    )

    # Corpus 2
    print("\n2. Low Diversity Corpus")
    features2, docs2 = metric.extract_features(corpus2)
    Z2 = metric.calculate_similarities(features2)
    diversity2 = metric(corpus2)
    print(f"   POS sequences: {features2}")
    print(f"   Diversity: {diversity2:.3f}")

    visualize_similarity_matrix(
        Z2,
        [f"POS {i+1}" for i in range(len(corpus2))],
        "POS Sequence Similarity Matrix: Low Diversity Corpus\n"
        "(Different POS patterns)",
        "pos_matrix_corpus2.png",
    )

    return {
        "Metric": "POS Sequence",
        "High Paraphrase": diversity1,
        "Low Diversity": diversity2,
    }


def create_summary_visualization(all_results):
    """Create summary comparison of all metrics."""
    print("\n" + "="*70)
    print("SUMMARY VISUALIZATION")
    print("="*70)

    # Create DataFrame
    df = pd.DataFrame(all_results)
    print("\nDiversity Scores:")
    print(df.to_string(index=False))

    # Create comparison plot
    visualize_diversity_comparison(
        df,
        "Diversity Comparison Across Metrics and Corpora",
        "diversity_comparison.png",
    )

    # Create normalized comparison
    df_norm = df.copy()
    df_norm["High Paraphrase"] = df["High Paraphrase"] / df["High Paraphrase"].max()
    df_norm["Low Diversity"] = df["Low Diversity"] / df["Low Diversity"].max()

    visualize_diversity_comparison(
        df_norm,
        "Normalized Diversity Comparison (scaled to max=1.0)",
        "diversity_comparison_normalized.png",
    )


def main():
    """Run all tests and create visualizations."""
    print("="*70)
    print("LINGUISTIC DIVERSITY - VISUALIZATION AND TESTING")
    print("="*70)
    print(f"\nOutput directory: {output_dir.absolute()}")

    all_results = []

    # Test each metric
    all_results.append(test_document_semantics())
    all_results.append(test_token_semantics())
    all_results.append(test_dependency_parse())
    all_results.append(test_pos_sequence())

    # Create summary
    create_summary_visualization(all_results)

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")

    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("""
1. SIMILARITY MATRICES show how similar each pair of items is:
   - Bright colors (yellow/red) = high similarity
   - Dark colors (dark red) = low similarity
   - Diagonal is always 1.0 (perfect self-similarity)

2. HIGH PARAPHRASE CORPUS ("massive earth", "enormous globe", etc.):
   - High semantic similarity (similar meanings)
   - High syntactic similarity (same structure: DET+ADJ+NOUN)
   - Low diversity overall

3. LOW DIVERSITY CORPUS ("human right", "were right", "a right"):
   - Lower semantic similarity (different meanings of 'right')
   - Different syntactic structures
   - Higher diversity despite shared word

4. DIVERSITY SCORES represent the "effective number" of distinct items:
   - Score of 2.5 ≈ equivalent to having 2.5 completely distinct items
   - Lower score = more similar/redundant items
   - Higher score = more diverse/distinct items
    """)


if __name__ == "__main__":
    main()
