"""Zero-Shot Authorship Verification using Linguistic Diversity Delta.

This script implements authorship verification by computing the difference in
diversity scores between paired texts. The hypothesis is that texts by the same
author should have similar diversity profiles (low delta), while texts by
different authors should have divergent profiles (high delta).
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path to import linguistic_diversity locally
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from linguistic_diversity.diversities import DocumentSemantics, DependencyParse


def split_into_sentences(text):
    """Split text into sentences using simple heuristics."""
    import re
    # Simple sentence splitter - can be improved with spaCy
    sentences = re.split(r'[.!?]+\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def compute_mattr(text, window_size=50):
    """Compute Moving-Average Type-Token Ratio (MATTR).

    Args:
        text: Input text string
        window_size: Window size for moving average

    Returns:
        MATTR score (float)
    """
    # Tokenize
    tokens = text.lower().split()

    if len(tokens) < window_size:
        # For short texts, use simple TTR
        if len(tokens) == 0:
            return 0.0
        return len(set(tokens)) / len(tokens)

    # Compute TTR for each window
    ttrs = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        ttr = len(set(window)) / len(window)
        ttrs.append(ttr)

    # Return average
    return np.mean(ttrs)


def compute_text_profile(text, semantic_metric, syntactic_metric, min_sentences=3):
    """Compute linguistic diversity profile for a single text.

    Args:
        text: Input text string
        semantic_metric: DocumentSemantics metric instance
        syntactic_metric: DependencyParse metric instance
        min_sentences: Minimum number of sentences required

    Returns:
        Dictionary with scores: {semantic, syntactic, lexical}
        Returns None if text is too short
    """
    # Split into sentences
    sentences = split_into_sentences(text)

    # Filter by minimum length
    if len(sentences) < min_sentences:
        return None

    # Compute semantic diversity (internal diversity across sentences)
    try:
        semantic_score = semantic_metric(sentences)
    except Exception as e:
        print(f"Error computing semantic score: {e}")
        semantic_score = 0.0

    # Compute syntactic diversity (internal diversity across sentence structures)
    try:
        syntactic_score = syntactic_metric(sentences)
    except Exception as e:
        print(f"Error computing syntactic score: {e}")
        syntactic_score = 0.0

    # Compute lexical diversity (MATTR)
    try:
        lexical_score = compute_mattr(text)
    except Exception as e:
        print(f"Error computing lexical score: {e}")
        lexical_score = 0.0

    return {
        'semantic': semantic_score,
        'syntactic': syntactic_score,
        'lexical': lexical_score
    }


def compute_fingerprint_delta(profile1, profile2):
    """Compute the delta (difference) between two diversity profiles.

    Args:
        profile1: Diversity profile for text 1
        profile2: Diversity profile for text 2

    Returns:
        Dictionary with deltas: {delta_sem, delta_syn, delta_lex, total_dist}
    """
    if profile1 is None or profile2 is None:
        return None

    delta_sem = abs(profile1['semantic'] - profile2['semantic'])
    delta_syn = abs(profile1['syntactic'] - profile2['syntactic'])
    delta_lex = abs(profile1['lexical'] - profile2['lexical'])

    total_dist = delta_sem + delta_syn + delta_lex

    return {
        'delta_sem': delta_sem,
        'delta_syn': delta_syn,
        'delta_lex': delta_lex,
        'total_dist': total_dist
    }


def main():
    """Main execution function."""

    print("=" * 80)
    print("Zero-Shot Authorship Verification using Linguistic Diversity Delta")
    print("=" * 80)

    # Get the script directory for relative paths
    script_dir = Path(__file__).parent

    # Step 1: Download dataset
    print("\n[1/6] Loading dataset from HuggingFace...")
    try:
        # Load the validation split
        dataset = load_dataset("swan07/authorship-verification", split="validation")
        print(f"   ✓ Loaded {len(dataset)} pairs from validation split")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return

    # Step 2: Initialize metrics
    print("\n[2/6] Initializing diversity metrics...")

    # Semantic metric (DocumentSemantics)
    print("   - Initializing DocumentSemantics...")
    semantic_metric = DocumentSemantics({
        'model_name': 'all-MiniLM-L6-v2',  # Faster model
        'verbose': False
    })

    # Syntactic metric (DependencyParse)
    print("   - Initializing DependencyParse...")
    syntactic_metric = DependencyParse({
        'similarity_type': 'ldp',  # Fast method
        'verbose': False
    })

    print("   ✓ Metrics initialized")

    # Step 3: Filter dataset
    print("\n[3/6] Filtering dataset (minimum 3 sentences per text)...")
    min_sentences = 3
    valid_pairs = []

    for i, example in enumerate(dataset):
        text1 = example['text1']
        text2 = example['text2']
        label = example['same']

        # Check sentence count
        sentences1 = split_into_sentences(text1)
        sentences2 = split_into_sentences(text2)

        if len(sentences1) >= min_sentences and len(sentences2) >= min_sentences:
            valid_pairs.append({
                'id': i,
                'text1': text1,
                'text2': text2,
                'label': label
            })

    print(f"   ✓ Filtered to {len(valid_pairs)} valid pairs")

    # Optional: Limit to a subset for faster testing
    # valid_pairs = valid_pairs[:500]
    # print(f"   ! Using subset of {len(valid_pairs)} pairs for testing")

    # Step 4: Compute fingerprint deltas
    print("\n[4/6] Computing fingerprint deltas...")
    results = []

    for pair in tqdm(valid_pairs, desc="   Processing pairs"):
        # Compute profiles
        profile1 = compute_text_profile(
            pair['text1'],
            semantic_metric,
            syntactic_metric,
            min_sentences
        )

        profile2 = compute_text_profile(
            pair['text2'],
            semantic_metric,
            syntactic_metric,
            min_sentences
        )

        # Compute delta
        delta = compute_fingerprint_delta(profile1, profile2)

        if delta is not None:
            results.append({
                'id': pair['id'],
                'label': pair['label'],
                'delta_sem': delta['delta_sem'],
                'delta_syn': delta['delta_syn'],
                'delta_lex': delta['delta_lex'],
                'total_dist': delta['total_dist']
            })

    print(f"   ✓ Computed {len(results)} fingerprint deltas")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Save results
    output_file = script_dir / 'output/verification/fingerprint_scores.csv'
    df.to_csv(output_file, index=False)
    print(f"   ✓ Saved results to {output_file}")

    # Step 5: Generate visualizations
    print("\n[5/6] Generating visualizations...")

    # Create boxplot
    plt.figure(figsize=(10, 6))

    # Prepare data for boxplot
    same_author = df[df['label'] == 1]['total_dist']
    diff_author = df[df['label'] == 0]['total_dist']

    data_to_plot = [same_author, diff_author]
    labels = ['Same Author', 'Different Author']

    # Create boxplot
    bp = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)

    # Customize colors
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')

    plt.ylabel('Total Fingerprint Distance')
    plt.title('Linguistic Diversity Delta: Same vs Different Authors')
    plt.grid(axis='y', alpha=0.3)

    # Add statistics
    plt.text(1, plt.ylim()[1] * 0.95,
             f'Mean: {same_author.mean():.3f}\nMedian: {same_author.median():.3f}',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.text(2, plt.ylim()[1] * 0.95,
             f'Mean: {diff_author.mean():.3f}\nMedian: {diff_author.median():.3f}',
             ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plot_file = script_dir / 'output/verification/separation_plot.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved boxplot to {plot_file}")

    # Step 6: Calculate ROC-AUC
    print("\n[6/6] Calculating ROC-AUC score...")

    # For ROC-AUC, we need to invert the distance
    # (lower distance = higher probability of same author)
    # We'll use negative distance as the score
    y_true = df['label'].values
    y_score = -df['total_dist'].values  # Negative because lower distance = same author

    # Calculate AUC
    auc = roc_auc_score(y_true, y_score)
    print(f"   ✓ ROC-AUC Score: {auc:.4f}")

    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Generate report
    report = f"""
Authorship Verification Evaluation Report
==========================================

Dataset Statistics:
- Total pairs processed: {len(df)}
- Same author pairs: {(df['label'] == 1).sum()}
- Different author pairs: {(df['label'] == 0).sum()}

Fingerprint Distance Statistics:
Same Author Pairs:
- Mean: {same_author.mean():.4f}
- Median: {same_author.median():.4f}
- Std: {same_author.std():.4f}

Different Author Pairs:
- Mean: {diff_author.mean():.4f}
- Median: {diff_author.median():.4f}
- Std: {diff_author.std():.4f}

Performance Metrics:
- ROC-AUC Score: {auc:.4f}
- Benchmark (random guessing): 0.5000

Interpretation:
{
    "EXCELLENT (>0.80): Highly reliable forensic signal" if auc > 0.80 else
    "GOOD (0.70-0.80): Viable forensic metric" if auc > 0.70 else
    "MODERATE (0.65-0.70): Shows promise, needs refinement" if auc > 0.65 else
    "WEAK (0.50-0.65): Limited discriminative power" if auc > 0.50 else
    "FAILED (<0.50): No better than random guessing"
}

Conclusion:
{"The linguistic diversity delta shows strong potential for authorship verification!" if auc > 0.65 else
 "The metric shows some signal but may need additional features or refinement."}
"""

    report_file = script_dir / 'output/verification/report.txt'
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"   ✓ Saved report to {report_file}")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nROC-AUC Score: {auc:.4f}")
    print(f"Check {script_dir / 'output/verification/'} for detailed results and visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
