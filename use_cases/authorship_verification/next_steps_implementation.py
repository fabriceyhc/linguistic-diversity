"""Next Steps Implementation: Enhanced Lexical Metrics + Inter-text Similarity.

This script implements:
1. IMMEDIATE: Enhanced lexical diversity metrics (Uber Index, Summer's S, hapax ratios)
2. IMMEDIATE: Lexical-only model (dropping semantic/syntactic)
3. STRATEGIC: Inter-text similarity paradigm shift
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Add the src directory to the path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "src"))

from linguistic_diversity.diversities import DocumentSemantics, DependencyParse

# Set style
sns.set_style("whitegrid")


# ============================================================================
# ENHANCED LEXICAL METRICS
# ============================================================================

def tokenize(text):
    """Simple tokenization."""
    import re
    # Remove punctuation and split
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def compute_ttr(tokens, window_size=None):
    """Compute Type-Token Ratio."""
    if len(tokens) == 0:
        return 0.0

    if window_size is None:
        # Simple TTR
        return len(set(tokens)) / len(tokens)
    else:
        # Moving-average TTR (MATTR)
        if len(tokens) < window_size:
            return len(set(tokens)) / len(tokens)

        ttrs = []
        for i in range(len(tokens) - window_size + 1):
            window = tokens[i:i + window_size]
            ttr = len(set(window)) / len(window)
            ttrs.append(ttr)
        return np.mean(ttrs)


def compute_uber_index(tokens):
    """Compute Uber Index (more stable than TTR).

    Uber = (log(N)^2) / (log(N) - log(V))
    where N = total tokens, V = vocabulary size
    """
    if len(tokens) == 0:
        return 0.0

    N = len(tokens)
    V = len(set(tokens))

    if V == N:  # All unique
        return 1.0

    log_N = np.log(N)
    log_V = np.log(V)

    if log_N == log_V:  # Avoid division by zero
        return 1.0

    uber = (log_N ** 2) / (log_N - log_V)
    return uber


def compute_summers_s(tokens):
    """Compute Summer's S Index.

    S = log(log(V)) / log(log(N))
    where N = total tokens, V = vocabulary size
    """
    if len(tokens) < 4:  # Need at least 4 tokens for log(log(N))
        return 0.0

    N = len(tokens)
    V = len(set(tokens))

    if V < 2:  # Need at least 2 unique tokens
        return 0.0

    log_log_N = np.log(np.log(N))
    log_log_V = np.log(np.log(V))

    if log_log_N == 0:
        return 0.0

    s = log_log_V / log_log_N
    return s


def compute_hapax_ratio(tokens):
    """Compute ratio of hapax legomena (words appearing once)."""
    if len(tokens) == 0:
        return 0.0

    freq = Counter(tokens)
    hapax_count = sum(1 for count in freq.values() if count == 1)
    return hapax_count / len(tokens)


def compute_dis_legomena_ratio(tokens):
    """Compute ratio of dis legomena (words appearing twice)."""
    if len(tokens) == 0:
        return 0.0

    freq = Counter(tokens)
    dis_count = sum(1 for count in freq.values() if count == 2)
    return dis_count / len(tokens)


def compute_word_length_stats(tokens):
    """Compute word length statistics."""
    if len(tokens) == 0:
        return {'mean': 0.0, 'std': 0.0, 'max': 0}

    lengths = [len(token) for token in tokens]
    return {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'max': np.max(lengths)
    }


def compute_enhanced_lexical_profile(text):
    """Compute comprehensive lexical profile."""
    tokens = tokenize(text)

    if len(tokens) < 10:  # Too short
        return None

    profile = {
        'mattr_25': compute_ttr(tokens, window_size=25),
        'mattr_50': compute_ttr(tokens, window_size=50),
        'mattr_100': compute_ttr(tokens, window_size=100),
        'uber_index': compute_uber_index(tokens),
        'summers_s': compute_summers_s(tokens),
        'hapax_ratio': compute_hapax_ratio(tokens),
        'dis_legomena_ratio': compute_dis_legomena_ratio(tokens),
        'word_length_mean': compute_word_length_stats(tokens)['mean'],
        'word_length_std': compute_word_length_stats(tokens)['std'],
    }

    return profile


def compute_lexical_delta(profile1, profile2):
    """Compute delta between two lexical profiles."""
    if profile1 is None or profile2 is None:
        return None

    deltas = {}
    for key in profile1.keys():
        deltas[key] = abs(profile1[key] - profile2[key])

    # Compute composite distance (equal weights for now)
    total = sum(deltas.values())
    deltas['total'] = total

    return deltas


# ============================================================================
# INTER-TEXT SIMILARITY (PARADIGM SHIFT)
# ============================================================================

def compute_semantic_similarity(text1, text2, semantic_metric):
    """Compute semantic similarity between two texts directly.

    Instead of comparing diversity within each text, compare the texts directly.
    """
    try:
        # Encode both texts
        embeddings1 = semantic_metric.model.encode([text1], convert_to_numpy=True)
        embeddings2 = semantic_metric.model.encode([text2], convert_to_numpy=True)

        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(embeddings1, embeddings2)[0, 0]

        # Convert similarity to distance
        distance = 1 - similarity

        return distance
    except Exception as e:
        print(f"Error computing semantic similarity: {e}")
        return None


def compute_syntactic_similarity(text1, text2, syntactic_metric):
    """Compute syntactic similarity between two texts directly.

    Compare the parse tree structures of the two texts.
    """
    try:
        # For simplicity, we'll compare the diversity of the combined text
        # vs the average of individual diversities
        # Lower difference = more similar syntactic style

        # This is still indirect, but better than comparing internal diversity
        combined = [text1, text2]

        # Extract features for both texts
        features, _ = syntactic_metric.extract_features(combined)

        if len(features) < 2:
            return None

        # Compute similarity between the two feature representations
        # For LDP embeddings, use cosine similarity
        if isinstance(features, np.ndarray) and features.ndim == 2:
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity([features[0]], [features[1]])[0, 0]
            distance = 1 - sim
            return distance
        else:
            # For graph-based features, return None for now
            return None

    except Exception as e:
        print(f"Error computing syntactic similarity: {e}")
        return None


def compute_lexical_similarity(text1, text2):
    """Compute lexical similarity between two texts directly.

    Compare vocabulary overlap and usage patterns.
    """
    try:
        tokens1 = set(tokenize(text1))
        tokens2 = set(tokenize(text2))

        if len(tokens1) == 0 or len(tokens2) == 0:
            return None

        # Jaccard distance
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        if union == 0:
            return None

        jaccard_similarity = intersection / union
        jaccard_distance = 1 - jaccard_similarity

        return jaccard_distance

    except Exception as e:
        print(f"Error computing lexical similarity: {e}")
        return None


# ============================================================================
# MAIN EVALUATION
# ============================================================================

def load_baseline_results():
    """Load baseline results for comparison."""
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / 'output/verification/fingerprint_scores.csv')
    return df


def evaluate_enhanced_lexical(output_dir):
    """Evaluate enhanced lexical metrics only."""
    print("\n" + "="*80)
    print("APPROACH 1: Enhanced Lexical Metrics")
    print("="*80)

    script_dir = Path(__file__).parent

    # Check if we already computed this
    cache_file = output_dir / 'enhanced_lexical_scores.csv'

    if cache_file.exists():
        print(f"\nLoading cached results from {cache_file}")
        df = pd.read_csv(cache_file)
    else:
        print("\nComputing enhanced lexical profiles (this will take a while)...")

        # Load the original dataset
        from datasets import load_dataset
        dataset = load_dataset("swan07/authorship-verification", split="validation")

        results = []

        for i, example in enumerate(tqdm(dataset, desc="Processing pairs")):
            if i >= 1000:  # Limit for faster testing
                break

            text1 = example['text1']
            text2 = example['text2']
            label = example['same']

            # Compute profiles
            profile1 = compute_enhanced_lexical_profile(text1)
            profile2 = compute_enhanced_lexical_profile(text2)

            if profile1 is None or profile2 is None:
                continue

            # Compute deltas
            deltas = compute_lexical_delta(profile1, profile2)

            if deltas is not None:
                result = {'id': i, 'label': label}
                result.update(deltas)
                results.append(result)

        df = pd.DataFrame(results)
        df.to_csv(cache_file, index=False)
        print(f"Saved results to {cache_file}")

    # Evaluate
    print(f"\nProcessed {len(df)} pairs")

    # Individual metrics
    metrics = [col for col in df.columns if col not in ['id', 'label', 'total']]

    print("\nIndividual metric performance:")
    individual_aucs = {}
    for metric in metrics:
        y_true = df['label'].values
        y_score = -df[metric].values
        auc = roc_auc_score(y_true, y_score)
        individual_aucs[metric] = auc
        print(f"  {metric:25s}: {auc:.4f}")

    # Total composite
    y_true = df['label'].values
    y_score = -df['total'].values
    total_auc = roc_auc_score(y_true, y_score)
    print(f"\n  {'COMPOSITE (all metrics)':25s}: {total_auc:.4f}")

    # Learn optimal weights
    X = df[metrics].values
    y = df['label'].values
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict_proba(X)[:, 1]
    learned_auc = roc_auc_score(y, y_pred)

    print(f"  {'LEARNED WEIGHTS':25s}: {learned_auc:.4f}")

    # Show learned weights
    print("\nLearned weights:")
    weights = model.coef_[0]
    abs_weights = np.abs(weights)
    importance = abs_weights / abs_weights.sum()
    for metric, weight, imp in sorted(zip(metrics, weights, importance), key=lambda x: x[2], reverse=True):
        print(f"  {metric:25s}: {weight:+.4f} ({imp:.1%} importance)")

    return {
        'individual': individual_aucs,
        'composite': total_auc,
        'learned': learned_auc,
        'best_individual': max(individual_aucs.values())
    }


def evaluate_inter_text_similarity(output_dir):
    """Evaluate inter-text similarity approach."""
    print("\n" + "="*80)
    print("APPROACH 2: Inter-Text Similarity (Paradigm Shift)")
    print("="*80)

    cache_file = output_dir / 'inter_text_similarity_scores.csv'

    if cache_file.exists():
        print(f"\nLoading cached results from {cache_file}")
        df = pd.read_csv(cache_file)
    else:
        print("\nComputing inter-text similarities (this will take a while)...")

        # Initialize metrics
        print("Initializing metrics...")
        semantic_metric = DocumentSemantics({
            'model_name': 'all-MiniLM-L6-v2',
            'verbose': False
        })

        syntactic_metric = DependencyParse({
            'similarity_type': 'ldp',
            'verbose': False
        })

        # Load dataset
        from datasets import load_dataset
        dataset = load_dataset("swan07/authorship-verification", split="validation")

        results = []

        for i, example in enumerate(tqdm(dataset, desc="Processing pairs")):
            if i >= 1000:  # Limit for faster testing
                break

            text1 = example['text1']
            text2 = example['text2']
            label = example['same']

            # Compute similarities
            sem_sim = compute_semantic_similarity(text1, text2, semantic_metric)
            syn_sim = compute_syntactic_similarity(text1, text2, syntactic_metric)
            lex_sim = compute_lexical_similarity(text1, text2)

            if sem_sim is not None and lex_sim is not None:
                total_dist = sem_sim + (syn_sim if syn_sim is not None else 0) + lex_sim

                results.append({
                    'id': i,
                    'label': label,
                    'semantic_dist': sem_sim,
                    'syntactic_dist': syn_sim if syn_sim is not None else np.nan,
                    'lexical_dist': lex_sim,
                    'total_dist': total_dist
                })

        df = pd.DataFrame(results)
        df.to_csv(cache_file, index=False)
        print(f"Saved results to {cache_file}")

    # Evaluate
    print(f"\nProcessed {len(df)} pairs")

    # Individual dimensions
    print("\nIndividual dimension performance:")
    individual_aucs = {}

    for dim in ['semantic_dist', 'syntactic_dist', 'lexical_dist']:
        if df[dim].notna().sum() > 0:
            df_valid = df[df[dim].notna()]
            y_true = df_valid['label'].values
            y_score = -df_valid[dim].values  # Negative because lower distance = same author
            auc = roc_auc_score(y_true, y_score)
            individual_aucs[dim] = auc
            print(f"  {dim:25s}: {auc:.4f}")

    # Total composite
    y_true = df['label'].values
    y_score = -df['total_dist'].values
    total_auc = roc_auc_score(y_true, y_score)
    print(f"\n  {'COMPOSITE':25s}: {total_auc:.4f}")

    # Learn optimal weights (only on valid data)
    df_valid = df.dropna()
    X = df_valid[['semantic_dist', 'lexical_dist']].values  # Drop syntactic if mostly NaN
    y = df_valid['label'].values
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict_proba(X)[:, 1]
    learned_auc = roc_auc_score(y, y_pred)

    print(f"  {'LEARNED WEIGHTS':25s}: {learned_auc:.4f}")

    return {
        'individual': individual_aucs,
        'composite': total_auc,
        'learned': learned_auc,
        'best_individual': max(individual_aucs.values())
    }


def create_comparison_plot(baseline_auc, enhanced_results, similarity_results, output_dir):
    """Create comprehensive comparison plot."""
    print("\n" + "="*80)
    print("Generating Comparison Visualizations")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Overall approach comparison
    ax = axes[0, 0]
    approaches = [
        'Baseline\n(Diversity Delta)',
        'Enhanced Lexical\n(9 metrics)',
        'Enhanced Lexical\n(Learned Weights)',
        'Inter-Text Similarity\n(Composite)',
        'Inter-Text Similarity\n(Learned Weights)'
    ]
    aucs = [
        baseline_auc,
        enhanced_results['composite'],
        enhanced_results['learned'],
        similarity_results['composite'],
        similarity_results['learned']
    ]

    colors = ['gray', 'skyblue', 'dodgerblue', 'lightcoral', 'red']
    bars = ax.bar(range(len(approaches)), aucs, color=colors)
    ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.3, label='Random (0.50)')
    ax.axhline(y=0.65, color='green', linestyle='--', alpha=0.5, label='Viable (0.65)')
    ax.set_xticks(range(len(approaches)))
    ax.set_xticklabels(approaches, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Overall Approach Comparison', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim([0.45, max(aucs) + 0.05])

    # Add value labels
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.4f}',
                ha='center', va='bottom', fontsize=9)

    # 2. Enhanced lexical metrics breakdown
    ax = axes[0, 1]
    metrics = list(enhanced_results['individual'].keys())
    metric_aucs = list(enhanced_results['individual'].values())

    # Sort by AUC
    sorted_pairs = sorted(zip(metrics, metric_aucs), key=lambda x: x[1], reverse=True)
    metrics_sorted, aucs_sorted = zip(*sorted_pairs)

    y_pos = np.arange(len(metrics_sorted))
    bars = ax.barh(y_pos, aucs_sorted, color='skyblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics_sorted, fontsize=8)
    ax.set_xlabel('ROC-AUC')
    ax.set_title('Enhanced Lexical Metrics (Individual)', fontsize=12, fontweight='bold')
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=baseline_auc, color='red', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_auc:.3f})')
    ax.legend()

    # Add value labels
    for bar, auc in zip(bars, aucs_sorted):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f'{auc:.4f}',
                ha='left', va='center', fontsize=8)

    # 3. Paradigm comparison
    ax = axes[1, 0]
    paradigms = ['Diversity Delta\n(Baseline)', 'Enhanced Lexical\n(Best)', 'Inter-Text Similarity\n(Best)']
    paradigm_aucs = [
        baseline_auc,
        enhanced_results['learned'],
        similarity_results['learned']
    ]
    improvements = [
        0,
        ((enhanced_results['learned'] - baseline_auc) / baseline_auc) * 100,
        ((similarity_results['learned'] - baseline_auc) / baseline_auc) * 100
    ]

    bars = ax.bar(paradigms, paradigm_aucs, color=['gray', 'dodgerblue', 'red'])
    ax.axhline(y=0.65, color='green', linestyle='--', alpha=0.5, label='Viable threshold')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Paradigm Comparison (Best Performing Variants)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.set_ylim([0.45, max(paradigm_aucs) + 0.05])

    # Add value labels with improvement
    for bar, auc, imp in zip(bars, paradigm_aucs, improvements):
        height = bar.get_height()
        label = f'{auc:.4f}\n({imp:+.1f}%)'
        ax.text(bar.get_x() + bar.get_width()/2., height,
                label,
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Improvement summary table
    ax = axes[1, 1]
    ax.axis('off')

    summary_data = [
        ['Approach', 'ROC-AUC', 'vs Baseline', 'Status'],
        ['─' * 20, '─' * 10, '─' * 12, '─' * 15],
        ['Baseline', f'{baseline_auc:.4f}', '—', 'Weak'],
        ['Enhanced Lexical (Best)', f'{enhanced_results["learned"]:.4f}',
         f'{((enhanced_results["learned"]-baseline_auc)/baseline_auc)*100:+.1f}%',
         'Better' if enhanced_results['learned'] > baseline_auc else 'Worse'],
        ['Inter-Text Sim (Best)', f'{similarity_results["learned"]:.4f}',
         f'{((similarity_results["learned"]-baseline_auc)/baseline_auc)*100:+.1f}%',
         '✓ Viable' if similarity_results['learned'] >= 0.65 else 'Improved'],
        ['', '', '', ''],
        ['Target (Viable)', '0.6500', '—', 'Goal'],
    ]

    table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.35, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plot_file = output_dir / 'next_steps_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison plot to {plot_file}")


def main():
    """Main execution."""
    print("="*80)
    print("NEXT STEPS IMPLEMENTATION")
    print("Testing Enhanced Lexical Metrics + Inter-Text Similarity")
    print("="*80)

    script_dir = Path(__file__).parent
    output_dir = script_dir / 'output/verification'

    # Load baseline for comparison
    baseline_df = load_baseline_results()
    y_true = baseline_df['label'].values
    y_score = -baseline_df['total_dist'].values
    baseline_auc = roc_auc_score(y_true, y_score)
    print(f"\nBaseline ROC-AUC: {baseline_auc:.4f}")

    # Test enhanced lexical metrics
    enhanced_results = evaluate_enhanced_lexical(output_dir)

    # Test inter-text similarity
    similarity_results = evaluate_inter_text_similarity(output_dir)

    # Generate comparison
    create_comparison_plot(baseline_auc, enhanced_results, similarity_results, output_dir)

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\nBaseline (Diversity Delta):          {baseline_auc:.4f}")
    print(f"Enhanced Lexical (Learned Weights):  {enhanced_results['learned']:.4f} ({((enhanced_results['learned']-baseline_auc)/baseline_auc)*100:+.1f}%)")
    print(f"Inter-Text Similarity (Learned):     {similarity_results['learned']:.4f} ({((similarity_results['learned']-baseline_auc)/baseline_auc)*100:+.1f}%)")

    best_auc = max(baseline_auc, enhanced_results['learned'], similarity_results['learned'])

    if best_auc >= 0.70:
        print(f"\n✓✓✓ EXCELLENT: Achieved {best_auc:.4f} (viable for deployment)")
    elif best_auc >= 0.65:
        print(f"\n✓✓ SUCCESS: Achieved {best_auc:.4f} (meets viability threshold!)")
    elif best_auc >= 0.60:
        print(f"\n✓ PROGRESS: Achieved {best_auc:.4f} (significant improvement)")
    else:
        print(f"\n⚠ LIMITED: Best is {best_auc:.4f} (still needs work)")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
