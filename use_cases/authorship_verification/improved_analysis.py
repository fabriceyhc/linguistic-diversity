"""Improved Authorship Verification Analysis with Quick Wins.

This script implements several improvements over the baseline:
1. Analyzes individual dimension contributions
2. Tests weighted combinations
3. Implements relative distance normalization
4. Filters outliers
5. Compares all approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_results():
    """Load the baseline results."""
    script_dir = Path(__file__).parent
    df = pd.read_csv(script_dir / 'output/verification/fingerprint_scores.csv')
    print(f"Loaded {len(df)} pairs from baseline results")
    return df


def analyze_individual_dimensions(df):
    """Analyze which dimension contributes most signal."""
    print("\n" + "="*80)
    print("ANALYSIS 1: Individual Dimension Contributions")
    print("="*80)

    dimensions = ['delta_sem', 'delta_syn', 'delta_lex']
    results = {}

    for dim in dimensions:
        # Calculate AUC for this dimension alone
        y_true = df['label'].values
        y_score = -df[dim].values  # Negative because lower = same author

        auc = roc_auc_score(y_true, y_score)
        results[dim] = auc

        # Calculate statistics
        same_author = df[df['label'] == 1][dim]
        diff_author = df[df['label'] == 0][dim]

        print(f"\n{dim.upper()}:")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  Same author   - Mean: {same_author.mean():.4f}, Median: {same_author.median():.4f}")
        print(f"  Diff author   - Mean: {diff_author.mean():.4f}, Median: {diff_author.median():.4f}")
        print(f"  Median ratio: {diff_author.median() / same_author.median():.2f}x")
        print(f"  Effect size (Cohen's d): {cohens_d(same_author, diff_author):.4f}")

    # Find best dimension
    best_dim = max(results, key=results.get)
    print(f"\n{'='*80}")
    print(f"BEST INDIVIDUAL DIMENSION: {best_dim} (AUC = {results[best_dim]:.4f})")
    print(f"{'='*80}")

    return results, best_dim


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std


def test_weighted_combinations(df, dim_results):
    """Test different weighting schemes."""
    print("\n" + "="*80)
    print("ANALYSIS 2: Weighted Combinations")
    print("="*80)

    # Prepare data
    y_true = df['label'].values

    # Test different weighting schemes
    schemes = {
        'baseline_equal': [1, 1, 1],
        'auc_weighted': [
            dim_results['delta_sem'],
            dim_results['delta_syn'],
            dim_results['delta_lex']
        ],
        'best_only': [
            1 if 'delta_sem' == max(dim_results, key=dim_results.get) else 0,
            1 if 'delta_syn' == max(dim_results, key=dim_results.get) else 0,
            1 if 'delta_lex' == max(dim_results, key=dim_results.get) else 0
        ],
        'sem_heavy': [3, 1, 1],
        'syn_heavy': [1, 3, 1],
        'lex_heavy': [1, 1, 3],
        'sem_lex': [2, 0, 2],  # Drop syntactic
    }

    results = {}
    for name, weights in schemes.items():
        # Normalize weights
        w = np.array(weights) / np.sum(weights)

        # Compute weighted distance
        weighted_dist = (
            w[0] * df['delta_sem'] +
            w[1] * df['delta_syn'] +
            w[2] * df['delta_lex']
        )

        y_score = -weighted_dist.values
        auc = roc_auc_score(y_true, y_score)
        results[name] = {'auc': auc, 'weights': w}

        print(f"\n{name}:")
        print(f"  Weights: sem={w[0]:.3f}, syn={w[1]:.3f}, lex={w[2]:.3f}")
        print(f"  ROC-AUC: {auc:.4f}")

    # Find best scheme
    best_scheme = max(results, key=lambda x: results[x]['auc'])
    print(f"\n{'='*80}")
    print(f"BEST WEIGHTING SCHEME: {best_scheme} (AUC = {results[best_scheme]['auc']:.4f})")
    print(f"{'='*80}")

    return results, best_scheme


def learn_optimal_weights(df):
    """Learn optimal weights using logistic regression."""
    print("\n" + "="*80)
    print("ANALYSIS 3: Learned Optimal Weights (Logistic Regression)")
    print("="*80)

    # Prepare features and labels
    X = df[['delta_sem', 'delta_syn', 'delta_lex']].values
    y = df['label'].values

    # Train logistic regression (which learns optimal weights)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)

    # Get predictions
    y_score = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_score)

    # Get weights (coefficients)
    weights = model.coef_[0]

    print(f"\nLearned weights:")
    print(f"  Semantic:  {weights[0]:+.4f}")
    print(f"  Syntactic: {weights[1]:+.4f}")
    print(f"  Lexical:   {weights[2]:+.4f}")
    print(f"\nROC-AUC: {auc:.4f}")

    # Interpret weights
    abs_weights = np.abs(weights)
    importance = abs_weights / abs_weights.sum()
    print(f"\nRelative importance:")
    print(f"  Semantic:  {importance[0]:.1%}")
    print(f"  Syntactic: {importance[1]:.1%}")
    print(f"  Lexical:   {importance[2]:.1%}")

    return auc, weights


def apply_relative_normalization(df):
    """Apply relative distance normalization."""
    print("\n" + "="*80)
    print("ANALYSIS 4: Relative Distance Normalization")
    print("="*80)

    # For relative normalization, we need the original scores
    # Since we only have deltas, we'll approximate using:
    # relative_delta = delta / mean(score1, score2)
    # But we don't have score1 and score2 separately

    # Alternative: normalize by the sum
    # relative_delta = delta / (score1 + score2) ≈ delta / (2 * mean)

    # Since we can't reconstruct original scores, let's try:
    # Normalize each dimension by its own statistics

    df_norm = df.copy()

    for dim in ['delta_sem', 'delta_syn', 'delta_lex']:
        # Z-score normalization
        mean = df[dim].mean()
        std = df[dim].std()
        df_norm[dim + '_norm'] = (df[dim] - mean) / std

    # Compute normalized total distance
    df_norm['total_dist_norm'] = (
        df_norm['delta_sem_norm'] +
        df_norm['delta_syn_norm'] +
        df_norm['delta_lex_norm']
    )

    y_true = df_norm['label'].values
    y_score = -df_norm['total_dist_norm'].values
    auc = roc_auc_score(y_true, y_score)

    print(f"\nZ-score normalized ROC-AUC: {auc:.4f}")

    # Try min-max normalization
    for dim in ['delta_sem', 'delta_syn', 'delta_lex']:
        min_val = df[dim].min()
        max_val = df[dim].max()
        df_norm[dim + '_minmax'] = (df[dim] - min_val) / (max_val - min_val)

    df_norm['total_dist_minmax'] = (
        df_norm['delta_sem_minmax'] +
        df_norm['delta_syn_minmax'] +
        df_norm['delta_lex_minmax']
    )

    y_score_minmax = -df_norm['total_dist_minmax'].values
    auc_minmax = roc_auc_score(y_true, y_score_minmax)

    print(f"Min-max normalized ROC-AUC: {auc_minmax:.4f}")

    return df_norm, max(auc, auc_minmax)


def filter_outliers(df, n_std=3):
    """Filter outliers beyond n standard deviations."""
    print("\n" + "="*80)
    print(f"ANALYSIS 5: Outlier Filtering (>{n_std} std dev)")
    print("="*80)

    original_count = len(df)

    # Filter based on total_dist
    mean = df['total_dist'].mean()
    std = df['total_dist'].std()
    threshold = mean + n_std * std

    df_filtered = df[df['total_dist'] <= threshold].copy()
    removed_count = original_count - len(df_filtered)

    print(f"\nOriginal pairs: {original_count}")
    print(f"Removed outliers: {removed_count} ({100*removed_count/original_count:.1f}%)")
    print(f"Remaining pairs: {len(df_filtered)}")

    # Recompute AUC on filtered data
    y_true = df_filtered['label'].values
    y_score = -df_filtered['total_dist'].values
    auc = roc_auc_score(y_true, y_score)

    print(f"\nFiltered ROC-AUC: {auc:.4f}")

    # Show statistics
    same_author = df_filtered[df_filtered['label'] == 1]['total_dist']
    diff_author = df_filtered[df_filtered['label'] == 0]['total_dist']

    print(f"\nFiltered statistics:")
    print(f"  Same author   - Mean: {same_author.mean():.4f}, Median: {same_author.median():.4f}")
    print(f"  Diff author   - Mean: {diff_author.mean():.4f}, Median: {diff_author.median():.4f}")

    return df_filtered, auc


def combine_all_improvements(df):
    """Combine all improvements for best possible result."""
    print("\n" + "="*80)
    print("ANALYSIS 6: Combined Improvements")
    print("="*80)

    # Step 1: Filter outliers
    mean = df['total_dist'].mean()
    std = df['total_dist'].std()
    threshold = mean + 3 * std
    df_filtered = df[df['total_dist'] <= threshold].copy()

    print(f"Step 1: Filtered to {len(df_filtered)} pairs")

    # Step 2: Apply normalization
    for dim in ['delta_sem', 'delta_syn', 'delta_lex']:
        mean_val = df_filtered[dim].mean()
        std_val = df_filtered[dim].std()
        df_filtered[dim + '_norm'] = (df_filtered[dim] - mean_val) / std_val

    print("Step 2: Applied z-score normalization")

    # Step 3: Learn optimal weights on normalized data
    X = df_filtered[['delta_sem_norm', 'delta_syn_norm', 'delta_lex_norm']].values
    y = df_filtered['label'].values

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    y_score = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_score)

    print("Step 3: Learned optimal weights on filtered+normalized data")
    print(f"\nFinal combined ROC-AUC: {auc:.4f}")

    weights = model.coef_[0]
    print(f"\nOptimal weights:")
    print(f"  Semantic:  {weights[0]:+.4f}")
    print(f"  Syntactic: {weights[1]:+.4f}")
    print(f"  Lexical:   {weights[2]:+.4f}")

    return auc, df_filtered


def generate_comparison_report(baseline_auc, all_results):
    """Generate comprehensive comparison report."""
    print("\n" + "="*80)
    print("FINAL COMPARISON REPORT")
    print("="*80)

    print(f"\nBaseline (equal weights, no filtering): {baseline_auc:.4f}")
    print("\nImprovement Results:")

    improvements = []
    for name, auc in all_results.items():
        improvement = ((auc - baseline_auc) / baseline_auc) * 100
        improvements.append((name, auc, improvement))
        print(f"  {name:40s}: {auc:.4f} ({improvement:+.1f}%)")

    # Find best
    best = max(improvements, key=lambda x: x[1])
    print(f"\n{'='*80}")
    print(f"BEST APPROACH: {best[0]}")
    print(f"  ROC-AUC: {best[1]:.4f}")
    print(f"  Improvement: {best[2]:+.1f}% over baseline")
    print(f"{'='*80}")

    return improvements


def create_comparison_visualizations(df, df_filtered, output_dir):
    """Create comparison visualizations."""
    print("\n" + "="*80)
    print("Generating Comparison Visualizations")
    print("="*80)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Baseline boxplot
    ax = axes[0, 0]
    same_author = df[df['label'] == 1]['total_dist']
    diff_author = df[df['label'] == 0]['total_dist']
    ax.boxplot([same_author, diff_author], labels=['Same Author', 'Different Author'])
    ax.set_ylabel('Total Distance')
    ax.set_title('Baseline: Equal Weights, No Filtering')
    ax.grid(axis='y', alpha=0.3)

    # 2. Filtered boxplot
    ax = axes[0, 1]
    same_author_filt = df_filtered[df_filtered['label'] == 1]['total_dist']
    diff_author_filt = df_filtered[df_filtered['label'] == 0]['total_dist']
    ax.boxplot([same_author_filt, diff_author_filt], labels=['Same Author', 'Different Author'])
    ax.set_ylabel('Total Distance')
    ax.set_title('Improved: Outliers Filtered')
    ax.grid(axis='y', alpha=0.3)

    # 3. Individual dimensions comparison
    ax = axes[1, 0]
    dimensions = ['delta_sem', 'delta_syn', 'delta_lex']
    aucs = []
    for dim in dimensions:
        y_true = df['label'].values
        y_score = -df[dim].values
        auc = roc_auc_score(y_true, y_score)
        aucs.append(auc)

    bars = ax.bar(['Semantic', 'Syntactic', 'Lexical'], aucs, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.axhline(y=0.5, color='r', linestyle='--', label='Random (0.50)')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Individual Dimension Performance')
    ax.legend()
    ax.set_ylim([0.45, max(aucs) + 0.05])

    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.4f}',
                ha='center', va='bottom')

    # 4. Distribution comparison (histogram)
    ax = axes[1, 1]
    ax.hist(same_author_filt, bins=50, alpha=0.5, label='Same Author', color='blue', density=True)
    ax.hist(diff_author_filt, bins=50, alpha=0.5, label='Different Author', color='red', density=True)
    ax.set_xlabel('Total Distance (Filtered)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison (Outliers Removed)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / 'improved_comparison.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comparison visualization to {plot_file}")


def main():
    """Main execution."""
    print("="*80)
    print("IMPROVED AUTHORSHIP VERIFICATION ANALYSIS")
    print("="*80)

    # Load baseline results
    df = load_results()

    # Calculate baseline AUC
    y_true = df['label'].values
    y_score = -df['total_dist'].values
    baseline_auc = roc_auc_score(y_true, y_score)
    print(f"\nBaseline ROC-AUC: {baseline_auc:.4f}")

    all_results = {}

    # Analysis 1: Individual dimensions
    dim_results, best_dim = analyze_individual_dimensions(df)
    all_results['best_dimension_only'] = dim_results[best_dim]

    # Analysis 2: Weighted combinations
    weight_results, best_scheme = test_weighted_combinations(df, dim_results)
    all_results[f'weighted_{best_scheme}'] = weight_results[best_scheme]['auc']

    # Analysis 3: Learned weights
    learned_auc, learned_weights = learn_optimal_weights(df)
    all_results['learned_weights'] = learned_auc

    # Analysis 4: Normalization
    df_norm, norm_auc = apply_relative_normalization(df)
    all_results['normalized'] = norm_auc

    # Analysis 5: Outlier filtering
    df_filtered, filtered_auc = filter_outliers(df, n_std=3)
    all_results['filtered_outliers'] = filtered_auc

    # Analysis 6: Combined improvements
    combined_auc, df_combined = combine_all_improvements(df)
    all_results['combined_all'] = combined_auc

    # Generate comparison report
    improvements = generate_comparison_report(baseline_auc, all_results)

    # Create visualizations
    output_dir = Path(__file__).parent / 'output/verification'
    create_comparison_visualizations(df, df_filtered, output_dir)

    # Save detailed report
    report_lines = [
        "=" * 80,
        "IMPROVED AUTHORSHIP VERIFICATION - DETAILED REPORT",
        "=" * 80,
        "",
        f"Baseline ROC-AUC: {baseline_auc:.4f}",
        "",
        "=" * 80,
        "IMPROVEMENT RESULTS",
        "=" * 80,
        ""
    ]

    for name, auc, improvement in sorted(improvements, key=lambda x: x[1], reverse=True):
        report_lines.append(f"{name:40s}: {auc:.4f} ({improvement:+.1f}% over baseline)")

    best = max(improvements, key=lambda x: x[1])
    report_lines.extend([
        "",
        "=" * 80,
        f"BEST APPROACH: {best[0]}",
        f"  ROC-AUC: {best[1]:.4f}",
        f"  Improvement: {best[2]:+.1f}% over baseline",
        "=" * 80,
        "",
        "INDIVIDUAL DIMENSION PERFORMANCE:",
        "-" * 80,
    ])

    for dim, auc in sorted(dim_results.items(), key=lambda x: x[1], reverse=True):
        report_lines.append(f"  {dim}: {auc:.4f}")

    report_lines.extend([
        "",
        "LEARNED OPTIMAL WEIGHTS:",
        "-" * 80,
        f"  Semantic:  {learned_weights[0]:+.4f}",
        f"  Syntactic: {learned_weights[1]:+.4f}",
        f"  Lexical:   {learned_weights[2]:+.4f}",
        "",
        "OUTLIER FILTERING IMPACT:",
        "-" * 80,
        f"  Original pairs: {len(df)}",
        f"  Filtered pairs: {len(df_filtered)}",
        f"  Removed: {len(df) - len(df_filtered)} ({100*(len(df)-len(df_filtered))/len(df):.1f}%)",
        "",
        "=" * 80,
        "CONCLUSION",
        "=" * 80,
        "",
        f"The best improvement achieved an ROC-AUC of {best[1]:.4f}, which is",
        f"{best[2]:.1f}% better than the baseline of {baseline_auc:.4f}.",
        "",
    ])

    if best[1] >= 0.65:
        report_lines.append("✓ SUCCESS: The improved metric now meets the viability threshold (AUC ≥ 0.65)!")
    elif best[1] >= 0.60:
        report_lines.append("⚠ PROGRESS: Significant improvement but still below viability threshold.")
    else:
        report_lines.append("✗ LIMITED: Improvements are modest; may need more fundamental changes.")

    report_file = output_dir / 'improved_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n✓ Saved detailed report to {report_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
