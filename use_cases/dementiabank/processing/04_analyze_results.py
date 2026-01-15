#!/usr/bin/env python3
"""Statistical analysis of diversity metrics.

This script:
1. Loads raw scores
2. Computes descriptive statistics per group
3. Performs statistical tests (t-tests, Mann-Whitney U)
4. Calculates effect sizes (Cohen's d)
5. Evaluates success criteria
6. Generates statistical report
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import yaml
from scipy.stats import ttest_ind, mannwhitneyu, normaltest


def load_config():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size.

    Args:
        group1: First group (numpy array or series)
        group2: Second group (numpy array or series)

    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    return (group1.mean() - group2.mean()) / pooled_std


def interpret_effect_size(d):
    """Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def main():
    print("=" * 80)
    print("STEP 4: STATISTICAL ANALYSIS")
    print("=" * 80)

    # Load configuration
    config = load_config()
    stats_config = config['statistical']

    # Load raw scores
    input_path = Path(__file__).parent.parent / "output" / "scores" / "raw_scores.csv"

    if not input_path.exists():
        print(f"\n✗ Error: Raw scores not found: {input_path}")
        print("   Please run: python processing/03_compute_metrics.py first")
        sys.exit(1)

    print(f"\n1. Loading raw scores...")
    df = pd.read_csv(input_path)
    print(f"   ✓ Loaded {len(df)} subjects")

    # Identify metric columns
    metric_cols = [col for col in df.columns if col not in ['subject_id', 'label', 'n_sentences']]
    print(f"   ✓ {len(metric_cols)} metrics to analyze")

    # Split by group
    print(f"\n2. Splitting by group...")
    df_dementia = df[df['label'] == 'Dementia']
    df_control = df[df['label'] == 'Control']

    print(f"   Dementia: {len(df_dementia)} subjects")
    print(f"   Control: {len(df_control)} subjects")

    # Check minimum sample size
    min_size = stats_config['min_sample_size']
    if len(df_dementia) < min_size or len(df_control) < min_size:
        print(f"\n⚠ Warning: Sample size below minimum ({min_size})")
        print("   Statistical power may be insufficient")

    # Perform statistical tests
    print(f"\n3. Performing statistical tests...")
    print("   (t-test, Mann-Whitney U, effect size)")

    stats_results = []

    for metric in metric_cols:
        # Get valid data
        dementia = df_dementia[metric].dropna()
        control = df_control[metric].dropna()

        if len(dementia) < 2 or len(control) < 2:
            print(f"   ⚠ Skipping {metric}: insufficient data")
            continue

        # Descriptive statistics
        dem_mean = dementia.mean()
        dem_std = dementia.std()
        dem_median = dementia.median()

        con_mean = control.mean()
        con_std = control.std()
        con_median = control.median()

        # Percent difference
        if dem_mean != 0:
            delta_pct = ((con_mean - dem_mean) / dem_mean) * 100
        else:
            delta_pct = np.nan

        # Independent samples t-test
        t_stat, p_value = ttest_ind(control, dementia, equal_var=False)  # Welch's t-test

        # Mann-Whitney U (non-parametric)
        u_stat, p_value_mw = mannwhitneyu(control, dementia, alternative='two-sided')

        # Cohen's d effect size
        d = cohens_d(control, dementia)
        effect_interp = interpret_effect_size(d)

        # Check normality (Shapiro-Wilk for small samples, D'Agostino for larger)
        try:
            if len(dementia) > 20 and len(control) > 20:
                _, p_norm_dem = normaltest(dementia)
                _, p_norm_con = normaltest(control)
            else:
                p_norm_dem = np.nan
                p_norm_con = np.nan
        except:
            p_norm_dem = np.nan
            p_norm_con = np.nan

        # Determine if significant
        alpha = stats_config['alpha']
        significant_t = p_value < alpha
        significant_mw = p_value_mw < alpha

        # Direction of effect
        if con_mean > dem_mean:
            direction = "Control > Dementia"
        elif con_mean < dem_mean:
            direction = "Dementia > Control"
        else:
            direction = "No difference"

        stats_results.append({
            'metric': metric,
            'dementia_n': len(dementia),
            'dementia_mean': dem_mean,
            'dementia_std': dem_std,
            'dementia_median': dem_median,
            'control_n': len(control),
            'control_mean': con_mean,
            'control_std': con_std,
            'control_median': con_median,
            'delta_percent': delta_pct,
            't_statistic': t_stat,
            'p_value': p_value,
            'p_value_mw': p_value_mw,
            'cohens_d': d,
            'effect_size': effect_interp,
            'significant_ttest': significant_t,
            'significant_mw': significant_mw,
            'direction': direction,
        })

    stats_df = pd.DataFrame(stats_results)

    # Apply Bonferroni correction if specified
    if stats_config.get('use_bonferroni', False):
        print(f"\n   Applying Bonferroni correction...")
        corrected_alpha = alpha / len(stats_df)
        stats_df['significant_ttest_bonf'] = stats_df['p_value'] < corrected_alpha
        stats_df['significant_mw_bonf'] = stats_df['p_value_mw'] < corrected_alpha
        print(f"   Corrected alpha: {corrected_alpha:.4f}")

    print(f"   ✓ Analyzed {len(stats_df)} metrics")

    # Save summary statistics
    print(f"\n4. Saving results...")
    output_dir = Path(__file__).parent.parent / "output" / "scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_stats.csv"
    stats_df.to_csv(summary_path, index=False)
    print(f"   ✓ Saved to: {summary_path}")

    # Generate text report
    print(f"\n5. Generating statistical report...")

    report_path = output_dir / "statistical_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total subjects: {len(df)}\n")
        f.write(f"  Dementia: {len(df_dementia)}\n")
        f.write(f"  Control: {len(df_control)}\n")
        f.write(f"Metrics analyzed: {len(stats_df)}\n")
        f.write(f"Significance threshold: α = {alpha}\n")
        if stats_config.get('use_bonferroni'):
            f.write(f"Bonferroni corrected α = {alpha / len(stats_df):.4f}\n")
        f.write("\n")

        f.write("RESULTS BY METRIC\n")
        f.write("=" * 80 + "\n\n")

        for _, row in stats_df.iterrows():
            f.write(f"{row['metric'].upper()}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Dementia:  mean={row['dementia_mean']:.3f}, std={row['dementia_std']:.3f}, n={row['dementia_n']}\n")
            f.write(f"Control:   mean={row['control_mean']:.3f}, std={row['control_std']:.3f}, n={row['control_n']}\n")
            f.write(f"Difference: {row['delta_percent']:.1f}% ({row['direction']})\n")
            f.write(f"\n")
            f.write(f"Statistical Tests:\n")
            f.write(f"  t-test:          t={row['t_statistic']:.3f}, p={row['p_value']:.4f}")
            if row['significant_ttest']:
                f.write(" ***SIGNIFICANT***\n")
            else:
                f.write(" (not significant)\n")
            f.write(f"  Mann-Whitney U:  p={row['p_value_mw']:.4f}")
            if row['significant_mw']:
                f.write(" ***SIGNIFICANT***\n")
            else:
                f.write(" (not significant)\n")
            f.write(f"  Cohen's d:       d={row['cohens_d']:.3f} ({row['effect_size']} effect)\n")
            f.write("\n\n")

        # Evaluate success criteria
        f.write("=" * 80 + "\n")
        f.write("SUCCESS CRITERIA EVALUATION\n")
        f.write("=" * 80 + "\n\n")

        # Semantic metrics with significant drop
        sem_metrics = stats_df[stats_df['metric'].str.contains('semantic', case=False)]
        sem_success = sem_metrics[
            (sem_metrics['control_mean'] > sem_metrics['dementia_mean']) &
            (sem_metrics['significant_ttest']) &
            (sem_metrics['cohens_d'].abs() > 0.3)
        ]

        f.write("1. Semantic Diversity\n")
        f.write("-" * 80 + "\n")
        if len(sem_success) > 0:
            f.write("✓ SUCCESS: Semantic metrics show significant differences\n\n")
            for _, row in sem_success.iterrows():
                f.write(f"   {row['metric']}:\n")
                f.write(f"     Difference: {row['delta_percent']:.1f}% lower in Dementia\n")
                f.write(f"     p-value: {row['p_value']:.4f}\n")
                f.write(f"     Effect size: {row['cohens_d']:.3f} ({row['effect_size']})\n\n")
        else:
            f.write("✗ NO SUCCESS: No semantic metrics show significant differences\n\n")

        # Syntactic metrics with significant drop
        syn_metrics = stats_df[stats_df['metric'].str.contains('syntactic', case=False)]
        syn_success = syn_metrics[
            (syn_metrics['control_mean'] > syn_metrics['dementia_mean']) &
            (syn_metrics['significant_ttest']) &
            (syn_metrics['cohens_d'].abs() > 0.3)
        ]

        f.write("2. Syntactic Diversity\n")
        f.write("-" * 80 + "\n")
        if len(syn_success) > 0:
            f.write("✓ SUCCESS: Syntactic metrics show significant differences\n\n")
            for _, row in syn_success.iterrows():
                f.write(f"   {row['metric']}:\n")
                f.write(f"     Difference: {row['delta_percent']:.1f}% lower in Dementia\n")
                f.write(f"     p-value: {row['p_value']:.4f}\n")
                f.write(f"     Effect size: {row['cohens_d']:.3f} ({row['effect_size']})\n\n")
        else:
            f.write("✗ NO SUCCESS: No syntactic metrics show significant differences\n\n")

        # Overall significant metrics
        all_significant = stats_df[
            (stats_df['significant_ttest']) &
            (stats_df['cohens_d'].abs() > 0.3)
        ]

        f.write("3. All Significant Metrics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total metrics with significant differences: {len(all_significant)}\n\n")
        for _, row in all_significant.iterrows():
            f.write(f"   ✓ {row['metric']}: p={row['p_value']:.4f}, d={row['cohens_d']:.3f}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("OVERALL CONCLUSION\n")
        f.write("=" * 80 + "\n\n")

        if len(sem_success) > 0 or len(syn_success) > 0:
            f.write("✓ FRAMEWORK IS USEFUL for cognitive impairment detection\n\n")
            f.write("Rationale:\n")
            if len(sem_success) > 0:
                f.write(f"  • {len(sem_success)} semantic metric(s) show significant differences\n")
            if len(syn_success) > 0:
                f.write(f"  • {len(syn_success)} syntactic metric(s) show significant differences\n")
            f.write(f"  • Total {len(all_significant)} metrics meet success criteria\n")
            f.write("\nRecommendation: This framework can be used for cognitive impairment detection.\n")
            f.write("Metrics showing strongest signal should be prioritized in applications.\n")
        else:
            f.write("✗ FRAMEWORK NOT USEFUL for this specific task\n\n")
            f.write("Rationale:\n")
            f.write("  • No semantic or syntactic metrics show significant differences\n")
            f.write("  • Effect sizes are too small (< 0.3)\n")
            f.write("  • OR p-values are not significant (> 0.05)\n")
            f.write("\nRecommendation: This framework may not be suitable for cognitive\n")
            f.write("impairment detection using the Cookie Theft task, or additional\n")
            f.write("features/methods may be needed.\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"   ✓ Saved to: {report_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nSignificant metrics (p < {alpha}):")
    significant = stats_df[stats_df['significant_ttest']]
    if len(significant) > 0:
        for _, row in significant.iterrows():
            print(f"  ✓ {row['metric']}: p={row['p_value']:.4f}, d={row['cohens_d']:.3f} ({row['effect_size']})")
    else:
        print("  (none)")

    print("\nNext step:")
    print("  python processing/05_create_plots.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
