#!/usr/bin/env python3
"""Evaluate the composite dementia_detector metric on DementiaBank data.

This script:
1. Loads preprocessed DementiaBank data
2. Computes the composite "dementia_detector" metric for all subjects
3. Compares performance against individual metrics
4. Generates statistical analysis and visualization
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import os
import time
import pickle
import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm

# Set environment variables
os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)

from linguistic_diversity import UniversalLinguisticDiversity, get_preset_config


def compute_cohens_d(group1, group2):
    """Compute Cohen's d effect size.

    Args:
        group1: First group values
        group2: Second group values

    Returns:
        Cohen's d
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def main():
    print("=" * 80)
    print("COMPOSITE METRIC EVALUATION: DEMENTIA DETECTOR")
    print("=" * 80)

    # Load preprocessed data
    input_path = Path(__file__).parent / "input" / "preprocessed_data.pkl"

    if not input_path.exists():
        print(f"\n✗ Error: Preprocessed data not found: {input_path}")
        print("   Please run: python processing/02_preprocess.py first")
        sys.exit(1)

    print(f"\n1. Loading preprocessed data...")
    with open(input_path, 'rb') as f:
        df = pickle.load(f)

    print(f"   ✓ Loaded {len(df)} subjects")
    print(f"   Dementia: {(df['label'] == 'Dementia').sum()}")
    print(f"   Control: {(df['label'] == 'Control').sum()}")

    # Initialize composite metric
    print(f"\n2. Initializing composite 'dementia_detector' metric...")
    config = get_preset_config('dementia_detector')

    print(f"\n   Configuration:")
    print(f"   - Strategy: {config['strategy']}")
    print(f"   - Semantic weight: {config['semantic_weight']:.1%}")
    print(f"     • Document: {config['document_semantics_weight']:.1%}")
    print(f"     • Token: {config['token_semantics_weight']:.1%}")
    print(f"   - Syntactic weight: {config['syntactic_weight']:.1%}")
    print(f"     • Constituency: {config['constituency_parse_weight']:.1%}")
    print(f"   - Morphological: disabled (not significant)")
    print(f"   - Phonological: disabled (not significant)")

    # GPU configuration - use available GPU
    gpu_config = {
        'semantic_config': {'use_cuda': True, 'verbose': False},
        'syntactic_config': {'verbose': False, 'similarity_type': 'ldp'},
    }
    config.update(gpu_config)

    metric = UniversalLinguisticDiversity(config)
    print(f"   ✓ Metric initialized")

    # Compute composite scores
    print(f"\n3. Computing composite scores for {len(df)} subjects...")
    print(f"   (This will take a while...)")

    results = []
    errors = []
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Computing", ncols=80):
        subject_id = row['subject_id']
        label = row['label']
        sentences = row['sentences']

        try:
            score = metric(sentences)

            # Validate score
            if score is None or np.isnan(score) or np.isinf(score):
                raise ValueError(f"Invalid score: {score}")

            results.append({
                'subject_id': subject_id,
                'label': label,
                'composite_diversity': score
            })
        except Exception as e:
            errors.append({
                'subject_id': subject_id,
                'error': str(e)[:200]
            })
            results.append({
                'subject_id': subject_id,
                'label': label,
                'composite_diversity': np.nan
            })

    elapsed_time = time.time() - start_time

    results_df = pd.DataFrame(results)

    print(f"\n   ✓ Computation complete!")
    print(f"   Time elapsed: {elapsed_time / 60:.1f} minutes")
    print(f"   Average time per subject: {elapsed_time / len(df):.1f} seconds")

    if errors:
        print(f"   ⚠ Errors: {len(errors)}")

    # Statistical analysis
    print(f"\n4. Statistical analysis...")

    dementia_scores = results_df[results_df['label'] == 'Dementia']['composite_diversity'].dropna()
    control_scores = results_df[results_df['label'] == 'Control']['composite_diversity'].dropna()

    # Compute statistics
    dementia_mean = dementia_scores.mean()
    dementia_std = dementia_scores.std()
    dementia_median = dementia_scores.median()

    control_mean = control_scores.mean()
    control_std = control_scores.std()
    control_median = control_scores.median()

    # Effect size
    cohens_d = compute_cohens_d(control_scores, dementia_scores)

    # Statistical tests
    t_stat, p_value_t = stats.ttest_ind(control_scores, dementia_scores)
    _, p_value_mw = stats.mannwhitneyu(control_scores, dementia_scores, alternative='two-sided')

    # Percent difference
    delta_pct = ((control_mean - dementia_mean) / dementia_mean) * 100

    # Classify effect size
    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    print(f"\n   COMPOSITE 'DEMENTIA_DETECTOR' METRIC")
    print(f"   {'-' * 76}")
    print(f"   Dementia:  mean={dementia_mean:.3f}, std={dementia_std:.3f}, n={len(dementia_scores)}")
    print(f"   Control:   mean={control_mean:.3f}, std={control_std:.3f}, n={len(control_scores)}")
    print(f"   Difference: {delta_pct:.1f}% (Control > Dementia)")
    print(f"")
    print(f"   Statistical Tests:")
    print(f"     t-test:          t={t_stat:.3f}, p={p_value_t:.6f}", end="")
    if p_value_t < 0.001:
        print(" ***HIGHLY SIGNIFICANT***")
    elif p_value_t < 0.01:
        print(" **VERY SIGNIFICANT**")
    elif p_value_t < 0.05:
        print(" *SIGNIFICANT*")
    else:
        print(" (not significant)")

    print(f"     Mann-Whitney U:  p={p_value_mw:.6f}", end="")
    if p_value_mw < 0.001:
        print(" ***HIGHLY SIGNIFICANT***")
    elif p_value_mw < 0.01:
        print(" **VERY SIGNIFICANT**")
    elif p_value_mw < 0.05:
        print(" *SIGNIFICANT*")
    else:
        print(" (not significant)")

    print(f"     Cohen's d:       d={cohens_d:.3f} ({effect_size} effect)")

    # Compare with individual metrics
    print(f"\n5. Comparison with individual metrics...")

    # Load individual metric scores
    scores_path = Path(__file__).parent / "output" / "scores" / "raw_scores.csv"
    if scores_path.exists():
        individual_df = pd.read_csv(scores_path)

        print(f"\n   Performance Comparison (sorted by Cohen's d):")
        print(f"   {'-' * 76}")
        cohens_header = "Cohen's d"
        print(f"   {'Metric':<25} {cohens_header:>10} {'p-value':>12} {'Effect':<12}")
        print(f"   {'-' * 76}")

        # Collect results for all metrics including composite
        all_results = []

        # Add composite result
        all_results.append({
            'metric': 'COMPOSITE (dementia_detector)',
            'cohens_d': cohens_d,
            'p_value': p_value_t,
            'effect_size': effect_size
        })

        # Individual metrics
        metric_cols = [col for col in individual_df.columns
                      if col not in ['subject_id', 'label', 'n_sentences']]

        for metric in metric_cols:
            dem_vals = individual_df[individual_df['label'] == 'Dementia'][metric].dropna()
            con_vals = individual_df[individual_df['label'] == 'Control'][metric].dropna()

            if len(dem_vals) > 0 and len(con_vals) > 0:
                d = compute_cohens_d(con_vals, dem_vals)
                _, p = stats.ttest_ind(con_vals, dem_vals)

                if abs(d) < 0.2:
                    eff = "negligible"
                elif abs(d) < 0.5:
                    eff = "small"
                elif abs(d) < 0.8:
                    eff = "medium"
                else:
                    eff = "large"

                all_results.append({
                    'metric': metric,
                    'cohens_d': d,
                    'p_value': p,
                    'effect_size': eff
                })

        # Sort by Cohen's d (descending)
        all_results.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

        for r in all_results:
            sig_marker = ""
            if r['p_value'] < 0.001:
                sig_marker = " ***"
            elif r['p_value'] < 0.01:
                sig_marker = " **"
            elif r['p_value'] < 0.05:
                sig_marker = " *"

            print(f"   {r['metric']:<25} {r['cohens_d']:>10.3f} {r['p_value']:>12.6f}{sig_marker:<4} {r['effect_size']:<12}")

    # Save results
    print(f"\n6. Saving results...")
    output_dir = Path(__file__).parent / "output" / "scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "composite_scores.csv"
    results_df.to_csv(output_path, index=False)
    print(f"   ✓ Saved scores to: {output_path}")

    # Save summary
    summary_path = output_dir / "composite_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("COMPOSITE METRIC EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Configuration: dementia_detector preset\n")
        f.write(f"  - Semantic weight: {config['semantic_weight']:.1%}\n")
        f.write(f"  - Syntactic weight: {config['syntactic_weight']:.1%}\n")
        f.write(f"  - Strategy: {config['strategy']}\n\n")
        f.write("Results:\n")
        f.write(f"  Dementia: mean={dementia_mean:.3f}, std={dementia_std:.3f}, n={len(dementia_scores)}\n")
        f.write(f"  Control: mean={control_mean:.3f}, std={control_std:.3f}, n={len(control_scores)}\n")
        f.write(f"  Difference: {delta_pct:.1f}% (Control > Dementia)\n\n")
        f.write("Statistical Tests:\n")
        f.write(f"  t-test: t={t_stat:.3f}, p={p_value_t:.6f}\n")
        f.write(f"  Mann-Whitney U: p={p_value_mw:.6f}\n")
        f.write(f"  Cohen's d: {cohens_d:.3f} ({effect_size} effect)\n\n")
        f.write(f"Computation time: {elapsed_time / 60:.1f} minutes\n")
        f.write(f"Errors: {len(errors)}\n")

    print(f"   ✓ Saved summary to: {summary_path}")

    print("\n" + "=" * 80)
    print("✓ COMPOSITE METRIC EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nThe 'dementia_detector' composite metric:")
    print(f"  - Cohen's d: {cohens_d:.3f} ({effect_size} effect)")
    print(f"  - p-value: {p_value_t:.6f}")
    if p_value_t < 0.05:
        print(f"  - ✓ STATISTICALLY SIGNIFICANT")
    else:
        print(f"  - ✗ Not statistically significant")

    print(f"\nCompared to best individual metric (doc_semantic: d=0.621):")
    pct_change = ((cohens_d - 0.621) / 0.621) * 100
    if cohens_d > 0.621:
        print(f"  ✓ IMPROVED by {pct_change:.1f}%")
    else:
        print(f"  ✗ DECREASED by {abs(pct_change):.1f}%")

    print("=" * 80)


if __name__ == "__main__":
    main()
