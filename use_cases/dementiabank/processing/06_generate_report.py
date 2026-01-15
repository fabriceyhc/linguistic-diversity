#!/usr/bin/env python3
"""Generate comprehensive evaluation report.

This script:
1. Loads all results and statistics
2. Creates markdown report with findings
3. Includes visualizations
4. Provides actionable conclusions
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datetime import datetime


def main():
    print("=" * 80)
    print("STEP 6: GENERATE EVALUATION REPORT")
    print("=" * 80)

    # Load data
    scores_path = Path(__file__).parent.parent / "output" / "scores" / "raw_scores.csv"
    stats_path = Path(__file__).parent.parent / "output" / "scores" / "summary_stats.csv"

    if not scores_path.exists() or not stats_path.exists():
        print(f"\n✗ Error: Required data files not found")
        sys.exit(1)

    print(f"\n1. Loading results...")
    df = pd.read_csv(scores_path)
    stats_df = pd.read_csv(stats_path)
    print(f"   ✓ Loaded data")

    # Prepare report
    print(f"\n2. Generating report...")

    report_dir = Path(__file__).parent.parent / "output" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "evaluation_report.md"

    with open(report_path, 'w') as f:
        # Header
        f.write("# DementiaBank Cognitive Impairment Detection\n")
        f.write("## Linguistic Diversity Metrics Evaluation Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")

        n_total = len(df)
        n_dementia = (df['label'] == 'Dementia').sum()
        n_control = (df['label'] == 'Control').sum()
        n_metrics = len(stats_df)

        f.write(f"This report evaluates {n_metrics} linguistic diversity metrics on the DementiaBank dataset ")
        f.write(f"to determine their utility in distinguishing cognitively impaired speech from healthy controls.\n\n")

        f.write(f"**Dataset:**\n")
        f.write(f"- Total subjects: {n_total}\n")
        f.write(f"- Dementia group: {n_dementia}\n")
        f.write(f"- Control group: {n_control}\n")
        f.write(f"- Task: Cookie Theft picture description\n\n")

        # Key Findings
        significant = stats_df[stats_df['significant_ttest']]
        large_effect = stats_df[stats_df['cohens_d'].abs() > 0.8]

        f.write(f"**Key Findings:**\n")
        f.write(f"- {len(significant)} out of {n_metrics} metrics show statistically significant differences (p < 0.05)\n")
        f.write(f"- {len(large_effect)} metrics show large effect sizes (|d| > 0.8)\n")

        # Success determination
        sem_success = stats_df[
            stats_df['metric'].str.contains('semantic') &
            (stats_df['control_mean'] > stats_df['dementia_mean']) &
            (stats_df['significant_ttest']) &
            (stats_df['cohens_d'].abs() > 0.3)
        ]

        syn_success = stats_df[
            stats_df['metric'].str.contains('syntactic') &
            (stats_df['control_mean'] > stats_df['dementia_mean']) &
            (stats_df['significant_ttest']) &
            (stats_df['cohens_d'].abs() > 0.3)
        ]

        if len(sem_success) > 0 or len(syn_success) > 0:
            f.write(f"- **Conclusion:** ✅ Framework IS USEFUL for cognitive impairment detection\n\n")
        else:
            f.write(f"- **Conclusion:** ❌ Framework NOT USEFUL for this specific task\n\n")

        f.write("---\n\n")

        # Methodology
        f.write("## Methodology\n\n")

        f.write("### Data Processing\n\n")
        f.write("1. **Data Source:** MearaHe/dementiabank dataset from Hugging Face\n")
        f.write("2. **Task Selection:** Cookie Theft picture description task only\n")
        f.write("3. **Preprocessing:**\n")
        f.write("   - Removed transcription artifacts ([unintelligible], [laughter], etc.)\n")
        f.write("   - Segmented transcripts into sentences\n")
        f.write("   - Filtered for minimum 2 sentences per transcript\n")
        f.write("4. **Class Balancing:** Used all available data\n\n")

        f.write("### Metrics Evaluated\n\n")
        f.write("**Semantic Diversity:**\n")
        f.write("- Document-level: Sentence embeddings similarity\n")
        f.write("- Token-level: Contextualized word embeddings\n\n")

        f.write("**Syntactic Diversity:**\n")
        f.write("- Dependency parsing structures\n")
        f.write("- Constituency parsing trees\n\n")

        f.write("**Morphological Diversity:**\n")
        f.write("- Part-of-speech sequence patterns\n\n")

        f.write("**Phonological Diversity:**\n")
        f.write("- Phonemic sequences\n")
        f.write("- Rhythmic patterns (stress & weight)\n\n")

        f.write("**Lexical Diversity:**\n")
        f.write("- Type-Token Ratio (TTR)\n\n")

        f.write("### Statistical Analysis\n\n")
        f.write("- **Tests:** Independent samples t-test, Mann-Whitney U\n")
        f.write("- **Effect Size:** Cohen's d\n")
        f.write("- **Significance Threshold:** α = 0.05\n\n")

        f.write("---\n\n")

        # Results
        f.write("## Results\n\n")

        f.write("### Summary Statistics\n\n")
        f.write("| Metric | Control Mean (SD) | Dementia Mean (SD) | Difference | p-value | Cohen's d | Effect Size |\n")
        f.write("|--------|-------------------|-------------------|------------|---------|-----------|-------------|\n")

        for _, row in stats_df.iterrows():
            metric = row['metric'].replace('_', ' ').title()
            con = f"{row['control_mean']:.3f} ({row['control_std']:.3f})"
            dem = f"{row['dementia_mean']:.3f} ({row['dementia_std']:.3f})"
            diff = f"{row['delta_percent']:.1f}%"
            p = f"{row['p_value']:.4f}"
            d = f"{row['cohens_d']:.3f}"
            effect = row['effect_size'].title()

            sig_marker = ""
            if row['significant_ttest']:
                if row['p_value'] < 0.001:
                    sig_marker = "***"
                elif row['p_value'] < 0.01:
                    sig_marker = "**"
                else:
                    sig_marker = "*"

            f.write(f"| {metric} | {con} | {dem} | {diff} | {p}{sig_marker} | {d} | {effect} |\n")

        f.write("\n*Significance: * p<0.05, ** p<0.01, *** p<0.001*\n\n")

        # Significant Findings
        f.write("### Significant Findings\n\n")

        if len(significant) > 0:
            for _, row in significant.iterrows():
                f.write(f"#### {row['metric'].replace('_', ' ').title()}\n\n")

                if row['control_mean'] > row['dementia_mean']:
                    direction = "**Lower in Dementia group**"
                else:
                    direction = "**Higher in Dementia group**"

                f.write(f"- {direction}\n")
                f.write(f"- Difference: {row['delta_percent']:.1f}%\n")
                f.write(f"- Statistical significance: p = {row['p_value']:.4f}\n")
                f.write(f"- Effect size: {row['cohens_d']:.3f} ({row['effect_size']})\n")
                f.write(f"- Interpretation: {_interpret_finding(row)}\n\n")
        else:
            f.write("No metrics showed statistically significant differences between groups.\n\n")

        f.write("---\n\n")

        # Visualizations
        f.write("## Visualizations\n\n")
        f.write("The following plots are available in `output/plots/`:\n\n")
        f.write("1. **Individual Boxplots** (`{metric}_boxplot.png`):\n")
        f.write("   - Compare distributions between Dementia and Control groups\n")
        f.write("   - Show statistical significance and effect sizes\n\n")

        f.write("2. **Violin Plots** (`{metric}_violin.png`):\n")
        f.write("   - Display full distribution shapes\n")
        f.write("   - Reveal multimodality and outliers\n\n")

        f.write("3. **Summary Comparison** (`all_metrics_comparison.png`):\n")
        f.write("   - Side-by-side comparison of all metrics\n")
        f.write("   - Significance markers for quick assessment\n\n")

        f.write("4. **Correlation Heatmap** (`metrics_correlation_heatmap.png`):\n")
        f.write("   - Inter-metric correlations\n")
        f.write("   - Identify redundant vs. complementary measures\n\n")

        f.write("---\n\n")

        # Conclusions
        f.write("## Conclusions\n\n")

        if len(sem_success) > 0 or len(syn_success) > 0:
            f.write("### ✅ Framework IS USEFUL\n\n")

            f.write("The linguistic diversity framework successfully distinguishes between ")
            f.write("cognitively impaired speech and healthy controls.\n\n")

            f.write("**Evidence:**\n\n")

            if len(sem_success) > 0:
                f.write(f"- **Semantic diversity** shows significant reduction in Dementia group:\n")
                for _, row in sem_success.iterrows():
                    f.write(f"  - {row['metric']}: {row['delta_percent']:.1f}% lower (p={row['p_value']:.4f})\n")
                f.write("\n")

            if len(syn_success) > 0:
                f.write(f"- **Syntactic diversity** shows significant reduction in Dementia group:\n")
                for _, row in syn_success.iterrows():
                    f.write(f"  - {row['metric']}: {row['delta_percent']:.1f}% lower (p={row['p_value']:.4f})\n")
                f.write("\n")

            f.write("**Recommended Metrics:**\n\n")
            top_metrics = stats_df[stats_df['significant_ttest']].nlargest(3, 'cohens_d', keep='all')
            for _, row in top_metrics.iterrows():
                f.write(f"1. **{row['metric'].replace('_', ' ').title()}**\n")
                f.write(f"   - Effect size: {row['cohens_d']:.3f} ({row['effect_size']})\n")
                f.write(f"   - Use for: Primary cognitive impairment detection\n\n")

            f.write("**Applications:**\n\n")
            f.write("- Screening tool for cognitive decline\n")
            f.write("- Monitoring disease progression\n")
            f.write("- Treatment efficacy evaluation\n")
            f.write("- Research into linguistic markers of dementia\n\n")

        else:
            f.write("### ❌ Framework NOT USEFUL for This Task\n\n")

            f.write("The linguistic diversity metrics do not show sufficient discriminative power ")
            f.write("for this specific task (Cookie Theft description) and dataset.\n\n")

            f.write("**Possible Reasons:**\n\n")
            f.write("1. **Task limitations:** Cookie Theft may not elicit enough linguistic variability\n")
            f.write("2. **Sample size:** Insufficient statistical power\n")
            f.write("3. **Disease heterogeneity:** Dementia group may be too diverse\n")
            f.write("4. **Metric limitations:** Current metrics may not capture relevant features\n\n")

            f.write("**Recommendations:**\n\n")
            f.write("1. Try alternative tasks (e.g., narrative speech, conversation)\n")
            f.write("2. Combine metrics with other features (acoustic, timing)\n")
            f.write("3. Use more sophisticated models (machine learning, deep learning)\n")
            f.write("4. Consider disease subtyping (Alzheimer's vs. other dementias)\n\n")

        f.write("---\n\n")

        # Limitations
        f.write("## Limitations\n\n")
        f.write("1. **Single task:** Results limited to Cookie Theft description\n")
        f.write("2. **Cross-sectional:** No longitudinal tracking of individuals\n")
        f.write("3. **Binary classification:** Does not capture severity levels\n")
        f.write("4. **Transcription:** Relies on accurate transcripts\n")
        f.write("5. **Computational:** Some metrics may fail on very short transcripts\n\n")

        f.write("---\n\n")

        # References
        f.write("## References\n\n")
        f.write("- **Dataset:** MearaHe/dementiabank (Hugging Face)\n")
        f.write("- **Framework:** linguistic-diversity (Hill numbers approach)\n")
        f.write("- **Statistical Methods:** Welch's t-test, Mann-Whitney U, Cohen's d\n\n")

        f.write("---\n\n")
        f.write("*Report generated automatically by the evaluation pipeline*\n")

    print(f"   ✓ Saved to: {report_path}")

    print("\n" + "=" * 80)
    print("✓ REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nEvaluation report saved to:")
    print(f"  {report_path}")
    print("\nReview the report to see:")
    print("  - Success/failure determination")
    print("  - Recommended metrics")
    print("  - Statistical details")
    print("  - Visualizations")
    print("=" * 80)


def _interpret_finding(row):
    """Interpret a statistical finding.

    Args:
        row: DataFrame row with statistical results

    Returns:
        Interpretation string
    """
    metric = row['metric']
    diff = row['delta_percent']
    direction = "reduced" if row['control_mean'] > row['dementia_mean'] else "increased"

    if 'semantic' in metric.lower():
        if direction == "reduced":
            return f"Dementia group shows {abs(diff):.1f}% less semantic diversity, suggesting more repetitive or semantically constrained language."
        else:
            return f"Unexpected increase in semantic diversity in Dementia group. May warrant further investigation."

    elif 'syntactic' in metric.lower():
        if direction == "reduced":
            return f"Dementia group shows {abs(diff):.1f}% less syntactic diversity, indicating simpler or more stereotyped grammatical structures."
        else:
            return f"Unexpected increase in syntactic diversity in Dementia group. May reflect parsing errors or unusual syntax."

    elif 'morphological' in metric.lower():
        if direction == "reduced":
            return f"Dementia group shows {abs(diff):.1f}% less morphological diversity, suggesting fewer grammatical pattern variations."
        else:
            return f"Unexpected increase in morphological diversity in Dementia group."

    elif 'phonological' in metric.lower() or 'phonemic' in metric.lower() or 'rhythmic' in metric.lower():
        return f"Phonological diversity is {direction} by {abs(diff):.1f}% in Dementia group."

    elif 'lexical' in metric.lower() or 'ttr' in metric.lower():
        if direction == "reduced":
            return f"Dementia group shows {abs(diff):.1f}% lower lexical diversity (Type-Token Ratio), indicating more limited vocabulary usage."
        else:
            return f"Unexpected increase in lexical diversity in Dementia group."

    else:
        return f"{direction.capitalize()} by {abs(diff):.1f}% in Dementia group."


if __name__ == "__main__":
    main()
