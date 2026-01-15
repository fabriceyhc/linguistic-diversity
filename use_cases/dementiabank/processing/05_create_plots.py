#!/usr/bin/env python3
"""Create visualizations of results.

This script:
1. Loads raw scores and statistical summary
2. Creates boxplots for each metric
3. Creates summary comparison plots
4. Creates correlation heatmap
5. Saves all plots to output directory
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# Fix matplotlib C++ library issue - use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot

import matplotlib.pyplot as plt
import seaborn as sns
import yaml


def load_config():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_boxplot(df, metric, stats_row, output_path, config):
    """Create boxplot for a single metric.

    Args:
        df: DataFrame with raw scores
        metric: Metric name
        stats_row: Statistical summary row for this metric
        output_path: Path to save plot
        config: Configuration dictionary
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Boxplot
    sns.boxplot(
        data=df.dropna(subset=[metric]),
        x='label',
        y=metric,
        hue='label',
        order=['Control', 'Dementia'],
        palette={'Control': '#3498db', 'Dementia': '#e74c3c'},
        ax=ax,
        width=0.5,
        legend=False
    )

    # Add stripplot for individual points
    sns.stripplot(
        data=df.dropna(subset=[metric]),
        x='label',
        y=metric,
        order=['Control', 'Dementia'],
        color='black',
        alpha=0.3,
        size=3,
        ax=ax
    )

    # Add mean markers
    means = df.groupby('label')[metric].mean()
    ax.plot([0, 1], [means['Control'], means['Dementia']],
            'D-', color='darkgreen', markersize=10, linewidth=2.5,
            label='Group Means', zorder=10)

    # Statistical annotation
    p_val = stats_row['p_value']
    cohens = stats_row['cohens_d']

    if p_val < 0.001:
        sig_text = '***'
        sig_color = 'darkred'
    elif p_val < 0.01:
        sig_text = '**'
        sig_color = 'red'
    elif p_val < 0.05:
        sig_text = '*'
        sig_color = 'orange'
    else:
        sig_text = 'ns'
        sig_color = 'gray'

    # Create annotation box
    annotation = f"p = {p_val:.4f} {sig_text}\nCohen's d = {cohens:.3f}\n({stats_row['effect_size']} effect)"

    ax.text(
        0.98, 0.97,
        annotation,
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor=sig_color, linewidth=2)
    )

    # Labels and title
    metric_title = metric.replace('_', ' ').title()
    ax.set_title(f"{metric_title} Diversity", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('Diversity Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Group', fontsize=12, fontweight='bold')

    # Add grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config['output']['plots_dpi'], bbox_inches='tight')
    plt.close()


def create_summary_plot(stats_df, output_path, config):
    """Create summary comparison plot for all metrics.

    Args:
        stats_df: DataFrame with statistical summary
        output_path: Path to save plot
        config: Configuration dictionary
    """
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(stats_df))
    width = 0.35

    control_means = stats_df['control_mean'].values
    dementia_means = stats_df['dementia_mean'].values

    bars1 = ax.bar(x - width/2, control_means, width, label='Control',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, dementia_means, width, label='Dementia',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add significance markers
    for i, row in stats_df.iterrows():
        if row['significant_ttest']:
            max_val = max(row['control_mean'], row['dementia_mean'])
            # Add asterisks
            if row['p_value'] < 0.001:
                marker = '***'
            elif row['p_value'] < 0.01:
                marker = '**'
            else:
                marker = '*'

            ax.text(i, max_val * 1.05, marker, ha='center',
                   fontsize=16, color='red', fontweight='bold')

    # Labels
    ax.set_ylabel('Diversity Score', fontweight='bold', fontsize=13)
    ax.set_title('Linguistic Diversity: Dementia vs Control\nAll Metrics Comparison',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['metric'], rotation=45, ha='right', fontsize=11)
    ax.legend(fontsize=12, loc='upper left')

    # Grid
    ax.yaxis.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Add note about significance
    ax.text(0.02, 0.98, '* p<0.05  ** p<0.01  *** p<0.001',
           transform=ax.transAxes, fontsize=10, va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=config['output']['plots_dpi'], bbox_inches='tight')
    plt.close()


def create_violin_plot(df, metric, output_path, config):
    """Create violin plot for a metric (shows distribution).

    Args:
        df: DataFrame with raw scores
        metric: Metric name
        output_path: Path to save plot
        config: Configuration dictionary
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Violin plot
    sns.violinplot(
        data=df.dropna(subset=[metric]),
        x='label',
        y=metric,
        hue='label',
        order=['Control', 'Dementia'],
        palette={'Control': '#3498db', 'Dementia': '#e74c3c'},
        ax=ax,
        inner='box',
        legend=False
    )

    # Add individual points
    sns.stripplot(
        data=df.dropna(subset=[metric]),
        x='label',
        y=metric,
        order=['Control', 'Dementia'],
        color='black',
        alpha=0.2,
        size=2,
        ax=ax
    )

    metric_title = metric.replace('_', ' ').title()
    ax.set_title(f"{metric_title} Diversity Distribution", fontsize=14, fontweight='bold')
    ax.set_ylabel('Diversity Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Group', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=config['output']['plots_dpi'], bbox_inches='tight')
    plt.close()


def create_correlation_heatmap(df, output_path, config):
    """Create correlation heatmap between metrics.

    Args:
        df: DataFrame with raw scores
        output_path: Path to save plot
        config: Configuration dictionary
    """
    # Get metric columns
    metric_cols = [col for col in df.columns if col not in ['subject_id', 'label', 'n_sentences']]

    # Compute correlation matrix
    corr = df[metric_cols].corr()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlation'},
        ax=ax,
        vmin=-1,
        vmax=1
    )

    ax.set_title('Correlation Between Diversity Metrics', fontsize=16, fontweight='bold', pad=20)

    # Rotate labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=config['output']['plots_dpi'], bbox_inches='tight')
    plt.close()


def main():
    print("=" * 80)
    print("STEP 5: CREATE VISUALIZATIONS")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Load data
    scores_path = Path(__file__).parent.parent / "output" / "scores" / "raw_scores.csv"
    stats_path = Path(__file__).parent.parent / "output" / "scores" / "summary_stats.csv"

    if not scores_path.exists() or not stats_path.exists():
        print(f"\n✗ Error: Required data files not found")
        print("   Please run previous steps first")
        sys.exit(1)

    print(f"\n1. Loading data...")
    df = pd.read_csv(scores_path)
    stats_df = pd.read_csv(stats_path)
    print(f"   ✓ Loaded {len(df)} subjects")
    print(f"   ✓ Loaded {len(stats_df)} metrics with statistics")

    # Show which metrics have data
    metric_cols = [col for col in df.columns if col not in ['subject_id', 'label', 'n_sentences']]
    metrics_in_stats = set(stats_df['metric'].unique())
    metrics_in_scores = set(metric_cols)

    missing_stats = metrics_in_scores - metrics_in_stats
    if missing_stats:
        print(f"   ⚠ {len(missing_stats)} metrics without statistics (will be skipped):")
        for m in sorted(missing_stats):
            print(f"      - {m}")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "output" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    print(f"\n2. Creating individual boxplots...")
    for metric in metric_cols:
        # Check if stats exist for this metric
        stats_rows = stats_df[stats_df['metric'] == metric]
        if len(stats_rows) == 0:
            print(f"   ⏭️  {metric} (no statistics, skipping)")
            continue

        stats_row = stats_rows.iloc[0]
        output_path = output_dir / f"{metric}_boxplot.png"

        try:
            create_boxplot(df, metric, stats_row, output_path, config)
            print(f"   ✓ {metric}")
        except Exception as e:
            print(f"   ✗ {metric}: {e}")

    print(f"\n3. Creating individual violin plots...")
    for metric in metric_cols:
        # Check if stats exist for this metric
        stats_rows = stats_df[stats_df['metric'] == metric]
        if len(stats_rows) == 0:
            print(f"   ⏭️  {metric} (no statistics, skipping)")
            continue

        output_path = output_dir / f"{metric}_violin.png"

        try:
            create_violin_plot(df, metric, output_path, config)
            print(f"   ✓ {metric}")
        except Exception as e:
            print(f"   ✗ {metric}: {e}")

    print(f"\n4. Creating summary comparison plot...")
    summary_path = output_dir / "all_metrics_comparison.png"
    try:
        create_summary_plot(stats_df, summary_path, config)
        print(f"   ✓ Saved to: {summary_path.name}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print(f"\n5. Creating correlation heatmap...")
    corr_path = output_dir / "metrics_correlation_heatmap.png"
    try:
        create_correlation_heatmap(df, corr_path, config)
        print(f"   ✓ Saved to: {corr_path.name}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # List all created plots
    print(f"\n6. Summary of generated plots:")
    plots = sorted(output_dir.glob("*.png"))
    print(f"   Total plots created: {len(plots)}")
    for plot in plots:
        size_mb = plot.stat().st_size / 1024 / 1024
        print(f"   - {plot.name} ({size_mb:.2f} MB)")

    print("\n" + "=" * 80)
    print("✓ VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nAll plots saved to: {output_dir}")
    print("\nNext step:")
    print("  python processing/06_generate_report.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
