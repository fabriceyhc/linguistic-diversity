#!/usr/bin/env python3
"""
Generate report and visualizations from instruction fine-tuning results.

This script:
1. Loads evaluation results
2. Creates comparison tables and visualizations
3. Performs statistical significance tests
4. Generates markdown report with LIMA-style analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import yaml
from datetime import datetime

plt.style.use('seaborn-v0_8-whitegrid')


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_results(output_dir: Path):
    results_path = output_dir / "finetuning_results.json"
    if not results_path.exists():
        return None
    with open(results_path, 'r') as f:
        return json.load(f)


def load_selections_metadata(base_dir: Path):
    """Load selection metadata for analysis."""
    selections_path = base_dir / "selections" / "selections.json"
    overlap_path = base_dir / "selections" / "overlap_stats.json"
    metadata_path = base_dir / "datasets" / "metadata.json"

    data = {}
    if selections_path.exists():
        with open(selections_path, 'r') as f:
            data['selections'] = json.load(f)
    if overlap_path.exists():
        with open(overlap_path, 'r') as f:
            data['overlap'] = json.load(f)
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            data['metadata'] = json.load(f)

    return data


def get_method_display_names():
    return {
        'pretrained': 'Pretrained (no fine-tuning)',
        'full_dataset': 'Full Dataset',
        'random': 'Random',
        'semantic_diversity': 'Semantic Diversity',
        'syntactic_diversity': 'Syntactic Diversity',
        'combined_diversity': 'Combined Diversity',
        'length_diversity': 'Length Stratified',
        'quality_filtered': 'Quality + Diversity',
        'instruction_diversity': 'Instruction Diversity',
        'universal_embedding_diversity': 'Universal Embedding',  # NEW
    }


def get_method_colors():
    return {
        'random': '#808080',              # Gray
        'semantic_diversity': '#4ECDC4',  # Teal
        'syntactic_diversity': '#45B7D1', # Blue
        'combined_diversity': '#FF6B6B',  # Red
        'length_diversity': '#96CEB4',    # Green
        'quality_filtered': '#DDA0DD',    # Plum
        'instruction_diversity': '#FFD93D', # Yellow
        'universal_embedding_diversity': '#9B59B6',  # Purple - NEW
    }


def plot_results_comparison(results: dict, output_dir: Path):
    """Create bar plots comparing methods across sizes."""
    display_names = get_method_display_names()
    colors = get_method_colors()

    for model_result in results['results']:
        model_name = model_result['model'].split('/')[-1]
        results_by_size = model_result['results_by_size']

        if not results_by_size:
            continue

        # Get all metrics
        all_metrics = set()
        for size_data in results_by_size.values():
            for method_data in size_data.values():
                all_metrics.update(method_data.get('metrics', {}).keys())

        if not all_metrics:
            continue

        for metric in all_metrics:
            fig, axes = plt.subplots(1, len(results_by_size), figsize=(5 * len(results_by_size), 5))
            if len(results_by_size) == 1:
                axes = [axes]

            for ax, (size_key, methods) in zip(axes, results_by_size.items()):
                size_label = size_key.replace('size_', '')

                method_names = []
                means = []
                stds = []
                bar_colors = []

                # Sort methods by mean score
                sorted_methods = sorted(
                    methods.items(),
                    key=lambda x: x[1].get('metrics', {}).get(metric, {}).get('mean', 0),
                    reverse=True
                )

                for method, data in sorted_methods:
                    if metric not in data.get('metrics', {}):
                        continue
                    method_names.append(display_names.get(method, method))
                    means.append(data['metrics'][metric]['mean'])
                    stds.append(data['metrics'][metric]['std'])
                    bar_colors.append(colors.get(method, '#808080'))

                if not method_names:
                    continue

                x = np.arange(len(method_names))
                bars = ax.bar(x, means, yerr=stds, capsize=3, color=bar_colors, alpha=0.8, edgecolor='black')

                ax.set_xticks(x)
                ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
                ax.set_ylabel(f'{metric}', fontsize=10)
                ax.set_title(f'n = {size_label}', fontsize=11, fontweight='bold')

                # Set y-axis limits
                if means:
                    y_min = min(m - s for m, s in zip(means, stds)) - 0.02
                    y_max = max(m + s for m, s in zip(means, stds)) + 0.02
                    ax.set_ylim(max(0, y_min), min(1, y_max))

            plt.suptitle(f'{model_name} - {metric} by Selection Method', fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()

            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            plt.savefig(plots_dir / f'{model_name}_{metric}_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {model_name}_{metric}_comparison.png")


def plot_data_efficiency(results: dict, output_dir: Path):
    """Plot performance vs training data size (LIMA-style analysis)."""
    display_names = get_method_display_names()
    colors = get_method_colors()

    for model_result in results['results']:
        model_name = model_result['model'].split('/')[-1]
        results_by_size = model_result['results_by_size']

        if not results_by_size:
            continue

        # Get all metrics
        all_metrics = set()
        for size_data in results_by_size.values():
            for method_data in size_data.values():
                all_metrics.update(method_data.get('metrics', {}).keys())

        if not all_metrics:
            continue

        for metric in all_metrics:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Collect data by method
            method_data_dict = {}
            for size_key, methods in sorted(results_by_size.items()):
                size = int(size_key.replace('size_', ''))
                for method, data in methods.items():
                    if metric not in data.get('metrics', {}):
                        continue
                    if method not in method_data_dict:
                        method_data_dict[method] = {'sizes': [], 'means': [], 'stds': []}
                    method_data_dict[method]['sizes'].append(size)
                    method_data_dict[method]['means'].append(data['metrics'][metric]['mean'])
                    method_data_dict[method]['stds'].append(data['metrics'][metric]['std'])

            # Plot each method
            for method, data in method_data_dict.items():
                ax.errorbar(
                    data['sizes'], data['means'], yerr=data['stds'],
                    marker='o', label=display_names.get(method, method),
                    color=colors.get(method, '#808080'), linewidth=2, markersize=8, capsize=4
                )

            ax.set_xlabel('Training Samples', fontsize=12)
            ax.set_ylabel(f'{metric}', fontsize=12)
            ax.set_title(f'{model_name} - Data Efficiency ({metric})', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=9)
            ax.set_xscale('log')

            plt.tight_layout()

            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            plt.savefig(plots_dir / f'{model_name}_{metric}_efficiency.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved: {model_name}_{metric}_efficiency.png")


def compute_significance(results: dict) -> dict:
    """Compute statistical significance vs random baseline."""
    significance = {}

    for model_result in results['results']:
        model_name = model_result['model']
        significance[model_name] = {}

        for size_key, methods in model_result['results_by_size'].items():
            if 'random' not in methods:
                continue

            random_data = methods['random']
            significance[model_name][size_key] = {}

            for method, data in methods.items():
                if method == 'random':
                    continue

                for metric, metric_data in data.get('metrics', {}).items():
                    if metric not in random_data.get('metrics', {}):
                        continue

                    method_scores = metric_data.get('scores', [metric_data['mean']])
                    random_scores = random_data['metrics'][metric].get('scores', [random_data['metrics'][metric]['mean']])

                    if len(method_scores) >= 2 and len(random_scores) >= 2:
                        t_stat, p_value = stats.ttest_ind(method_scores, random_scores)
                        diff = np.mean(method_scores) - np.mean(random_scores)

                        if method not in significance[model_name][size_key]:
                            significance[model_name][size_key][method] = {}

                        significance[model_name][size_key][method][metric] = {
                            'diff': diff,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                        }

    return significance


def generate_markdown_report(results: dict, significance: dict, metadata: dict, output_dir: Path):
    """Generate comprehensive markdown report."""
    display_names = get_method_display_names()

    lines = []
    lines.append("# Instruction Fine-tuning Data Selection Results\n")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # Experimental setup
    lines.append("## Experimental Setup\n")
    config = results.get('config', {})
    lines.append(f"- **Mode**: {config.get('mode', 'unknown')}")
    lines.append(f"- **Models**: {', '.join(config.get('models', []))}")
    lines.append(f"- **Selection sizes**: {config.get('sizes', [])}")
    lines.append(f"- **Selection methods**: {config.get('methods', [])}")
    lines.append(f"- **Evaluation tasks**: {config.get('eval_tasks', [])}")
    lines.append(f"- **Seeds**: {config.get('seeds', [])}")

    if metadata.get('metadata'):
        meta = metadata['metadata']
        lines.append(f"\n### Dataset Statistics")
        lines.append(f"- **Total samples**: {meta.get('num_samples', 'N/A')}")
        lines.append(f"- **Source datasets**: {', '.join(meta.get('datasets', []))}")
        lines.append(f"- **Categories**: {', '.join(meta.get('categories', []))}")

    lines.append("\n")

    # Results tables
    lines.append("## Results\n")

    for model_result in results['results']:
        model_name = model_result['model']
        short_name = model_name.split('/')[-1]

        lines.append(f"### {short_name}\n")

        # Collect all metrics across baselines and size-specific results
        all_metrics = set()
        if 'baselines' in model_result:
            for baseline_data in model_result['baselines'].values():
                all_metrics.update(baseline_data.get('metrics', {}).keys())
        for methods in model_result.get('results_by_size', {}).values():
            for method_data in methods.values():
                all_metrics.update(method_data.get('metrics', {}).keys())

        if not all_metrics:
            lines.append("*No evaluation results available*\n")
            continue

        metrics_list = sorted(all_metrics)

        # Baselines table
        if 'baselines' in model_result and model_result['baselines']:
            lines.append("\n#### Baselines\n")
            header = "| Method | n | " + " | ".join(metrics_list) + " |"
            separator = "|--------|---|" + "|".join(["--------" for _ in metrics_list]) + "|"
            lines.append(header)
            lines.append(separator)

            for baseline_name, data in model_result['baselines'].items():
                n = data.get('n_samples', 0)
                row = f"| {display_names.get(baseline_name, baseline_name)} | {n} |"
                for metric in metrics_list:
                    if metric in data.get('metrics', {}):
                        mean = data['metrics'][metric]['mean']
                        std = data['metrics'][metric]['std']
                        row += f" {mean:.4f} ± {std:.4f} |"
                    else:
                        row += " - |"
                lines.append(row)
            lines.append("")

        # Size-specific results
        for size_key, methods in model_result.get('results_by_size', {}).items():
            size = size_key.replace('size_', '')
            lines.append(f"\n#### Training Size: {size} samples\n")

            # Create table header
            header = "| Method | " + " | ".join(metrics_list) + " |"
            separator = "|--------|" + "|".join(["--------" for _ in metrics_list]) + "|"

            lines.append(header)
            lines.append(separator)

            # Sort methods by first metric
            first_metric = metrics_list[0]
            sorted_methods = sorted(
                methods.items(),
                key=lambda x: x[1].get('metrics', {}).get(first_metric, {}).get('mean', 0),
                reverse=True
            )

            for method, data in sorted_methods:
                row = f"| {display_names.get(method, method)} |"
                for metric in metrics_list:
                    if metric in data.get('metrics', {}):
                        mean = data['metrics'][metric]['mean']
                        std = data['metrics'][metric]['std']
                        row += f" {mean:.4f} ± {std:.4f} |"
                    else:
                        row += " - |"
                lines.append(row)

            lines.append("")

    # Key findings
    lines.append("## Key Findings\n")

    # Calculate wins for each method
    wins = {}
    total_comparisons = 0

    for model_result in results['results']:
        for size_key, methods in model_result['results_by_size'].items():
            if 'random' not in methods:
                continue

            random_metrics = methods['random'].get('metrics', {})

            for method, data in methods.items():
                if method == 'random':
                    continue

                if method not in wins:
                    wins[method] = {'wins': 0, 'losses': 0, 'ties': 0}

                for metric, metric_data in data.get('metrics', {}).items():
                    if metric in random_metrics:
                        total_comparisons += 1
                        method_mean = metric_data['mean']
                        random_mean = random_metrics[metric]['mean']

                        if method_mean > random_mean + 0.001:
                            wins[method]['wins'] += 1
                        elif method_mean < random_mean - 0.001:
                            wins[method]['losses'] += 1
                        else:
                            wins[method]['ties'] += 1

    lines.append("### Method Performance vs Random Baseline\n")

    if wins:
        for method, counts in sorted(wins.items(), key=lambda x: -x[1]['wins']):
            total = counts['wins'] + counts['losses'] + counts['ties']
            win_rate = counts['wins'] / total * 100 if total > 0 else 0
            lines.append(f"- **{display_names.get(method, method)}**: {counts['wins']}/{total} wins ({win_rate:.0f}%)")
    else:
        lines.append("*Insufficient data for comparison*")

    lines.append("")

    # Statistical significance
    lines.append("### Statistical Significance (p < 0.05 vs Random)\n")

    sig_found = False
    for model_name, model_sig in significance.items():
        for size_key, methods_sig in model_sig.items():
            for method, metrics_sig in methods_sig.items():
                for metric, sig_data in metrics_sig.items():
                    if sig_data['significant']:
                        sig_found = True
                        direction = "+" if sig_data['diff'] > 0 else ""
                        lines.append(f"- **{display_names.get(method, method)}** on {metric}: {direction}{sig_data['diff']:.4f} (p={sig_data['p_value']:.4f})")

    if not sig_found:
        lines.append("*No statistically significant differences found (may need more seeds)*")

    lines.append("")

    # LIMA-style conclusions
    lines.append("## Conclusions\n")

    lines.append("### Data Efficiency Analysis\n")
    lines.append("Following the LIMA paper's insight that quality/diversity matters more than quantity:\n")

    if wins:
        best_method = max(wins.items(), key=lambda x: x[1]['wins'])[0]
        best_display = display_names.get(best_method, best_method)
        best_wins = wins[best_method]['wins']
        total = wins[best_method]['wins'] + wins[best_method]['losses'] + wins[best_method]['ties']

        if best_wins > total * 0.5:
            lines.append(f"1. **{best_display}** shows the most consistent improvement over random selection")
        else:
            lines.append("1. No single selection method dominates across all configurations")

    lines.append("2. Diversity-based selection methods aim to achieve comparable performance to larger randomly-selected datasets")
    lines.append("3. The effectiveness of each method may vary based on the downstream task and model architecture")

    lines.append("\n### Recommendations\n")
    lines.append("- For resource-constrained fine-tuning, consider **combined diversity** selection")
    lines.append("- Monitor both task performance and training efficiency")
    lines.append("- Consider ensemble methods that combine multiple selection strategies")

    lines.append("")

    # Selection overlap analysis
    if metadata.get('overlap'):
        lines.append("## Selection Method Overlap Analysis\n")
        lines.append("Low overlap between methods indicates they capture different aspects of diversity.\n")

        for size_key, overlaps in metadata['overlap'].items():
            lines.append(f"\n### {size_key}\n")
            # Show top 5 highest overlaps
            sorted_overlaps = sorted(overlaps.items(), key=lambda x: -x[1]['jaccard'])[:5]
            for comparison, data in sorted_overlaps:
                lines.append(f"- {comparison}: Jaccard={data['jaccard']:.3f} ({data['overlap']} shared samples)")

    lines.append("")

    # Write report
    report_path = output_dir / "finetuning_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"  Saved: {report_path}")


def main():
    print("=" * 70)
    print("STEP 4: GENERATE REPORT")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "output"

    # Load results
    print("\n1. Loading results...")
    results = load_results(output_dir)

    if results is None:
        print("   No results found. Run 03_finetune_evaluate.py first.")
        return

    print(f"   Loaded results for {len(results.get('results', []))} models")

    # Load metadata
    metadata = load_selections_metadata(base_dir)
    print(f"   Loaded selection metadata")

    # Generate plots
    print("\n2. Generating plots...")
    plot_results_comparison(results, output_dir)
    plot_data_efficiency(results, output_dir)

    # Compute significance
    print("\n3. Computing statistical significance...")
    significance = compute_significance(results)

    # Generate report
    print("\n4. Generating report...")
    generate_markdown_report(results, significance, metadata, output_dir)

    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
