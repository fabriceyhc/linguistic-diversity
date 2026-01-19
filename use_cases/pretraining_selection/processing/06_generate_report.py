#!/usr/bin/env python3
"""Generate evaluation report with visualizations.

This script:
1. Loads evaluation results
2. Creates comparative visualizations:
   - Training loss curves
   - Perplexity comparison
   - Repetition rate comparison
3. Generates comprehensive markdown report
4. Saves all outputs
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import json
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set environment variables
os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)

# Set matplotlib backend
plt.switch_backend('Agg')
sns.set_style("whitegrid")


def load_config():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def plot_training_curves(models_dir: Path, output_dir: Path, config: dict):
    """Plot training loss curves for all models.

    Args:
        models_dir: Models directory
        output_dir: Output directory for plots
        config: Configuration dict
    """
    print(f"\n1. Plotting training curves...")

    regimes = ['semantic_diversity', 'syntactic_diversity', 'morphological_diversity', 'phonological_diversity', 'composite_diversity', 'universal_diversity', 'random_baseline']

    for dataset_config in config['corpus']['datasets']:
        dataset_name = dataset_config['name']
        clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()

        fig, ax = plt.subplots(figsize=(10, 6))

        for regime in regimes:
            log_file = models_dir / clean_name / regime / "training_log.json"

            if not log_file.exists():
                continue

            with open(log_file, 'r') as f:
                log_data = json.load(f)

            steps = [entry['step'] for entry in log_data]
            losses = [entry['loss'] for entry in log_data]

            ax.plot(steps, losses, label=regime.replace('_', ' ').title(), linewidth=2)

        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'Training Loss Curves - {dataset_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save plot
        plot_file = output_dir / "plots" / f"training_curves_{clean_name}.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   ✓ Saved: {plot_file}")


def plot_perplexity_comparison(results: list, output_dir: Path):
    """Plot perplexity comparison across regimes.

    Args:
        results: List of evaluation results
        output_dir: Output directory for plots
    """
    print(f"\n2. Plotting perplexity comparison...")

    # Create dataframe
    data = []
    for result in results:
        data.append({
            'Dataset': result['dataset_name'].split('/')[-1],
            'Regime': result['regime'].replace('_', ' ').title(),
            'Perplexity': result['perplexity']
        })

    df = pd.DataFrame(data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = df['Dataset'].unique()
    regimes = df['Regime'].unique()
    x = np.arange(len(datasets))
    width = 0.2

    for i, regime in enumerate(regimes):
        regime_data = df[df['Regime'] == regime]
        perplexities = [regime_data[regime_data['Dataset'] == d]['Perplexity'].values[0]
                       if len(regime_data[regime_data['Dataset'] == d]) > 0 else 0
                       for d in datasets]

        ax.bar(x + i * width, perplexities, width, label=regime)

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perplexity (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Model Perplexity Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticks_labels(datasets)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Save plot
    plot_file = output_dir / "plots" / "perplexity_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {plot_file}")


def plot_repetition_comparison(results: list, output_dir: Path):
    """Plot repetition rate comparison across regimes.

    Args:
        results: List of evaluation results
        output_dir: Output directory for plots
    """
    print(f"\n3. Plotting repetition rate comparison...")

    # Create dataframe
    data = []
    for result in results:
        data.append({
            'Dataset': result['dataset_name'].split('/')[-1],
            'Regime': result['regime'].replace('_', ' ').title(),
            'Repetition Rate': result['avg_repetition_rate']
        })

    df = pd.DataFrame(data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    datasets = df['Dataset'].unique()
    regimes = df['Regime'].unique()
    x = np.arange(len(datasets))
    width = 0.2

    for i, regime in enumerate(regimes):
        regime_data = df[df['Regime'] == regime]
        rep_rates = [regime_data[regime_data['Dataset'] == d]['Repetition Rate'].values[0]
                    if len(regime_data[regime_data['Dataset'] == d]) > 0 else 0
                    for d in datasets]

        ax.bar(x + i * width, rep_rates, width, label=regime)

    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Repetition Rate (lower is better)', fontsize=12, fontweight='bold')
    ax.set_title('Generation Repetition Rate Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Save plot
    plot_file = output_dir / "plots" / "repetition_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✓ Saved: {plot_file}")


def generate_markdown_report(results: list, output_dir: Path, config: dict):
    """Generate comprehensive markdown report.

    Args:
        results: List of evaluation results
        output_dir: Output directory
        config: Configuration dict
    """
    print(f"\n4. Generating markdown report...")

    report = []

    # Header
    report.append("# Pretraining Data Selection: Linguistic Diversity Evaluation")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")

    # Executive Summary
    report.append("## Executive Summary\n")
    report.append("This report evaluates the hypothesis that training data selected for high linguistic ")
    report.append("diversity (semantic or syntactic) yields better downstream performance in small language ")
    report.append("models compared to random selection.\n")

    # Configuration
    report.append("\n## Experiment Configuration\n")
    report.append(f"- **Mode:** {config['mode']}\n")
    report.append(f"- **Datasets:** {', '.join([d['name'] for d in config['corpus']['datasets']])}\n")
    report.append(f"- **Model:** GPT-2 ({config['training']['model']['n_layers']} layers, "
                 f"{config['training']['model']['n_embed']} dim)\n")
    report.append(f"- **Training Steps:** {config['training']['training']['max_steps'][config['mode']]}\n")
    report.append(f"- **Selection Ratio:** {config['selection']['selection_ratio']:.1%}\n")

    # Results
    report.append("\n## Results\n")

    # Create results table
    report.append("### Perplexity and Generation Quality\n")
    report.append("| Dataset | Regime | Perplexity ↓ | Repetition Rate ↓ |")
    report.append("|---------|--------|-------------|-------------------|")

    for result in sorted(results, key=lambda x: (x['dataset_name'], x['regime'])):
        dataset = result['dataset_name'].split('/')[-1]
        regime = result['regime'].replace('_', ' ').title()
        ppl = f"{result['perplexity']:.2f}"
        rep = f"{result['avg_repetition_rate']:.4f}"
        report.append(f"| {dataset} | {regime} | {ppl} | {rep} |")

    # Analysis
    report.append("\n### Key Findings\n")

    # Group by dataset
    datasets = {}
    for result in results:
        ds_name = result['dataset_name']
        if ds_name not in datasets:
            datasets[ds_name] = []
        datasets[ds_name].append(result)

    for ds_name, ds_results in datasets.items():
        report.append(f"\n#### {ds_name}\n")

        # Find best by perplexity
        best_ppl = min(ds_results, key=lambda x: x['perplexity'])
        baseline = next((r for r in ds_results if r['regime'] == 'random_baseline'), None)

        if baseline:
            improvement = ((baseline['perplexity'] - best_ppl['perplexity']) / baseline['perplexity']) * 100
            report.append(f"- **Best regime:** {best_ppl['regime'].replace('_', ' ').title()} "
                         f"(Perplexity: {best_ppl['perplexity']:.2f})\n")
            report.append(f"- **Baseline:** Random (Perplexity: {baseline['perplexity']:.2f})\n")
            report.append(f"- **Improvement:** {improvement:+.2f}%\n")

            if improvement > 0:
                report.append(f"- **Conclusion:** ✅ Diversity-based selection **improved** performance\n")
            else:
                report.append(f"- **Conclusion:** ❌ Diversity-based selection did **not improve** performance\n")

    # Visualizations
    report.append("\n## Visualizations\n")
    report.append("### Training Curves\n")
    for dataset_config in config['corpus']['datasets']:
        clean_name = dataset_config['name'].replace('/', '_').replace('-', '_').lower()
        report.append(f"![Training Curves](./../plots/training_curves_{clean_name}.png)\n")

    report.append("\n### Perplexity Comparison\n")
    report.append("![Perplexity Comparison](./../plots/perplexity_comparison.png)\n")

    report.append("\n### Repetition Rate Comparison\n")
    report.append("![Repetition Comparison](./../plots/repetition_comparison.png)\n")

    # Sample Generations
    report.append("\n## Sample Generations\n")

    for result in results[:4]:  # Show first 4 models
        report.append(f"\n### {result['dataset_name']} - {result['regime'].replace('_', ' ').title()}\n")

        for i, gen in enumerate(result['generations'][:2]):  # Show 2 samples per model
            report.append(f"\n**Prompt:** {gen['prompt']}\n")
            report.append(f"**Generated:** {gen['generated']}\n")
            report.append(f"**Repetition Rate:** {gen['repetition_rate']:.4f}\n")

    # Conclusion
    report.append("\n## Conclusion\n")
    report.append("This experiment evaluated whether linguistic diversity-based data selection improves ")
    report.append("language model performance compared to random selection. The results ")
    report.append("suggest that:\n\n")

    # Calculate overall improvement
    all_diversity = [r for r in results if r['regime'] != 'random_baseline']
    all_baseline = [r for r in results if r['regime'] == 'random_baseline']

    if all_diversity and all_baseline:
        avg_div_ppl = np.mean([r['perplexity'] for r in all_diversity])
        avg_base_ppl = np.mean([r['perplexity'] for r in all_baseline])
        overall_improvement = ((avg_base_ppl - avg_div_ppl) / avg_base_ppl) * 100

        if overall_improvement > 2:
            report.append(f"- ✅ **Diversity-based selection showed {overall_improvement:.1f}% improvement** ")
            report.append("over random baseline on average\n")
            report.append("- The approach is **promising** for pretraining data curation\n")
        elif overall_improvement > 0:
            report.append(f"- ⚠️ Diversity-based selection showed modest {overall_improvement:.1f}% improvement\n")
            report.append("- Results are **mixed** and warrant further investigation\n")
        else:
            report.append(f"- ❌ Diversity-based selection did not improve over random baseline\n")
            report.append("- The approach may **not be effective** for this task/scale\n")

    # Write report
    report_file = output_dir / "report" / "pretraining_selection_report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"   ✓ Saved: {report_file}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("STEP 6: GENERATE REPORT")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Setup directories
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "report").mkdir(parents=True, exist_ok=True)

    # Load evaluation results
    results_file = output_dir / "evaluation_results.json"
    if not results_file.exists():
        print(f"\n✗ Evaluation results not found: {results_file}")
        print("   Please run 05_evaluate_models.py first")
        return

    with open(results_file, 'r') as f:
        results = json.load(f)

    print(f"\n✓ Loaded evaluation results: {len(results)} models")

    # Generate visualizations
    try:
        plot_training_curves(models_dir, output_dir, config)
        plot_perplexity_comparison(results, output_dir)
        plot_repetition_comparison(results, output_dir)
    except Exception as e:
        print(f"\n⚠ Warning: Failed to generate some plots: {e}")
        import traceback
        traceback.print_exc()

    # Generate markdown report
    generate_markdown_report(results, output_dir, config)

    print("\n" + "=" * 80)
    print("✓ REPORT GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
