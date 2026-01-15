#!/usr/bin/env python
"""Benchmark the Universal Linguistic Diversity metric and its components.

This script measures the execution time of each individual metric and the
combined universal metric to identify performance bottlenecks.
"""

import sys
import time
from typing import Any

import pandas as pd

sys.path.insert(0, "src")

from linguistic_diversity import (
    DependencyParse,
    DocumentSemantics,
    PartOfSpeechSequence,
    TokenSemantics,
    UniversalLinguisticDiversity,
)


def create_test_corpora() -> dict[str, list[str]]:
    """Create test corpora of different sizes."""
    # Small corpus (3 documents)
    small = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps above an idle hound.",
        "Swift russet predator vaults past lethargic beast.",
    ]

    # Medium corpus (10 documents)
    medium = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast auburn canine leaps above an idle hound.",
        "Swift russet predator vaults past lethargic beast.",
        "Artificial intelligence is transforming modern technology.",
        "Machine learning models require large amounts of training data.",
        "Neural networks can learn complex patterns from examples.",
        "Climate change poses significant challenges to humanity.",
        "Renewable energy sources are becoming increasingly viable.",
        "Sustainable development requires balancing economic and environmental needs.",
        "Global cooperation is essential for addressing planetary challenges.",
    ]

    # Large corpus (30 documents)
    large = medium * 3

    # Very large corpus (100 documents)
    very_large = medium * 10

    return {
        "small (3 docs)": small,
        "medium (10 docs)": medium,
        "large (30 docs)": large,
        "very_large (100 docs)": very_large,
    }


def benchmark_metric(
    metric: Any, corpus: list[str], name: str, warmup: bool = True
) -> tuple[float, float]:
    """Benchmark a single metric.

    Args:
        metric: The metric to benchmark
        corpus: The corpus to process
        name: Name of the metric
        warmup: Whether to do a warmup run

    Returns:
        Tuple of (execution_time, diversity_score)
    """
    # Warmup run (to load models, etc.)
    if warmup:
        try:
            _ = metric(corpus)
        except Exception:
            pass

    # Actual benchmark
    start_time = time.perf_counter()
    try:
        score = metric(corpus)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        return execution_time, score
    except Exception as e:
        print(f"  ⚠️  {name} failed: {e}")
        return 0.0, 0.0


def benchmark_all_metrics(corpus: list[str], corpus_name: str) -> pd.DataFrame:
    """Benchmark all individual metrics and the universal metric.

    Args:
        corpus: The corpus to process
        corpus_name: Name of the corpus (for display)

    Returns:
        DataFrame with benchmark results
    """
    print(f"\n{'='*80}")
    print(f"Benchmarking: {corpus_name}")
    print(f"{'='*80}")

    results = []

    # Configuration for faster benchmarking
    fast_config = {"verbose": False, "use_cuda": False, "batch_size": 8}

    # Define metrics to benchmark
    metrics_to_test = [
        ("TokenSemantics", TokenSemantics(fast_config), "Semantic"),
        (
            "DocumentSemantics",
            DocumentSemantics({**fast_config, "model_name": "all-MiniLM-L6-v2"}),
            "Semantic",
        ),
        ("DependencyParse", DependencyParse({"verbose": False}), "Syntactic"),
        ("PartOfSpeechSequence", PartOfSpeechSequence({"verbose": False}), "Morphological"),
    ]

    # Try optional metrics
    try:
        from linguistic_diversity import ConstituencyParse

        metrics_to_test.append(
            ("ConstituencyParse", ConstituencyParse({"verbose": False}), "Syntactic")
        )
    except ImportError:
        print("  ℹ️  ConstituencyParse not available (requires benepar)")

    try:
        from linguistic_diversity import Rhythmic

        metrics_to_test.append(("Rhythmic", Rhythmic({"verbose": False}), "Phonological"))
    except ImportError:
        print("  ℹ️  Rhythmic not available (requires cadences)")

    try:
        from linguistic_diversity import Phonemic

        metrics_to_test.append(("Phonemic", Phonemic({"verbose": False}), "Phonological"))
    except ImportError:
        print("  ℹ️  Phonemic not available (requires g2p_en)")

    # Benchmark each metric
    print("\nIndividual Metrics:")
    print("-" * 80)

    for metric_name, metric, branch in metrics_to_test:
        print(f"  Benchmarking {metric_name}...", end=" ", flush=True)
        exec_time, score = benchmark_metric(metric, corpus, metric_name, warmup=True)

        if exec_time > 0:
            print(f"✓ {exec_time:.3f}s (score: {score:.2f})")
            results.append(
                {
                    "Metric": metric_name,
                    "Branch": branch,
                    "Time (s)": exec_time,
                    "Score": score,
                    "Type": "Individual",
                }
            )
        else:
            print("✗ Failed")

    # Benchmark universal metric (minimal config - only required metrics)
    print("\nUniversal Metric (minimal config):")
    print("-" * 80)
    print("  Benchmarking UniversalLinguisticDiversity (minimal)...", end=" ", flush=True)

    minimal_config = {
        "verbose": False,
        "use_constituency_parse": False,
        "use_rhythmic": False,
        "use_phonemic": False,
        "semantic_config": fast_config,
    }
    universal_minimal = UniversalLinguisticDiversity(minimal_config)
    exec_time, score = benchmark_metric(
        universal_minimal, corpus, "UniversalLinguisticDiversity (minimal)", warmup=True
    )

    if exec_time > 0:
        print(f"✓ {exec_time:.3f}s (score: {score:.2f})")
        results.append(
            {
                "Metric": "Universal (minimal)",
                "Branch": "All",
                "Time (s)": exec_time,
                "Score": score,
                "Type": "Universal",
            }
        )

    # Benchmark universal metric (full config with all available metrics)
    print("\nUniversal Metric (full config):")
    print("-" * 80)
    print("  Benchmarking UniversalLinguisticDiversity (full)...", end=" ", flush=True)

    full_config = {
        "verbose": False,
        "use_constituency_parse": False,  # Optional
        "use_rhythmic": True,
        "use_phonemic": True,
        "semantic_config": fast_config,
    }
    universal_full = UniversalLinguisticDiversity(full_config)
    exec_time, score = benchmark_metric(
        universal_full, corpus, "UniversalLinguisticDiversity (full)", warmup=True
    )

    if exec_time > 0:
        print(f"✓ {exec_time:.3f}s (score: {score:.2f})")
        results.append(
            {
                "Metric": "Universal (full)",
                "Branch": "All",
                "Time (s)": exec_time,
                "Score": score,
                "Type": "Universal",
            }
        )

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame, corpus_name: str) -> None:
    """Analyze and display benchmark results.

    Args:
        df: DataFrame with benchmark results
        corpus_name: Name of the corpus
    """
    print(f"\n{'='*80}")
    print(f"Results Summary: {corpus_name}")
    print(f"{'='*80}\n")

    # Display full results table
    print("All Metrics:")
    print("-" * 80)
    display_df = df.copy()
    display_df["Time (s)"] = display_df["Time (s)"].apply(lambda x: f"{x:.3f}")
    display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.2f}")
    print(display_df.to_string(index=False))

    # Individual metrics only
    individual_df = df[df["Type"] == "Individual"].copy()

    if len(individual_df) > 0:
        print("\n" + "="*80)
        print("Performance Analysis")
        print("="*80)

        # Sort by time
        individual_df_sorted = individual_df.sort_values("Time (s)", ascending=False)

        print("\nSlowest to Fastest:")
        print("-" * 80)
        total_time = individual_df["Time (s)"].sum()
        for idx, row in individual_df_sorted.iterrows():
            percentage = (row["Time (s)"] / total_time) * 100
            print(
                f"  {row['Metric']:30s} {row['Time (s)']:6.3f}s ({percentage:5.1f}%) [{row['Branch']}]"
            )

        print(f"\n  {'TOTAL (sum of individual)':30s} {total_time:6.3f}s (100.0%)")

        # Compare with universal metrics
        universal_df = df[df["Type"] == "Universal"]
        if len(universal_df) > 0:
            print("\nUniversal Metric Comparison:")
            print("-" * 80)
            for idx, row in universal_df.iterrows():
                overhead = row["Time (s)"] - total_time
                overhead_pct = (overhead / total_time) * 100 if total_time > 0 else 0
                print(
                    f"  {row['Metric']:30s} {row['Time (s)']:6.3f}s "
                    f"(overhead: {overhead:+.3f}s, {overhead_pct:+.1f}%)"
                )

        # Branch-level aggregation
        print("\nBy Linguistic Branch:")
        print("-" * 80)
        branch_times = individual_df.groupby("Branch")["Time (s)"].sum().sort_values(
            ascending=False
        )
        for branch, branch_time in branch_times.items():
            percentage = (branch_time / total_time) * 100
            count = len(individual_df[individual_df["Branch"] == branch])
            print(
                f"  {branch:20s} {branch_time:6.3f}s ({percentage:5.1f}%) "
                f"[{count} metric{'s' if count > 1 else ''}]"
            )

        # Recommendations
        print("\n" + "="*80)
        print("Optimization Recommendations")
        print("="*80)

        slowest = individual_df_sorted.iloc[0]
        fastest = individual_df_sorted.iloc[-1]

        print(f"\n📊 Performance Insights:")
        print(f"  • Slowest metric: {slowest['Metric']} ({slowest['Time (s)']:.3f}s)")
        print(f"  • Fastest metric: {fastest['Metric']} ({fastest['Time (s)']:.3f}s)")
        print(
            f"  • Speed ratio: {slowest['Time (s)'] / fastest['Time (s)']:.1f}x difference"
        )

        # Identify bottlenecks (>30% of total time)
        bottlenecks = individual_df_sorted[
            (individual_df_sorted["Time (s)"] / total_time) > 0.30
        ]

        if len(bottlenecks) > 0:
            print(f"\n⚠️  Performance Bottlenecks (>30% of total time):")
            for idx, row in bottlenecks.iterrows():
                percentage = (row["Time (s)"] / total_time) * 100
                print(f"  • {row['Metric']} ({percentage:.1f}% of total time)")
            print(
                "\n  Consider excluding these for faster computation using:"
            )
            for idx, row in bottlenecks.iterrows():
                metric_flag = f"use_{row['Metric'].lower().replace('semantics', 'semantics').replace('parse', 'parse').replace('sequence', 'sequence')}"
                # Map metric names to config flags
                config_mapping = {
                    "tokensemantics": "use_token_semantics",
                    "documentsemantics": "use_document_semantics",
                    "dependencyparse": "use_dependency_parse",
                    "constituencyparse": "use_constituency_parse",
                    "partofpeechsequence": "use_pos_sequence",
                    "rhythmic": "use_rhythmic",
                    "phonemic": "use_phonemic",
                }
                clean_name = row['Metric'].lower().replace(" ", "")
                config_flag = config_mapping.get(clean_name, f"use_{clean_name}")
                print(f"    config['{config_flag}'] = False")
        else:
            print("\n✓ No major bottlenecks detected (all metrics <30% of total time)")

        # Speed recommendations
        print(f"\n🚀 Speed Optimization Options:")
        print(f"  1. Minimal config (core metrics only):")
        print(f"     • Excludes: ConstituencyParse, Rhythmic, Phonemic")
        print(f"     • Use: get_preset_config('minimal')")

        # Calculate potential speedup by excluding slowest metric
        if len(individual_df) > 1:
            without_slowest = total_time - slowest["Time (s)"]
            speedup = total_time / without_slowest
            print(
                f"  2. Exclude slowest metric ({slowest['Metric']}):"
            )
            print(
                f"     • Speedup: {speedup:.2f}x faster ({without_slowest:.3f}s vs {total_time:.3f}s)"
            )

        # Calculate potential speedup by using only fastest 3
        if len(individual_df) >= 3:
            fastest_3 = individual_df_sorted.tail(3)
            fastest_3_time = fastest_3["Time (s)"].sum()
            speedup = total_time / fastest_3_time
            print(f"  3. Use only 3 fastest metrics:")
            print(f"     • Metrics: {', '.join(fastest_3['Metric'].tolist())}")
            print(
                f"     • Speedup: {speedup:.2f}x faster ({fastest_3_time:.3f}s vs {total_time:.3f}s)"
            )


def main():
    """Run comprehensive benchmarks."""
    print("="*80)
    print("Universal Linguistic Diversity - Performance Benchmark")
    print("="*80)
    print("\nThis benchmark measures the execution time of each metric component")
    print("to identify performance bottlenecks and optimization opportunities.")

    # Create test corpora
    corpora = create_test_corpora()

    # Store all results
    all_results = {}

    # Benchmark each corpus size
    for corpus_name, corpus in corpora.items():
        df = benchmark_all_metrics(corpus, corpus_name)
        all_results[corpus_name] = df
        analyze_results(df, corpus_name)

    # Scaling analysis
    print("\n" + "="*80)
    print("Scaling Analysis")
    print("="*80)
    print("\nHow execution time scales with corpus size:\n")

    # Create scaling comparison table
    scaling_data = []
    for corpus_name, df in all_results.items():
        # Get corpus size
        size = int(corpus_name.split("(")[1].split()[0])

        # Individual metrics
        for idx, row in df[df["Type"] == "Individual"].iterrows():
            scaling_data.append(
                {
                    "Corpus Size": size,
                    "Metric": row["Metric"],
                    "Time (s)": row["Time (s)"],
                }
            )

        # Universal metrics
        for idx, row in df[df["Type"] == "Universal"].iterrows():
            scaling_data.append(
                {
                    "Corpus Size": size,
                    "Metric": row["Metric"],
                    "Time (s)": row["Time (s)"],
                }
            )

    scaling_df = pd.DataFrame(scaling_data)

    # Pivot table for better visualization
    pivot_df = scaling_df.pivot(index="Metric", columns="Corpus Size", values="Time (s)")
    print(pivot_df.to_string(float_format=lambda x: f"{x:.3f}"))

    # Calculate scaling factors (time ratio between corpus sizes)
    print("\n\nScaling Factors (time ratio relative to smallest corpus):")
    print("-" * 80)
    sizes = sorted(pivot_df.columns)
    smallest_size = sizes[0]

    for size in sizes:
        if size == smallest_size:
            continue
        ratio = size / smallest_size
        print(f"\n{size} docs / {smallest_size} docs = {ratio:.1f}x more documents:")
        for metric in pivot_df.index:
            time_ratio = pivot_df.loc[metric, size] / pivot_df.loc[metric, smallest_size]
            print(f"  {metric:30s} {time_ratio:5.2f}x slower")

    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
