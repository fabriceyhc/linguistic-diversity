#!/usr/bin/env python3
"""Compute linguistic diversity metrics for all subjects.

This script:
1. Loads preprocessed data
2. Initializes all diversity metrics
3. Computes metrics for each subject
4. Handles errors gracefully
5. Saves raw scores and error log
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import os
import time
import subprocess
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm

# Set environment variables
os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)


def find_available_gpu():
    """Find an available GPU with low utilization.

    Returns:
        str: GPU ID (e.g., "0", "1") or None if no GPU available
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )

        gpu_info = result.stdout.strip().split('\n')

        # Find GPU with lowest utilization
        best_gpu = None
        lowest_util = 100

        for line in gpu_info:
            parts = line.split(',')
            gpu_id = parts[0].strip()
            utilization = int(parts[1].strip())
            mem_free = int(parts[2].strip())

            # Consider GPU available if utilization < 30% and has at least 10GB free
            if utilization < 30 and mem_free > 10000:
                if utilization < lowest_util:
                    lowest_util = utilization
                    best_gpu = gpu_id

        return best_gpu

    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

from linguistic_diversity import (
    DocumentSemantics,
    TokenSemantics,
    DependencyParse,
    ConstituencyParse,
    PartOfSpeechSequence,
    Rhythmic,
    Phonemic,
)


def load_config():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_metrics(config):
    """Initialize all enabled metrics.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of metric_name -> metric_object
    """
    metrics = {}
    metrics_config = config['metrics']

    print("\nInitializing metrics...")

    # Document-level semantic
    if metrics_config['semantic_document']['enabled']:
        try:
            print("   - Document Semantics...", end=' ')
            metrics['doc_semantic'] = DocumentSemantics({
                'model_name': metrics_config['semantic_document']['model_name'],
                'use_cuda': metrics_config['semantic_document']['use_cuda'],
                'verbose': metrics_config['semantic_document']['verbose'],
            })
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    # Token-level semantic
    if metrics_config['semantic_token']['enabled']:
        try:
            print("   - Token Semantics...", end=' ')
            metrics['token_semantic'] = TokenSemantics({
                'model_name': metrics_config['semantic_token']['model_name'],
                'use_cuda': metrics_config['semantic_token']['use_cuda'],
                'remove_stopwords': metrics_config['semantic_token']['remove_stopwords'],
                'verbose': metrics_config['semantic_token']['verbose'],
            })
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    # Syntactic - Dependency
    if metrics_config['syntactic_dependency']['enabled']:
        try:
            print("   - Dependency Parse...", end=' ')
            metrics['syntactic_dep'] = DependencyParse({
                'similarity_type': metrics_config['syntactic_dependency']['similarity_type'],
                'verbose': metrics_config['syntactic_dependency']['verbose'],
            })
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    # Syntactic - Constituency
    if metrics_config['syntactic_constituency']['enabled']:
        try:
            print("   - Constituency Parse...", end=' ')
            metrics['syntactic_const'] = ConstituencyParse({
                'similarity_type': metrics_config['syntactic_constituency']['similarity_type'],
                'verbose': metrics_config['syntactic_constituency']['verbose'],
            })
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    # Morphological
    if metrics_config['morphological']['enabled']:
        try:
            print("   - Morphological (POS)...", end=' ')
            metrics['morphological'] = PartOfSpeechSequence({
                'verbose': metrics_config['morphological']['verbose'],
            })
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    # Phonological - Phonemic
    if metrics_config['phonological_phonemic']['enabled']:
        try:
            print("   - Phonemic...", end=' ')
            metrics['phonemic'] = Phonemic({
                'backend': metrics_config['phonological_phonemic']['backend'],
                'verbose': metrics_config['phonological_phonemic']['verbose'],
            })
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    # Phonological - Rhythmic
    if metrics_config['phonological_rhythmic']['enabled']:
        try:
            print("   - Rhythmic...", end=' ')
            metrics['rhythmic'] = Rhythmic({
                'verbose': metrics_config['phonological_rhythmic']['verbose'],
            })
            print("✓")
        except Exception as e:
            print(f"✗ ({e})")

    print(f"\n   ✓ Initialized {len(metrics)} metrics")

    return metrics


def compute_lexical_ttr(sentences):
    """Compute Type-Token Ratio (TTR).

    Args:
        sentences: List of sentences

    Returns:
        TTR score (float)
    """
    all_text = ' '.join(sentences).lower()
    tokens = all_text.split()

    if len(tokens) == 0:
        return np.nan

    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)

    return unique_tokens / total_tokens


def main():
    print("=" * 80)
    print("STEP 3: COMPUTE DIVERSITY METRICS")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Find and set available GPU
    print(f"\nGPU Configuration:")
    available_gpu = find_available_gpu()

    if available_gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = available_gpu
        print(f"   ✓ Found available GPU: {available_gpu}")
        print(f"   CUDA_VISIBLE_DEVICES: {available_gpu}")
        use_gpu = True
    else:
        print(f"   ⚠ No available GPU found (all busy or none detected)")
        print(f"   Running on CPU (this will be slower)")
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        use_gpu = False
        # Update config to disable GPU
        config['metrics']['semantic_document']['use_cuda'] = False
        config['metrics']['semantic_token']['use_cuda'] = False

    # Load preprocessed data
    input_path = Path(__file__).parent.parent / "input" / "preprocessed_data.pkl"

    if not input_path.exists():
        print(f"\n✗ Error: Preprocessed data not found: {input_path}")
        print("   Please run: python processing/02_preprocess.py first")
        sys.exit(1)

    print(f"\n1. Loading preprocessed data...")
    df = pd.read_pickle(input_path)
    print(f"   ✓ Loaded {len(df)} subjects")
    print(f"   Dementia: {(df['label'] == 'Dementia').sum()}")
    print(f"   Control: {(df['label'] == 'Control').sum()}")

    # Initialize metrics
    print("\n2. Initializing metrics...")
    metrics = initialize_metrics(config)

    if len(metrics) == 0:
        print("\n✗ Error: No metrics initialized successfully")
        print("   Please check configuration and dependencies")
        sys.exit(1)

    # Compute metrics for each subject
    print(f"\n3. Computing metrics for {len(df)} subjects...")
    print("   (This will take a while...)")
    print("   Progress:")

    results = []
    errors_log = []
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Computing", ncols=80):
        subject_id = row['subject_id']
        label = row['label']
        sentences = row['sentences']
        n_sentences = row['n_sentences']

        scores = {
            'subject_id': subject_id,
            'label': label,
            'n_sentences': n_sentences,
        }

        # Compute each metric
        for metric_name, metric_obj in metrics.items():
            try:
                score = metric_obj(sentences)

                # Check if score is valid
                if score is None or np.isnan(score) or np.isinf(score):
                    raise ValueError(f"Metric returned invalid score: {score}")

                scores[metric_name] = score
            except Exception as e:
                scores[metric_name] = np.nan
                # Get full traceback for debugging
                import traceback
                error_msg = f"{str(e)}\n{traceback.format_exc()}"
                errors_log.append({
                    'subject_id': subject_id,
                    'metric': metric_name,
                    'error': error_msg[:500],  # Store more error info
                })

        # Compute lexical TTR
        if config['metrics']['lexical_ttr']['enabled']:
            try:
                scores['lexical_ttr'] = compute_lexical_ttr(sentences)
            except Exception as e:
                scores['lexical_ttr'] = np.nan
                errors_log.append({
                    'subject_id': subject_id,
                    'metric': 'lexical_ttr',
                    'error': str(e)[:200],
                })

        results.append(scores)

    elapsed_time = time.time() - start_time

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    print(f"\n   ✓ Computation complete!")
    print(f"   Time elapsed: {elapsed_time / 60:.1f} minutes")
    print(f"   Average time per subject: {elapsed_time / len(df):.1f} seconds")

    # Check for missing values
    print("\n4. Data quality check...")
    metric_cols = [col for col in results_df.columns if col not in ['subject_id', 'label', 'n_sentences']]

    missing_counts = results_df[metric_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print("   ⚠ Missing values detected:")
        for metric, count in missing_counts.items():
            if count > 0:
                pct = (count / len(results_df)) * 100
                print(f"      {metric}: {count} ({pct:.1f}%)")
    else:
        print("   ✓ No missing values!")

    # Summary statistics
    print("\n5. Metric summary:")
    print("   " + "-" * 76)
    for metric in metric_cols:
        valid_data = results_df[metric].dropna()
        if len(valid_data) > 0:
            print(f"   {metric}:")
            print(f"      Mean: {valid_data.mean():.3f}, Std: {valid_data.std():.3f}")
            print(f"      Range: [{valid_data.min():.3f}, {valid_data.max():.3f}]")
        else:
            print(f"   {metric}: No valid data")
    print("   " + "-" * 76)

    # Save results
    print("\n6. Saving results...")
    output_dir = Path(__file__).parent.parent / "output" / "scores"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw scores
    scores_path = output_dir / "raw_scores.csv"
    results_df.to_csv(scores_path, index=False)
    print(f"   ✓ Saved raw scores to: {scores_path}")

    # Save error log if there were errors
    if errors_log:
        errors_df = pd.DataFrame(errors_log)
        error_path = output_dir / "error_log.csv"
        errors_df.to_csv(error_path, index=False)
        print(f"   ⚠ Saved error log to: {error_path}")
        print(f"      Total errors: {len(errors_log)}")

    # Save computation metadata
    metadata = {
        'n_subjects': len(df),
        'n_dementia': (df['label'] == 'Dementia').sum(),
        'n_control': (df['label'] == 'Control').sum(),
        'n_metrics': len(metric_cols),
        'computation_time_minutes': elapsed_time / 60,
        'errors': len(errors_log),
    }

    metadata_path = output_dir / "computation_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("COMPUTATION METADATA\n")
        f.write("=" * 80 + "\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"   ✓ Saved metadata to: {metadata_path}")

    print("\n" + "=" * 80)
    print("✓ METRIC COMPUTATION COMPLETE")
    print("=" * 80)
    print(f"\nComputed {len(metric_cols)} metrics for {len(results_df)} subjects")
    print(f"Time elapsed: {elapsed_time / 60:.1f} minutes")
    if errors_log:
        print(f"⚠ {len(errors_log)} computation errors (see error_log.csv)")
    print("\nNext step:")
    print("  python processing/04_analyze_results.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
