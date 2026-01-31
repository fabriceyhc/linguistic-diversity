#!/usr/bin/env python3
"""Run the Universal Embedding Diversity experiment.

This script runs only the new universal embedding diversity selection
and its subsequent training/evaluation steps, without re-running
the existing selection regimes.

Usage:
    python run_universal_embedding_experiment.py [--select-only] [--train-only] [--eval-only]

Options:
    --select-only   Only run the selection step
    --train-only    Only run the training step (assumes selection is done)
    --eval-only     Only run the evaluation step (assumes training is done)
    --all           Run all steps (default)
"""

import sys
import os
from pathlib import Path
import argparse

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set environment variables
os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)


def run_selection():
    """Run only the universal embedding diversity selection."""
    print("=" * 80)
    print("STEP 1: UNIVERSAL EMBEDDING DIVERSITY SELECTION")
    print("=" * 80)

    import json
    import pickle
    import yaml
    import numpy as np
    from tqdm import tqdm

    from linguistic_diversity import (
        UniversalLinguisticDiversity,
        DIVERSITY_EMBEDDING_METRICS,
        select_diverse_texts,
    )

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    base_dir = Path(__file__).parent
    input_dir = base_dir / "input"
    datasets_dir = base_dir / "datasets"

    selection_ratio = config['selection']['selection_ratio']
    shard_size = config['selection']['shard_size']

    # Get embedding config
    embedding_config = config.get('selection', {}).get('universal_embedding', {})
    selection_method = embedding_config.get('selection_method', 'facility_location')

    for dataset_config in config['corpus']['datasets']:
        dataset_name = dataset_config['name']
        clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()

        print(f"\n{'=' * 80}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 80}")

        # Load documents
        train_file = input_dir / f"{clean_name}_train.pkl"
        if not train_file.exists():
            print(f"   Train file not found: {train_file}")
            print(f"   Run 01_download_data.py and 02_extract_features.py first")
            continue

        with open(train_file, 'rb') as f:
            documents = pickle.load(f)

        n_items = len(documents)
        n_select = int(n_items * selection_ratio)

        print(f"   Total documents: {n_items}")
        print(f"   Selection target: {n_select} ({selection_ratio:.0%})")

        # Initialize metric
        metric = UniversalLinguisticDiversity({
            'use_constituency_parse': embedding_config.get('use_constituency_parse', False),
            'use_rhythmic': embedding_config.get('use_rhythmic', False),
            'use_phonemic': embedding_config.get('use_phonemic', False),
            'verbose': False,
        })

        print(f"\n   Computing diversity embeddings...")

        # Compute embeddings in batches
        batch_size = 100
        n_metrics = len(DIVERSITY_EMBEDDING_METRICS)
        embeddings = np.zeros((n_items, n_metrics), dtype=np.float64)

        for i in tqdm(range(0, n_items, batch_size), desc="   Computing embeddings"):
            batch_end = min(i + batch_size, n_items)
            for j in range(i, batch_end):
                doc = documents[j]
                if doc and doc.strip():
                    try:
                        emb = metric._compute_single_doc_embedding([doc], normalize=True)
                        embeddings[j] = emb
                    except Exception:
                        pass

        print(f"   Embeddings shape: {embeddings.shape}")
        print(f"   Running {selection_method} selection...")

        # For large datasets, use hierarchical selection
        if n_items > shard_size:
            print(f"   Using hierarchical selection...")
            # Stage 1: Select from shards
            n_shards = (n_items + shard_size - 1) // shard_size
            per_shard_select = max(1, (n_select * 2) // n_shards)

            shard_winners = []
            shard_winner_indices = []

            for shard_idx in tqdm(range(n_shards), desc="   Shard selection"):
                start_idx = shard_idx * shard_size
                end_idx = min((shard_idx + 1) * shard_size, n_items)

                shard_embeddings = embeddings[start_idx:end_idx]

                try:
                    k = min(per_shard_select, end_idx - start_idx)
                    result = select_diverse_texts(
                        shard_embeddings,
                        n_select=k,
                        method=selection_method,
                        verbose=False,
                    )

                    global_indices = start_idx + result.indices
                    shard_winner_indices.extend(global_indices.tolist())

                    for local_idx in result.indices:
                        shard_winners.append(shard_embeddings[local_idx])
                except Exception as e:
                    print(f"   Warning: Shard {shard_idx} failed: {e}")

            # Stage 2: Final selection
            winner_embeddings = np.array(shard_winners)
            result = select_diverse_texts(
                winner_embeddings,
                n_select=min(n_select, len(shard_winner_indices)),
                method=selection_method,
                verbose=True,
            )

            selected_indices = np.array([shard_winner_indices[i] for i in result.indices])
        else:
            result = select_diverse_texts(
                embeddings,
                n_select=n_select,
                method=selection_method,
                verbose=True,
            )
            selected_indices = result.indices

        print(f"   Selected {len(selected_indices)} documents")
        print(f"   Coverage per metric: {result.coverage_per_metric}")

        # Save selection
        regime_dir = datasets_dir / "universal_embedding_diversity" / clean_name
        regime_dir.mkdir(parents=True, exist_ok=True)

        # Save indices
        indices_file = regime_dir / "selected_indices.json"
        with open(indices_file, 'w') as f:
            json.dump(selected_indices.tolist(), f)
        print(f"   Saved indices: {indices_file}")

        # Save corpus
        selected_docs = [documents[i] for i in selected_indices]
        corpus_file = regime_dir / "corpus.jsonl"
        with open(corpus_file, 'w') as f:
            for doc in selected_docs:
                json.dump({"text": doc}, f)
                f.write('\n')
        print(f"   Saved corpus: {corpus_file}")

        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'regime': 'universal_embedding_diversity',
            'n_selected': len(selected_indices),
            'selection_method': selection_method,
            'coverage_per_metric': result.coverage_per_metric.tolist(),
            'metrics': DIVERSITY_EMBEDDING_METRICS,
        }

        metadata_file = regime_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   Saved metadata: {metadata_file}")

    print("\n" + "=" * 80)
    print("SELECTION COMPLETE")
    print("=" * 80)


def run_training():
    """Run training for universal embedding diversity regime only."""
    print("=" * 80)
    print("STEP 2: TRAINING MODELS")
    print("=" * 80)

    import subprocess
    result = subprocess.run(
        ['python', 'processing/04_train_models.py'],
        cwd=Path(__file__).parent
    )
    return result.returncode == 0


def run_evaluation():
    """Run evaluation for universal embedding diversity regime only."""
    print("=" * 80)
    print("STEP 3: EVALUATION")
    print("=" * 80)

    import subprocess

    # Run diversity and perplexity evaluation
    print("\nRunning diversity and perplexity evaluation...")
    subprocess.run(
        ['python', 'processing/05_evaluate_models_and_diversity.py'],
        cwd=Path(__file__).parent
    )

    # Run GLUE evaluation (encoder)
    print("\nRunning GLUE evaluation...")
    subprocess.run(
        ['python', 'processing/06b_evaluate_encoder_glue.py'],
        cwd=Path(__file__).parent
    )

    # Run decoder benchmarks
    print("\nRunning decoder benchmarks...")
    subprocess.run(
        ['python', 'processing/06c_evaluate_decoder_lmeval.py'],
        cwd=Path(__file__).parent
    )

    # Run encoder-decoder tasks
    print("\nRunning encoder-decoder tasks...")
    subprocess.run(
        ['python', 'processing/06d_evaluate_encdec_tasks.py'],
        cwd=Path(__file__).parent
    )

    # Generate report
    print("\nGenerating report...")
    subprocess.run(
        ['python', 'processing/07_generate_report.py'],
        cwd=Path(__file__).parent
    )


def main():
    parser = argparse.ArgumentParser(description='Run Universal Embedding Diversity experiment')
    parser.add_argument('--select-only', action='store_true', help='Only run selection')
    parser.add_argument('--train-only', action='store_true', help='Only run training')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--all', action='store_true', help='Run all steps (default)')

    args = parser.parse_args()

    # Default to all if no specific option given
    run_all = args.all or not (args.select_only or args.train_only or args.eval_only)

    if args.select_only or run_all:
        run_selection()

    if args.train_only or run_all:
        run_training()

    if args.eval_only or run_all:
        run_evaluation()

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
