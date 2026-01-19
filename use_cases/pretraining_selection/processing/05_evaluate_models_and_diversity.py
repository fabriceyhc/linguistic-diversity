#!/usr/bin/env python3
"""Combined evaluation: corpus diversity + model perplexity.

This script:
1. Measures corpus diversity using scaled estimation (memory efficient, no GPU)
2. Evaluates trained models one at a time (GPU, aggressive memory cleanup)
3. Combines results and generates summary

Usage: CUDA_VISIBLE_DEVICES=0 python 05_evaluate_all.py
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import os
import gc
import json
import pickle
import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional

# Set environment variables
os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)


def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_config():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_corpus_texts(datasets_dir: Path, regime_name: str, dataset_name: str) -> list:
    """Load corpus texts for a regime."""
    clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()
    corpus_file = datasets_dir / regime_name / clean_name / "corpus.jsonl"

    if not corpus_file.exists():
        return []

    documents = []
    with open(corpus_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            documents.append(data['text'])

    return documents


def load_test_data(input_dir: Path, dataset_name: str, max_docs: int = 30) -> list:
    """Load a small subset of test dataset."""
    clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()
    test_file = input_dir / f"{clean_name}_test.pkl"

    if not test_file.exists():
        return []

    with open(test_file, 'rb') as f:
        test_docs = pickle.load(f)

    return test_docs[:max_docs]


# =============================================================================
# PART 1: Diversity Measurement (CPU only, uses scaled estimation)
# =============================================================================

def measure_all_diversity(datasets_dir: Path, dataset_name: str, regimes: list) -> Dict[str, dict]:
    """Measure diversity for all regimes using scaled estimation.

    Returns dict mapping regime -> diversity results.
    """
    print(f"\n{'=' * 80}")
    print("PART 1: MEASURING CORPUS DIVERSITY (Scaled Estimation)")
    print(f"{'=' * 80}")
    print("Using DiversityMetric.estimate_diversity() for memory-efficient measurement\n")

    from linguistic_diversity import UniversalLinguisticDiversity

    diversity_results = {}

    for regime in regimes:
        print(f"\n   [{regimes.index(regime)+1}/{len(regimes)}] {regime}")

        corpus = load_corpus_texts(datasets_dir, regime, dataset_name)

        if not corpus:
            print(f"      ⚠ Corpus not found")
            diversity_results[regime] = {'error': 'Corpus not found'}
            continue

        print(f"      Corpus size: {len(corpus)} documents")

        try:
            metric = UniversalLinguisticDiversity({
                'strategy': 'hierarchical',
                'verbose': False,
                'use_constituency_parse': False,
            })

            # Use scaled estimation with tiny samples for speed
            # Sample sizes: 5 -> 10 -> 20 -> 40 (4 sizes × 2 trials = 8 measurements)
            result = metric.estimate_diversity(
                corpus,
                base_sample_size=5,
                max_sample_size=40,
                num_trials=2,
                growth_factor=2.0,
                verbose=True,  # Show progress
            )

            diversity_results[regime] = {
                'universal_diversity': result.diversity,
                'diversity_std': result.std,
                'projected_uncertainty_95': list(result.projected_uncertainty_95),
                'method': result.method,
                'model': result.model,
                'model_params': result.model_params,
                'corpus_size': len(corpus),
                'fit_rmse': result.fit_rmse,
            }

            print(f"      ✓ Final: {result.diversity:.4f} ± {result.std:.4f} ({result.method})")

            del metric, result

        except Exception as e:
            print(f"      ✗ Error: {e}")
            diversity_results[regime] = {'error': str(e)}

        del corpus
        clear_memory()

    return diversity_results


# =============================================================================
# PART 2: Model Evaluation (GPU, one model at a time)
# =============================================================================

def evaluate_single_model(model_dir: Path, model_type: str, test_docs: list, max_length: int = 256) -> dict:
    """Evaluate a single model with minimal memory usage."""

    from model_loader import load_model, MODEL_REGISTRY

    model_info = MODEL_REGISTRY[model_type]
    results = {'model_type': model_type, 'model_id': model_info.model_id}

    try:
        tokenizer, model, model_config = load_model(model_type)
        device = next(model.parameters()).device
        model.eval()
    except Exception as e:
        return {'error': f"Failed to load model: {e}"}

    # Find checkpoint
    checkpoint_path = model_dir / "model_final.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "pytorch_model.bin"
    if not checkpoint_path.exists():
        checkpoints = list(model_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            checkpoint_path = sorted(checkpoints)[-1]

    if not checkpoint_path.exists():
        del model, tokenizer
        clear_memory()
        return {'error': f"No checkpoint found"}

    # Load weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        del checkpoint
        clear_memory()
    except Exception as e:
        del model, tokenizer
        clear_memory()
        return {'error': f"Failed to load checkpoint: {e}"}

    # Evaluate perplexity
    total_loss = 0.0
    total_tokens = 0
    mlm_probability = 0.15

    with torch.no_grad():
        for doc in tqdm(test_docs, desc="      Eval", leave=False):
            try:
                if model_type == "encoder":
                    inputs = tokenizer(doc, padding=True, truncation=True,
                                       max_length=max_length, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    labels = input_ids.clone()

                    probability_matrix = torch.full(labels.shape, mlm_probability)
                    special_tokens_mask = tokenizer.get_special_tokens_mask(
                        labels[0].tolist(), already_has_special_tokens=True)
                    special_tokens_mask = torch.tensor([special_tokens_mask], dtype=torch.bool)
                    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
                    masked_indices = torch.bernoulli(probability_matrix).bool()
                    labels[~masked_indices] = -100

                    if tokenizer.mask_token_id is not None:
                        input_ids_masked = input_ids.clone()
                        input_ids_masked[masked_indices] = tokenizer.mask_token_id
                    else:
                        input_ids_masked = input_ids

                    outputs = model(input_ids=input_ids_masked, attention_mask=attention_mask, labels=labels.to(device))
                    if outputs.loss is not None:
                        num_masked = masked_indices.sum().item()
                        if num_masked > 0:
                            total_loss += outputs.loss.item() * num_masked
                            total_tokens += num_masked

                elif model_type == "decoder":
                    inputs = tokenizer(doc, padding=True, truncation=True,
                                       max_length=max_length, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    labels = input_ids.clone()
                    if tokenizer.pad_token_id is not None:
                        labels[labels == tokenizer.pad_token_id] = -100

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    if outputs.loss is not None:
                        num_tokens = (labels != -100).sum().item()
                        total_loss += outputs.loss.item() * num_tokens
                        total_tokens += num_tokens

                elif model_type == "encoder-decoder":
                    inputs = tokenizer(f"denoise: {doc}", text_target=doc,
                                       padding=True, truncation=True,
                                       max_length=max_length, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    labels = inputs["labels"].to(device)
                    if tokenizer.pad_token_id is not None:
                        labels[labels == tokenizer.pad_token_id] = -100

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    if outputs.loss is not None:
                        num_tokens = (labels != -100).sum().item()
                        total_loss += outputs.loss.item() * num_tokens
                        total_tokens += num_tokens

            except Exception as e:
                continue

    # Calculate perplexity
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
    else:
        avg_loss = float('inf')
        perplexity = float('inf')

    if model_type == "encoder":
        results['pseudo_perplexity'] = perplexity
    else:
        results['perplexity'] = perplexity
    results['avg_loss'] = avg_loss
    results['total_tokens'] = total_tokens

    del model, tokenizer
    clear_memory()

    return results


def evaluate_all_models(
    models_dir: Path,
    input_dir: Path,
    datasets_dir: Path,
    dataset_name: str,
    regimes: list,
    model_types: list,
    training_results: dict,
    diversity_results: Dict[str, dict],
) -> list:
    """Evaluate all models one at a time."""

    print(f"\n{'=' * 80}")
    print("PART 2: EVALUATING MODELS")
    print(f"{'=' * 80}")

    clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()

    # Load test data
    test_docs = load_test_data(input_dir, dataset_name, max_docs=30)
    if test_docs:
        print(f"\n   Loaded {len(test_docs)} test documents")
    else:
        test_docs = load_corpus_texts(datasets_dir, 'random_baseline', dataset_name)[:30]
        if test_docs:
            print(f"\n   Using {len(test_docs)} corpus documents as test set")

    all_results = []
    total_models = len(regimes) * len(model_types)
    current = 0

    for regime in regimes:
        for model_type in model_types:
            current += 1
            model_dir = models_dir / clean_name / regime / model_type

            print(f"\n   [{current}/{total_models}] {regime} / {model_type}")

            if not model_dir.exists():
                print(f"      ✗ Model not found")
                continue

            results = evaluate_single_model(model_dir, model_type, test_docs)
            results['dataset_name'] = dataset_name
            results['regime'] = regime

            # Add training loss
            if dataset_name in training_results:
                if regime in training_results[dataset_name]:
                    if model_type in training_results[dataset_name][regime]:
                        train_info = training_results[dataset_name][regime][model_type]
                        results['training_loss'] = train_info.get('final_loss')
                        results['training_steps'] = train_info.get('total_steps')

            # Add diversity info
            if regime in diversity_results and 'universal_diversity' in diversity_results[regime]:
                results['corpus_diversity'] = diversity_results[regime]['universal_diversity']
                results['diversity_std'] = diversity_results[regime].get('diversity_std')
                results['diversity_method'] = diversity_results[regime].get('method')

            if 'error' not in results:
                ppl_key = 'pseudo_perplexity' if model_type == 'encoder' else 'perplexity'
                print(f"      ✓ {ppl_key}: {results.get(ppl_key, 'N/A'):.2f}")
            else:
                print(f"      ✗ {results['error']}")

            all_results.append(results)
            clear_memory()

    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("STEP 5: EVALUATE MODELS + MEASURE DIVERSITY")
    print("=" * 80)

    config = load_config()

    # Setup directories
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "input"
    datasets_dir = base_dir / "datasets"
    models_dir = base_dir / "models"
    output_dir = base_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training results
    training_results_file = models_dir / "training_results.json"
    training_results = {}
    if training_results_file.exists():
        with open(training_results_file, 'r') as f:
            training_results = json.load(f)
        print(f"\n   ✓ Loaded training results")

    model_types = ['encoder', 'decoder', 'encoder-decoder']
    regimes = [
        'semantic_diversity',
        'syntactic_diversity',
        'morphological_diversity',
        'phonological_diversity',
        # 'composite_diversity',  # Disabled - redundant with universal
        'universal_diversity',
        'random_baseline',
        'full_dataset',  # No subsampling - train on all data
    ]

    all_eval_results = []
    all_diversity_results = []

    for dataset_config in config['corpus']['datasets']:
        dataset_name = dataset_config['name']

        print(f"\n{'#' * 80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#' * 80}")

        # PART 1: Measure diversity (CPU only)
        diversity_results = measure_all_diversity(datasets_dir, dataset_name, regimes)

        # Store diversity results
        for regime, div_data in diversity_results.items():
            div_record = {'dataset_name': dataset_name, 'regime': regime}
            div_record.update(div_data)
            all_diversity_results.append(div_record)

        # PART 2: Evaluate models (GPU)
        eval_results = evaluate_all_models(
            models_dir, input_dir, datasets_dir, dataset_name,
            regimes, model_types, training_results, diversity_results
        )
        all_eval_results.extend(eval_results)

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================

    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")

    # Save evaluation results
    eval_file = output_dir / "evaluation_results.json"
    with open(eval_file, 'w') as f:
        json.dump(all_eval_results, f, indent=2, default=str)
    print(f"   ✓ Saved: {eval_file}")

    # Save diversity results
    diversity_file = output_dir / "diversity_results.json"
    with open(diversity_file, 'w') as f:
        json.dump(all_diversity_results, f, indent=2, default=str)
    print(f"   ✓ Saved: {diversity_file}")

    # Create summary CSV
    import pandas as pd
    summary_data = []
    for r in all_eval_results:
        if 'error' in r:
            continue
        row = {
            'Dataset': r.get('dataset_name', ''),
            'Regime': r.get('regime', ''),
            'Model': r.get('model_type', ''),
            'Perplexity': r.get('perplexity', r.get('pseudo_perplexity', None)),
            'Eval Loss': r.get('avg_loss', None),
            'Train Loss': r.get('training_loss', None),
            'Corpus Diversity': r.get('corpus_diversity', None),
            'Diversity Std': r.get('diversity_std', None),
        }
        summary_data.append(row)

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"   ✓ Saved: {summary_file}")

        print(f"\n{'=' * 80}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 80}\n")
        print(summary_df.to_string(index=False))

    # Print diversity summary
    print(f"\n{'=' * 80}")
    print("DIVERSITY SUMMARY")
    print(f"{'=' * 80}\n")
    print(f"{'Regime':<30} {'Diversity':>12} {'± Std':>10} {'Method':>15}")
    print("-" * 70)
    for r in all_diversity_results:
        ud = r.get('universal_diversity')
        std = r.get('diversity_std', 0)
        method = r.get('method', 'N/A')
        if ud is not None:
            print(f"{r['regime']:<30} {ud:>12.4f} {std:>10.4f} {method:>15}")
        else:
            print(f"{r['regime']:<30} {'ERROR':>12} {'-':>10} {r.get('error', '')[:15]:>15}")

    print("\n" + "=" * 80)
    print("✓ EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
