#!/usr/bin/env python3
"""Evaluate decoder models on standard LM benchmarks.

This script:
1. Loads trained decoder models (GPT-style)
2. Evaluates on commonsense/reasoning benchmarks
3. Reports accuracy for each task

Benchmarks:
- HellaSwag: Commonsense sentence completion
- ARC-Easy: Science reasoning (easy)
- ARC-Challenge: Science reasoning (hard)
- PIQA: Physical intuition
- WinoGrande: Coreference resolution
- BoolQ: Yes/no QA

Uses lm-evaluation-harness or manual implementation.

Usage: CUDA_VISIBLE_DEVICES=0 python 06c_evaluate_decoder_lmeval.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import os
import gc
import json
import yaml
import torch
import numpy as np
from typing import Dict, List, Optional
from tqdm import tqdm

os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_trained_decoder(model_dir: Path, device: str = 'cuda'):
    """Load a trained decoder model."""
    from model_loader import load_model

    tokenizer, model, model_config = load_model('decoder')

    checkpoint_path = model_dir / "model_final.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "pytorch_model.bin"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint in {model_dir}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return tokenizer, model


def load_pretrained_decoder(device: str = 'cuda'):
    """Load the original pretrained decoder (no additional pretraining)."""
    from model_loader import load_model

    tokenizer, model, model_config = load_model('decoder')
    model = model.to(device)
    model.eval()

    return tokenizer, model


def score_completion(model, tokenizer, context: str, completion: str, device: str) -> float:
    """Score a completion given context using log probability."""
    full_text = context + completion
    context_ids = tokenizer.encode(context, return_tensors='pt').to(device)
    full_ids = tokenizer.encode(full_text, return_tensors='pt').to(device)

    # Get completion token positions
    context_len = context_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    # Calculate log prob of completion tokens
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    completion_log_prob = 0.0
    for i in range(context_len, full_ids.shape[1]):
        token_id = full_ids[0, i]
        completion_log_prob += log_probs[0, i - 1, token_id].item()

    # Normalize by length
    completion_len = full_ids.shape[1] - context_len
    if completion_len > 0:
        completion_log_prob /= completion_len

    return completion_log_prob


def evaluate_hellaswag(model, tokenizer, device: str, max_samples: Optional[int] = None, seed: int = 42) -> Dict:
    """Evaluate on HellaSwag benchmark."""
    from datasets import load_dataset

    print(f"\n      Task: HellaSwag (seed={seed})")

    dataset = load_dataset('hellaswag', trust_remote_code=True)
    eval_data = dataset['validation']

    if max_samples is not None and len(eval_data) > max_samples:
        eval_data = eval_data.shuffle(seed=seed).select(range(max_samples))

    print(f"      Samples: {len(eval_data)}")

    correct = 0
    total = 0

    for item in tqdm(eval_data, desc="      Evaluating", leave=False):
        context = item['ctx']
        endings = item['endings']
        label = int(item['label'])

        # Score each ending
        scores = []
        for ending in endings:
            score = score_completion(model, tokenizer, context, " " + ending, device)
            scores.append(score)

        pred = np.argmax(scores)
        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"      ✓ Accuracy: {accuracy:.4f}")

    return {'task': 'hellaswag', 'accuracy': accuracy, 'samples': total}


def evaluate_arc(model, tokenizer, device: str, split: str = 'easy', max_samples: Optional[int] = None, seed: int = 42) -> Dict:
    """Evaluate on ARC benchmark."""
    from datasets import load_dataset

    task_name = f"ARC-{split.capitalize()}"
    print(f"\n      Task: {task_name} (seed={seed})")

    dataset_name = f"ARC-{'Easy' if split == 'easy' else 'Challenge'}"
    dataset = load_dataset('ai2_arc', dataset_name, trust_remote_code=True)
    eval_data = dataset['validation']

    if max_samples is not None and len(eval_data) > max_samples:
        eval_data = eval_data.shuffle(seed=seed).select(range(max_samples))

    print(f"      Samples: {len(eval_data)}")

    correct = 0
    total = 0

    for item in tqdm(eval_data, desc="      Evaluating", leave=False):
        question = item['question']
        choices = item['choices']
        answer_key = item['answerKey']

        # Map answer key to index
        labels = choices['label']
        texts = choices['text']

        try:
            label_idx = labels.index(answer_key)
        except ValueError:
            # Sometimes answer_key is numeric
            label_idx = int(answer_key) - 1 if answer_key.isdigit() else 0

        # Score each choice
        scores = []
        for choice_text in texts:
            context = f"Question: {question}\nAnswer:"
            score = score_completion(model, tokenizer, context, " " + choice_text, device)
            scores.append(score)

        pred = np.argmax(scores)
        if pred == label_idx:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"      ✓ Accuracy: {accuracy:.4f}")

    return {'task': f'arc_{split}', 'accuracy': accuracy, 'samples': total}


def evaluate_piqa(model, tokenizer, device: str, max_samples: Optional[int] = None, seed: int = 42) -> Dict:
    """Evaluate on PIQA benchmark."""
    from datasets import load_dataset

    print(f"\n      Task: PIQA (seed={seed})")

    dataset = load_dataset('piqa', trust_remote_code=True)
    eval_data = dataset['validation']

    if max_samples is not None and len(eval_data) > max_samples:
        eval_data = eval_data.shuffle(seed=seed).select(range(max_samples))

    print(f"      Samples: {len(eval_data)}")

    correct = 0
    total = 0

    for item in tqdm(eval_data, desc="      Evaluating", leave=False):
        goal = item['goal']
        sol1 = item['sol1']
        sol2 = item['sol2']
        label = item['label']

        context = f"Goal: {goal}\nSolution:"

        score1 = score_completion(model, tokenizer, context, " " + sol1, device)
        score2 = score_completion(model, tokenizer, context, " " + sol2, device)

        pred = 0 if score1 > score2 else 1
        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"      ✓ Accuracy: {accuracy:.4f}")

    return {'task': 'piqa', 'accuracy': accuracy, 'samples': total}


def evaluate_winogrande(model, tokenizer, device: str, max_samples: Optional[int] = None, seed: int = 42) -> Dict:
    """Evaluate on WinoGrande benchmark."""
    from datasets import load_dataset

    print(f"\n      Task: WinoGrande (seed={seed})")

    dataset = load_dataset('winogrande', 'winogrande_m', trust_remote_code=True)
    eval_data = dataset['validation']

    if max_samples is not None and len(eval_data) > max_samples:
        eval_data = eval_data.shuffle(seed=seed).select(range(max_samples))

    print(f"      Samples: {len(eval_data)}")

    correct = 0
    total = 0

    for item in tqdm(eval_data, desc="      Evaluating", leave=False):
        sentence = item['sentence']
        option1 = item['option1']
        option2 = item['option2']
        label = int(item['answer']) - 1  # 1-indexed to 0-indexed

        # Replace _ with each option
        sent1 = sentence.replace('_', option1)
        sent2 = sentence.replace('_', option2)

        # Score full sentences
        score1 = score_completion(model, tokenizer, "", sent1, device)
        score2 = score_completion(model, tokenizer, "", sent2, device)

        pred = 0 if score1 > score2 else 1
        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"      ✓ Accuracy: {accuracy:.4f}")

    return {'task': 'winogrande', 'accuracy': accuracy, 'samples': total}


def evaluate_boolq(model, tokenizer, device: str, max_samples: Optional[int] = None, seed: int = 42) -> Dict:
    """Evaluate on BoolQ benchmark."""
    from datasets import load_dataset

    print(f"\n      Task: BoolQ (seed={seed})")

    dataset = load_dataset('boolq', trust_remote_code=True)
    eval_data = dataset['validation']

    if max_samples is not None and len(eval_data) > max_samples:
        eval_data = eval_data.shuffle(seed=seed).select(range(max_samples))

    print(f"      Samples: {len(eval_data)}")

    correct = 0
    total = 0

    for item in tqdm(eval_data, desc="      Evaluating", leave=False):
        passage = item['passage']
        question = item['question']
        label = item['answer']  # True or False

        context = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

        score_yes = score_completion(model, tokenizer, context, " Yes", device)
        score_no = score_completion(model, tokenizer, context, " No", device)

        pred = score_yes > score_no  # True if yes, False if no
        if pred == label:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"      ✓ Accuracy: {accuracy:.4f}")

    return {'task': 'boolq', 'accuracy': accuracy, 'samples': total}


def evaluate_decoder_on_benchmarks(
    model_dir: Optional[Path],
    regime: str,
    device: str = 'cuda',
    max_samples: Optional[int] = 500,  # Use subsets for multi-seed evaluation
    num_seeds: int = 3,
    seeds: List[int] = None,
    use_pretrained: bool = False,
) -> Dict:
    """Evaluate a decoder model on all benchmarks with multiple seeds for statistical significance."""

    if seeds is None:
        seeds = [42, 123, 456][:num_seeds]

    print(f"\n   {'=' * 60}")
    print(f"   Evaluating: {regime}")
    print(f"   Seeds: {seeds}, Samples per seed: {max_samples}")
    print(f"   {'=' * 60}")

    try:
        if use_pretrained:
            tokenizer, model = load_pretrained_decoder(device)
            print(f"   ✓ Loaded original pretrained model (no additional pretraining)")
        else:
            tokenizer, model = load_trained_decoder(model_dir, device)
            print(f"   ✓ Loaded model from {model_dir}")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return {'regime': regime, 'error': str(e)}

    results = {
        'regime': regime,
        'tasks': {},
        'num_seeds': num_seeds,
        'seeds': seeds,
    }

    benchmark_names = ['hellaswag', 'arc_easy', 'piqa', 'boolq']

    for bench_name in benchmark_names:
        task_scores = []

        for seed in seeds:
            try:
                if bench_name == 'hellaswag':
                    task_result = evaluate_hellaswag(model, tokenizer, device, max_samples, seed)
                elif bench_name == 'arc_easy':
                    task_result = evaluate_arc(model, tokenizer, device, 'easy', max_samples, seed)
                elif bench_name == 'piqa':
                    task_result = evaluate_piqa(model, tokenizer, device, max_samples, seed)
                elif bench_name == 'boolq':
                    task_result = evaluate_boolq(model, tokenizer, device, max_samples, seed)

                task_scores.append(task_result['accuracy'])
            except Exception as e:
                print(f"      ✗ {bench_name} (seed={seed}) failed: {e}")

            clear_memory()

        if task_scores:
            results['tasks'][bench_name] = {
                'task': bench_name,
                'scores': task_scores,
                'mean': float(np.mean(task_scores)),
                'std': float(np.std(task_scores)),
                'min': float(np.min(task_scores)),
                'max': float(np.max(task_scores)),
            }
            print(f"      {bench_name.upper()}: {results['tasks'][bench_name]['mean']:.4f} ± {results['tasks'][bench_name]['std']:.4f}")
        else:
            results['tasks'][bench_name] = {'error': 'All seeds failed'}

    # Calculate average across tasks
    task_means = [t['mean'] for t in results['tasks'].values() if 'mean' in t]
    if task_means:
        results['average_accuracy'] = float(np.mean(task_means))
        # Compute std of averages per seed
        avg_per_seed = []
        for i in range(num_seeds):
            seed_scores = [t['scores'][i] for t in results['tasks'].values() if 'scores' in t and len(t['scores']) > i]
            if seed_scores:
                avg_per_seed.append(np.mean(seed_scores))
        if avg_per_seed:
            results['average_std'] = float(np.std(avg_per_seed))
        print(f"\n   Average accuracy: {results['average_accuracy']:.4f} ± {results.get('average_std', 0):.4f}")

    del model, tokenizer
    clear_memory()

    return results


def main():
    print("=" * 80)
    print("STEP 6c: DECODER BENCHMARK EVALUATION")
    print("=" * 80)

    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n   Device: {device}")

    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    output_dir = base_dir / "output"

    regimes = [
        'pretrained_baseline',  # Original pretrained model (no additional pretraining)
        'semantic_diversity',
        'syntactic_diversity',
        'morphological_diversity',
        'phonological_diversity',
        'universal_diversity',
        'random_baseline',
        'full_dataset',  # No subsampling - train on all data
    ]

    # Load existing results to skip completed regimes
    results_file = output_dir / "decoder_benchmark_results.json"
    existing_results = []
    completed_regimes = set()

    if results_file.exists():
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        for r in existing_results:
            if 'error' not in r and 'tasks' in r:
                tasks = r['tasks']
                if any('mean' in t for t in tasks.values()):
                    completed_regimes.add((r.get('dataset', ''), r['regime']))
        print(f"   Found {len(completed_regimes)} completed regimes, will skip them")

    all_results = [r for r in existing_results if (r.get('dataset', ''), r['regime']) in completed_regimes]

    for dataset_config in config['corpus']['datasets']:
        dataset_name = dataset_config['name']
        clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()

        print(f"\n{'#' * 80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#' * 80}")

        for regime in regimes:
            if (dataset_name, regime) in completed_regimes:
                print(f"\n   ✓ Skipping (already evaluated): {regime}")
                continue

            # Handle pretrained baseline specially
            if regime == 'pretrained_baseline':
                results = evaluate_decoder_on_benchmarks(
                    model_dir=None,
                    regime=regime,
                    device=device,
                    max_samples=500,
                    num_seeds=3,
                    use_pretrained=True,
                )
                results['dataset'] = dataset_name
                all_results.append(results)
                continue

            model_dir = models_dir / clean_name / regime / "decoder"

            if not model_dir.exists():
                print(f"\n   ✗ Model not found: {model_dir}")
                continue

            results = evaluate_decoder_on_benchmarks(
                model_dir=model_dir,
                regime=regime,
                device=device,
                max_samples=500,  # Use subsets for multi-seed statistical evaluation
                num_seeds=3,
            )
            results['dataset'] = dataset_name
            all_results.append(results)

    # Save results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")

    results_file = output_dir / "decoder_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"   ✓ Saved: {results_file}")

    # Print summary with mean ± std
    print(f"\n{'=' * 80}")
    print("DECODER BENCHMARK SUMMARY (mean ± std across seeds)")
    print(f"{'=' * 80}\n")

    def fmt_score(task_data):
        if 'mean' in task_data:
            return f"{task_data['mean']:.3f}±{task_data['std']:.3f}"
        return "N/A"

    print(f"{'Regime':<25} {'HellaSwag':>12} {'ARC-E':>12} {'PIQA':>12} {'BoolQ':>12} {'Avg':>12}")
    print("-" * 95)

    for r in all_results:
        if 'error' in r:
            continue
        regime = r['regime']
        tasks = r.get('tasks', {})
        hs = fmt_score(tasks.get('hellaswag', {}))
        arc = fmt_score(tasks.get('arc_easy', {}))
        piqa = fmt_score(tasks.get('piqa', {}))
        boolq = fmt_score(tasks.get('boolq', {}))
        avg = f"{r.get('average_accuracy', 0):.3f}±{r.get('average_std', 0):.3f}"
        print(f"{regime:<25} {hs:>12} {arc:>12} {piqa:>12} {boolq:>12} {avg:>12}")

    # Statistical significance tests
    print(f"\n{'=' * 80}")
    print("STATISTICAL SIGNIFICANCE (paired t-test vs random_baseline)")
    print(f"{'=' * 80}\n")

    from scipy.stats import ttest_rel

    # Find random baseline
    baseline_result = next((r for r in all_results if r['regime'] == 'random_baseline' and 'error' not in r), None)
    benchmark_names = ['hellaswag', 'arc_easy', 'piqa', 'boolq']

    if baseline_result:
        print(f"{'Regime':<25} {'Task':>12} {'Diff':>10} {'p-value':>10} {'Sig?':>8}")
        print("-" * 70)

        for r in all_results:
            if 'error' in r or r['regime'] == 'random_baseline':
                continue

            regime = r['regime']
            tasks = r.get('tasks', {})

            for task_name in benchmark_names:
                if task_name not in tasks or task_name not in baseline_result['tasks']:
                    continue

                regime_scores = tasks[task_name].get('scores', [])
                baseline_scores = baseline_result['tasks'][task_name].get('scores', [])

                if len(regime_scores) >= 2 and len(baseline_scores) >= 2:
                    min_len = min(len(regime_scores), len(baseline_scores))
                    t_stat, p_value = ttest_rel(regime_scores[:min_len], baseline_scores[:min_len])
                    diff = np.mean(regime_scores) - np.mean(baseline_scores)
                    sig = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
                    print(f"{regime:<25} {task_name:>12} {diff:>+10.4f} {p_value:>10.4f} {sig:>8}")
    else:
        print("   No random baseline found for comparison.")

    print("\n   * p < 0.05, ** p < 0.01")
    print("\n" + "=" * 80)
    print("✓ DECODER EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
