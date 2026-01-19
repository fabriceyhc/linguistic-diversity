#!/usr/bin/env python3
"""Evaluate encoder-decoder models on generation tasks.

This script:
1. Loads trained encoder-decoder models (T5-style)
2. Evaluates on summarization and QA tasks
3. Reports ROUGE (summarization) and F1/EM (QA)

Tasks:
- XSum: Abstractive summarization (ROUGE-1, ROUGE-2, ROUGE-L)
- SQuAD v2: Extractive QA (F1, Exact Match)
- SAMSum: Dialogue summarization (ROUGE)

Usage: CUDA_VISIBLE_DEVICES=0 python 06d_evaluate_encdec_tasks.py
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
import re
import string
from collections import Counter

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


def load_trained_encdec(model_dir: Path, device: str = 'cuda'):
    """Load a trained encoder-decoder model."""
    from model_loader import load_model

    tokenizer, model, model_config = load_model('encoder-decoder')

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


# =============================================================================
# ROUGE Scoring (simple implementation)
# =============================================================================

def get_ngrams(text: str, n: int) -> Counter:
    """Get n-grams from text."""
    words = text.lower().split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngrams.append(tuple(words[i:i+n]))
    return Counter(ngrams)


def rouge_n(hypothesis: str, reference: str, n: int) -> Dict[str, float]:
    """Calculate ROUGE-N score."""
    hyp_ngrams = get_ngrams(hypothesis, n)
    ref_ngrams = get_ngrams(reference, n)

    if not ref_ngrams or not hyp_ngrams:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    overlap = sum((hyp_ngrams & ref_ngrams).values())

    precision = overlap / sum(hyp_ngrams.values()) if hyp_ngrams else 0
    recall = overlap / sum(ref_ngrams.values()) if ref_ngrams else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def rouge_l(hypothesis: str, reference: str) -> Dict[str, float]:
    """Calculate ROUGE-L score using LCS."""
    hyp_words = hypothesis.lower().split()
    ref_words = reference.lower().split()

    if not hyp_words or not ref_words:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    # LCS using dynamic programming
    m, n = len(hyp_words), len(ref_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp_words[i-1] == ref_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs_length = dp[m][n]

    precision = lcs_length / len(hyp_words) if hyp_words else 0
    recall = lcs_length / len(ref_words) if ref_words else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L."""
    r1 = rouge_n(hypothesis, reference, 1)
    r2 = rouge_n(hypothesis, reference, 2)
    rl = rouge_l(hypothesis, reference)

    return {
        'rouge1': r1['f1'],
        'rouge2': r2['f1'],
        'rougeL': rl['f1'],
    }


# =============================================================================
# QA Metrics
# =============================================================================

def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return f1


# =============================================================================
# Task Evaluations
# =============================================================================

def evaluate_xsum(model, tokenizer, device: str, max_samples: Optional[int] = None, seed: int = 42) -> Dict:
    """Evaluate on XSum summarization."""
    from datasets import load_dataset

    print(f"\n      Task: XSum (seed={seed})")

    dataset = load_dataset('xsum', trust_remote_code=True)
    eval_data = dataset['validation']

    if max_samples is not None and len(eval_data) > max_samples:
        eval_data = eval_data.shuffle(seed=seed).select(range(max_samples))

    print(f"      Samples: {len(eval_data)}")

    all_rouge1 = []
    all_rouge2 = []
    all_rougeL = []

    for item in tqdm(eval_data, desc="      Evaluating", leave=False):
        document = item['document']
        reference = item['summary']

        # Generate summary
        input_text = f"summarize: {document}"
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
            )

        hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compute ROUGE
        scores = compute_rouge(hypothesis, reference)
        all_rouge1.append(scores['rouge1'])
        all_rouge2.append(scores['rouge2'])
        all_rougeL.append(scores['rougeL'])

    results = {
        'task': 'xsum',
        'rouge1': float(np.mean(all_rouge1)),
        'rouge2': float(np.mean(all_rouge2)),
        'rougeL': float(np.mean(all_rougeL)),
        'samples': len(eval_data),
    }

    print(f"      ✓ ROUGE-1: {results['rouge1']:.4f}, ROUGE-2: {results['rouge2']:.4f}, ROUGE-L: {results['rougeL']:.4f}")

    return results


def evaluate_samsum(model, tokenizer, device: str, max_samples: Optional[int] = None, seed: int = 42) -> Dict:
    """Evaluate on SAMSum dialogue summarization."""
    from datasets import load_dataset

    print(f"\n      Task: SAMSum (seed={seed})")

    dataset = load_dataset('samsum', trust_remote_code=True)
    eval_data = dataset['validation']

    if max_samples is not None and len(eval_data) > max_samples:
        eval_data = eval_data.shuffle(seed=seed).select(range(max_samples))

    print(f"      Samples: {len(eval_data)}")

    all_rouge1 = []
    all_rouge2 = []
    all_rougeL = []

    for item in tqdm(eval_data, desc="      Evaluating", leave=False):
        dialogue = item['dialogue']
        reference = item['summary']

        input_text = f"summarize dialogue: {dialogue}"
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
            )

        hypothesis = tokenizer.decode(outputs[0], skip_special_tokens=True)

        scores = compute_rouge(hypothesis, reference)
        all_rouge1.append(scores['rouge1'])
        all_rouge2.append(scores['rouge2'])
        all_rougeL.append(scores['rougeL'])

    results = {
        'task': 'samsum',
        'rouge1': float(np.mean(all_rouge1)),
        'rouge2': float(np.mean(all_rouge2)),
        'rougeL': float(np.mean(all_rougeL)),
        'samples': len(eval_data),
    }

    print(f"      ✓ ROUGE-1: {results['rouge1']:.4f}, ROUGE-2: {results['rouge2']:.4f}, ROUGE-L: {results['rougeL']:.4f}")

    return results


def evaluate_squad(model, tokenizer, device: str, max_samples: Optional[int] = None, seed: int = 42) -> Dict:
    """Evaluate on SQuAD v2 QA."""
    from datasets import load_dataset

    print(f"\n      Task: SQuAD v2 (seed={seed})")

    dataset = load_dataset('squad_v2', trust_remote_code=True)
    eval_data = dataset['validation']

    # Filter to answerable questions for simplicity
    eval_data = eval_data.filter(lambda x: len(x['answers']['text']) > 0)

    if max_samples is not None and len(eval_data) > max_samples:
        eval_data = eval_data.shuffle(seed=seed).select(range(max_samples))

    print(f"      Samples: {len(eval_data)}")

    all_f1 = []
    all_em = []

    for item in tqdm(eval_data, desc="      Evaluating", leave=False):
        context = item['context']
        question = item['question']
        answers = item['answers']['text']

        input_text = f"question: {question} context: {context}"
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                num_beams=4,
                early_stopping=True,
            )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Compare to all valid answers, take best score
        best_f1 = max(compute_f1(prediction, ans) for ans in answers)
        best_em = max(compute_exact_match(prediction, ans) for ans in answers)

        all_f1.append(best_f1)
        all_em.append(best_em)

    results = {
        'task': 'squad_v2',
        'f1': float(np.mean(all_f1)),
        'exact_match': float(np.mean(all_em)),
        'samples': len(eval_data),
    }

    print(f"      ✓ F1: {results['f1']:.4f}, EM: {results['exact_match']:.4f}")

    return results


def evaluate_encdec_on_tasks(
    model_dir: Path,
    regime: str,
    device: str = 'cuda',
    max_samples: Optional[int] = 300,  # Use subsets for multi-seed evaluation
    num_seeds: int = 3,
    seeds: List[int] = None,
) -> Dict:
    """Evaluate encoder-decoder model on all tasks with multiple seeds for statistical significance."""

    if seeds is None:
        seeds = [42, 123, 456][:num_seeds]

    print(f"\n   {'=' * 60}")
    print(f"   Evaluating: {regime}")
    print(f"   Seeds: {seeds}, Samples per seed: {max_samples}")
    print(f"   {'=' * 60}")

    try:
        tokenizer, model = load_trained_encdec(model_dir, device)
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

    task_names = ['xsum', 'samsum', 'squad_v2']

    for task_name in task_names:
        task_scores = []  # Will store primary metric (rougeL for summarization, f1 for QA)
        task_results_by_seed = []

        for seed in seeds:
            try:
                if task_name == 'xsum':
                    task_result = evaluate_xsum(model, tokenizer, device, max_samples, seed)
                    task_scores.append(task_result['rougeL'])
                elif task_name == 'samsum':
                    task_result = evaluate_samsum(model, tokenizer, device, max_samples, seed)
                    task_scores.append(task_result['rougeL'])
                elif task_name == 'squad_v2':
                    task_result = evaluate_squad(model, tokenizer, device, max_samples, seed)
                    task_scores.append(task_result['f1'])

                task_results_by_seed.append(task_result)
            except Exception as e:
                print(f"      ✗ {task_name} (seed={seed}) failed: {e}")

            clear_memory()

        if task_scores:
            metric_name = 'f1' if task_name == 'squad_v2' else 'rougeL'
            results['tasks'][task_name] = {
                'task': task_name,
                'metric': metric_name,
                'scores': task_scores,
                'mean': float(np.mean(task_scores)),
                'std': float(np.std(task_scores)),
                'min': float(np.min(task_scores)),
                'max': float(np.max(task_scores)),
                'runs': task_results_by_seed,
            }
            print(f"      {task_name.upper()}: {results['tasks'][task_name]['mean']:.4f} ± {results['tasks'][task_name]['std']:.4f}")
        else:
            results['tasks'][task_name] = {'error': 'All seeds failed'}

    # Calculate average ROUGE-L across summarization tasks
    rouge_means = [t['mean'] for t in results['tasks'].values() if 'mean' in t and t.get('metric') == 'rougeL']
    if rouge_means:
        results['average_rougeL'] = float(np.mean(rouge_means))
        # Compute std of averages per seed
        avg_per_seed = []
        for i in range(num_seeds):
            seed_scores = [t['scores'][i] for t in results['tasks'].values()
                          if 'scores' in t and len(t['scores']) > i and t.get('metric') == 'rougeL']
            if seed_scores:
                avg_per_seed.append(np.mean(seed_scores))
        if avg_per_seed:
            results['average_rougeL_std'] = float(np.std(avg_per_seed))
        print(f"\n   Average ROUGE-L: {results['average_rougeL']:.4f} ± {results.get('average_rougeL_std', 0):.4f}")

    del model, tokenizer
    clear_memory()

    return results


def main():
    print("=" * 80)
    print("STEP 6d: ENCODER-DECODER TASK EVALUATION")
    print("=" * 80)

    config = load_config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n   Device: {device}")

    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / "models"
    output_dir = base_dir / "output"

    regimes = [
        'semantic_diversity',
        'syntactic_diversity',
        'morphological_diversity',
        'phonological_diversity',
        'universal_diversity',
        'random_baseline',
        'full_dataset',  # No subsampling - train on all data
    ]

    # Load existing results to skip completed regimes
    results_file = output_dir / "encdec_task_results.json"
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

            model_dir = models_dir / clean_name / regime / "encoder-decoder"

            if not model_dir.exists():
                print(f"\n   ✗ Model not found: {model_dir}")
                continue

            results = evaluate_encdec_on_tasks(
                model_dir=model_dir,
                regime=regime,
                device=device,
                max_samples=300,  # Use subsets for multi-seed statistical evaluation
                num_seeds=3,
            )
            results['dataset'] = dataset_name
            all_results.append(results)

    # Save results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")

    results_file = output_dir / "encdec_task_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"   ✓ Saved: {results_file}")

    # Print summary with mean ± std
    print(f"\n{'=' * 80}")
    print("ENCODER-DECODER TASK SUMMARY (mean ± std across seeds)")
    print(f"{'=' * 80}\n")

    def fmt_score(task_data):
        if 'mean' in task_data:
            return f"{task_data['mean']:.3f}±{task_data['std']:.3f}"
        return "N/A"

    print(f"{'Regime':<25} {'XSum-RL':>12} {'SAMSum-RL':>12} {'SQuAD-F1':>12} {'Avg-RL':>12}")
    print("-" * 80)

    for r in all_results:
        if 'error' in r:
            continue
        regime = r['regime']
        tasks = r.get('tasks', {})
        xsum = fmt_score(tasks.get('xsum', {}))
        samsum = fmt_score(tasks.get('samsum', {}))
        squad = fmt_score(tasks.get('squad_v2', {}))
        avg = f"{r.get('average_rougeL', 0):.3f}±{r.get('average_rougeL_std', 0):.3f}"
        print(f"{regime:<25} {xsum:>12} {samsum:>12} {squad:>12} {avg:>12}")

    # Statistical significance tests
    print(f"\n{'=' * 80}")
    print("STATISTICAL SIGNIFICANCE (paired t-test vs random_baseline)")
    print(f"{'=' * 80}\n")

    from scipy.stats import ttest_rel

    # Find random baseline
    baseline_result = next((r for r in all_results if r['regime'] == 'random_baseline' and 'error' not in r), None)
    task_names = ['xsum', 'samsum', 'squad_v2']

    if baseline_result:
        print(f"{'Regime':<25} {'Task':>12} {'Diff':>10} {'p-value':>10} {'Sig?':>8}")
        print("-" * 70)

        for r in all_results:
            if 'error' in r or r['regime'] == 'random_baseline':
                continue

            regime = r['regime']
            tasks = r.get('tasks', {})

            for task_name in task_names:
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
    print("✓ ENCODER-DECODER EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
