#!/usr/bin/env python3
"""Evaluate encoder models on GLUE benchmark.

This script:
1. Loads trained encoder models (BERT-style)
2. Fine-tunes on GLUE tasks (or evaluates with probing)
3. Reports accuracy/F1 for each task

GLUE tasks:
- CoLA: Linguistic acceptability (Matthews correlation)
- SST-2: Sentiment (accuracy)
- MRPC: Paraphrase detection (F1/accuracy)
- STS-B: Semantic similarity (Pearson/Spearman)
- QQP: Question paraphrase (F1/accuracy)
- MNLI: Natural language inference (accuracy)
- QNLI: Question NLI (accuracy)
- RTE: Recognizing textual entailment (accuracy)

Usage: CUDA_VISIBLE_DEVICES=0 python 06b_evaluate_encoder_glue.py
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
from dataclasses import dataclass

os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class GLUETask:
    name: str
    num_labels: int
    metric: str  # 'accuracy', 'f1', 'matthews', 'pearson'
    text_fields: tuple  # Which fields contain text


GLUE_TASKS = {
    'cola': GLUETask('cola', 2, 'matthews', ('sentence',)),
    'sst2': GLUETask('sst2', 2, 'accuracy', ('sentence',)),
    'mrpc': GLUETask('mrpc', 2, 'f1', ('sentence1', 'sentence2')),
    'stsb': GLUETask('stsb', 1, 'pearson', ('sentence1', 'sentence2')),  # regression
    'qqp': GLUETask('qqp', 2, 'f1', ('question1', 'question2')),
    'mnli': GLUETask('mnli', 3, 'accuracy', ('premise', 'hypothesis')),
    'qnli': GLUETask('qnli', 2, 'accuracy', ('question', 'sentence')),
    'rte': GLUETask('rte', 2, 'accuracy', ('sentence1', 'sentence2')),
}

# Subset of tasks for faster evaluation
QUICK_TASKS = ['cola', 'sst2', 'mrpc', 'rte']


def load_trained_encoder(model_dir: Path, device: str = 'cuda'):
    """Load a trained encoder model."""
    from model_loader import load_model

    tokenizer, model, model_config = load_model('encoder')

    # Find checkpoint
    checkpoint_path = model_dir / "model_final.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_dir / "pytorch_model.bin"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint in {model_dir}")

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return tokenizer, model


def load_pretrained_encoder(device: str = 'cuda'):
    """Load the original pretrained encoder (no additional pretraining)."""
    from model_loader import load_model

    tokenizer, model, model_config = load_model('encoder')
    model = model.to(device)
    model.eval()

    return tokenizer, model


def evaluate_glue_task(
    model,
    tokenizer,
    task_name: str,
    device: str = 'cuda',
    max_samples: Optional[int] = None,  # None = use full dataset
    batch_size: int = 16,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    seed: int = 42,
) -> Dict:
    """Fine-tune and evaluate on a single GLUE task."""
    import random
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    from scipy.stats import pearsonr, spearmanr

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    task = GLUE_TASKS[task_name]
    print(f"\n      Task: {task_name.upper()} (seed={seed})")

    # Load dataset
    if task_name == 'mnli':
        dataset = load_dataset('glue', 'mnli', trust_remote_code=False)
        eval_split = 'validation_matched'
    else:
        dataset = load_dataset('glue', task_name, trust_remote_code=False)
        eval_split = 'validation'

    train_data = dataset['train']
    eval_data = dataset[eval_split]

    # Optionally limit samples for speed
    if max_samples is not None:
        if len(train_data) > max_samples:
            train_data = train_data.shuffle(seed=42).select(range(max_samples))
        if len(eval_data) > max_samples // 2:
            eval_data = eval_data.shuffle(seed=42).select(range(max_samples // 2))

    print(f"      Train: {len(train_data)}, Eval: {len(eval_data)}")

    # Tokenize
    def tokenize_fn(examples):
        if len(task.text_fields) == 1:
            return tokenizer(
                examples[task.text_fields[0]],
                truncation=True,
                max_length=128,
                padding=False,
            )
        else:
            return tokenizer(
                examples[task.text_fields[0]],
                examples[task.text_fields[1]],
                truncation=True,
                max_length=128,
                padding=False,
            )

    train_data = train_data.map(tokenize_fn, batched=True, remove_columns=train_data.column_names)
    eval_data = eval_data.map(tokenize_fn, batched=True, remove_columns=eval_data.column_names)

    # Add labels back
    train_data = train_data.add_column('labels', dataset['train']['label'][:len(train_data)])
    eval_data = eval_data.add_column('labels', dataset[eval_split]['label'][:len(eval_data)])

    train_data.set_format('torch')
    eval_data.set_format('torch')

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, collate_fn=collator)

    # Create classification head
    from transformers import AutoModelForSequenceClassification, AutoConfig

    # Get base model config
    base_config = model.config

    # Create classifier model with same base but classification head
    classifier = AutoModelForSequenceClassification.from_pretrained(
        base_config._name_or_path,
        num_labels=task.num_labels,
        problem_type="regression" if task_name == 'stsb' else "single_label_classification",
    )

    # Copy encoder weights from our trained model
    # Handle different model architectures (BERT vs ModernBERT vs others)
    if hasattr(classifier, 'bert') and hasattr(model, 'bert'):
        classifier.bert.load_state_dict(model.bert.state_dict())
    elif hasattr(classifier, 'model') and hasattr(model, 'model'):
        classifier.model.load_state_dict(model.model.state_dict())
    elif hasattr(classifier, 'roberta') and hasattr(model, 'roberta'):
        classifier.roberta.load_state_dict(model.roberta.state_dict())
    else:
        # Try to copy the base model weights generically
        # Get the encoder/backbone from both models
        classifier_state = classifier.state_dict()
        model_state = model.state_dict()

        # Copy matching keys (excluding classifier head)
        for key in model_state:
            if key in classifier_state and classifier_state[key].shape == model_state[key].shape:
                classifier_state[key] = model_state[key]

        classifier.load_state_dict(classifier_state)

    classifier = classifier.to(device)

    # Fine-tune with early stopping
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    patience = 2
    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None

    for epoch in range(num_epochs):
        # Training
        classifier.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = classifier(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation loss
        classifier.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = classifier(**batch)
                val_loss += outputs.loss.item()
        avg_val_loss = val_loss / len(eval_loader)

        print(f"      Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state_dict = {k: v.cpu().clone() for k, v in classifier.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch+1}")
                break

    # Restore best model
    if best_state_dict is not None:
        classifier.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})

    epochs_trained = epoch + 1

    # Evaluate
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')

            outputs = classifier(**batch)

            if task_name == 'stsb':
                preds = outputs.logits.squeeze().cpu().numpy()
            else:
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    # Calculate metric
    if task.metric == 'accuracy':
        score = accuracy_score(all_labels, all_preds)
    elif task.metric == 'f1':
        score = f1_score(all_labels, all_preds, average='binary')
    elif task.metric == 'matthews':
        score = matthews_corrcoef(all_labels, all_preds)
    elif task.metric == 'pearson':
        score = pearsonr(all_labels, all_preds)[0]

    print(f"      ✓ {task.metric}: {score:.4f}")

    # Cleanup
    del classifier, optimizer
    clear_memory()

    return {
        'task': task_name,
        'metric': task.metric,
        'score': float(score),
        'train_samples': len(train_data),
        'eval_samples': len(eval_data),
        'epochs_trained': epochs_trained,
        'best_val_loss': float(best_val_loss),
    }


def evaluate_encoder_on_glue(
    model_dir: Optional[Path],
    regime: str,
    device: str = 'cuda',
    tasks: List[str] = None,
    num_seeds: int = 3,
    seeds: List[int] = None,
    use_pretrained: bool = False,
) -> Dict:
    """Evaluate an encoder model on GLUE tasks with multiple seeds for statistical significance."""

    if tasks is None:
        tasks = QUICK_TASKS

    if seeds is None:
        seeds = [42, 123, 456][:num_seeds]

    print(f"\n   {'=' * 60}")
    print(f"   Evaluating: {regime}")
    print(f"   Seeds: {seeds}")
    print(f"   {'=' * 60}")

    try:
        if use_pretrained:
            tokenizer, model = load_pretrained_encoder(device)
            print(f"   ✓ Loaded original pretrained model (no additional pretraining)")
        else:
            tokenizer, model = load_trained_encoder(model_dir, device)
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

    for task_name in tasks:
        task_scores = []
        task_results_by_seed = []

        for seed in seeds:
            try:
                task_result = evaluate_glue_task(model, tokenizer, task_name, device, seed=seed)
                task_scores.append(task_result['score'])
                task_results_by_seed.append(task_result)
            except Exception as e:
                print(f"      ✗ {task_name} (seed={seed}) failed: {e}")

            clear_memory()

        if task_scores:
            results['tasks'][task_name] = {
                'task': task_name,
                'metric': GLUE_TASKS[task_name].metric,
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

    # Calculate average score across tasks
    task_means = [t['mean'] for t in results['tasks'].values() if 'mean' in t]
    if task_means:
        results['average_score'] = float(np.mean(task_means))
        # Also compute std of the averages per seed
        avg_per_seed = []
        for i in range(num_seeds):
            seed_scores = [t['scores'][i] for t in results['tasks'].values() if 'scores' in t and len(t['scores']) > i]
            if seed_scores:
                avg_per_seed.append(np.mean(seed_scores))
        if avg_per_seed:
            results['average_std'] = float(np.std(avg_per_seed))
        print(f"\n   Average score: {results['average_score']:.4f} ± {results.get('average_std', 0):.4f}")

    del model, tokenizer
    clear_memory()

    return results


def main():
    print("=" * 80)
    print("STEP 6b: ENCODER GLUE EVALUATION")
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
        'universal_embedding_diversity',  # NEW: diversity embeddings + submodular optimization
        'random_baseline',
        'full_dataset',  # No subsampling - train on all data
    ]

    # Load existing results to skip completed regimes
    results_file = output_dir / "glue_results.json"
    existing_results = []
    completed_regimes = set()

    if results_file.exists():
        with open(results_file, 'r') as f:
            existing_results = json.load(f)
        # Track which regimes are already done (have 'mean' in tasks, not just single run)
        for r in existing_results:
            if 'error' not in r and 'tasks' in r:
                tasks = r['tasks']
                # Check if this has multi-seed results (has 'mean' key)
                if any('mean' in t for t in tasks.values()):
                    completed_regimes.add((r.get('dataset', ''), r['regime']))
        print(f"   Found {len(completed_regimes)} completed regimes, will skip them")

    # Keep only completed multi-seed results, discard old single-run results
    all_results = [r for r in existing_results if (r.get('dataset', ''), r['regime']) in completed_regimes]

    for dataset_config in config['corpus']['datasets']:
        dataset_name = dataset_config['name']
        clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()

        print(f"\n{'#' * 80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#' * 80}")

        for regime in regimes:
            # Skip if already completed
            if (dataset_name, regime) in completed_regimes:
                print(f"\n   ✓ Skipping (already evaluated): {regime}")
                continue

            # Handle pretrained baseline specially
            if regime == 'pretrained_baseline':
                results = evaluate_encoder_on_glue(
                    model_dir=None,
                    regime=regime,
                    device=device,
                    tasks=QUICK_TASKS,
                    use_pretrained=True,
                )
                results['dataset'] = dataset_name
                all_results.append(results)
                continue

            model_dir = models_dir / clean_name / regime / "encoder"

            if not model_dir.exists():
                print(f"\n   ✗ Model not found: {model_dir}")
                continue

            results = evaluate_encoder_on_glue(
                model_dir=model_dir,
                regime=regime,
                device=device,
                tasks=QUICK_TASKS,  # CoLA, SST-2, MRPC, RTE
            )
            results['dataset'] = dataset_name
            all_results.append(results)

    # Save results
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}")

    results_file = output_dir / "glue_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"   ✓ Saved: {results_file}")

    # Print summary with mean ± std
    print(f"\n{'=' * 80}")
    print("GLUE SUMMARY (mean ± std across seeds)")
    print(f"{'=' * 80}\n")

    def fmt_score(task_data):
        if 'mean' in task_data:
            return f"{task_data['mean']:.3f}±{task_data['std']:.3f}"
        return "N/A"

    print(f"{'Regime':<25} {'CoLA':>12} {'SST-2':>12} {'MRPC':>12} {'RTE':>12} {'Avg':>12}")
    print("-" * 90)

    for r in all_results:
        if 'error' in r:
            continue
        regime = r['regime']
        tasks = r.get('tasks', {})
        cola = fmt_score(tasks.get('cola', {}))
        sst2 = fmt_score(tasks.get('sst2', {}))
        mrpc = fmt_score(tasks.get('mrpc', {}))
        rte = fmt_score(tasks.get('rte', {}))
        avg = f"{r.get('average_score', 0):.3f}±{r.get('average_std', 0):.3f}"
        print(f"{regime:<25} {cola:>12} {sst2:>12} {mrpc:>12} {rte:>12} {avg:>12}")

    # Statistical significance tests
    print(f"\n{'=' * 80}")
    print("STATISTICAL SIGNIFICANCE (paired t-test vs random_baseline)")
    print(f"{'=' * 80}\n")

    from scipy.stats import ttest_rel

    # Find random baseline
    baseline_result = next((r for r in all_results if r['regime'] == 'random_baseline' and 'error' not in r), None)

    if baseline_result:
        print(f"{'Regime':<25} {'Task':>10} {'Diff':>10} {'p-value':>10} {'Sig?':>8}")
        print("-" * 70)

        for r in all_results:
            if 'error' in r or r['regime'] == 'random_baseline':
                continue

            regime = r['regime']
            tasks = r.get('tasks', {})

            for task_name in QUICK_TASKS:
                if task_name not in tasks or task_name not in baseline_result['tasks']:
                    continue

                regime_scores = tasks[task_name].get('scores', [])
                baseline_scores = baseline_result['tasks'][task_name].get('scores', [])

                if len(regime_scores) >= 2 and len(baseline_scores) >= 2:
                    # Paired t-test
                    min_len = min(len(regime_scores), len(baseline_scores))
                    t_stat, p_value = ttest_rel(regime_scores[:min_len], baseline_scores[:min_len])
                    diff = np.mean(regime_scores) - np.mean(baseline_scores)
                    sig = "*" if p_value < 0.05 else ("**" if p_value < 0.01 else "")
                    print(f"{regime:<25} {task_name:>10} {diff:>+10.4f} {p_value:>10.4f} {sig:>8}")
    else:
        print("   No random baseline found for comparison.")

    print("\n   * p < 0.05, ** p < 0.01")
    print("\n" + "=" * 80)
    print("✓ GLUE EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
