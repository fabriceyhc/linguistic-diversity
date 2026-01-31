#!/usr/bin/env python3
"""
Fine-tune and evaluate models on diversity-selected instruction subsets.

This script:
1. Loads pretrained decoder models
2. Fine-tunes with LoRA on selected training subsets
3. Evaluates on standard benchmarks (via lm-evaluation-harness)
4. Compares performance across selection methods

Inspired by LIMA: testing if diversity-selected small subsets can match larger datasets.
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
import random
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
from datetime import datetime

os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_instruction(sample: Dict, tokenizer) -> str:
    """Format instruction-response pair for training."""
    instruction = sample['instruction']
    response = sample['response']

    # Use chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass

    # Fallback to simple format
    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def create_training_dataset(samples: List[Dict], selected_indices: List[int], tokenizer, max_length: int = 2048):
    """Create a training dataset from selected samples."""
    from torch.utils.data import Dataset

    class InstructionDataset(Dataset):
        def __init__(self, samples, indices, tokenizer, max_length):
            self.samples = [samples[i] for i in indices]
            self.tokenizer = tokenizer
            self.max_length = max_length
            # Pre-tokenize all samples
            self.encodings = []
            for sample in tqdm(self.samples, desc="      Tokenizing"):
                text = format_instruction(sample, tokenizer)
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors='pt',
                )
                # Squeeze to remove batch dimension
                self.encodings.append({
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                })

        def __len__(self):
            return len(self.encodings)

        def __getitem__(self, idx):
            item = self.encodings[idx]
            # For causal LM, labels = input_ids (with -100 for padding)
            labels = item['input_ids'].clone()
            labels[item['attention_mask'] == 0] = -100
            return {
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask'],
                'labels': labels,
            }

    return InstructionDataset(samples, selected_indices, tokenizer, max_length)


def setup_lora_model(model_name: str, config: dict, device: str = 'cuda'):
    """Setup model with LoRA adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lora_config = config['finetuning']['lora_config']

    print(f"   Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with 4-bit quantization for efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Setup LoRA
    peft_config = LoraConfig(
        r=int(lora_config['r']),
        lora_alpha=int(lora_config['lora_alpha']),
        lora_dropout=float(lora_config['lora_dropout']),
        target_modules=lora_config['target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return tokenizer, model


def finetune_model(
    model,
    tokenizer,
    train_dataset,
    config: dict,
    output_dir: Path,
    seed: int = 42,
):
    """Fine-tune model on the training dataset."""
    from transformers import TrainingArguments, Trainer, default_data_collator

    set_seed(seed)
    ft_config = config['finetuning']

    # Use default collator since data is already padded
    data_collator = default_data_collator

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=int(ft_config['num_epochs']),
        per_device_train_batch_size=int(ft_config['batch_size']),
        gradient_accumulation_steps=int(ft_config['gradient_accumulation_steps']),
        learning_rate=float(ft_config['learning_rate']),
        weight_decay=float(ft_config['weight_decay']),
        warmup_ratio=float(ft_config['warmup_ratio']),
        logging_steps=10,
        save_strategy="no",  # Don't save intermediate checkpoints
        fp16=True,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print(f"   Training for {ft_config['num_epochs']} epochs...")
    trainer.train()

    return model


def evaluate_model_lmeval(model, tokenizer, model_name: str, tasks: List[str], output_dir: Path):
    """Evaluate model using lm-evaluation-harness."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    print(f"   Evaluating on: {tasks}")

    # Wrap model for lm-eval
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=8,
    )

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,  # Zero-shot for instruction-tuned models
        batch_size=8,
    )

    # Extract scores
    scores = {}
    for task in tasks:
        if task in results['results']:
            task_results = results['results'][task]
            # Get the main metric for each task
            if 'acc' in task_results:
                scores[task] = task_results['acc']
            elif 'acc_norm' in task_results:
                scores[task] = task_results['acc_norm']
            elif 'exact_match' in task_results:
                scores[task] = task_results['exact_match']

    return scores


def evaluate_model_simple(model, tokenizer, samples: List[Dict], n_eval: int = 100):
    """Simple evaluation: generate responses and compute basic metrics."""
    model.eval()

    # Sample evaluation examples
    rng = np.random.RandomState(42)
    eval_indices = rng.choice(len(samples), min(n_eval, len(samples)), replace=False)

    results = {
        'avg_response_length': 0,
        'non_empty_rate': 0,
        'samples': [],
    }

    total_length = 0
    non_empty = 0

    for idx in tqdm(eval_indices, desc="   Evaluating"):
        sample = samples[idx]
        instruction = sample['instruction']

        # Format prompt
        if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": instruction}]
            try:
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

        total_length += len(response.split())
        if response.strip():
            non_empty += 1

        if len(results['samples']) < 5:
            results['samples'].append({
                'instruction': instruction[:200],
                'response': response[:500],
            })

    results['avg_response_length'] = total_length / len(eval_indices)
    results['non_empty_rate'] = non_empty / len(eval_indices)

    return results


def _save_intermediate_results(output_dir, results, mode, models, sizes, methods, seeds, eval_tasks):
    """Save intermediate results for crash recovery."""
    results_file = output_dir / "finetuning_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'mode': mode,
                'models': [m['name'] for m in models],
                'sizes': sizes,
                'methods': methods,
                'seeds': seeds,
                'eval_tasks': eval_tasks,
            },
            'results': results,
            'status': 'in_progress',
        }, f, indent=2)


def load_existing_results(output_dir: Path) -> dict:
    """Load existing results if available."""
    results_file = output_dir / "finetuning_results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return None


def get_completed_runs(existing_results: dict) -> set:
    """Extract set of completed (model, size, method, seed) tuples."""
    completed = set()
    if not existing_results or 'results' not in existing_results:
        return completed

    for model_result in existing_results['results']:
        model_name = model_result['model']
        for size_key, methods in model_result.get('results_by_size', {}).items():
            for method, data in methods.items():
                # Check if this method has scores (indicating completion)
                if data.get('metrics'):
                    for metric_data in data['metrics'].values():
                        if 'scores' in metric_data:
                            for i, _ in enumerate(metric_data['scores']):
                                # Infer seed from position (assumes consistent ordering)
                                completed.add((model_name, size_key, method, i))
                            break
    return completed


def main():
    print("=" * 70)
    print("STEP 3: FINE-TUNE AND EVALUATE")
    print("=" * 70)

    config = load_config()
    mode = config.get('mode', 'quick')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Mode: {mode}")

    base_dir = Path(__file__).parent.parent
    datasets_dir = base_dir / "datasets"
    selections_dir = base_dir / "selections"
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)

    # Load existing results for checkpointing
    existing_results = load_existing_results(output_dir)
    completed_runs = get_completed_runs(existing_results)
    if completed_runs:
        print(f"\nFound {len(completed_runs)} completed runs. Will skip these.")

    # Load samples and selections
    print("\n1. Loading data...")
    with open(datasets_dir / "samples.json", 'r') as f:
        samples = json.load(f)
    print(f"   Total samples: {len(samples)}")

    with open(selections_dir / "selections.json", 'r') as f:
        selections = json.load(f)
    print(f"   Selection sizes: {list(selections.keys())}")

    # Get configuration
    seeds = config['finetuning']['seeds']
    ft_config = config['finetuning']
    n_total_samples = len(samples)

    # Determine which model to use based on mode
    if mode == 'quick':
        models_to_train = [config['models']['small'][0]]
        sizes_to_test = [list(selections.keys())[0]]  # Only smallest size
        # Quick mode: pretrained, random, and one diversity method
        methods_to_test = ['pretrained_baseline', 'random', 'semantic_diversity']
        seeds = [seeds[0]]  # Only one seed
        include_full_dataset = False
    else:
        models_to_train = config['models']['small'] + config['models'].get('medium', [])
        sizes_to_test = list(selections.keys())
        # Full mode: all methods including pretrained baseline
        methods_to_test = ['pretrained_baseline', 'random'] + [
            m for m in selections[sizes_to_test[0]].keys() if m != 'random'
        ]
        include_full_dataset = True

    print(f"\n2. Configuration:")
    print(f"   Models: {[m['name'] for m in models_to_train]}")
    print(f"   Sizes: {sizes_to_test}")
    print(f"   Methods: {methods_to_test}")
    print(f"   Seeds: {seeds}")
    print(f"   Include full dataset baseline: {include_full_dataset}")

    # Evaluation tasks (use simpler tasks for quick mode)
    if mode == 'quick':
        eval_tasks = ['hellaswag']
    else:
        eval_tasks = ['hellaswag', 'arc_easy', 'winogrande', 'piqa']

    all_results = []

    def run_evaluation(model, tokenizer, model_name, method, size_key, run_output_dir):
        """Run evaluation and return scores."""
        try:
            eval_scores = evaluate_model_lmeval(
                model, tokenizer, model_name, eval_tasks, run_output_dir
            )
        except Exception as e:
            print(f"         Warning: lm-eval failed: {e}")
            print(f"         Falling back to simple evaluation...")
            eval_scores = evaluate_model_simple(model, tokenizer, samples)
        return eval_scores

    def aggregate_scores(method_scores):
        """Aggregate scores across seeds."""
        aggregated = {}
        if method_scores:
            all_keys = set()
            for scores in method_scores:
                all_keys.update(scores.keys())
            for key in all_keys:
                values = [s.get(key, 0) for s in method_scores if key in s]
                if values:
                    aggregated[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'scores': values,
                    }
        return aggregated

    for model_config in models_to_train:
        model_name = model_config['name']
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")

        model_results = {
            'model': model_name,
            'baselines': {},
            'results_by_size': {},
        }

        # =====================================================================
        # BASELINE 1: Pretrained model (no fine-tuning)
        # =====================================================================
        if 'pretrained_baseline' in methods_to_test:
            print(f"\n   BASELINE: Pretrained (no fine-tuning)")
            run_key = (model_name, 'baseline', 'pretrained', 0)

            if run_key in completed_runs:
                print(f"      SKIPPED (already completed)")
                # Load existing
                if existing_results:
                    for mr in existing_results['results']:
                        if mr['model'] == model_name and 'baselines' in mr:
                            if 'pretrained' in mr['baselines']:
                                model_results['baselines']['pretrained'] = mr['baselines']['pretrained']
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                print(f"      Loading pretrained model...")
                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
                )

                run_output_dir = output_dir / f"{model_name.split('/')[-1]}_pretrained_baseline"
                run_output_dir.mkdir(exist_ok=True, parents=True)

                eval_scores = run_evaluation(model, tokenizer, model_name, 'pretrained', 'baseline', run_output_dir)
                model_results['baselines']['pretrained'] = {
                    'n_samples': 0,
                    'metrics': {k: {'mean': v, 'std': 0.0, 'scores': [v]} for k, v in eval_scores.items()},
                }
                print(f"      Results: {eval_scores}")

                del model, tokenizer
                clear_memory()

        # =====================================================================
        # BASELINE 2: Full dataset fine-tuning
        # =====================================================================
        if include_full_dataset:
            print(f"\n   BASELINE: Full dataset fine-tuning (n={n_total_samples})")

            method_scores = []
            for seed_idx, seed in enumerate(seeds):
                run_key = (model_name, 'baseline', 'full_dataset', seed_idx)

                if run_key in completed_runs:
                    print(f"      Seed: {seed} - SKIPPED (already completed)")
                    if existing_results:
                        for mr in existing_results['results']:
                            if mr['model'] == model_name and 'baselines' in mr:
                                if 'full_dataset' in mr['baselines']:
                                    existing_metrics = mr['baselines']['full_dataset'].get('metrics', {})
                                    if existing_metrics:
                                        score_dict = {}
                                        for metric, mdata in existing_metrics.items():
                                            if 'scores' in mdata and seed_idx < len(mdata['scores']):
                                                score_dict[metric] = mdata['scores'][seed_idx]
                                        if score_dict:
                                            method_scores.append(score_dict)
                    continue

                print(f"      Seed: {seed}")
                tokenizer, model = setup_lora_model(model_name, config, device)

                all_indices = list(range(n_total_samples))
                train_dataset = create_training_dataset(
                    samples, all_indices, tokenizer,
                    max_length=int(ft_config['max_seq_length'])
                )
                print(f"         Training samples: {len(train_dataset)}")

                run_output_dir = output_dir / f"{model_name.split('/')[-1]}_full_dataset_seed{seed}"
                run_output_dir.mkdir(exist_ok=True, parents=True)

                model = finetune_model(model, tokenizer, train_dataset, config, output_dir=run_output_dir, seed=seed)
                eval_scores = run_evaluation(model, tokenizer, model_name, 'full_dataset', 'baseline', run_output_dir)
                method_scores.append(eval_scores)
                print(f"         Results: {eval_scores}")

                del model, tokenizer
                clear_memory()

            if method_scores:
                model_results['baselines']['full_dataset'] = {
                    'n_samples': n_total_samples,
                    'metrics': aggregate_scores(method_scores),
                }

        # =====================================================================
        # SIZE-SPECIFIC: Random and Diversity-based selection methods
        # =====================================================================
        for size_key in sizes_to_test:
            size_num = int(size_key.replace('size_', ''))
            print(f"\n   Size: {size_key} ({size_num} samples)")
            model_results['results_by_size'][size_key] = {}

            # Get methods to test (exclude pretrained_baseline which is handled above)
            size_methods = [m for m in methods_to_test if m != 'pretrained_baseline']

            for method in size_methods:
                if method not in selections[size_key]:
                    continue

                print(f"\n      Method: {method}")
                selected_indices = selections[size_key][method]

                method_scores = []

                for seed_idx, seed in enumerate(seeds):
                    run_key = (model_name, size_key, method, seed_idx)
                    if run_key in completed_runs:
                        print(f"         Seed: {seed} - SKIPPED (already completed)")
                        if existing_results:
                            for mr in existing_results['results']:
                                if mr['model'] == model_name:
                                    if size_key in mr.get('results_by_size', {}):
                                        if method in mr['results_by_size'][size_key]:
                                            existing_metrics = mr['results_by_size'][size_key][method].get('metrics', {})
                                            if existing_metrics:
                                                score_dict = {}
                                                for metric, mdata in existing_metrics.items():
                                                    if 'scores' in mdata and seed_idx < len(mdata['scores']):
                                                        score_dict[metric] = mdata['scores'][seed_idx]
                                                if score_dict:
                                                    method_scores.append(score_dict)
                        continue

                    print(f"         Seed: {seed}")
                    tokenizer, model = setup_lora_model(model_name, config, device)

                    train_dataset = create_training_dataset(
                        samples, selected_indices, tokenizer,
                        max_length=int(ft_config['max_seq_length'])
                    )
                    print(f"         Training samples: {len(train_dataset)}")

                    run_output_dir = output_dir / f"{model_name.split('/')[-1]}_{size_key}_{method}_seed{seed}"
                    run_output_dir.mkdir(exist_ok=True, parents=True)

                    model = finetune_model(model, tokenizer, train_dataset, config, output_dir=run_output_dir, seed=seed)
                    eval_scores = run_evaluation(model, tokenizer, model_name, method, size_key, run_output_dir)
                    method_scores.append(eval_scores)
                    print(f"         Results: {eval_scores}")

                    del model, tokenizer
                    clear_memory()

                model_results['results_by_size'][size_key][method] = {
                    'n_samples': len(selected_indices),
                    'metrics': aggregate_scores(method_scores),
                }

                print(f"      Aggregated: {model_results['results_by_size'][size_key][method]['metrics']}")

                # Incremental save after each method completes
                _save_intermediate_results(
                    output_dir, all_results + [model_results],
                    mode, models_to_train, sizes_to_test, methods_to_test, seeds, eval_tasks
                )

        all_results.append(model_results)

    # Final save
    results_file = output_dir / "finetuning_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': {
                'mode': mode,
                'models': [m['name'] for m in models_to_train],
                'sizes': sizes_to_test,
                'methods': methods_to_test,
                'seeds': seeds,
                'eval_tasks': eval_tasks,
                'include_full_dataset': include_full_dataset,
                'total_samples': n_total_samples,
            },
            'results': all_results,
            'status': 'completed',
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {results_file}")
    print(f"{'=' * 70}")

    # Print summary
    print("\nSUMMARY")
    print("=" * 70)

    for model_result in all_results:
        print(f"\n{model_result['model']}:")
        print("-" * 60)

        # Print baselines first
        if 'baselines' in model_result and model_result['baselines']:
            print("\n  BASELINES:")
            for baseline_name, data in model_result['baselines'].items():
                n = data['n_samples']
                metrics_str = ", ".join(
                    f"{k}: {v['mean']:.4f}±{v['std']:.4f}"
                    for k, v in data['metrics'].items()
                )
                print(f"    {baseline_name:25s} (n={n:6d}): {metrics_str}")

        # Print size-specific results
        for size_key, methods in model_result['results_by_size'].items():
            print(f"\n  {size_key}:")

            # Sort by mean score of first metric
            sorted_methods = []
            for method, data in methods.items():
                metrics = data['metrics']
                if metrics:
                    first_metric = list(metrics.keys())[0]
                    mean_score = metrics[first_metric]['mean']
                else:
                    mean_score = 0
                sorted_methods.append((method, data, mean_score))

            sorted_methods.sort(key=lambda x: -x[2])

            for method, data, _ in sorted_methods:
                n = data['n_samples']
                metrics_str = ", ".join(
                    f"{k}: {v['mean']:.4f}±{v['std']:.4f}"
                    for k, v in data['metrics'].items()
                )
                print(f"    {method:25s} (n={n:6d}): {metrics_str}")

    print("\n" + "=" * 70)
    print("FINE-TUNING EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
