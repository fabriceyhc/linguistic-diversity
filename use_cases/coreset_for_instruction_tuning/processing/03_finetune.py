#!/usr/bin/env python3
"""
Step 3: Fine-tuning with LoRA (Unsloth-optimized)

This script:
1. Loads base models (Mistral-7B-v0.3 or Llama-3-8B)
2. Applies LoRA adapters
3. Fine-tunes on selected subsets:
   - Full dataset (baseline)
   - Random 10% selection (baseline)
   - Diversity 10% selection (treatment)
4. Saves adapters for evaluation

Uses Unsloth for 2-5x faster training when available.
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
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_config() -> dict:
    """Load experiment configuration."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class TrainingResult:
    """Result of a training run."""
    model_name: str
    dataset_name: str
    selection_method: str
    n_samples: int
    seed: int
    adapter_path: str
    training_time_seconds: float
    peak_memory_gb: float
    train_loss: float
    config: Dict


def check_unsloth_available() -> bool:
    """Check if Unsloth is available."""
    try:
        import unsloth
        return True
    except ImportError:
        return False


def load_model_unsloth(model_name: str, config: dict):
    """Load model with Unsloth for faster training."""
    from unsloth import FastLanguageModel

    lora_config = config['finetuning']['lora']
    quant_config = config['finetuning']['quantization']

    print(f"   Loading {model_name} with Unsloth...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=config['finetuning']['training']['max_seq_length'],
        dtype=None,  # Auto-detect
        load_in_4bit=quant_config['load_in_4bit'],
    )

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config['r'],
        target_modules=lora_config['target_modules'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth optimization
        random_state=42,
    )

    return model, tokenizer


def load_model_standard(model_name: str, config: dict):
    """Load model with standard HuggingFace + PEFT."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lora_config = config['finetuning']['lora']
    quant_config = config['finetuning']['quantization']

    print(f"   Loading {model_name} with standard HuggingFace...")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config['load_in_4bit'],
        bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=getattr(torch, quant_config['bnb_4bit_compute_dtype']),
        bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant'],
    )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # LoRA
    peft_config = LoraConfig(
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        target_modules=lora_config['target_modules'],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer


def format_instruction_alpaca(
    instruction: str,
    input_text: str,
    output: str,
    tokenizer,
) -> str:
    """Format in Alpaca style for training."""
    # Use chat template if available
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        # Combine instruction and input for chat format
        user_content = instruction
        if input_text:
            user_content = f"{instruction}\n\nInput: {input_text}"

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass

    # Fallback to Alpaca format
    if input_text:
        text = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        text = f"""### Instruction:
{instruction}

### Response:
{output}"""

    return text


def load_training_data(jsonl_path: Path) -> List[Dict]:
    """Load training data from JSONL file."""
    samples = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def create_dataset(samples: List[Dict], tokenizer, max_length: int):
    """Create a training dataset with pre-tokenized and padded samples."""
    from torch.utils.data import Dataset
    from tqdm import tqdm

    class InstructionDataset(Dataset):
        def __init__(self, samples, tokenizer, max_length):
            self.tokenizer = tokenizer
            self.max_length = max_length
            # Pre-tokenize all samples with padding
            self.encodings = []
            for sample in tqdm(samples, desc="      Tokenizing"):
                text = format_instruction_alpaca(
                    sample['instruction'],
                    sample.get('input', ''),
                    sample['output'],
                    tokenizer,
                )
                encoding = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors='pt',
                )
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

    return InstructionDataset(samples, tokenizer, max_length)


def train_with_unsloth(
    model,
    tokenizer,
    train_data: List[Dict],
    config: dict,
    output_dir: Path,
    seed: int = 42,
) -> Dict:
    """Train using Unsloth's optimized trainer."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    set_seed(seed)
    train_config = config['finetuning']['training']

    # Prepare dataset for SFTTrainer
    def formatting_prompts_func(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            text = format_instruction_alpaca(
                examples['instruction'][i],
                examples.get('input', [''] * len(examples['instruction']))[i],
                examples['output'][i],
                tokenizer,
            )
            texts.append(text)
        return {"text": texts}

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config['num_epochs'],
        per_device_train_batch_size=train_config['batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        warmup_ratio=train_config['warmup_ratio'],
        logging_steps=10,
        save_strategy="no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",  # Memory efficient
        seed=seed,
        report_to="none",  # Disable W&B for now
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=train_config['max_seq_length'],
        args=training_args,
    )

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    print(f"   Training for {train_config['num_epochs']} epoch(s)...")
    start_time = time.time()

    # Track GPU memory
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = trainer.train()

    training_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    return {
        'train_loss': result.training_loss,
        'training_time_seconds': training_time,
        'peak_memory_gb': peak_memory,
    }


def train_standard(
    model,
    tokenizer,
    train_data: List[Dict],
    config: dict,
    output_dir: Path,
    seed: int = 42,
) -> Dict:
    """Train using standard HuggingFace Trainer."""
    from transformers import TrainingArguments, Trainer, default_data_collator

    set_seed(seed)
    train_config = config['finetuning']['training']

    # Create dataset (pre-tokenized with padding)
    dataset = create_dataset(train_data, tokenizer, train_config['max_seq_length'])

    # Use default collator since data is already padded
    data_collator = default_data_collator

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config['num_epochs'],
        per_device_train_batch_size=train_config['batch_size'],
        gradient_accumulation_steps=train_config['gradient_accumulation_steps'],
        learning_rate=train_config['learning_rate'],
        weight_decay=train_config['weight_decay'],
        warmup_ratio=train_config['warmup_ratio'],
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        seed=seed,
        report_to="none",
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"   Training for {train_config['num_epochs']} epoch(s)...")
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    result = trainer.train()

    training_time = time.time() - start_time
    peak_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    return {
        'train_loss': result.training_loss,
        'training_time_seconds': training_time,
        'peak_memory_gb': peak_memory,
    }


def save_adapter(model, tokenizer, output_path: Path, use_unsloth: bool) -> None:
    """Save LoRA adapter."""
    output_path.mkdir(parents=True, exist_ok=True)

    if use_unsloth:
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
    else:
        model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))


def main():
    """Main fine-tuning pipeline."""
    print("=" * 70)
    print("STEP 3: FINE-TUNING WITH LORA")
    print("=" * 70)

    config = load_config()
    mode = config.get('mode', 'pilot')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nDevice: {device}")
    print(f"Mode: {mode}")

    # Check Unsloth availability
    use_unsloth = config['finetuning']['use_unsloth'] and check_unsloth_available()
    if config['finetuning']['use_unsloth'] and not use_unsloth:
        print("   Warning: Unsloth requested but not available, using standard training")
    print(f"Using Unsloth: {use_unsloth}")

    base_dir = Path(__file__).parent.parent
    selections_dir = base_dir / "selections"
    output_dir = base_dir / "output"
    adapters_dir = output_dir / "adapters"
    adapters_dir.mkdir(parents=True, exist_ok=True)

    # Load selection summary
    summary_file = selections_dir / "selection_summary.json"
    if not summary_file.exists():
        print(f"   ERROR: Selection summary not found: {summary_file}")
        print(f"   Run 02_select_subsets.py first")
        return

    with open(summary_file, 'r') as f:
        selection_summary = json.load(f)

    # Determine models and datasets to train
    if mode == 'pilot':
        models_to_train = config['pilot']['models'][:1]  # First model only for pilot
        datasets_to_train = ['alpaca_gpt4']
    else:
        models_to_train = config['pilot']['models']
        datasets_to_train = list(selection_summary['selections'].keys())

    seeds = config['finetuning']['seeds']  # Use all configured seeds for statistical significance

    print(f"\nModels: {[m['name'] for m in models_to_train]}")
    print(f"Datasets: {datasets_to_train}")
    print(f"Seeds: {seeds}")

    # Selection methods to train on
    methods_to_train = ['full', 'random', 'diversity']

    all_results = []

    for model_config in models_to_train:
        model_name = model_config['name']
        model_short_name = model_name.split('/')[-1]

        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"{'=' * 70}")

        for dataset_name in datasets_to_train:
            print(f"\n   Dataset: {dataset_name}")
            print(f"   {'-' * 50}")

            for method in methods_to_train:
                # Find training data
                if method == 'full':
                    jsonl_path = selections_dir / f"{dataset_name}_full.jsonl"
                else:
                    jsonl_path = selections_dir / f"{dataset_name}_{method}.jsonl"

                if not jsonl_path.exists():
                    print(f"      {method}: SKIPPED (file not found)")
                    continue

                # Load training data
                train_data = load_training_data(jsonl_path)
                n_samples = len(train_data)

                print(f"\n      Method: {method} (n={n_samples})")

                for seed in seeds:
                    print(f"         Seed: {seed}")

                    # Check if adapter already exists
                    adapter_name = f"{model_short_name}_{dataset_name}_{method}_seed{seed}"
                    adapter_path = adapters_dir / adapter_name

                    if adapter_path.exists():
                        print(f"         Adapter exists, skipping: {adapter_path}")
                        continue

                    # Load model
                    if use_unsloth:
                        model, tokenizer = load_model_unsloth(model_name, config)
                        train_fn = train_with_unsloth
                    else:
                        model, tokenizer = load_model_standard(model_name, config)
                        train_fn = train_standard

                    # Train
                    run_output_dir = output_dir / "runs" / adapter_name
                    run_output_dir.mkdir(parents=True, exist_ok=True)

                    train_result = train_fn(
                        model, tokenizer, train_data, config,
                        run_output_dir, seed
                    )

                    print(f"         Loss: {train_result['train_loss']:.4f}")
                    print(f"         Time: {train_result['training_time_seconds']:.1f}s")
                    print(f"         Peak Memory: {train_result['peak_memory_gb']:.2f} GB")

                    # Save adapter
                    print(f"         Saving adapter to: {adapter_path}")
                    save_adapter(model, tokenizer, adapter_path, use_unsloth)

                    # Record result
                    result = TrainingResult(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        selection_method=method,
                        n_samples=n_samples,
                        seed=seed,
                        adapter_path=str(adapter_path),
                        training_time_seconds=train_result['training_time_seconds'],
                        peak_memory_gb=train_result['peak_memory_gb'],
                        train_loss=train_result['train_loss'],
                        config={
                            'lora': config['finetuning']['lora'],
                            'training': config['finetuning']['training'],
                        },
                    )
                    all_results.append(asdict(result))

                    # Cleanup
                    del model, tokenizer
                    clear_memory()

    # Save results
    results_file = output_dir / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'use_unsloth': use_unsloth,
            'results': all_results,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print("TRAINING RESULTS SUMMARY")
    print("=" * 70)

    # Group by model and dataset
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        key = (r['model_name'].split('/')[-1], r['dataset_name'])
        grouped[key].append(r)

    for (model, dataset), results in grouped.items():
        print(f"\n{model} / {dataset}:")
        for r in sorted(results, key=lambda x: x['n_samples']):
            print(f"   {r['selection_method']:12s}: n={r['n_samples']:6d}, "
                  f"loss={r['train_loss']:.4f}, time={r['training_time_seconds']:.0f}s")

    # Compute savings
    print(f"\n{'=' * 70}")
    print("COMPUTE SAVINGS")
    print("=" * 70)

    for (model, dataset), results in grouped.items():
        full_result = next((r for r in results if r['selection_method'] == 'full'), None)
        div_result = next((r for r in results if r['selection_method'] == 'diversity'), None)

        if full_result and div_result:
            time_savings = full_result['training_time_seconds'] - div_result['training_time_seconds']
            time_savings_pct = time_savings / full_result['training_time_seconds'] * 100

            print(f"\n{model} / {dataset}:")
            print(f"   Full training time: {full_result['training_time_seconds']:.0f}s")
            print(f"   Diversity (10%) time: {div_result['training_time_seconds']:.0f}s")
            print(f"   Time saved: {time_savings:.0f}s ({time_savings_pct:.1f}%)")

    print("\n" + "=" * 70)
    print("FINE-TUNING COMPLETE")
    print(f"Results saved to: {results_file}")
    print(f"Adapters saved to: {adapters_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
