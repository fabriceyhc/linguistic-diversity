#!/usr/bin/env python3
"""Train language models on selected subsets.

This script trains three model architectures on each selection regime:
1. Encoder-only (ModernBERT) - Masked Language Modeling
2. Decoder-only (Llama 3.2 1B) - Causal Language Modeling
3. Encoder-decoder (Flan-T5 Large) - Seq2Seq Language Modeling
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import gc
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from torch.utils.data import Dataset, DataLoader

from model_loader import load_model, TokenizationWrapper, MODEL_REGISTRY

# Set environment variables
os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)


def load_config():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TextDataset(Dataset):
    """Dataset for text data supporting different model architectures."""

    def __init__(
        self,
        corpus_file: Path,
        tokenizer,
        model_type: str,
        max_length: int = 512,
        mlm_probability: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_length = max_length
        self.mlm_probability = mlm_probability

        # Load documents
        self.documents = []
        with open(corpus_file, 'r') as f:
            for line in f:
                doc = json.loads(line)
                text = doc.get('text', '')
                if text.strip():
                    self.documents.append(text)

        print(f"   Loaded {len(self.documents)} documents")

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        text = self.documents[idx]

        if self.model_type == "encoder":
            # For MLM, we'll handle masking in collate_fn
            return {"text": text}
        elif self.model_type == "decoder":
            # For causal LM
            return {"text": text}
        elif self.model_type == "encoder-decoder":
            # For seq2seq, create input-output pairs
            # Use denoising objective: corrupt input, reconstruct original
            return {"text": text, "target": text}

        return {"text": text}


class DataCollator:
    """Collator that handles tokenization and masking for different architectures."""

    def __init__(
        self,
        tokenizer,
        model_type: str,
        max_length: int = 512,
        mlm_probability: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.max_length = max_length
        self.mlm_probability = mlm_probability

    def __call__(self, batch):
        texts = [item["text"] for item in batch]

        if self.model_type == "encoder":
            return self._collate_encoder(texts)
        elif self.model_type == "decoder":
            return self._collate_decoder(texts)
        elif self.model_type == "encoder-decoder":
            targets = [item.get("target", item["text"]) for item in batch]
            return self._collate_encoder_decoder(texts, targets)

    def _collate_encoder(self, texts):
        """Collate for MLM (BERT-style)."""
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create labels (copy of input_ids)
        labels = inputs["input_ids"].clone()

        # Create MLM mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Don't mask special tokens
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # Don't mask padding
        if self.tokenizer.pad_token_id is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        # Create masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Set labels to -100 for non-masked tokens (ignored in loss)
        labels[~masked_indices] = -100

        # 80% of masked tokens -> [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
            inputs["input_ids"][indices_replaced] = self.tokenizer.mask_token_id

        # 10% of masked tokens -> random token
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs["input_ids"][indices_random] = random_words[indices_random]

        # 10% of masked tokens -> keep original (already done)

        inputs["labels"] = labels
        return inputs

    def _collate_decoder(self, texts):
        """Collate for causal LM (GPT-style)."""
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # For causal LM, labels are same as input_ids
        # Model handles shifting internally
        inputs["labels"] = inputs["input_ids"].clone()

        # Mask padding tokens in labels
        if self.tokenizer.pad_token_id is not None:
            inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100

        return inputs

    def _collate_encoder_decoder(self, texts, targets):
        """Collate for seq2seq (T5-style)."""
        # Add task prefix for T5
        prefixed_texts = [f"denoise: {t}" for t in texts]

        inputs = self.tokenizer(
            prefixed_texts,
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Mask padding in labels
        if self.tokenizer.pad_token_id is not None:
            inputs["labels"][inputs["labels"] == self.tokenizer.pad_token_id] = -100

        return inputs


def create_optimizer(model, config: dict):
    """Create optimizer and scheduler."""
    opt_config = config['training']['optimizer']
    train_config = config['training']['training']
    mode = config['mode']

    # Filter parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params,
        lr=opt_config['learning_rate'],
        weight_decay=opt_config['weight_decay'],
        betas=(opt_config['beta1'], opt_config['beta2']),
    )

    # Create scheduler
    max_steps = train_config['max_steps'][mode]
    warmup_steps = opt_config['warmup_steps']

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.1, 0.5 * (1 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def train_step(model, batch, optimizer, scheduler, device, gradient_clip: float, accumulation_steps: int = 1, step: int = 0):
    """Perform a single training step with gradient accumulation."""
    model.train()

    # Move batch to device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss / accumulation_steps  # Scale loss for accumulation

    # Backward pass
    loss.backward()

    # Only update weights every accumulation_steps
    if (step + 1) % accumulation_steps == 0:
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return loss.item() * accumulation_steps  # Return unscaled loss for logging


def save_checkpoint(model, optimizer, scheduler, step, loss, output_dir: Path, is_final: bool = False):
    """Save model checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_final:
        checkpoint_path = output_dir / "model_final.pt"
    else:
        checkpoint_path = output_dir / f"checkpoint_{step}.pt"

    # For large models, just save the model state dict
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }, checkpoint_path)

    # Also save just the model for easier loading
    if is_final:
        model_path = output_dir / "pytorch_model.bin"
        torch.save(model.state_dict(), model_path)


def train_model(
    corpus_file: Path,
    output_dir: Path,
    model_type: str,
    regime_name: str,
    dataset_name: str,
    config: dict,
) -> dict:
    """Train a single model on a corpus.

    Args:
        corpus_file: Path to corpus JSONL file
        output_dir: Output directory for checkpoints
        model_type: One of 'encoder', 'decoder', 'encoder-decoder'
        regime_name: Name of selection regime
        dataset_name: Name of dataset
        config: Configuration dict

    Returns:
        Dictionary of training results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = MODEL_REGISTRY[model_type]
    print(f"\n{'=' * 80}")
    print(f"Training: {dataset_name} / {regime_name} / {model_type}")
    print(f"Model: {model_info.model_id}")
    print(f"{'=' * 80}")

    # Load model and tokenizer
    print(f"\n1. Loading model...")
    tokenizer, model, model_config = load_model(model_type)
    device = next(model.parameters()).device

    # Create dataset and dataloader
    print(f"\n2. Creating dataloader...")
    dataset = TextDataset(
        corpus_file=corpus_file,
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=model_config.max_length,
    )

    collator = DataCollator(
        tokenizer=tokenizer,
        model_type=model_type,
        max_length=model_config.max_length,
    )

    train_config = config['training']['training']
    dataloader = DataLoader(
        dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Avoid multiprocessing issues
    )
    print(f"   ✓ {len(dataloader)} batches per epoch")

    # Create optimizer
    print(f"\n3. Setting up optimizer...")
    optimizer, scheduler = create_optimizer(model, config)

    # Training configuration
    mode = config['mode']
    max_steps = train_config['max_steps'][mode]
    save_interval = train_config['save_interval']
    log_interval = train_config['log_interval']
    gradient_clip = train_config['gradient_clip']
    accumulation_steps = train_config['gradient_accumulation_steps']
    early_stopping_patience = train_config.get('early_stopping_patience', 500)
    early_stopping_min_delta = train_config.get('early_stopping_min_delta', 0.001)

    print(f"   Max steps: {max_steps}")
    print(f"   Gradient accumulation: {accumulation_steps}")
    print(f"   Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")

    # Training loop
    print(f"\n4. Training...")
    print(f"   {'- ' * 39}")

    global_step = 0
    epoch = 0
    training_losses = []
    best_loss = float('inf')
    steps_without_improvement = 0
    early_stopped = False

    dataloader_iterator = iter(dataloader)
    pbar = tqdm(total=max_steps, desc="   Training")

    while global_step < max_steps:
        # Get next batch
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(dataloader)
            batch = next(dataloader_iterator)
            epoch += 1

        # Training step
        loss = train_step(model, batch, optimizer, scheduler, device, gradient_clip, accumulation_steps, global_step)

        global_step += 1
        pbar.update(1)

        # Log
        if global_step % log_interval == 0:
            pbar.set_postfix({'loss': f'{loss:.4f}', 'epoch': epoch})
            training_losses.append({
                'step': global_step,
                'loss': loss
            })

            # Early stopping check
            if loss < best_loss - early_stopping_min_delta:
                best_loss = loss
                steps_without_improvement = 0
            else:
                steps_without_improvement += log_interval

            if steps_without_improvement >= early_stopping_patience:
                print(f"\n   Early stopping at step {global_step}")
                early_stopped = True
                break

        # Save checkpoint
        if global_step % save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, loss, output_dir)

            log_file = output_dir / "training_log.json"
            with open(log_file, 'w') as f:
                json.dump(training_losses, f, indent=2)

    pbar.close()

    # Save final checkpoint
    print(f"\n5. Saving final checkpoint...")
    save_checkpoint(model, optimizer, scheduler, global_step, loss, output_dir, is_final=True)

    # Save training log
    log_file = output_dir / "training_log.json"
    with open(log_file, 'w') as f:
        json.dump(training_losses, f, indent=2)

    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'regime': regime_name,
        'model_type': model_type,
        'model_id': model_info.model_id,
        'total_steps': global_step,
        'total_epochs': epoch,
        'final_loss': loss,
        'early_stopped': early_stopped,
        'best_loss': best_loss,
    }

    metadata_file = output_dir / "training_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 80}")
    print(f"✓ Training complete: {dataset_name} / {regime_name} / {model_type}")
    print(f"  Final loss: {loss:.4f}, Steps: {global_step}")
    print(f"{'=' * 80}")

    # Clean up GPU memory
    del model
    del optimizer
    del scheduler
    del dataloader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'final_loss': loss,
        'total_steps': global_step,
        'early_stopped': early_stopped,
    }


def main():
    """Main execution function."""
    print("=" * 80)
    print("STEP 4: TRAIN MODELS (Multi-Architecture)")
    print("=" * 80)

    # Load configuration
    config = load_config()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Setup directories
    base_dir = Path(__file__).parent.parent
    datasets_dir = base_dir / "datasets"
    models_dir = base_dir / "models"

    # Model types to train
    model_types = ['encoder', 'decoder', 'encoder-decoder']

    # Selection regimes
    regimes = [
        'semantic_diversity',
        'syntactic_diversity',
        'morphological_diversity',
        'phonological_diversity',
        'composite_diversity',
        'universal_diversity',
        'random_baseline',
        'full_dataset',  # No subsampling - train on all data
    ]

    # Train models
    results = {}

    for dataset_config in config['corpus']['datasets']:
        dataset_name = dataset_config['name']
        clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()

        results[dataset_name] = {}

        for regime in regimes:
            results[dataset_name][regime] = {}

            for model_type in model_types:
                try:
                    # Find corpus file
                    corpus_file = datasets_dir / regime / clean_name / "corpus.jsonl"

                    if not corpus_file.exists():
                        print(f"\n✗ Corpus not found: {corpus_file}")
                        continue

                    # Output directory
                    output_dir = models_dir / clean_name / regime / model_type

                    # Skip if model already trained
                    final_checkpoint = output_dir / "model_final.pt"
                    if final_checkpoint.exists():
                        print(f"\n✓ Skipping (already trained): {regime} / {model_type}")
                        continue

                    # Train model
                    result = train_model(
                        corpus_file=corpus_file,
                        output_dir=output_dir,
                        model_type=model_type,
                        regime_name=regime,
                        dataset_name=dataset_name,
                        config=config,
                    )

                    results[dataset_name][regime][model_type] = result

                except Exception as e:
                    print(f"\n{'=' * 80}")
                    print(f"✗ Failed: {dataset_name} / {regime} / {model_type}")
                    print(f"  Error: {e}")
                    print(f"{'=' * 80}")
                    import traceback
                    traceback.print_exc()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

    # Save results
    print(f"\n{'=' * 80}")
    print("TRAINING SUMMARY")
    print(f"{'=' * 80}")

    for dataset_name, dataset_results in results.items():
        print(f"\n{dataset_name}:")
        for regime, regime_results in dataset_results.items():
            print(f"  {regime}:")
            for model_type, result in regime_results.items():
                if isinstance(result, dict):
                    print(f"    {model_type:20s}: loss={result['final_loss']:.4f}, steps={result['total_steps']}")

    results_file = models_dir / "training_results.json"
    models_dir.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results: {results_file}")
    print("\n" + "=" * 80)
    print("✓ MODEL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
