"""Utility functions for model training and evaluation.

This module provides:
- GPT-2 model initialization with custom configuration
- Multi-GPU setup utilities
- Checkpoint saving/loading
- Training and evaluation step functions
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


class TextDataset(Dataset):
    """Simple dataset for text corpus."""

    def __init__(self, texts: List[str], tokenizer: GPT2Tokenizer, context_length: int):
        """
        Args:
            texts: List of document strings
            tokenizer: GPT-2 tokenizer
            context_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.context_length = context_length

        # Tokenize all texts
        print(f"   Tokenizing {len(texts)} documents...")
        self.examples = []

        for text in texts:
            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=True)

            # Split into chunks of context_length
            for i in range(0, len(tokens) - context_length + 1, context_length):
                chunk = tokens[i:i + context_length]
                if len(chunk) == context_length:
                    self.examples.append(torch.tensor(chunk))

        print(f"   ✓ Created {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def create_model(config: Dict) -> Tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Create GPT-2 model with custom configuration.

    Args:
        config: Model configuration dict

    Returns:
        tuple: (model, tokenizer)
    """
    model_config = config['training']['model']

    # Create GPT-2 configuration
    gpt_config = GPT2Config(
        vocab_size=model_config['vocab_size'],
        n_positions=model_config['context_length'],
        n_embd=model_config['n_embed'],
        n_layer=model_config['n_layers'],
        n_head=model_config['n_heads'],
        resid_pdrop=model_config['dropout'],
        embd_pdrop=model_config['dropout'],
        attn_pdrop=model_config['dropout'],
    )

    # Initialize model
    model = GPT2LMHeadModel(gpt_config)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {n_params / 1e6:.1f}M")

    return model, tokenizer


def setup_multi_gpu(model: nn.Module, config: Dict) -> nn.Module:
    """Setup model for multi-GPU training.

    Args:
        model: PyTorch model
        config: Configuration dict

    Returns:
        Model wrapped for multi-GPU if enabled
    """
    multi_gpu_config = config['training']['multi_gpu']

    if not multi_gpu_config['enabled']:
        return model

    if not torch.cuda.is_available():
        print("   Warning: Multi-GPU requested but CUDA not available")
        return model

    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        print(f"   Only {n_gpus} GPU available, using single GPU")
        return model

    print(f"   Setting up {n_gpus} GPUs")

    backend = multi_gpu_config['backend']
    if backend == 'DataParallel':
        model = nn.DataParallel(model)
    elif backend == 'DistributedDataParallel':
        # For simplicity, we use DataParallel in this implementation
        # DistributedDataParallel requires more setup
        print("   Note: Using DataParallel instead of DistributedDataParallel for simplicity")
        model = nn.DataParallel(model)

    return model


def create_dataloader(corpus_file: Path, tokenizer: GPT2Tokenizer, config: Dict, shuffle: bool = True) -> DataLoader:
    """Create dataloader from corpus file.

    Args:
        corpus_file: Path to corpus JSONL file
        tokenizer: GPT-2 tokenizer
        config: Configuration dict
        shuffle: Whether to shuffle data

    Returns:
        DataLoader
    """
    # Load corpus
    texts = []
    with open(corpus_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            texts.append(data['text'])

    # Create dataset
    context_length = config['training']['model']['context_length']
    dataset = TextDataset(texts, tokenizer, context_length)

    # Create dataloader
    batch_size = config['training']['training']['batch_size']
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return dataloader


def create_optimizer(model: nn.Module, config: Dict) -> Tuple[torch.optim.Optimizer, object]:
    """Create optimizer and learning rate scheduler.

    Args:
        model: PyTorch model
        config: Configuration dict

    Returns:
        tuple: (optimizer, scheduler)
    """
    opt_config = config['training']['optimizer']
    train_config = config['training']['training']

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_config['learning_rate'],
        weight_decay=opt_config['weight_decay'],
        betas=(opt_config['beta1'], opt_config['beta2'])
    )

    # Create scheduler
    mode = config['mode']
    max_steps = train_config['max_steps'][mode]
    warmup_steps = opt_config['warmup_steps']

    if opt_config['lr_schedule'] == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps
        )

    return optimizer, scheduler


def train_step(model: nn.Module, batch: torch.Tensor, optimizer: torch.optim.Optimizer,
               scheduler: object, device: str, gradient_clip: float = 1.0) -> float:
    """Perform single training step.

    Args:
        model: PyTorch model
        batch: Input batch
        optimizer: Optimizer
        scheduler: LR scheduler
        device: Device to use
        gradient_clip: Gradient clipping value

    Returns:
        Loss value
    """
    model.train()

    # Move batch to device
    batch = batch.to(device)

    # Forward pass
    outputs = model(batch, labels=batch)
    loss = outputs.loss

    # Handle multi-GPU (loss is mean of all GPUs)
    if isinstance(model, nn.DataParallel):
        loss = loss.mean()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

    # Update weights
    optimizer.step()
    scheduler.step()

    return loss.item()


def evaluate(model: nn.Module, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: PyTorch model
        dataloader: Evaluation dataloader
        device: Device to use

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            outputs = model(batch, labels=batch)
            loss = outputs.loss

            # Handle multi-GPU
            if isinstance(model, nn.DataParallel):
                loss = loss.mean()

            total_loss += loss.item() * batch.size(0) * batch.size(1)
            total_tokens += batch.size(0) * batch.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return {
        'loss': avg_loss,
        'perplexity': perplexity
    }


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: object, step: int, loss: float,
                   output_dir: Path, is_final: bool = False) -> None:
    """Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: LR scheduler
        step: Current training step
        loss: Current loss
        output_dir: Output directory
        is_final: Whether this is the final checkpoint
    """
    # Handle DataParallel
    model_to_save = model.module if isinstance(model, nn.DataParallel) else model

    checkpoint = {
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
    }

    if is_final:
        checkpoint_file = output_dir / "final_checkpoint.pt"
    else:
        checkpoint_file = output_dir / f"checkpoint_step_{step}.pt"

    torch.save(checkpoint, checkpoint_file)
    print(f"   ✓ Saved checkpoint: {checkpoint_file}")


def load_checkpoint(checkpoint_path: Path, model: nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None,
                   scheduler: Optional[object] = None) -> int:
    """Load model checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        model: PyTorch model
        optimizer: Optimizer (optional)
        scheduler: LR scheduler (optional)

    Returns:
        Training step of loaded checkpoint
    """
    checkpoint = torch.load(checkpoint_path)

    # Handle DataParallel
    model_to_load = model.module if isinstance(model, nn.DataParallel) else model
    model_to_load.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['step']


def count_tokens(corpus_file: Path, tokenizer: GPT2Tokenizer) -> int:
    """Count total tokens in corpus.

    Args:
        corpus_file: Path to corpus JSONL file
        tokenizer: GPT-2 tokenizer

    Returns:
        Total token count
    """
    total_tokens = 0

    with open(corpus_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            tokens = tokenizer.encode(data['text'])
            total_tokens += len(tokens)

    return total_tokens
