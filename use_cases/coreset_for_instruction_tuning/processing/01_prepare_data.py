#!/usr/bin/env python3
"""
Step 1: Data Preparation for Coreset Selection

This script:
1. Downloads instruction datasets from HuggingFace
2. Generates semantic embeddings using sentence-transformers
3. Computes length statistics for bias monitoring
4. Saves processed data for selection algorithms
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import os
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_config() -> dict:
    """Load experiment configuration."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class DatasetStats:
    """Statistics for a processed dataset."""
    name: str
    num_samples: int
    avg_instruction_length: float
    avg_response_length: float
    median_instruction_length: float
    median_response_length: float
    min_response_length: int
    max_response_length: int
    embedding_dim: int
    timestamp: str


def format_alpaca_instruction(item: dict, config: dict) -> Tuple[str, str, str]:
    """
    Format an Alpaca-style instruction item.

    Returns:
        (instruction, input_context, response)
    """
    text_field = config.get('text_field', 'instruction')
    input_field = config.get('input_field', 'input')
    response_field = config.get('response_field', 'output')

    instruction = item.get(text_field, '')
    input_context = item.get(input_field, '')
    response = item.get(response_field, '')

    # Ensure strings
    instruction = str(instruction) if instruction else ''
    input_context = str(input_context) if input_context else ''
    response = str(response) if response else ''

    return instruction, input_context, response


def format_orca_instruction(item: dict, config: dict) -> Tuple[str, str, str]:
    """Format an OpenOrca-style instruction item."""
    text_field = config.get('text_field', 'question')
    response_field = config.get('response_field', 'response')

    # OpenOrca has system_prompt, question, response
    system_prompt = item.get('system_prompt', '')
    instruction = item.get(text_field, '')
    response = item.get(response_field, '')

    # Combine system prompt with instruction if present
    if system_prompt:
        instruction = f"{system_prompt}\n\n{instruction}"

    return str(instruction), '', str(response)


def format_ultrachat_instruction(item: dict, config: dict) -> Tuple[str, str, str]:
    """Format an UltraChat-style conversation item."""
    messages = item.get('messages', [])

    if not messages:
        return '', '', ''

    # Extract first user message as instruction, first assistant as response
    instruction = ''
    response = ''

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        if role == 'user' and not instruction:
            instruction = content
        elif role == 'assistant' and not response:
            response = content

        if instruction and response:
            break

    return str(instruction), '', str(response)


def load_dataset_hf(dataset_name: str, split: str, max_samples: Optional[int] = None):
    """Load dataset from HuggingFace."""
    from datasets import load_dataset

    print(f"   Loading {dataset_name} (split: {split})...")

    try:
        dataset = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"   Warning: Failed to load {dataset_name}: {e}")
        return None

    print(f"   Loaded {len(dataset)} samples")

    # Sample if needed
    if max_samples and len(dataset) > max_samples:
        print(f"   Sampling {max_samples} from {len(dataset)}")
        indices = np.random.RandomState(42).choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(indices)

    return dataset


def process_dataset(
    dataset,
    dataset_config: dict,
    dataset_name: str,
) -> List[Dict]:
    """
    Process a dataset into standardized format.

    Returns list of dicts with keys:
        - instruction: The user instruction/question
        - input: Optional additional context
        - response: The expected response
        - source: Dataset source name
        - original_idx: Original index in dataset
    """
    samples = []

    # Determine format based on dataset name
    if 'alpaca' in dataset_name.lower():
        format_fn = format_alpaca_instruction
    elif 'orca' in dataset_name.lower():
        format_fn = format_orca_instruction
    elif 'ultrachat' in dataset_name.lower():
        format_fn = format_ultrachat_instruction
    else:
        format_fn = format_alpaca_instruction  # Default to Alpaca format

    for idx, item in enumerate(tqdm(dataset, desc=f"   Processing {dataset_name}")):
        instruction, input_context, response = format_fn(item, dataset_config)

        # Skip empty samples
        if not instruction.strip() or not response.strip():
            continue

        samples.append({
            'instruction': instruction.strip(),
            'input': input_context.strip(),
            'response': response.strip(),
            'source': dataset_name,
            'original_idx': idx,
        })

    return samples


def generate_embeddings(
    samples: List[Dict],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Generate semantic embeddings for instruction + response pairs.

    We embed the combined instruction + response to capture the full
    semantic content of each training example.
    """
    from sentence_transformers import SentenceTransformer

    print(f"\n   Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Combine instruction and response for embedding
    # This captures the full semantic content of the training example
    texts = []
    for sample in samples:
        combined = sample['instruction']
        if sample['input']:
            combined += f"\n{sample['input']}"
        combined += f"\n{sample['response']}"
        texts.append(combined)

    print(f"   Generating embeddings for {len(texts)} samples...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 normalize for cosine similarity
    )

    return embeddings


def generate_instruction_embeddings(
    samples: List[Dict],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Generate embeddings for instructions only.

    This captures task diversity (what the model is asked to do)
    separately from response diversity.
    """
    from sentence_transformers import SentenceTransformer

    print(f"\n   Loading embedding model for instructions: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Embed just the instruction (+input if present)
    texts = []
    for sample in samples:
        text = sample['instruction']
        if sample['input']:
            text += f"\n{sample['input']}"
        texts.append(text)

    print(f"   Generating instruction embeddings for {len(texts)} samples...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings


def compute_length_stats(samples: List[Dict]) -> Dict:
    """
    Compute length statistics for bias monitoring.

    Per spec warning: "Diversity metrics sometimes favor long, verbose outputs"
    """
    instruction_lengths = [len(s['instruction'].split()) for s in samples]
    response_lengths = [len(s['response'].split()) for s in samples]

    stats = {
        'instruction': {
            'mean': float(np.mean(instruction_lengths)),
            'median': float(np.median(instruction_lengths)),
            'std': float(np.std(instruction_lengths)),
            'min': int(np.min(instruction_lengths)),
            'max': int(np.max(instruction_lengths)),
            'p25': float(np.percentile(instruction_lengths, 25)),
            'p75': float(np.percentile(instruction_lengths, 75)),
        },
        'response': {
            'mean': float(np.mean(response_lengths)),
            'median': float(np.median(response_lengths)),
            'std': float(np.std(response_lengths)),
            'min': int(np.min(response_lengths)),
            'max': int(np.max(response_lengths)),
            'p25': float(np.percentile(response_lengths, 25)),
            'p75': float(np.percentile(response_lengths, 75)),
        },
    }

    return stats


def main():
    """Main data preparation pipeline."""
    print("=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)

    config = load_config()
    mode = config.get('mode', 'pilot')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\nDevice: {device}")
    print(f"Mode: {mode}")

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "data"
    output_dir.mkdir(exist_ok=True)

    # Determine which dataset(s) to process
    if mode == 'pilot':
        datasets_to_process = [config['pilot']['dataset']]
    else:
        datasets_to_process = [config['pilot']['dataset']] + config['scaleup']['datasets']

    selection_config = config['selection']
    embedding_model = selection_config['embedding_model']
    batch_size = selection_config['embedding_batch_size']

    for ds_config in datasets_to_process:
        dataset_name = ds_config['name']
        short_name = dataset_name.split('/')[-1].lower().replace('-', '_')

        print(f"\n{'=' * 70}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 70}")

        # Check if already processed
        embeddings_file = output_dir / f"{short_name}_embeddings.npy"
        if embeddings_file.exists():
            print(f"   Found existing data for {short_name}, skipping...")
            continue

        # Load dataset
        split = ds_config.get('split', 'train')
        max_samples = ds_config.get('max_samples')

        dataset = load_dataset_hf(dataset_name, split, max_samples)
        if dataset is None:
            print(f"   ERROR: Could not load {dataset_name}")
            continue

        # Process into standardized format
        samples = process_dataset(dataset, ds_config, dataset_name)
        print(f"   Processed {len(samples)} valid samples")

        if len(samples) == 0:
            print(f"   ERROR: No valid samples extracted")
            continue

        # Generate embeddings (combined instruction + response)
        embeddings = generate_embeddings(samples, embedding_model, batch_size, device)
        print(f"   Embedding shape: {embeddings.shape}")

        # Generate instruction-only embeddings (for task diversity)
        instruction_embeddings = generate_instruction_embeddings(
            samples, embedding_model, batch_size, device
        )
        print(f"   Instruction embedding shape: {instruction_embeddings.shape}")

        # Compute length statistics
        length_stats = compute_length_stats(samples)
        print(f"\n   Length Statistics:")
        print(f"      Instruction: mean={length_stats['instruction']['mean']:.1f}, "
              f"median={length_stats['instruction']['median']:.1f}")
        print(f"      Response: mean={length_stats['response']['mean']:.1f}, "
              f"median={length_stats['response']['median']:.1f}")

        # Create dataset stats
        stats = DatasetStats(
            name=dataset_name,
            num_samples=len(samples),
            avg_instruction_length=length_stats['instruction']['mean'],
            avg_response_length=length_stats['response']['mean'],
            median_instruction_length=length_stats['instruction']['median'],
            median_response_length=length_stats['response']['median'],
            min_response_length=length_stats['response']['min'],
            max_response_length=length_stats['response']['max'],
            embedding_dim=embeddings.shape[1],
            timestamp=datetime.now().isoformat(),
        )

        # Save outputs
        print(f"\n   Saving outputs to {output_dir}...")

        # Samples (JSON)
        with open(output_dir / f"{short_name}_samples.json", 'w') as f:
            json.dump(samples, f, indent=2)

        # Embeddings (numpy)
        np.save(output_dir / f"{short_name}_embeddings.npy", embeddings)
        np.save(output_dir / f"{short_name}_instruction_embeddings.npy", instruction_embeddings)

        # Stats and metadata
        metadata = {
            'stats': asdict(stats),
            'length_stats': length_stats,
            'config': ds_config,
            'embedding_model': embedding_model,
        }
        with open(output_dir / f"{short_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   Saved: {short_name}_samples.json ({len(samples)} samples)")
        print(f"   Saved: {short_name}_embeddings.npy ({embeddings.shape})")
        print(f"   Saved: {short_name}_instruction_embeddings.npy ({instruction_embeddings.shape})")
        print(f"   Saved: {short_name}_metadata.json")

        # Clean up
        del dataset, samples, embeddings, instruction_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
