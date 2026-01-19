#!/usr/bin/env python3
"""Download and preprocess corpus data for pretraining selection experiments.

This script:
1. Downloads TinyStories and FineWeb-Edu datasets from Hugging Face
2. Applies basic cleaning and filtering
3. Splits into train/test sets
4. Saves preprocessed data for feature extraction
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import re
import pickle
import pandas as pd
import yaml
from datasets import load_dataset
from tqdm import tqdm

# Set environment variables for temporary directories
os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)


def load_config():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clean_text(text):
    """Basic text cleaning.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def filter_document(text, min_length=50, max_length=5000):
    """Check if document meets filtering criteria.

    Args:
        text: Document text
        min_length: Minimum character length
        max_length: Maximum character length

    Returns:
        bool: True if document passes filters
    """
    if not text or not isinstance(text, str):
        return False

    length = len(text)
    if length < min_length or length > max_length:
        return False

    # Check for reasonable word count (at least 10 words)
    words = text.split()
    if len(words) < 10:
        return False

    return True


def download_dataset(dataset_config, max_samples, config):
    """Download and process a single dataset.

    Args:
        dataset_config: Dataset configuration dict
        max_samples: Maximum number of samples to keep
        config: Full configuration dict

    Returns:
        List of cleaned documents
    """
    dataset_name = dataset_config['name']
    text_column = dataset_config['text_column']

    print(f"\n{'=' * 80}")
    print(f"Processing: {dataset_name}")
    print(f"{'=' * 80}")

    # Download dataset
    print(f"\n1. Downloading from Hugging Face...")
    try:
        if 'subset' in dataset_config:
            dataset = load_dataset(dataset_name, dataset_config['subset'], split='train', streaming=True)
            print(f"   ✓ Loaded dataset (streaming mode): {dataset_name}/{dataset_config['subset']}")
        else:
            dataset = load_dataset(dataset_name, split='train')
            print(f"   ✓ Loaded dataset: {dataset_name}")
    except Exception as e:
        print(f"   ✗ Error downloading dataset: {e}")
        print("\n   Troubleshooting:")
        print("   - Check internet connection")
        print("   - Verify dataset name is correct")
        print("   - Try: pip install --upgrade datasets")
        raise

    # Process documents
    print(f"\n2. Cleaning and filtering documents...")
    documents = []
    min_length = config['corpus']['min_length']
    max_length = config['corpus']['max_length']

    # Handle streaming datasets
    is_streaming = hasattr(dataset, 'take')

    if is_streaming:
        # For streaming datasets, take more than needed to account for filtering
        sample_iterator = dataset.take(max_samples * 3)
        iterator = iter(sample_iterator)
    else:
        iterator = iter(dataset)

    pbar = tqdm(total=max_samples, desc="   Processing")

    for sample in iterator:
        if len(documents) >= max_samples:
            break

        try:
            # Extract text
            text = sample[text_column]

            # Clean
            text = clean_text(text)

            # Filter
            if filter_document(text, min_length, max_length):
                documents.append(text)
                pbar.update(1)

        except (KeyError, TypeError) as e:
            continue

    pbar.close()

    print(f"   ✓ Kept {len(documents)} documents after filtering")

    return documents


def split_and_save(documents, dataset_name, train_ratio, output_dir):
    """Split into train/test and save.

    Args:
        documents: List of document strings
        dataset_name: Name of the dataset (for file naming)
        train_ratio: Fraction of data for training
        output_dir: Output directory path
    """
    import numpy as np

    print(f"\n3. Splitting into train/test...")
    n_docs = len(documents)
    n_train = int(n_docs * train_ratio)

    # Shuffle
    indices = np.random.permutation(n_docs)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_docs = [documents[i] for i in train_indices]
    test_docs = [documents[i] for i in test_indices]

    print(f"   Train: {len(train_docs)} documents")
    print(f"   Test:  {len(test_docs)} documents")

    # Save
    print(f"\n4. Saving preprocessed data...")

    # Clean dataset name for filename
    clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()

    train_file = output_dir / f"{clean_name}_train.pkl"
    test_file = output_dir / f"{clean_name}_test.pkl"

    with open(train_file, 'wb') as f:
        pickle.dump(train_docs, f)
    print(f"   ✓ Saved: {train_file}")

    with open(test_file, 'wb') as f:
        pickle.dump(test_docs, f)
    print(f"   ✓ Saved: {test_file}")

    # Also save metadata
    metadata = {
        'dataset_name': dataset_name,
        'n_train': len(train_docs),
        'n_test': len(test_docs),
        'train_file': str(train_file),
        'test_file': str(test_file),
    }

    metadata_file = output_dir / f"{clean_name}_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✓ Saved metadata: {metadata_file}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("STEP 1: DOWNLOAD AND PREPROCESS CORPUS DATA")
    print("=" * 80)

    # Load configuration
    config = load_config()
    mode = config['mode']
    max_samples = config['corpus']['max_samples'][mode]
    train_ratio = config['corpus']['train_test_split']

    print(f"\nConfiguration:")
    print(f"  Mode: {mode}")
    print(f"  Max samples per dataset: {max_samples}")
    print(f"  Train/test split: {train_ratio:.1%} / {1-train_ratio:.1%}")

    # Set random seed for reproducibility
    import numpy as np
    np.random.seed(42)

    # Ensure output directory exists
    output_dir = Path(__file__).parent.parent / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download and process each dataset
    for dataset_config in config['corpus']['datasets']:
        try:
            documents = download_dataset(dataset_config, max_samples, config)
            split_and_save(documents, dataset_config['name'], train_ratio, output_dir)
            print(f"\n{'=' * 80}")
            print(f"✓ Successfully processed: {dataset_config['name']}")
            print(f"{'=' * 80}")
        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"✗ Failed to process: {dataset_config['name']}")
            print(f"  Error: {e}")
            print(f"{'=' * 80}")
            continue

    print("\n" + "=" * 80)
    print("✓ DATA DOWNLOAD AND PREPROCESSING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
