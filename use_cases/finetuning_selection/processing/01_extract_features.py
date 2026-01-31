#!/usr/bin/env python3
"""
Extract linguistic features from instruction fine-tuning datasets.

This script:
1. Loads instruction datasets from HuggingFace
2. Extracts semantic embeddings (sentence-transformers)
3. Extracts syntactic features (spaCy dependency parsing)
4. Computes quality signals (perplexity, length ratios)
5. Saves features for diversity-based selection
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
from datasets import load_dataset

os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_instruction_response(item: dict, dataset_config: dict) -> Tuple[str, str]:
    """Extract instruction and response from a dataset item."""
    text_field = dataset_config.get('text_field', 'instruction')
    response_field = dataset_config.get('response_field', 'output')

    # Handle different dataset formats
    if text_field == 'conversations':
        # LIMA format: list of conversation turns
        convs = item.get('conversations', [])
        if len(convs) >= 2:
            instruction = convs[0] if isinstance(convs[0], str) else str(convs[0])
            response = convs[1] if isinstance(convs[1], str) else str(convs[1])
        elif len(convs) == 1:
            instruction = convs[0] if isinstance(convs[0], str) else str(convs[0])
            response = ""
        else:
            instruction, response = "", ""
    elif text_field == 'data':
        # UltraChat format: list of messages
        data = item.get('data', [])
        if len(data) >= 2:
            instruction = data[0] if isinstance(data[0], str) else str(data[0])
            response = data[1] if isinstance(data[1], str) else str(data[1])
        else:
            instruction, response = str(data), ""
    else:
        # Standard format with separate fields
        instruction = item.get(text_field, '')
        response = item.get(response_field, '')

        # Handle nested structures
        if isinstance(instruction, list):
            instruction = ' '.join(str(x) for x in instruction)
        if isinstance(response, list):
            response = ' '.join(str(x) for x in response)

    return str(instruction), str(response)


def load_instruction_dataset(dataset_config: dict, max_samples: int = None) -> List[Dict]:
    """Load and format an instruction dataset."""
    name = dataset_config['name']
    split = dataset_config.get('split', 'train')
    max_samples = max_samples or dataset_config.get('max_samples', 10000)

    print(f"   Loading {name}...")

    try:
        dataset = load_dataset(name, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"   Warning: Failed to load {name}: {e}")
        return []

    # Sample if needed
    if len(dataset) > max_samples:
        indices = np.random.RandomState(42).choice(len(dataset), max_samples, replace=False)
        dataset = dataset.select(indices)

    # Extract instruction-response pairs
    samples = []
    for item in tqdm(dataset, desc=f"   Processing {name}"):
        instruction, response = extract_instruction_response(item, dataset_config)
        if instruction.strip():  # Only include if we have an instruction
            samples.append({
                'instruction': instruction,
                'response': response,
                'source': name,
            })

    print(f"   Loaded {len(samples)} samples from {name}")
    return samples


def extract_semantic_features(samples: List[Dict], config: dict, device: str = 'cuda') -> np.ndarray:
    """Extract semantic embeddings for instruction+response pairs."""
    from sentence_transformers import SentenceTransformer

    model_name = config['features']['semantic']['model']
    batch_size = config['features']['semantic']['batch_size']

    print(f"   Loading semantic model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    # Encode instruction + response together for better semantic representation
    texts = []
    for sample in samples:
        combined = f"{sample['instruction']} [SEP] {sample['response']}"
        texts.append(combined)

    print(f"   Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings


def extract_instruction_embeddings(samples: List[Dict], config: dict, device: str = 'cuda') -> np.ndarray:
    """Extract embeddings for instructions only (for diversity in task space)."""
    from sentence_transformers import SentenceTransformer

    model_name = config['features']['instruction_specific']['model']
    batch_size = config['features']['semantic']['batch_size']

    print(f"   Loading instruction model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    instructions = [sample['instruction'] for sample in samples]

    print(f"   Encoding {len(instructions)} instructions...")
    embeddings = model.encode(
        instructions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    return embeddings


def extract_syntactic_features(samples: List[Dict], config: dict) -> np.ndarray:
    """Extract syntactic features from responses using spaCy."""
    import spacy
    from collections import Counter

    print(f"   Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("   Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Define feature dimensions
    pos_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
                'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
    dep_labels = ['ROOT', 'acl', 'acomp', 'advcl', 'advmod', 'agent', 'amod',
                  'appos', 'attr', 'aux', 'auxpass', 'case', 'cc', 'ccomp',
                  'compound', 'conj', 'csubj', 'dative', 'dep', 'det', 'dobj',
                  'expl', 'intj', 'mark', 'meta', 'neg', 'nmod', 'npadvmod',
                  'nsubj', 'nsubjpass', 'nummod', 'oprd', 'parataxis', 'pcomp',
                  'pobj', 'poss', 'preconj', 'predet', 'prep', 'prt', 'punct',
                  'quantmod', 'relcl', 'xcomp']

    pos_to_idx = {tag: i for i, tag in enumerate(pos_tags)}
    dep_to_idx = {label: i for i, label in enumerate(dep_labels)}

    features = []

    # Process responses (where the model learns generation patterns)
    responses = [sample['response'][:10000] for sample in samples]  # Truncate long responses

    for doc in tqdm(nlp.pipe(responses, batch_size=64), total=len(responses), desc="   Syntactic"):
        # POS distribution
        pos_counts = Counter(token.pos_ for token in doc)
        pos_vec = np.zeros(len(pos_tags))
        total_tokens = len(doc) if len(doc) > 0 else 1
        for tag, count in pos_counts.items():
            if tag in pos_to_idx:
                pos_vec[pos_to_idx[tag]] = count / total_tokens

        # Dependency distribution
        dep_counts = Counter(token.dep_ for token in doc)
        dep_vec = np.zeros(len(dep_labels))
        for label, count in dep_counts.items():
            if label in dep_to_idx:
                dep_vec[dep_to_idx[label]] = count / total_tokens

        # Additional structural features
        avg_depth = np.mean([abs(token.head.i - token.i) for token in doc]) if len(doc) > 0 else 0
        num_sentences = len(list(doc.sents)) if len(doc) > 0 else 0
        avg_sent_len = total_tokens / max(num_sentences, 1)

        # Combine features
        extra = np.array([avg_depth, num_sentences / max(total_tokens, 1), avg_sent_len / 100])
        feature_vec = np.concatenate([pos_vec, dep_vec, extra])
        features.append(feature_vec)

    return np.array(features)


def compute_quality_signals(samples: List[Dict]) -> np.ndarray:
    """Compute quality signals for filtering/weighting."""
    signals = []

    for sample in tqdm(samples, desc="   Quality signals"):
        instruction = sample['instruction']
        response = sample['response']

        # Length features
        inst_len = len(instruction.split())
        resp_len = len(response.split())
        length_ratio = resp_len / max(inst_len, 1)

        # Simple quality heuristics
        has_code = '```' in response or 'def ' in response or 'function ' in response
        has_steps = any(f"{i}." in response or f"{i})" in response for i in range(1, 10))
        has_explanation = len(response) > 100 and resp_len > 20

        # Format diversity
        has_bullets = '- ' in response or '* ' in response
        has_headers = '#' in response or '**' in response

        signal = np.array([
            inst_len / 100,  # Normalized instruction length
            resp_len / 500,  # Normalized response length
            length_ratio / 10,  # Response/instruction ratio
            float(has_code),
            float(has_steps),
            float(has_explanation),
            float(has_bullets),
            float(has_headers),
        ])
        signals.append(signal)

    return np.array(signals)


def check_existing_features(output_dir: Path) -> bool:
    """Check if all required feature files exist."""
    required_files = [
        "samples.json",
        "semantic_features.npy",
        "instruction_features.npy",
        "syntactic_features.npy",
        "quality_signals.npy",
        "metadata.json",
    ]
    return all((output_dir / f).exists() for f in required_files)


def main():
    print("=" * 70)
    print("STEP 1: EXTRACT FEATURES FROM INSTRUCTION DATASETS")
    print("=" * 70)

    config = load_config()
    mode = config.get('mode', 'quick')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    print(f"Mode: {mode}")

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "datasets"
    output_dir.mkdir(exist_ok=True)

    # Check if features already exist
    if check_existing_features(output_dir):
        print("\nFeatures already exist. Checking if they match current config...")
        with open(output_dir / "metadata.json", 'r') as f:
            existing_metadata = json.load(f)

        if existing_metadata.get('mode') == mode:
            print("Existing features match current mode. Skipping extraction.")
            print(f"  Samples: {existing_metadata.get('num_samples', 'N/A')}")
            print(f"  Datasets: {existing_metadata.get('datasets', [])}")
            print("\nTo re-extract, delete the datasets/ directory.")
            return
        else:
            print(f"Mode changed ({existing_metadata.get('mode')} -> {mode}). Re-extracting...")

    # Determine which datasets to process based on mode
    if mode == 'quick':
        # Quick mode: only process first dataset from each category
        datasets_to_process = []
        for category, datasets in config['datasets'].items():
            if datasets:
                ds = datasets[0].copy()
                ds['category'] = category
                ds['max_samples'] = min(ds.get('max_samples', 5000), 5000)  # Cap at 5k for quick
                datasets_to_process.append(ds)
    else:
        # Full mode: process all datasets
        datasets_to_process = []
        for category, datasets in config['datasets'].items():
            for ds in datasets:
                ds_copy = ds.copy()
                ds_copy['category'] = category
                datasets_to_process.append(ds_copy)

    print(f"\nProcessing {len(datasets_to_process)} datasets")

    # Load all samples
    all_samples = []
    for ds_config in datasets_to_process:
        print(f"\n{'=' * 70}")
        print(f"Loading: {ds_config['name']} ({ds_config['category']})")
        print(f"{'=' * 70}")

        samples = load_instruction_dataset(ds_config)
        all_samples.extend(samples)

    print(f"\n{'=' * 70}")
    print(f"Total samples loaded: {len(all_samples)}")
    print(f"{'=' * 70}")

    if len(all_samples) == 0:
        print("ERROR: No samples loaded. Check dataset configurations.")
        return

    # Save samples
    print("\n1. Saving samples...")
    samples_file = output_dir / "samples.json"
    with open(samples_file, 'w') as f:
        json.dump(all_samples, f, indent=2)
    print(f"   Saved {len(all_samples)} samples to {samples_file}")

    # Extract semantic features (instruction + response)
    print("\n2. Extracting semantic features...")
    semantic_features = extract_semantic_features(all_samples, config, device)
    np.save(output_dir / "semantic_features.npy", semantic_features)
    print(f"   Shape: {semantic_features.shape}")

    # Extract instruction embeddings (for task diversity)
    print("\n3. Extracting instruction embeddings...")
    instruction_features = extract_instruction_embeddings(all_samples, config, device)
    np.save(output_dir / "instruction_features.npy", instruction_features)
    print(f"   Shape: {instruction_features.shape}")

    # Extract syntactic features
    print("\n4. Extracting syntactic features...")
    syntactic_features = extract_syntactic_features(all_samples, config)
    np.save(output_dir / "syntactic_features.npy", syntactic_features)
    print(f"   Shape: {syntactic_features.shape}")

    # Compute quality signals
    print("\n5. Computing quality signals...")
    quality_signals = compute_quality_signals(all_samples)
    np.save(output_dir / "quality_signals.npy", quality_signals)
    print(f"   Shape: {quality_signals.shape}")

    # Save metadata
    metadata = {
        'num_samples': len(all_samples),
        'datasets': [ds['name'] for ds in datasets_to_process],
        'categories': list(config['datasets'].keys()),
        'semantic_dim': semantic_features.shape[1],
        'instruction_dim': instruction_features.shape[1],
        'syntactic_dim': syntactic_features.shape[1],
        'quality_dim': quality_signals.shape[1],
        'mode': mode,
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Print source distribution
    print("\n6. Source distribution:")
    source_counts = {}
    for sample in all_samples:
        source = sample['source']
        source_counts[source] = source_counts.get(source, 0) + 1

    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"   {source}: {count} samples")

    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION COMPLETE")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
