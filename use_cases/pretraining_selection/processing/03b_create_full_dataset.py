#!/usr/bin/env python3
"""Create full_dataset corpus (no subsampling) from existing data."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import json
import yaml
import pickle

def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("=" * 80)
    print("CREATING FULL_DATASET CORPUS")
    print("=" * 80)
    
    config = load_config()
    base_dir = Path(__file__).parent.parent
    datasets_dir = base_dir / "datasets"
    input_dir = base_dir / "input"
    
    for dataset_config in config['corpus']['datasets']:
        dataset_name = dataset_config['name']
        clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()
        
        print(f"\nProcessing: {dataset_name}")
        
        # Load original training documents from pickle
        train_file = input_dir / f"{clean_name}_train.pkl"
        if not train_file.exists():
            print(f"   ✗ Training data not found: {train_file}")
            continue
            
        with open(train_file, 'rb') as f:
            documents = pickle.load(f)
        
        n_items = len(documents)
        print(f"   Total training documents: {n_items}")
        
        # Create full_dataset directory
        regime_dir = datasets_dir / "full_dataset" / clean_name
        regime_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all indices
        full_indices = list(range(n_items))
        indices_file = regime_dir / "selected_indices.json"
        with open(indices_file, 'w') as f:
            json.dump(full_indices, f)
        print(f"   ✓ Saved indices: {indices_file}")
        
        # Save corpus (all documents)
        corpus_file = regime_dir / "corpus.jsonl"
        with open(corpus_file, 'w') as f:
            for doc in documents:
                json.dump({"text": doc}, f)
                f.write('\n')
        print(f"   ✓ Saved corpus: {corpus_file} ({n_items} documents)")
        
        # Save metadata
        metadata = {
            'dataset_name': dataset_name,
            'regime': 'full_dataset',
            'n_total': n_items,
            'n_selected': n_items,
            'selection_ratio': 1.0,
        }
        metadata_file = regime_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   ✓ Saved metadata: {metadata_file}")
    
    print("\n" + "=" * 80)
    print("✓ FULL_DATASET CORPUS CREATED")
    print("=" * 80)

if __name__ == "__main__":
    main()
