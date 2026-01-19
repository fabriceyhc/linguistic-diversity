#!/usr/bin/env python3
"""Train models on full_dataset only."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Import the training function from the main script
from processing.train_model import train_model_on_corpus
import yaml
import json

def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    print("=" * 80)
    print("TRAINING MODELS ON FULL_DATASET")
    print("=" * 80)
    
    config = load_config()
    base_dir = Path(__file__).parent.parent
    datasets_dir = base_dir / "datasets"
    models_dir = base_dir / "models"
    
    model_types = ['encoder', 'decoder', 'encoder-decoder']
    regime = 'full_dataset'
    
    for dataset_config in config['corpus']['datasets']:
        dataset_name = dataset_config['name']
        clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()
        
        print(f"\n{'#' * 80}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#' * 80}")
        
        corpus_file = datasets_dir / regime / clean_name / "corpus.jsonl"
        
        if not corpus_file.exists():
            print(f"   ✗ Corpus not found: {corpus_file}")
            continue
        
        for model_type in model_types:
            output_dir = models_dir / clean_name / regime / model_type
            
            print(f"\n   Training: {model_type}")
            print(f"   Output: {output_dir}")
            
            try:
                from processing.model_utils import train_model
                result = train_model(
                    corpus_file=corpus_file,
                    output_dir=output_dir,
                    model_type=model_type,
                    config=config,
                    dataset_name=dataset_name,
                    regime_name=regime,
                )
                print(f"   ✓ Training complete: {model_type}")
            except Exception as e:
                print(f"   ✗ Training failed: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("✓ FULL_DATASET TRAINING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
