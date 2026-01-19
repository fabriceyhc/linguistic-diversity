#!/usr/bin/env python3
"""Select diverse subsets using submodular optimization.

This script:
1. Loads extracted features (semantic and syntactic)
2. Implements hierarchical merge strategy for scalability
3. Applies different selection regimes:
   - Semantic diversity (Facility Location)
   - Syntactic diversity (Feature-based)
   - Morphological diversity (Feature-based)
   - Phonological diversity (Feature-based)
   - Composite diversity (two-step)
   - Universal diversity (weighted combination)
   - Library diversity (using actual diversity metrics)
   - Random baseline
4. Saves selected document indices and corpus subsets
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import os
import json
import pickle
import numpy as np
import yaml
from scipy.sparse import load_npz, vstack
from apricot import FacilityLocationSelection, SaturatedCoverageSelection
from tqdm import tqdm

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


def load_features(features_dir, dataset_name):
    """Load extracted features.

    Args:
        features_dir: Features directory path
        dataset_name: Name of dataset

    Returns:
        dict: Dictionary of all feature types
    """
    clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()

    features = {}

    # Load semantic embeddings
    semantic_file = features_dir / f"{clean_name}_semantic_embeddings.npy"
    features['semantic'] = np.load(semantic_file)
    print(f"   ✓ Loaded semantic embeddings: {features['semantic'].shape}")

    # Load syntactic features
    syntactic_file = features_dir / f"{clean_name}_syntactic_features.npz"
    features['syntactic'] = load_npz(syntactic_file)
    print(f"   ✓ Loaded syntactic features: {features['syntactic'].shape}")

    # Load morphological features
    morphological_file = features_dir / f"{clean_name}_morphological_features.npz"
    features['morphological'] = load_npz(morphological_file)
    print(f"   ✓ Loaded morphological features: {features['morphological'].shape}")

    # Load phonological features
    phonological_file = features_dir / f"{clean_name}_phonological_features.npz"
    features['phonological'] = load_npz(phonological_file)
    print(f"   ✓ Loaded phonological features: {features['phonological'].shape}")

    return features


def load_documents(input_dir, dataset_name):
    """Load original documents.

    Args:
        input_dir: Input directory path
        dataset_name: Name of dataset

    Returns:
        List of document strings
    """
    clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()
    train_file = input_dir / f"{clean_name}_train.pkl"

    with open(train_file, 'rb') as f:
        documents = pickle.load(f)

    return documents


def hierarchical_selection(features, n_select, shard_size, selection_function, is_sparse=False):
    """Hierarchical merge strategy for scalable submodular selection.

    Args:
        features: Feature matrix (dense or sparse)
        n_select: Number of items to select
        shard_size: Size of each shard
        selection_function: Function that performs selection on a shard
        is_sparse: Whether features are sparse

    Returns:
        Array of selected indices
    """
    n_items = features.shape[0]

    print(f"   Total items: {n_items}")
    print(f"   Target selection: {n_select}")
    print(f"   Shard size: {shard_size}")

    # If small enough, select directly
    if n_items <= shard_size:
        print(f"   Dataset small enough - selecting directly")
        selected = selection_function(features, n_select)
        return selected

    # Stage 1: Process shards
    print(f"   Stage 1: Processing shards...")
    n_shards = (n_items + shard_size - 1) // shard_size
    per_shard_select = max(1, n_select // n_shards)  # Select proportionally from each shard

    shard_winners = []
    shard_winner_indices = []

    for i in tqdm(range(n_shards), desc="   Shard selection"):
        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, n_items)

        # Get shard
        if is_sparse:
            shard_features = features[start_idx:end_idx]
        else:
            shard_features = features[start_idx:end_idx]

        # Select from shard
        try:
            shard_selected = selection_function(shard_features, per_shard_select)

            # Convert to global indices
            global_indices = start_idx + shard_selected
            shard_winner_indices.extend(global_indices.tolist())

        except Exception as e:
            print(f"   Warning: Shard {i} failed with error: {e}")
            continue

    print(f"   ✓ Stage 1 complete: {len(shard_winner_indices)} items from {n_shards} shards")

    # Stage 2: Select from winners
    print(f"   Stage 2: Final selection from winners...")

    # Get features for winners
    if is_sparse:
        winner_features = features[shard_winner_indices]
    else:
        winner_features = features[shard_winner_indices]

    # Final selection
    final_selected_local = selection_function(winner_features, n_select)

    # Convert back to original indices
    final_selected = np.array([shard_winner_indices[i] for i in final_selected_local])

    print(f"   ✓ Stage 2 complete: {len(final_selected)} final selections")

    return final_selected


def select_semantic_diversity(embeddings, n_select, shard_size, config):
    """Select documents for semantic diversity using Facility Location.

    Args:
        embeddings: Semantic embedding matrix
        n_select: Number of documents to select
        shard_size: Shard size for hierarchical selection
        config: Configuration dict

    Returns:
        Array of selected indices
    """
    print(f"\n   Method: Facility Location Selection")

    optimizer = config['selection']['facility_location']['optimizer']

    def selection_fn(features, k):
        selector = FacilityLocationSelection(
            k,
            metric='euclidean',
            optimizer=optimizer,
            verbose=False
        )
        selector.fit(features)
        return selector.ranking

    selected = hierarchical_selection(
        embeddings,
        n_select,
        shard_size,
        selection_fn,
        is_sparse=False
    )

    return selected


def select_syntactic_diversity(syntactic_features, n_select, shard_size, config):
    """Select documents for syntactic diversity using Saturated Coverage Selection.

    Args:
        syntactic_features: Syntactic feature sparse matrix
        n_select: Number of documents to select
        shard_size: Shard size for hierarchical selection
        config: Configuration dict

    Returns:
        Array of selected indices
    """
    print(f"\n   Method: Saturated Coverage Selection")

    def selection_fn(features, k):
        selector = SaturatedCoverageSelection(
            k,
            optimizer='lazy',
            verbose=False
        )
        selector.fit(features.toarray())
        return selector.ranking

    selected = hierarchical_selection(
        syntactic_features,
        n_select,
        shard_size,
        selection_fn,
        is_sparse=True
    )

    return selected


def select_morphological_diversity(morphological_features, n_select, shard_size, config):
    """Select documents for morphological diversity using Saturated Coverage Selection.

    Args:
        morphological_features: Morphological feature sparse matrix
        n_select: Number of documents to select
        shard_size: Shard size for hierarchical selection
        config: Configuration dict

    Returns:
        Array of selected indices
    """
    print(f"\n   Method: Morphological Saturated Coverage Selection")

    def selection_fn(features, k):
        selector = SaturatedCoverageSelection(
            k,
            optimizer='lazy',
            verbose=False
        )
        selector.fit(features.toarray())
        return selector.ranking

    selected = hierarchical_selection(
        morphological_features,
        n_select,
        shard_size,
        selection_fn,
        is_sparse=True
    )

    return selected


def select_phonological_diversity(phonological_features, n_select, shard_size, config):
    """Select documents for phonological diversity using Saturated Coverage Selection.

    Args:
        phonological_features: Phonological feature sparse matrix
        n_select: Number of documents to select
        shard_size: Shard size for hierarchical selection
        config: Configuration dict

    Returns:
        Array of selected indices
    """
    print(f"\n   Method: Phonological Saturated Coverage Selection")

    def selection_fn(features, k):
        selector = SaturatedCoverageSelection(
            k,
            optimizer='lazy',
            verbose=False
        )
        selector.fit(features.toarray())
        return selector.ranking

    selected = hierarchical_selection(
        phonological_features,
        n_select,
        shard_size,
        selection_fn,
        is_sparse=True
    )

    return selected


def select_composite_diversity(features_dict, n_select, shard_size, config):
    """Select documents using composite diversity (two-step).

    Args:
        features_dict: Dictionary of all feature types
        n_select: Number of documents to select
        shard_size: Shard size for hierarchical selection
        config: Configuration dict

    Returns:
        Array of selected indices
    """
    print(f"\n   Method: Composite (Two-step) Selection")

    first_stage = config['selection']['composite']['first_stage']
    second_stage = config['selection']['composite']['second_stage']
    first_ratio = config['selection']['composite']['first_stage_ratio']

    n_first_stage = int(features_dict['semantic'].shape[0] * first_ratio)

    print(f"   First stage ({first_stage}): Select {n_first_stage} documents")

    # Stage 1: Select using first feature type
    if first_stage == 'semantic':
        first_selected = select_semantic_diversity(features_dict['semantic'], n_first_stage, shard_size, config)
    elif first_stage == 'syntactic':
        first_selected = select_syntactic_diversity(features_dict['syntactic'], n_first_stage, shard_size, config)
    elif first_stage == 'morphological':
        first_selected = select_morphological_diversity(features_dict['morphological'], n_first_stage, shard_size, config)
    elif first_stage == 'phonological':
        first_selected = select_phonological_diversity(features_dict['phonological'], n_first_stage, shard_size, config)

    print(f"   ✓ First stage complete: {len(first_selected)} documents")

    # Stage 2: Select using second feature type
    print(f"   Second stage ({second_stage}): Select {n_select} from {len(first_selected)} documents")

    # Get features for first stage winners
    if second_stage == 'semantic':
        second_features = features_dict['semantic'][first_selected]
        second_selected_local = select_semantic_diversity(second_features, n_select, shard_size, config)
    elif second_stage == 'syntactic':
        second_features = features_dict['syntactic'][first_selected]
        second_selected_local = select_syntactic_diversity(second_features, n_select, shard_size, config)
    elif second_stage == 'morphological':
        second_features = features_dict['morphological'][first_selected]
        second_selected_local = select_morphological_diversity(second_features, n_select, shard_size, config)
    elif second_stage == 'phonological':
        second_features = features_dict['phonological'][first_selected]
        second_selected_local = select_phonological_diversity(second_features, n_select, shard_size, config)

    # Convert back to original indices
    final_selected = first_selected[second_selected_local]

    print(f"   ✓ Second stage complete: {len(final_selected)} documents")

    return final_selected


def select_universal_diversity(features_dict, n_select, shard_size, config):
    """Select documents using universal diversity (all branches combined).

    Args:
        features_dict: Dictionary of all feature types
        n_select: Number of documents to select
        shard_size: Shard size for hierarchical selection
        config: Configuration dict

    Returns:
        Array of selected indices
    """
    print(f"\n   Method: Universal (Multi-branch) Selection")

    weights = config['selection']['universal']['weights']
    method = config['selection']['universal']['combination_method']

    print(f"   Weights: {weights}")
    print(f"   Combination method: {method}")

    # Normalize features and combine
    from sklearn.preprocessing import normalize
    from scipy.sparse import hstack, csr_matrix

    if method == 'weighted_sum':
        # Weight and concatenate all features
        combined_features = []

        # Semantic (dense)
        sem = features_dict['semantic'] * weights['semantic']
        combined_features.append(sem)

        # Syntactic (sparse)
        syn = features_dict['syntactic'].toarray() * weights['syntactic']
        combined_features.append(syn)

        # Morphological (sparse)
        mor = features_dict['morphological'].toarray() * weights['morphological']
        combined_features.append(mor)

        # Phonological (sparse)
        pho = features_dict['phonological'].toarray() * weights['phonological']
        combined_features.append(pho)

        # Concatenate
        combined = np.hstack(combined_features)

        print(f"   Combined feature shape: {combined.shape}")

        # Use Facility Location on combined features
        def selection_fn(features, k):
            selector = FacilityLocationSelection(
                k,
                metric='euclidean',
                optimizer='lazy',
                verbose=False
            )
            selector.fit(features)
            return selector.ranking

        selected = hierarchical_selection(
            combined,
            n_select,
            shard_size,
            selection_fn,
            is_sparse=False
        )

    return selected


def select_random_baseline(n_items, n_select):
    """Random baseline selection.

    Args:
        n_items: Total number of items
        n_select: Number to select

    Returns:
        Array of selected indices
    """
    print(f"\n   Method: Random Baseline")
    print(f"   Randomly selecting {n_select} from {n_items} documents")

    np.random.seed(42)
    selected = np.random.choice(n_items, size=n_select, replace=False)

    return selected


def save_subset(selected_indices, documents, regime_name, datasets_dir, dataset_name):
    """Save selected subset.

    Args:
        selected_indices: Selected document indices
        documents: Full document list
        regime_name: Name of selection regime
        datasets_dir: Datasets directory path
        dataset_name: Name of dataset
    """
    clean_name = dataset_name.replace('/', '_').replace('-', '_').lower()

    # Create regime directory
    regime_dir = datasets_dir / regime_name / clean_name
    regime_dir.mkdir(parents=True, exist_ok=True)

    # Save indices
    indices_file = regime_dir / "selected_indices.json"
    with open(indices_file, 'w') as f:
        json.dump(selected_indices.tolist(), f)
    print(f"   ✓ Saved indices: {indices_file}")

    # Save corpus
    selected_docs = [documents[i] for i in selected_indices]

    corpus_file = regime_dir / "corpus.jsonl"
    with open(corpus_file, 'w') as f:
        for doc in selected_docs:
            json.dump({"text": doc}, f)
            f.write('\n')
    print(f"   ✓ Saved corpus: {corpus_file}")

    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'regime': regime_name,
        'n_selected': len(selected_indices),
        'indices_file': str(indices_file),
        'corpus_file': str(corpus_file),
    }

    metadata_file = regime_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved metadata: {metadata_file}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("STEP 3: SELECT DIVERSE SUBSETS")
    print("=" * 80)

    # Load configuration
    config = load_config()

    shard_size = config['selection']['shard_size']
    selection_ratio = config['selection']['selection_ratio']

    # Setup directories
    base_dir = Path(__file__).parent.parent
    input_dir = base_dir / "input"
    features_dir = base_dir / "features"
    datasets_dir = base_dir / "datasets"

    # Process each dataset
    for dataset_config in config['corpus']['datasets']:
        dataset_name = dataset_config['name']

        print(f"\n{'=' * 80}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 80}")

        try:
            # Load features
            print(f"\n1. Loading features...")
            features_dict = load_features(features_dir, dataset_name)

            # Load documents
            print(f"\n2. Loading documents...")
            documents = load_documents(input_dir, dataset_name)
            print(f"   ✓ Loaded {len(documents)} documents")

            n_items = len(documents)
            n_select = int(n_items * selection_ratio)
            print(f"\n3. Selection target: {n_select} documents ({selection_ratio:.1%})")

            # Semantic diversity
            print(f"\n4. Semantic Diversity Selection")
            print(f"   {'- ' * 39}")
            semantic_selected = select_semantic_diversity(features_dict['semantic'], n_select, shard_size, config)
            save_subset(semantic_selected, documents, "semantic_diversity", datasets_dir, dataset_name)

            # Syntactic diversity
            print(f"\n5. Syntactic Diversity Selection")
            print(f"   {'- ' * 39}")
            syntactic_selected = select_syntactic_diversity(features_dict['syntactic'], n_select, shard_size, config)
            save_subset(syntactic_selected, documents, "syntactic_diversity", datasets_dir, dataset_name)

            # Morphological diversity
            print(f"\n6. Morphological Diversity Selection")
            print(f"   {'- ' * 39}")
            morphological_selected = select_morphological_diversity(features_dict['morphological'], n_select, shard_size, config)
            save_subset(morphological_selected, documents, "morphological_diversity", datasets_dir, dataset_name)

            # Phonological diversity
            print(f"\n7. Phonological Diversity Selection")
            print(f"   {'- ' * 39}")
            phonological_selected = select_phonological_diversity(features_dict['phonological'], n_select, shard_size, config)
            save_subset(phonological_selected, documents, "phonological_diversity", datasets_dir, dataset_name)

            # Composite diversity
            print(f"\n8. Composite Diversity Selection")
            print(f"   {'- ' * 39}")
            composite_selected = select_composite_diversity(features_dict, n_select, shard_size, config)
            save_subset(composite_selected, documents, "composite_diversity", datasets_dir, dataset_name)

            # Universal diversity
            print(f"\n9. Universal Diversity Selection")
            print(f"   {'- ' * 39}")
            universal_selected = select_universal_diversity(features_dict, n_select, shard_size, config)
            save_subset(universal_selected, documents, "universal_diversity", datasets_dir, dataset_name)

            # Random baseline
            print(f"\n10. Random Baseline Selection")
            print(f"   {'- ' * 39}")
            random_selected = select_random_baseline(n_items, n_select)
            save_subset(random_selected, documents, "random_baseline", datasets_dir, dataset_name)

            # Full dataset (no subsampling - use all data)
            print(f"\n11. Full Dataset (No Subsampling)")
            print(f"   {'- ' * 39}")
            print(f"   Saving ALL {n_items} documents (no selection)")
            full_indices = np.arange(n_items)
            save_subset(full_indices, documents, "full_dataset", datasets_dir, dataset_name)

            print(f"\n{'=' * 80}")
            print(f"✓ Successfully processed: {dataset_name}")
            print(f"{'=' * 80}")

        except Exception as e:
            print(f"\n{'=' * 80}")
            print(f"✗ Failed to process: {dataset_name}")
            print(f"  Error: {e}")
            print(f"{'=' * 80}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("✓ SUBSET SELECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
