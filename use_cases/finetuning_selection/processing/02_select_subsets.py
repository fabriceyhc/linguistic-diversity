#!/usr/bin/env python3
"""
Select diverse subsets from instruction fine-tuning data.

This script implements multiple selection strategies:
1. Random selection (baseline)
2. Semantic diversity (facility location on embeddings)
3. Syntactic diversity (feature-based submodular)
4. Combined diversity (weighted semantic + syntactic)
5. Length diversity (stratified by response length)
6. Quality-filtered (filter by quality signals, then diverse select)

Inspired by LIMA: "Less Is More for Alignment"
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import yaml
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

# Import our diversity embedding selection
from linguistic_diversity import (
    UniversalLinguisticDiversity,
    DIVERSITY_EMBEDDING_METRICS,
    select_diverse_texts,
)


def load_config():
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def random_selection(n_samples: int, n_select: int, seed: int = 42) -> np.ndarray:
    """Random selection baseline."""
    rng = np.random.RandomState(seed)
    return rng.choice(n_samples, size=min(n_select, n_samples), replace=False)


def facility_location_selection(
    features: np.ndarray,
    n_select: int,
    seed: int = 42,
    verbose: bool = True
) -> np.ndarray:
    """
    Greedy facility location for diversity maximization.
    Selects points that maximize coverage of the feature space.
    """
    n_samples = features.shape[0]
    n_select = min(n_select, n_samples)
    rng = np.random.RandomState(seed)

    # Normalize features
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    features_norm = features / norms

    selected = []
    remaining = set(range(n_samples))

    # Start with random point
    first = rng.choice(list(remaining))
    selected.append(first)
    remaining.remove(first)

    # Track minimum distance to selected set for each point
    selected_features = features_norm[selected]
    min_sim_to_selected = features_norm @ selected_features.T
    min_sim_to_selected = min_sim_to_selected.flatten()

    # Greedy selection
    iterator = range(n_select - 1)
    if verbose:
        iterator = tqdm(iterator, desc="      Facility Location")

    for _ in iterator:
        if not remaining:
            break

        remaining_list = list(remaining)

        # Find point with minimum similarity to selected set (most diverse)
        remaining_sims = min_sim_to_selected[remaining_list]
        best_local_idx = remaining_sims.argmin()
        best_idx = remaining_list[best_local_idx]

        selected.append(best_idx)
        remaining.remove(best_idx)

        # Update minimum similarities
        new_sims = features_norm @ features_norm[best_idx:best_idx+1].T
        min_sim_to_selected = np.maximum(min_sim_to_selected, new_sims.flatten())

    return np.array(selected)


def feature_submodular_selection(
    features: np.ndarray,
    n_select: int,
    seed: int = 42,
    verbose: bool = True
) -> np.ndarray:
    """
    Feature-based submodular selection using sqrt-concave function.
    Encourages coverage of different feature dimensions.
    """
    n_samples, n_features = features.shape
    n_select = min(n_select, n_samples)
    rng = np.random.RandomState(seed)

    # Normalize features to [0, 1] range
    features_min = features.min(axis=0)
    features_max = features.max(axis=0)
    features_range = features_max - features_min
    features_range = np.where(features_range == 0, 1, features_range)
    features_norm = (features - features_min) / features_range

    selected = []
    remaining = set(range(n_samples))

    # Track cumulative feature coverage
    coverage = np.zeros(n_features)

    iterator = range(n_select)
    if verbose:
        iterator = tqdm(iterator, desc="      Submodular")

    for _ in iterator:
        if not remaining:
            break

        remaining_list = list(remaining)
        remaining_features = features_norm[remaining_list]

        # Current value: sum of sqrt(coverage)
        current_value = np.sum(np.sqrt(coverage))

        # Compute gain for adding each remaining sample
        new_coverage = coverage + remaining_features
        new_values = np.sum(np.sqrt(new_coverage), axis=1)
        gains = new_values - current_value

        # Select best
        best_local_idx = gains.argmax()
        best_idx = remaining_list[best_local_idx]

        selected.append(best_idx)
        remaining.remove(best_idx)
        coverage += features_norm[best_idx]

    return np.array(selected)


def combined_diversity_selection(
    semantic_features: np.ndarray,
    syntactic_features: np.ndarray,
    n_select: int,
    semantic_weight: float = 0.7,
    syntactic_weight: float = 0.3,
    seed: int = 42,
    verbose: bool = True
) -> np.ndarray:
    """
    Combined semantic and syntactic diversity selection.
    Uses weighted combination of both feature spaces.
    """
    n_samples = semantic_features.shape[0]
    n_select = min(n_select, n_samples)
    rng = np.random.RandomState(seed)

    # Normalize each feature space
    def normalize(features):
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return features / norms

    semantic_norm = normalize(semantic_features)
    syntactic_norm = normalize(syntactic_features)

    selected = []
    remaining = set(range(n_samples))

    # Start with random point
    first = rng.choice(list(remaining))
    selected.append(first)
    remaining.remove(first)

    # Track similarities
    sem_selected = semantic_norm[selected]
    syn_selected = syntactic_norm[selected]

    max_sem_sim = semantic_norm @ sem_selected.T
    max_syn_sim = syntactic_norm @ syn_selected.T

    iterator = range(n_select - 1)
    if verbose:
        iterator = tqdm(iterator, desc="      Combined Diversity")

    for _ in iterator:
        if not remaining:
            break

        remaining_list = list(remaining)

        # Weighted combination of similarities
        combined_sim = (
            semantic_weight * max_sem_sim[remaining_list].max(axis=1) +
            syntactic_weight * max_syn_sim[remaining_list].max(axis=1)
        )

        # Select point with minimum combined similarity
        best_local_idx = combined_sim.argmin()
        best_idx = remaining_list[best_local_idx]

        selected.append(best_idx)
        remaining.remove(best_idx)

        # Update similarities
        new_sem_sim = semantic_norm @ semantic_norm[best_idx:best_idx+1].T
        new_syn_sim = syntactic_norm @ syntactic_norm[best_idx:best_idx+1].T

        max_sem_sim = np.maximum(max_sem_sim, new_sem_sim)
        max_syn_sim = np.maximum(max_syn_sim, new_syn_sim)

    return np.array(selected)


def length_stratified_selection(
    samples: List[Dict],
    n_select: int,
    seed: int = 42
) -> np.ndarray:
    """
    Length-stratified selection.
    Ensures selected samples cover different response length ranges.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(samples)
    n_select = min(n_select, n_samples)

    lengths = np.array([len(s['response'].split()) for s in samples])

    # Create length bins (quartiles)
    percentiles = [0, 25, 50, 75, 100]
    bins = np.percentile(lengths, percentiles)

    # Assign samples to bins
    bin_indices = np.digitize(lengths, bins[1:-1])

    # Select proportionally from each bin
    selected = []
    unique_bins = sorted(set(bin_indices))
    samples_per_bin = n_select // len(unique_bins)

    for bin_id in unique_bins:
        bin_samples = np.where(bin_indices == bin_id)[0]
        n_from_bin = min(samples_per_bin, len(bin_samples))
        selected.extend(rng.choice(bin_samples, size=n_from_bin, replace=False))

    # Fill remaining slots randomly
    remaining = set(range(n_samples)) - set(selected)
    n_remaining = n_select - len(selected)
    if n_remaining > 0 and remaining:
        selected.extend(rng.choice(list(remaining), size=min(n_remaining, len(remaining)), replace=False))

    return np.array(selected[:n_select])


def quality_filtered_selection(
    semantic_features: np.ndarray,
    quality_signals: np.ndarray,
    n_select: int,
    quality_percentile: float = 50.0,
    seed: int = 42,
    verbose: bool = True
) -> np.ndarray:
    """
    Quality-filtered diversity selection.
    First filters by quality signals, then applies diversity selection.
    """
    n_samples = semantic_features.shape[0]
    n_select = min(n_select, n_samples)

    # Compute quality score (weighted sum of signals)
    # Higher weight for: explanation, steps, reasonable length ratio
    weights = np.array([0.1, 0.2, 0.1, 0.15, 0.2, 0.15, 0.05, 0.05])
    if quality_signals.shape[1] != len(weights):
        weights = np.ones(quality_signals.shape[1]) / quality_signals.shape[1]

    quality_scores = quality_signals @ weights

    # Filter to top quality samples
    threshold = np.percentile(quality_scores, quality_percentile)
    high_quality_mask = quality_scores >= threshold
    high_quality_indices = np.where(high_quality_mask)[0]

    if len(high_quality_indices) < n_select:
        # If not enough high quality samples, lower threshold
        high_quality_indices = np.argsort(quality_scores)[-n_select * 2:]

    # Apply diversity selection on filtered set
    filtered_features = semantic_features[high_quality_indices]
    selected_in_filtered = facility_location_selection(
        filtered_features, n_select, seed=seed, verbose=verbose
    )

    # Map back to original indices
    return high_quality_indices[selected_in_filtered]


def universal_embedding_diversity_selection(
    samples: List[Dict],
    n_select: int,
    selection_method: str = "facility_location",
    seed: int = 42,
    verbose: bool = True,
) -> np.ndarray:
    """
    Universal linguistic diversity embedding selection.

    Computes diversity embeddings capturing multiple linguistic dimensions
    (semantic, syntactic, morphological, phonological) and uses submodular
    optimization to select samples covering all dimensions.

    Args:
        samples: List of sample dicts with 'instruction' and 'response' keys
        n_select: Number of samples to select
        selection_method: 'facility_location', 'max_min', or 'balanced'
        seed: Random seed
        verbose: Print progress

    Returns:
        Array of selected indices
    """
    n_samples = len(samples)
    n_select = min(n_select, n_samples)

    # Initialize metric (using fast options)
    metric = UniversalLinguisticDiversity({
        'use_constituency_parse': False,
        'use_rhythmic': False,
        'use_phonemic': False,
        'verbose': False,
    })

    print(f"      Computing diversity embeddings for {n_samples} samples...")

    # Compute diversity embeddings
    n_metrics = len(DIVERSITY_EMBEDDING_METRICS)
    diversity_embeddings = np.zeros((n_samples, n_metrics), dtype=np.float64)

    batch_size = 100
    iterator = range(0, n_samples, batch_size)
    if verbose:
        iterator = tqdm(iterator, desc="      Universal Embedding")

    for i in iterator:
        batch_end = min(i + batch_size, n_samples)
        for j in range(i, batch_end):
            sample = samples[j]
            # Combine instruction and response
            text = f"{sample.get('instruction', '')} {sample.get('response', '')}"
            if text.strip():
                try:
                    emb = metric._compute_single_doc_embedding([text], normalize=True)
                    diversity_embeddings[j] = emb
                except Exception:
                    pass

    print(f"      Running {selection_method} selection...")

    # Apply submodular selection
    result = select_diverse_texts(
        diversity_embeddings,
        n_select=n_select,
        method=selection_method,
        seed=seed,
        verbose=verbose,
    )

    print(f"      Coverage per metric: {result.coverage_per_metric}")

    return result.indices


def check_existing_selections(output_dir: Path, target_sizes: List[int]) -> bool:
    """Check if selections already exist for all target sizes."""
    selections_file = output_dir / "selections.json"
    if not selections_file.exists():
        return False

    with open(selections_file, 'r') as f:
        existing = json.load(f)

    # Check if all target sizes are present
    expected_keys = [f"size_{size}" for size in target_sizes]
    return all(key in existing for key in expected_keys)


def main():
    print("=" * 70)
    print("STEP 2: SELECT DIVERSE SUBSETS")
    print("=" * 70)

    config = load_config()
    base_dir = Path(__file__).parent.parent
    datasets_dir = base_dir / "datasets"
    output_dir = base_dir / "selections"
    output_dir.mkdir(exist_ok=True)

    # Get target sizes from config
    target_sizes = config['selection']['target_sizes']

    # Check if selections already exist
    if check_existing_selections(output_dir, target_sizes):
        print("\nSelections already exist for all target sizes.")
        with open(output_dir / "selections.json", 'r') as f:
            existing = json.load(f)
        print(f"  Sizes: {list(existing.keys())}")
        print(f"  Methods: {list(existing[list(existing.keys())[0]].keys())}")
        print("\nTo re-select, delete the selections/ directory.")
        return

    # Load features
    print("\n1. Loading features...")
    semantic_features = np.load(datasets_dir / "semantic_features.npy")
    instruction_features = np.load(datasets_dir / "instruction_features.npy")
    syntactic_features = np.load(datasets_dir / "syntactic_features.npy")
    quality_signals = np.load(datasets_dir / "quality_signals.npy")

    with open(datasets_dir / "samples.json", 'r') as f:
        samples = json.load(f)

    n_samples = len(samples)
    print(f"   Total samples: {n_samples}")
    print(f"   Semantic features: {semantic_features.shape}")
    print(f"   Instruction features: {instruction_features.shape}")
    print(f"   Syntactic features: {syntactic_features.shape}")
    print(f"   Quality signals: {quality_signals.shape}")

    print(f"\n2. Target subset sizes: {target_sizes}")

    # Get selection method configs
    method_configs = {m['name']: m for m in config['selection']['methods']}

    selections = {}

    for target_size in target_sizes:
        size_key = f"size_{target_size}"
        print(f"\n{'=' * 70}")
        print(f"Selecting {target_size} samples")
        print(f"{'=' * 70}")

        if target_size > n_samples:
            print(f"   Warning: target size {target_size} > available samples {n_samples}")
            target_size = n_samples

        selections[size_key] = {}

        # 1. Random selection
        print(f"\n   Random selection...")
        random_idx = random_selection(n_samples, target_size)
        selections[size_key]['random'] = random_idx.tolist()
        print(f"      Selected {len(random_idx)} samples")

        # 2. Semantic diversity (facility location on combined embeddings)
        print(f"\n   Semantic diversity selection...")
        semantic_idx = facility_location_selection(semantic_features, target_size)
        selections[size_key]['semantic_diversity'] = semantic_idx.tolist()
        print(f"      Selected {len(semantic_idx)} samples")

        # 3. Syntactic diversity (submodular on syntactic features)
        print(f"\n   Syntactic diversity selection...")
        syntactic_idx = feature_submodular_selection(syntactic_features, target_size)
        selections[size_key]['syntactic_diversity'] = syntactic_idx.tolist()
        print(f"      Selected {len(syntactic_idx)} samples")

        # 4. Combined diversity
        print(f"\n   Combined diversity selection...")
        combined_config = method_configs.get('combined_diversity', {})
        weights = combined_config.get('weights', {'semantic': 0.7, 'syntactic': 0.3})
        combined_idx = combined_diversity_selection(
            semantic_features, syntactic_features, target_size,
            semantic_weight=weights['semantic'],
            syntactic_weight=weights['syntactic']
        )
        selections[size_key]['combined_diversity'] = combined_idx.tolist()
        print(f"      Selected {len(combined_idx)} samples")

        # 5. Length diversity
        print(f"\n   Length diversity selection...")
        length_idx = length_stratified_selection(samples, target_size)
        selections[size_key]['length_diversity'] = length_idx.tolist()
        print(f"      Selected {len(length_idx)} samples")

        # 6. Quality-filtered diversity
        print(f"\n   Quality-filtered selection...")
        quality_idx = quality_filtered_selection(
            semantic_features, quality_signals, target_size
        )
        selections[size_key]['quality_filtered'] = quality_idx.tolist()
        print(f"      Selected {len(quality_idx)} samples")

        # 7. Instruction-space diversity (for task coverage)
        print(f"\n   Instruction diversity selection...")
        instruction_idx = facility_location_selection(instruction_features, target_size)
        selections[size_key]['instruction_diversity'] = instruction_idx.tolist()
        print(f"      Selected {len(instruction_idx)} samples")

        # 8. Universal embedding diversity (multi-dimensional linguistic diversity)
        print(f"\n   Universal embedding diversity selection...")
        universal_idx = universal_embedding_diversity_selection(
            samples, target_size, selection_method='facility_location'
        )
        selections[size_key]['universal_embedding_diversity'] = universal_idx.tolist()
        print(f"      Selected {len(universal_idx)} samples")

    # Save selections
    selections_file = output_dir / "selections.json"
    with open(selections_file, 'w') as f:
        json.dump(selections, f, indent=2)
    print(f"\n   Saved selections to {selections_file}")

    # Compute and save overlap statistics
    print("\n3. Computing selection overlap statistics...")
    overlap_stats = {}

    for size_key, methods in selections.items():
        overlap_stats[size_key] = {}
        method_names = list(methods.keys())

        for i, m1 in enumerate(method_names):
            for m2 in method_names[i+1:]:
                set1 = set(methods[m1])
                set2 = set(methods[m2])
                overlap = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = overlap / union if union > 0 else 0
                overlap_stats[size_key][f"{m1}_vs_{m2}"] = {
                    'overlap': overlap,
                    'jaccard': round(jaccard, 3)
                }

    with open(output_dir / "overlap_stats.json", 'w') as f:
        json.dump(overlap_stats, f, indent=2)

    # Print summary
    print("\n4. Selection Summary:")
    print("-" * 60)
    for size_key, methods in selections.items():
        print(f"\n{size_key}:")
        for method, indices in methods.items():
            print(f"   {method:25s}: {len(indices)} samples")

    print("\n5. Overlap Statistics (Jaccard similarity):")
    print("-" * 60)
    for size_key, stats in overlap_stats.items():
        print(f"\n{size_key}:")
        for comparison, data in sorted(stats.items(), key=lambda x: -x[1]['jaccard'])[:5]:
            print(f"   {comparison}: {data['jaccard']:.3f} ({data['overlap']} shared)")

    print("\n" + "=" * 70)
    print("SUBSET SELECTION COMPLETE")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
