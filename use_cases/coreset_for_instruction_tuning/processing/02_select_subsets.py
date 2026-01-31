#!/usr/bin/env python3
"""
Step 2: Diversity-Based Subset Selection

This script implements selection algorithms for creating diverse coresets:
1. Random selection (baseline)
2. Greedy Diversity Maximization (for smaller datasets < 100k)
3. K-Means Clustering + Centroid Selection (for larger datasets)

The goal is to select ~10% of the data that maximizes diversity,
preserving the "essence" of the full dataset.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import json
import yaml
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from abc import ABC, abstractmethod

# Import our diversity embedding selection
from linguistic_diversity import (
    UniversalLinguisticDiversity,
    DIVERSITY_EMBEDDING_METRICS,
    select_diverse_texts,
)


def load_config() -> dict:
    """Load experiment configuration."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@dataclass
class SelectionResult:
    """Result of a selection algorithm."""
    method: str
    indices: List[int]
    n_selected: int
    n_total: int
    selection_ratio: float
    diversity_score: Optional[float]
    length_stats: Dict
    compute_time_seconds: float
    metadata: Dict


class Selector(ABC):
    """Abstract base class for selection algorithms."""

    @abstractmethod
    def select(
        self,
        embeddings: np.ndarray,
        n_select: int,
        seed: int = 42,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Select indices from embeddings.

        Args:
            embeddings: N x D embedding matrix (assumed L2 normalized)
            n_select: Number of samples to select
            seed: Random seed for reproducibility

        Returns:
            Tuple of (selected_indices, metadata_dict)
        """
        pass


class RandomSelector(Selector):
    """Random selection baseline."""

    def select(
        self,
        embeddings: np.ndarray,
        n_select: int,
        seed: int = 42,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        rng = np.random.RandomState(seed)
        n_samples = len(embeddings)
        n_select = min(n_select, n_samples)

        indices = rng.choice(n_samples, size=n_select, replace=False)

        metadata = {
            'method': 'random',
            'seed': seed,
        }

        return indices, metadata


class GreedyDiversitySelector(Selector):
    """
    Greedy Diversity Maximization Selector.

    Algorithm:
    1. Start with an empty set
    2. Iteratively add the point that is most dissimilar to the current set
       (maximizes minimum distance to selected points)
    3. Continue until target size is reached

    This is a facility location / max-min diversity algorithm that
    maximizes coverage of the embedding space.

    Time complexity: O(n_select * n_samples) with optimizations
    Best for: Datasets < 100k samples
    """

    def __init__(self, use_instruction_embeddings: bool = False):
        """
        Args:
            use_instruction_embeddings: If True, use instruction-only embeddings
                for selection (task diversity). Otherwise use combined embeddings.
        """
        self.use_instruction_embeddings = use_instruction_embeddings

    def select(
        self,
        embeddings: np.ndarray,
        n_select: int,
        seed: int = 42,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """Greedy max-min diversity selection."""
        from tqdm import tqdm

        rng = np.random.RandomState(seed)
        n_samples = embeddings.shape[0]
        n_select = min(n_select, n_samples)

        # Ensure embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = embeddings / norms

        selected = []
        remaining = set(range(n_samples))

        # Start with a random point
        first_idx = rng.choice(list(remaining))
        selected.append(first_idx)
        remaining.remove(first_idx)

        # Track maximum similarity to selected set for each point
        # Lower similarity = more diverse
        # Using similarity instead of distance for efficiency (dot product)
        max_sim_to_selected = embeddings_norm @ embeddings_norm[first_idx:first_idx+1].T
        max_sim_to_selected = max_sim_to_selected.flatten()

        # Greedy selection
        iterator = range(n_select - 1)
        if verbose:
            iterator = tqdm(iterator, desc="   Greedy diversity selection")

        for _ in iterator:
            if not remaining:
                break

            remaining_list = list(remaining)

            # Find point with minimum similarity to selected set (most diverse)
            remaining_sims = max_sim_to_selected[remaining_list]
            best_local_idx = remaining_sims.argmin()
            best_idx = remaining_list[best_local_idx]

            selected.append(best_idx)
            remaining.remove(best_idx)

            # Update maximum similarities
            new_sims = embeddings_norm @ embeddings_norm[best_idx:best_idx+1].T
            max_sim_to_selected = np.maximum(max_sim_to_selected, new_sims.flatten())

        indices = np.array(selected)

        # Compute diversity score (average pairwise distance in selected set)
        selected_embeddings = embeddings_norm[indices]
        sim_matrix = selected_embeddings @ selected_embeddings.T
        n_sel = len(indices)
        avg_sim = (sim_matrix.sum() - n_sel) / (n_sel * (n_sel - 1)) if n_sel > 1 else 0
        diversity_score = 1 - avg_sim  # Convert similarity to diversity

        metadata = {
            'method': 'greedy_diversity',
            'seed': seed,
            'diversity_score': float(diversity_score),
            'avg_pairwise_similarity': float(avg_sim),
        }

        return indices, metadata


class ClusteringSelector(Selector):
    """
    K-Means Clustering + Centroid Selection.

    For large datasets (> 100k), greedy selection becomes too slow.
    Instead:
    1. Cluster embeddings into K clusters (K = target_size)
    2. Select the point closest to each cluster centroid

    This provides a representative sample that covers the data distribution.

    Time complexity: O(n_samples * K * iterations) - much faster than greedy for large N
    Best for: Datasets > 100k samples
    """

    def __init__(self, use_minibatch: bool = True, batch_size: int = 10000):
        """
        Args:
            use_minibatch: Use MiniBatchKMeans for very large datasets
            batch_size: Batch size for MiniBatchKMeans
        """
        self.use_minibatch = use_minibatch
        self.batch_size = batch_size

    def select(
        self,
        embeddings: np.ndarray,
        n_select: int,
        seed: int = 42,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """Clustering-based selection."""
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.metrics import pairwise_distances_argmin_min

        n_samples = embeddings.shape[0]
        n_select = min(n_select, n_samples)

        print(f"   Clustering {n_samples} samples into {n_select} clusters...")

        # Choose clustering algorithm based on dataset size
        if self.use_minibatch and n_samples > 50000:
            print(f"   Using MiniBatchKMeans (batch_size={self.batch_size})")
            kmeans = MiniBatchKMeans(
                n_clusters=n_select,
                random_state=seed,
                batch_size=self.batch_size,
                n_init=3,
                verbose=1 if verbose else 0,
            )
        else:
            print(f"   Using standard KMeans")
            kmeans = KMeans(
                n_clusters=n_select,
                random_state=seed,
                n_init=10,
                verbose=1 if verbose else 0,
            )

        # Fit clustering
        kmeans.fit(embeddings)
        cluster_centers = kmeans.cluster_centers_

        # Find point closest to each centroid
        print(f"   Finding closest points to centroids...")
        closest_indices, distances = pairwise_distances_argmin_min(
            cluster_centers, embeddings
        )

        indices = closest_indices.astype(int)

        # Compute diversity metrics
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = embeddings / norms

        selected_embeddings = embeddings_norm[indices]
        sim_matrix = selected_embeddings @ selected_embeddings.T
        n_sel = len(indices)
        avg_sim = (sim_matrix.sum() - n_sel) / (n_sel * (n_sel - 1)) if n_sel > 1 else 0
        diversity_score = 1 - avg_sim

        metadata = {
            'method': 'clustering',
            'algorithm': 'MiniBatchKMeans' if self.use_minibatch and n_samples > 50000 else 'KMeans',
            'n_clusters': n_select,
            'seed': seed,
            'inertia': float(kmeans.inertia_),
            'diversity_score': float(diversity_score),
            'avg_pairwise_similarity': float(avg_sim),
            'avg_distance_to_centroid': float(np.mean(distances)),
        }

        return indices, metadata


class HybridSelector(Selector):
    """
    Hybrid selector that chooses algorithm based on dataset size.

    - For N < threshold: Use greedy diversity maximization
    - For N >= threshold: Use clustering-based selection
    """

    def __init__(self, threshold: int = 100000):
        self.threshold = threshold
        self.greedy = GreedyDiversitySelector()
        self.clustering = ClusteringSelector()

    def select(
        self,
        embeddings: np.ndarray,
        n_select: int,
        seed: int = 42,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        n_samples = embeddings.shape[0]

        if n_samples < self.threshold:
            print(f"   Using greedy selection (N={n_samples} < threshold={self.threshold})")
            return self.greedy.select(embeddings, n_select, seed, verbose)
        else:
            print(f"   Using clustering selection (N={n_samples} >= threshold={self.threshold})")
            return self.clustering.select(embeddings, n_select, seed, verbose)


class UniversalEmbeddingSelector(Selector):
    """
    Universal Linguistic Diversity Embedding Selector.

    Uses diversity embeddings that capture multiple linguistic dimensions:
    - Token semantics
    - Document semantics
    - Dependency parse structures
    - Constituency parse structures
    - Part-of-speech sequences
    - Rhythmic patterns
    - Phonemic patterns

    Then applies submodular optimization to select samples that
    maximize coverage across all diversity dimensions.
    """

    def __init__(
        self,
        selection_method: str = "facility_location",
        use_constituency_parse: bool = False,
        use_rhythmic: bool = False,
        use_phonemic: bool = False,
    ):
        """
        Args:
            selection_method: 'facility_location', 'max_min', or 'balanced'
            use_constituency_parse: Include constituency parsing (slower)
            use_rhythmic: Include rhythmic analysis
            use_phonemic: Include phonemic analysis
        """
        self.selection_method = selection_method
        self.metric = UniversalLinguisticDiversity({
            'use_constituency_parse': use_constituency_parse,
            'use_rhythmic': use_rhythmic,
            'use_phonemic': use_phonemic,
            'verbose': False,
        })

    def select(
        self,
        embeddings: np.ndarray,
        n_select: int,
        seed: int = 42,
        verbose: bool = True,
        samples: Optional[List[Dict]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Select using universal diversity embeddings.

        Args:
            embeddings: Semantic embeddings (used as fallback, but we compute diversity embeddings)
            n_select: Number to select
            seed: Random seed
            verbose: Print progress
            samples: List of sample dicts with 'instruction' and 'response' keys

        Returns:
            Tuple of (indices, metadata)
        """
        from tqdm import tqdm

        n_samples = embeddings.shape[0]
        n_select = min(n_select, n_samples)

        if samples is None:
            print("   Warning: No text samples provided, falling back to semantic embeddings")
            # Fall back to greedy selection on semantic embeddings
            greedy = GreedyDiversitySelector()
            return greedy.select(embeddings, n_select, seed, verbose)

        print(f"   Computing universal diversity embeddings for {n_samples} samples...")

        # Compute diversity embeddings
        n_metrics = len(DIVERSITY_EMBEDDING_METRICS)
        diversity_embeddings = np.zeros((n_samples, n_metrics), dtype=np.float64)

        batch_size = 100
        iterator = range(0, n_samples, batch_size)
        if verbose:
            iterator = tqdm(iterator, desc="   Computing diversity embeddings")

        for i in iterator:
            batch_end = min(i + batch_size, n_samples)
            for j in range(i, batch_end):
                sample = samples[j]
                # Combine instruction and response for diversity analysis
                text = f"{sample.get('instruction', '')} {sample.get('response', '')}"
                if text.strip():
                    try:
                        emb = self.metric._compute_single_doc_embedding([text], normalize=True)
                        diversity_embeddings[j] = emb
                    except Exception:
                        pass

        print(f"   Running {self.selection_method} selection...")

        # Apply submodular selection
        result = select_diverse_texts(
            diversity_embeddings,
            n_select=n_select,
            method=self.selection_method,
            seed=seed,
            verbose=verbose,
        )

        indices = result.indices

        # Compute diversity metrics for selected set
        selected_embeddings = diversity_embeddings[indices]
        norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        selected_norm = selected_embeddings / norms

        sim_matrix = selected_norm @ selected_norm.T
        n_sel = len(indices)
        avg_sim = (sim_matrix.sum() - n_sel) / (n_sel * (n_sel - 1)) if n_sel > 1 else 0
        diversity_score = 1 - avg_sim

        metadata = {
            'method': 'universal_embedding_diversity',
            'selection_algorithm': self.selection_method,
            'seed': seed,
            'diversity_score': float(diversity_score),
            'avg_pairwise_similarity': float(avg_sim),
            'coverage_per_metric': result.coverage_per_metric.tolist(),
            'metrics': DIVERSITY_EMBEDDING_METRICS,
        }

        return indices, metadata


def compute_selection_length_stats(samples: List[Dict], indices: np.ndarray) -> Dict:
    """
    Compute length statistics for selected samples.

    This is critical for monitoring length bias - per spec warning:
    "Diversity metrics sometimes favor long, verbose outputs"
    """
    selected_samples = [samples[i] for i in indices]

    instruction_lengths = [len(s['instruction'].split()) for s in selected_samples]
    response_lengths = [len(s['response'].split()) for s in selected_samples]

    return {
        'instruction': {
            'mean': float(np.mean(instruction_lengths)),
            'median': float(np.median(instruction_lengths)),
            'std': float(np.std(instruction_lengths)),
            'min': int(np.min(instruction_lengths)),
            'max': int(np.max(instruction_lengths)),
        },
        'response': {
            'mean': float(np.mean(response_lengths)),
            'median': float(np.median(response_lengths)),
            'std': float(np.std(response_lengths)),
            'min': int(np.min(response_lengths)),
            'max': int(np.max(response_lengths)),
        },
        'n_selected': len(indices),
    }


def check_length_bias(
    selected_stats: Dict,
    full_stats: Dict,
    max_ratio: float = 3.0
) -> Dict:
    """
    Check for length bias in selection.

    Returns warning if selected samples are significantly longer than random.
    """
    selected_resp_mean = selected_stats['response']['mean']
    full_resp_mean = full_stats['response']['mean']

    ratio = selected_resp_mean / full_resp_mean if full_resp_mean > 0 else 1.0

    return {
        'selected_response_mean': selected_resp_mean,
        'full_response_mean': full_resp_mean,
        'length_ratio': ratio,
        'has_length_bias': ratio > max_ratio,
        'warning': f"Length bias detected! Selected samples are {ratio:.1f}x longer than average"
                   if ratio > max_ratio else None,
    }


def save_selection_jsonl(
    samples: List[Dict],
    indices: np.ndarray,
    output_path: Path,
) -> None:
    """Save selected samples in JSONL format for training."""
    with open(output_path, 'w') as f:
        for idx in indices:
            sample = samples[idx]
            # Format for instruction tuning
            record = {
                'instruction': sample['instruction'],
                'input': sample.get('input', ''),
                'output': sample['response'],
            }
            f.write(json.dumps(record) + '\n')


def main():
    """Main selection pipeline."""
    import time

    print("=" * 70)
    print("STEP 2: DIVERSITY-BASED SUBSET SELECTION")
    print("=" * 70)

    config = load_config()
    mode = config.get('mode', 'pilot')

    print(f"\nMode: {mode}")

    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    output_dir = base_dir / "selections"
    output_dir.mkdir(exist_ok=True)

    # Get selection configuration
    selection_config = config['selection']
    length_config = selection_config.get('length_normalization', {})
    max_length_ratio = length_config.get('max_length_ratio', 3.0)

    # Determine target fraction
    if mode == 'pilot':
        target_fraction = config['pilot'].get('target_fraction', 0.10)
        datasets_to_process = [config['pilot']['dataset']['name']]
    else:
        target_fraction = 0.10  # Default 10%
        datasets_to_process = [config['pilot']['dataset']['name']]
        datasets_to_process += [d['name'] for d in config['scaleup']['datasets']]

    # Initialize selectors
    selectors = {
        'random': RandomSelector(),
        'diversity': HybridSelector(threshold=100000),
        'universal_embedding': UniversalEmbeddingSelector(
            selection_method='facility_location',
            use_constituency_parse=False,
            use_rhythmic=False,
            use_phonemic=False,
        ),
    }

    all_selections = {}

    for dataset_name in datasets_to_process:
        short_name = dataset_name.split('/')[-1].lower().replace('-', '_')

        print(f"\n{'=' * 70}")
        print(f"Processing: {dataset_name}")
        print(f"{'=' * 70}")

        # Load data
        samples_file = data_dir / f"{short_name}_samples.json"
        embeddings_file = data_dir / f"{short_name}_embeddings.npy"
        metadata_file = data_dir / f"{short_name}_metadata.json"

        if not samples_file.exists():
            print(f"   Samples file not found: {samples_file}")
            print(f"   Run 01_prepare_data.py first")
            continue

        print(f"\n   Loading data...")
        with open(samples_file, 'r') as f:
            samples = json.load(f)
        embeddings = np.load(embeddings_file)
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        n_samples = len(samples)
        n_select = int(n_samples * target_fraction)

        print(f"   Total samples: {n_samples}")
        print(f"   Target selection: {n_select} ({target_fraction*100:.0f}%)")

        full_length_stats = metadata['length_stats']

        dataset_selections = {
            'dataset': dataset_name,
            'n_total': n_samples,
            'n_select': n_select,
            'target_fraction': target_fraction,
            'methods': {},
        }

        # Create selections
        for method_name, selector in selectors.items():
            print(f"\n   Method: {method_name}")
            print(f"   {'-' * 50}")

            start_time = time.time()

            # Select (pass samples for universal embedding selector)
            if isinstance(selector, UniversalEmbeddingSelector):
                indices, select_metadata = selector.select(
                    embeddings, n_select, seed=42, verbose=True, samples=samples
                )
            else:
                indices, select_metadata = selector.select(
                    embeddings, n_select, seed=42, verbose=True
                )

            compute_time = time.time() - start_time
            print(f"   Selection completed in {compute_time:.1f}s")

            # Compute length stats
            length_stats = compute_selection_length_stats(samples, indices)

            # Check for length bias
            bias_check = check_length_bias(length_stats, full_length_stats, max_length_ratio)
            if bias_check['has_length_bias']:
                print(f"   WARNING: {bias_check['warning']}")
            else:
                print(f"   Length ratio: {bias_check['length_ratio']:.2f}x (OK)")

            # Create result
            result = SelectionResult(
                method=method_name,
                indices=indices.tolist(),
                n_selected=len(indices),
                n_total=n_samples,
                selection_ratio=len(indices) / n_samples,
                diversity_score=select_metadata.get('diversity_score'),
                length_stats=length_stats,
                compute_time_seconds=compute_time,
                metadata=select_metadata,
            )

            # Save JSONL for training
            jsonl_path = output_dir / f"{short_name}_{method_name}.jsonl"
            save_selection_jsonl(samples, indices, jsonl_path)
            print(f"   Saved: {jsonl_path}")

            dataset_selections['methods'][method_name] = asdict(result)

        # Also create full dataset JSONL (for baseline training)
        full_indices = np.arange(n_samples)
        full_jsonl_path = output_dir / f"{short_name}_full.jsonl"
        save_selection_jsonl(samples, full_indices, full_jsonl_path)
        print(f"\n   Saved full dataset: {full_jsonl_path}")

        # Add full dataset stats
        dataset_selections['methods']['full'] = {
            'method': 'full',
            'indices': full_indices.tolist(),
            'n_selected': n_samples,
            'n_total': n_samples,
            'selection_ratio': 1.0,
            'length_stats': full_length_stats,
        }

        all_selections[short_name] = dataset_selections

    # Save selection summary
    summary_file = output_dir / "selection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'mode': mode,
            'selections': all_selections,
        }, f, indent=2)
    print(f"\n   Saved summary: {summary_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SELECTION SUMMARY")
    print("=" * 70)

    for ds_name, ds_selections in all_selections.items():
        print(f"\n{ds_name}:")
        print(f"  Total: {ds_selections['n_total']}, Target: {ds_selections['n_select']}")

        for method_name, method_data in ds_selections['methods'].items():
            n_sel = method_data['n_selected']
            div_score = method_data.get('diversity_score', 'N/A')
            if isinstance(div_score, float):
                div_score = f"{div_score:.4f}"

            resp_len = method_data['length_stats']['response']['mean']
            print(f"  {method_name:15s}: n={n_sel:6d}, diversity={div_score}, "
                  f"avg_response_len={resp_len:.1f}")

    print("\n" + "=" * 70)
    print("SUBSET SELECTION COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
