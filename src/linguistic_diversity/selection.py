"""Submodular optimization for diversity-based text selection.

This module provides algorithms for selecting a subset of texts that
maximizes coverage across multiple linguistic diversity dimensions.

The key insight is that instead of using a single diversity metric,
we treat each diversity dimension as a separate objective and use
submodular optimization to find texts that collectively cover all dimensions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import numpy.typing as npt


@dataclass
class SelectionResult:
    """Result of a diversity-based selection algorithm.

    Attributes:
        indices: Indices of selected items in the original corpus.
        scores: Final objective scores for selected set.
        n_selected: Number of items selected.
        method: Name of selection algorithm used.
        coverage_per_metric: Coverage achieved for each diversity dimension.
        metadata: Additional algorithm-specific information.
    """

    indices: npt.NDArray[np.intp]
    scores: npt.NDArray[np.float64]
    n_selected: int
    method: str
    coverage_per_metric: npt.NDArray[np.float64]
    metadata: dict[str, Any] = field(default_factory=dict)


class DiversitySelector(ABC):
    """Abstract base class for diversity-based selection algorithms."""

    @abstractmethod
    def select(
        self,
        embeddings: npt.NDArray[np.float64],
        n_select: int,
        **kwargs: Any,
    ) -> SelectionResult:
        """Select a diverse subset of items.

        Args:
            embeddings: Diversity embeddings of shape (n_items, n_metrics).
            n_select: Number of items to select.
            **kwargs: Algorithm-specific parameters.

        Returns:
            SelectionResult with selected indices and metadata.
        """
        ...


class FacilityLocationSelector(DiversitySelector):
    """Facility Location based selection for multi-metric diversity.

    This implements a greedy submodular optimization approach that treats
    each diversity metric as a separate facility location problem and
    combines them to ensure coverage across all dimensions.

    The facility location objective for a set S is:
        F(S) = sum_i max_{j in S} similarity(i, j)

    This encourages selecting items that collectively "cover" the entire
    space, ensuring diversity across all dimensions.

    For multi-metric diversity, we compute a combined similarity that
    considers all diversity dimensions, ensuring the selected set has
    high coverage across semantic, syntactic, morphological, and
    phonological diversity.
    """

    def __init__(
        self,
        metric_weights: npt.NDArray[np.float64] | None = None,
        similarity_fn: str = "cosine",
    ):
        """Initialize the facility location selector.

        Args:
            metric_weights: Optional weights for each diversity metric.
                Shape (n_metrics,). If None, uniform weights are used.
            similarity_fn: Similarity function to use. Options:
                - "cosine": Cosine similarity (default)
                - "rbf": RBF (Gaussian) kernel
                - "linear": Linear (dot product)
        """
        self.metric_weights = metric_weights
        self.similarity_fn = similarity_fn

    def select(
        self,
        embeddings: npt.NDArray[np.float64],
        n_select: int,
        seed: int = 42,
        verbose: bool = False,
    ) -> SelectionResult:
        """Select diverse items using greedy facility location.

        Args:
            embeddings: Diversity embeddings of shape (n_items, n_metrics).
            n_select: Number of items to select.
            seed: Random seed for tie-breaking.
            verbose: Whether to print progress.

        Returns:
            SelectionResult with selected indices.
        """
        n_items, n_metrics = embeddings.shape
        n_select = min(n_select, n_items)

        if verbose:
            print(f"Selecting {n_select} items from {n_items} using facility location...")

        # Apply metric weights if provided
        if self.metric_weights is not None:
            weights = np.asarray(self.metric_weights)
            if weights.shape[0] != n_metrics:
                raise ValueError(
                    f"metric_weights has {weights.shape[0]} elements, "
                    f"expected {n_metrics}"
                )
            embeddings = embeddings * np.sqrt(weights)

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = embeddings / norms

        # Compute similarity matrix
        if self.similarity_fn == "cosine":
            similarity = embeddings_norm @ embeddings_norm.T
        elif self.similarity_fn == "rbf":
            # RBF kernel: exp(-gamma * ||x - y||^2)
            sq_dists = (
                np.sum(embeddings**2, axis=1, keepdims=True)
                + np.sum(embeddings**2, axis=1)
                - 2 * embeddings @ embeddings.T
            )
            gamma = 1.0 / n_metrics
            similarity = np.exp(-gamma * np.maximum(sq_dists, 0))
        elif self.similarity_fn == "linear":
            similarity = embeddings @ embeddings.T
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_fn}")

        # Greedy facility location
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        remaining = set(range(n_items))

        # Track max similarity to selected set for each item
        # Initially no items selected, so max_sim = 0
        max_sim_to_selected = np.zeros(n_items)

        # Marginal gains for each item
        marginal_gains = similarity.sum(axis=0)  # Initial: adding item i covers sum of similarities

        for step in range(n_select):
            if not remaining:
                break

            # Find item with maximum marginal gain
            remaining_list = list(remaining)
            gains = marginal_gains[remaining_list]

            # Handle ties randomly
            max_gain = gains.max()
            candidates = [remaining_list[i] for i, g in enumerate(gains) if g >= max_gain - 1e-10]
            best_idx = rng.choice(candidates)

            selected.append(best_idx)
            remaining.remove(best_idx)

            # Update max similarities and marginal gains
            new_sims = similarity[best_idx]
            improvement = np.maximum(new_sims - max_sim_to_selected, 0)
            max_sim_to_selected = np.maximum(max_sim_to_selected, new_sims)

            # Marginal gain of adding item j is now:
            # sum_i max(sim(i,j), max_sim_to_selected[i]) - sum_i max_sim_to_selected[i]
            # = sum_i max(sim(i,j) - max_sim_to_selected[i], 0)
            for j in remaining:
                marginal_gains[j] = np.sum(
                    np.maximum(similarity[j] - max_sim_to_selected, 0)
                )

            if verbose and (step + 1) % max(1, n_select // 10) == 0:
                print(f"  Selected {step + 1}/{n_select} items")

        indices = np.array(selected, dtype=np.intp)

        # Compute coverage per metric
        # Coverage = average max similarity achieved per metric
        selected_embeddings = embeddings_norm[indices]
        coverage = np.zeros(n_metrics)
        for m in range(n_metrics):
            metric_emb = embeddings_norm[:, m : m + 1]
            selected_metric = selected_embeddings[:, m : m + 1]
            # Max similarity of each item to selected set in this metric
            sim_to_selected = (metric_emb @ selected_metric.T).max(axis=1)
            coverage[m] = sim_to_selected.mean()

        # Final objective value
        final_score = max_sim_to_selected.sum()

        return SelectionResult(
            indices=indices,
            scores=max_sim_to_selected[indices],
            n_selected=len(indices),
            method="facility_location",
            coverage_per_metric=coverage,
            metadata={
                "total_coverage": float(final_score),
                "similarity_fn": self.similarity_fn,
                "seed": seed,
            },
        )


class MaxMinDiversitySelector(DiversitySelector):
    """Max-Min Diversity selection across multiple metrics.

    This selector finds items that maximize the minimum pairwise distance
    in the diversity embedding space. It ensures that selected items are
    spread out across all diversity dimensions.

    Unlike facility location (which maximizes coverage), max-min diversity
    maximizes the spread of selected items.
    """

    def __init__(
        self,
        metric_weights: npt.NDArray[np.float64] | None = None,
    ):
        """Initialize the max-min diversity selector.

        Args:
            metric_weights: Optional weights for each diversity metric.
        """
        self.metric_weights = metric_weights

    def select(
        self,
        embeddings: npt.NDArray[np.float64],
        n_select: int,
        seed: int = 42,
        verbose: bool = False,
    ) -> SelectionResult:
        """Select diverse items using greedy max-min diversity.

        Args:
            embeddings: Diversity embeddings of shape (n_items, n_metrics).
            n_select: Number of items to select.
            seed: Random seed for initialization.
            verbose: Whether to print progress.

        Returns:
            SelectionResult with selected indices.
        """
        n_items, n_metrics = embeddings.shape
        n_select = min(n_select, n_items)

        if verbose:
            print(f"Selecting {n_select} items from {n_items} using max-min diversity...")

        # Apply metric weights if provided
        if self.metric_weights is not None:
            weights = np.asarray(self.metric_weights)
            embeddings = embeddings * np.sqrt(weights)

        # Normalize for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = embeddings / norms

        # Use similarity (higher = more similar = less diverse)
        # Max-min diversity: maximize min distance = minimize max similarity
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        remaining = set(range(n_items))

        # Start with random item
        first = rng.choice(list(remaining))
        selected.append(first)
        remaining.remove(first)

        # Track max similarity to selected set (we want to minimize this)
        max_sim_to_selected = embeddings_norm @ embeddings_norm[first : first + 1].T
        max_sim_to_selected = max_sim_to_selected.flatten()

        for step in range(1, n_select):
            if not remaining:
                break

            remaining_list = list(remaining)
            sims = max_sim_to_selected[remaining_list]

            # Select item with minimum similarity to selected set (most diverse)
            best_local_idx = sims.argmin()
            best_idx = remaining_list[best_local_idx]

            selected.append(best_idx)
            remaining.remove(best_idx)

            # Update max similarities
            new_sims = embeddings_norm @ embeddings_norm[best_idx : best_idx + 1].T
            max_sim_to_selected = np.maximum(max_sim_to_selected, new_sims.flatten())

            if verbose and (step + 1) % max(1, n_select // 10) == 0:
                print(f"  Selected {step + 1}/{n_select} items")

        indices = np.array(selected, dtype=np.intp)

        # Compute coverage per metric
        selected_embeddings = embeddings_norm[indices]
        coverage = np.zeros(n_metrics)
        for m in range(n_metrics):
            # Variance in this metric dimension
            coverage[m] = np.std(selected_embeddings[:, m])

        # Compute diversity score (average pairwise distance)
        if len(indices) > 1:
            selected_sims = selected_embeddings @ selected_embeddings.T
            n_sel = len(indices)
            avg_sim = (selected_sims.sum() - n_sel) / (n_sel * (n_sel - 1))
            diversity_score = 1 - avg_sim
        else:
            diversity_score = 0.0

        return SelectionResult(
            indices=indices,
            scores=1 - max_sim_to_selected[indices],  # Convert sim to diversity
            n_selected=len(indices),
            method="max_min_diversity",
            coverage_per_metric=coverage,
            metadata={
                "diversity_score": float(diversity_score),
                "seed": seed,
            },
        )


class BalancedCoverageSelector(DiversitySelector):
    """Balanced coverage selection ensuring all metrics are covered.

    This selector explicitly balances coverage across all diversity metrics,
    preventing any single metric from dominating the selection. It uses a
    round-robin approach where at each step, it selects an item that most
    improves the least-covered metric.

    This is useful when you want guaranteed minimum coverage across all
    linguistic diversity dimensions.
    """

    def __init__(
        self,
        min_coverage_per_metric: float = 0.5,
    ):
        """Initialize the balanced coverage selector.

        Args:
            min_coverage_per_metric: Target minimum coverage for each metric
                before switching to overall optimization.
        """
        self.min_coverage_per_metric = min_coverage_per_metric

    def select(
        self,
        embeddings: npt.NDArray[np.float64],
        n_select: int,
        seed: int = 42,
        verbose: bool = False,
    ) -> SelectionResult:
        """Select items with balanced coverage across all metrics.

        Args:
            embeddings: Diversity embeddings of shape (n_items, n_metrics).
            n_select: Number of items to select.
            seed: Random seed.
            verbose: Whether to print progress.

        Returns:
            SelectionResult with selected indices.
        """
        n_items, n_metrics = embeddings.shape
        n_select = min(n_select, n_items)

        if verbose:
            print(f"Selecting {n_select} items with balanced coverage...")

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings_norm = embeddings / norms

        rng = np.random.default_rng(seed)
        selected: list[int] = []
        remaining = set(range(n_items))

        # Track coverage per metric
        # Coverage[m] = max similarity to selected set for metric m, averaged over all items
        per_metric_max_sim = np.zeros((n_items, n_metrics))

        for step in range(n_select):
            if not remaining:
                break

            remaining_list = list(remaining)

            # Compute current coverage per metric
            current_coverage = per_metric_max_sim.mean(axis=0)

            # Find least covered metric (that's below threshold)
            below_threshold = current_coverage < self.min_coverage_per_metric
            if below_threshold.any():
                # Focus on least covered metric
                target_metric = np.argmin(
                    np.where(below_threshold, current_coverage, np.inf)
                )
            else:
                # All metrics above threshold, optimize overall
                target_metric = None

            # Score each candidate
            best_idx = None
            best_score = -np.inf

            for idx in remaining_list:
                # Compute improvement in coverage if we add this item
                candidate_sims = (
                    embeddings_norm @ embeddings_norm[idx : idx + 1].T
                ).flatten()

                if target_metric is not None:
                    # Score = improvement in target metric
                    new_coverage = np.maximum(
                        per_metric_max_sim[:, target_metric], candidate_sims
                    )
                    score = new_coverage.mean() - current_coverage[target_metric]
                else:
                    # Score = improvement in min coverage across all metrics
                    improvements = []
                    for m in range(n_metrics):
                        new_cov_m = np.maximum(
                            per_metric_max_sim[:, m],
                            candidate_sims * embeddings_norm[idx, m],
                        )
                        improvements.append(new_cov_m.mean() - current_coverage[m])
                    score = min(improvements)  # Maximize the minimum improvement

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is None:
                best_idx = rng.choice(remaining_list)

            selected.append(best_idx)
            remaining.remove(best_idx)

            # Update per-metric similarities
            new_sims = (
                embeddings_norm @ embeddings_norm[best_idx : best_idx + 1].T
            ).flatten()
            for m in range(n_metrics):
                per_metric_max_sim[:, m] = np.maximum(
                    per_metric_max_sim[:, m],
                    new_sims * abs(embeddings_norm[best_idx, m]),
                )

            if verbose and (step + 1) % max(1, n_select // 10) == 0:
                coverage = per_metric_max_sim.mean(axis=0)
                print(f"  Selected {step + 1}/{n_select}, coverage: {coverage}")

        indices = np.array(selected, dtype=np.intp)
        final_coverage = per_metric_max_sim.mean(axis=0)

        return SelectionResult(
            indices=indices,
            scores=per_metric_max_sim[indices].mean(axis=1),
            n_selected=len(indices),
            method="balanced_coverage",
            coverage_per_metric=final_coverage,
            metadata={
                "min_coverage_threshold": self.min_coverage_per_metric,
                "final_min_coverage": float(final_coverage.min()),
                "seed": seed,
            },
        )


def select_diverse_texts(
    embeddings: npt.NDArray[np.float64],
    n_select: int,
    method: str = "facility_location",
    metric_weights: npt.NDArray[np.float64] | None = None,
    seed: int = 42,
    verbose: bool = False,
    **kwargs: Any,
) -> SelectionResult:
    """Select a diverse subset of texts based on diversity embeddings.

    This is a convenience function that wraps the selector classes.

    Args:
        embeddings: Diversity embeddings of shape (n_items, n_metrics).
            Each row is a diversity embedding from
            UniversalLinguisticDiversity.compute_diversity_embedding().
        n_select: Number of items to select.
        method: Selection algorithm. Options:
            - "facility_location": Maximize coverage (default)
            - "max_min": Maximize minimum pairwise distance
            - "balanced": Balance coverage across all metrics
        metric_weights: Optional weights for each diversity metric.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress.
        **kwargs: Additional algorithm-specific parameters.

    Returns:
        SelectionResult with selected indices and metadata.

    Example:
        >>> from linguistic_diversity import UniversalLinguisticDiversity
        >>> from linguistic_diversity.selection import select_diverse_texts
        >>>
        >>> metric = UniversalLinguisticDiversity()
        >>> corpus = ["Text 1", "Text 2", "Text 3", ...]
        >>> embeddings = metric.compute_corpus_diversity_embeddings(corpus)
        >>> result = select_diverse_texts(embeddings, n_select=100)
        >>> selected_texts = [corpus[i] for i in result.indices]
    """
    selectors: dict[str, Callable[..., DiversitySelector]] = {
        "facility_location": lambda: FacilityLocationSelector(
            metric_weights=metric_weights,
            similarity_fn=kwargs.get("similarity_fn", "cosine"),
        ),
        "max_min": lambda: MaxMinDiversitySelector(metric_weights=metric_weights),
        "balanced": lambda: BalancedCoverageSelector(
            min_coverage_per_metric=kwargs.get("min_coverage_per_metric", 0.5),
        ),
    }

    if method not in selectors:
        raise ValueError(
            f"Unknown method '{method}'. Available: {list(selectors.keys())}"
        )

    selector = selectors[method]()
    return selector.select(embeddings, n_select, seed=seed, verbose=verbose)
