"""Core metric classes for linguistic diversity measurement."""

from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt

try:
    from scipy import optimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class MetricConfig:
    """Base configuration for metrics."""

    q: float = 1.0  # Diversity order parameter
    normalize: bool = False  # Normalize diversity by number of species
    verbose: bool = False


@dataclass
class ScaledEstimationResult:
    """Result of scaled diversity estimation.

    This dataclass holds the results of diversity estimation using sampling
    and extrapolation, similar to rarefaction curves in ecology.

    Attributes:
        diversity: Estimated diversity value (either direct or extrapolated)
        std: Projected uncertainty of the estimate (from sampling variance).
            Note: This is a heuristic projection, not a true statistical
            confidence interval derived from parameter covariance.
        projected_uncertainty_95: 95% projected uncertainty bounds [lower, upper].
            Based on sampling variance scaled by extrapolation distance.
        method: Either "direct" (measured full corpus) or "extrapolation"
        model: Name of the fitted growth curve model (if extrapolated)
        model_params: Fitted parameters for the growth curve model
        sample_sizes: List of sample sizes used for measurements
        sample_means: Mean diversity at each sample size
        sample_stds: Std of diversity at each sample size
        corpus_size: Total size of the corpus
        fit_rmse: RMSE of the curve fit (if extrapolated)
    """
    diversity: float
    std: float = 0.0
    projected_uncertainty_95: Tuple[float, float] = (0.0, 0.0)
    method: str = "direct"
    model: Optional[str] = None
    model_params: Optional[List[float]] = None
    sample_sizes: List[int] = field(default_factory=list)
    sample_means: List[float] = field(default_factory=list)
    sample_stds: List[float] = field(default_factory=list)
    corpus_size: int = 0
    fit_rmse: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'diversity': self.diversity,
            'std': self.std,
            'projected_uncertainty_95': list(self.projected_uncertainty_95),
            'method': self.method,
            'model': self.model,
            'model_params': self.model_params,
            'sample_sizes': self.sample_sizes,
            'sample_means': self.sample_means,
            'sample_stds': self.sample_stds,
            'corpus_size': self.corpus_size,
            'fit_rmse': self.fit_rmse,
        }

    def plot(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """Visualize the sampling points and fitted curve.

        Plots observed diversity at each sample size with error bars,
        the fitted growth curve, and the extrapolated estimate.

        Args:
            save_path: Optional path to save the figure.
            show: Whether to display the plot (default True).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib is required for plotting. Install with: pip install matplotlib")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot observed samples with error bars
        if self.sample_sizes and self.sample_means:
            ax.errorbar(
                self.sample_sizes, self.sample_means,
                yerr=self.sample_stds if self.sample_stds else None,
                fmt='o', markersize=8, capsize=5,
                label='Observed samples', color='blue'
            )

        # Plot fitted curve if we have model params
        if self.model and self.model_params and self.sample_sizes:
            x_curve = np.linspace(
                min(self.sample_sizes) * 0.8,
                self.corpus_size * 1.1,
                200
            )

            # Reconstruct the fitted curve
            a, b, c = self.model_params
            if self.model == 'logarithmic':
                y_curve = a * np.log(x_curve + b) + c
            elif self.model == 'power_law':
                y_curve = a * np.power(x_curve, b) + c
            elif self.model == 'asymptotic':
                y_curve = a * (1 - np.exp(-x_curve / b)) + c
            else:  # linear fallback
                y_curve = a * x_curve + b

            ax.plot(x_curve, y_curve, '--', color='orange',
                    label=f'Fitted curve ({self.model})', linewidth=2)

        # Plot extrapolated point with uncertainty
        if self.corpus_size and self.diversity:
            ax.errorbar(
                [self.corpus_size], [self.diversity],
                yerr=[[self.diversity - self.projected_uncertainty_95[0]],
                      [self.projected_uncertainty_95[1] - self.diversity]],
                fmt='s', markersize=10, capsize=5,
                label=f'Extrapolated (n={self.corpus_size})', color='red'
            )

        ax.set_xscale('log')
        ax.set_xlabel('Sample Size (log scale)', fontsize=12)
        ax.set_ylabel('Diversity', fontsize=12)
        ax.set_title(f'Diversity Scaling Analysis\n'
                     f'Method: {self.method}, Model: {self.model or "N/A"}, '
                     f'RMSE: {self.fit_rmse:.4f}', fontsize=11)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()


class Metric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, config: MetricConfig | dict[str, Any] | None = None) -> None:
        """Initialize metric with configuration.

        Args:
            config: Metric configuration as MetricConfig or dict.
        """
        if config is None:
            config = {}
        if isinstance(config, dict):
            config = self._config_class()(**{**self._default_config(), **config})
        self.config = config

    @classmethod
    def _config_class(cls) -> type[MetricConfig]:
        """Return the configuration class for this metric."""
        return MetricConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        """Return default configuration as dict."""
        return {}

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> float:
        """Compute the metric."""
        ...


class DiversityMetric(Metric):
    """Base class for diversity metrics.

    This class provides both precise and estimated diversity computation:

    - Precise: Call the metric directly with `metric(corpus)` or `metric.diversity(corpus)`
      to compute exact diversity on the full corpus.

    - Estimated: Use `metric.estimate_diversity(corpus)` for large corpora. This uses
      sampling at multiple scales and curve fitting to extrapolate diversity,
      similar to rarefaction curves in ecology or Heaps' law in linguistics.
    """

    @abstractmethod
    def __call__(self, corpus: list[str]) -> float:
        """Compute diversity for a corpus (precise calculation).

        Args:
            corpus: List of text documents.

        Returns:
            Diversity score.
        """
        ...

    def estimate_diversity(
        self,
        corpus: list[str],
        base_sample_size: int = 50,
        max_sample_size: int = 200,
        num_trials: int = 2,
        growth_factor: float = 2.0,
        random_seed: int = 42,
        verbose: bool = True,
    ) -> ScaledEstimationResult:
        """Estimate diversity using sampling and extrapolation.

        For large corpora, this method estimates diversity by:
        1. Taking random samples at increasing sizes (e.g., 50, 100, 200)
        2. Measuring diversity at each sample size (multiple trials for variance)
        3. Fitting a growth curve (logarithmic, power-law, or asymptotic)
        4. Extrapolating to the full corpus size

        This approach is similar to:
        - Rarefaction curves in ecology (species accumulation)
        - Heaps' law for vocabulary growth in linguistics
        - Bootstrap estimation methods in statistics

        Args:
            corpus: List of text documents.
            base_sample_size: Starting sample size (default: 50).
            max_sample_size: Maximum sample size to actually measure (default: 200).
            num_trials: Number of random trials per sample size for variance (default: 2).
            growth_factor: Factor to increase sample size by each step (default: 2.0).
            random_seed: Random seed for reproducibility (default: 42).
            verbose: Whether to print progress information (default: True).

        Returns:
            ScaledEstimationResult with estimated diversity, confidence intervals,
            and curve fitting details.

        Example:
            >>> metric = TokenSemantics()
            >>> result = metric.estimate_diversity(large_corpus, max_sample_size=200)
            >>> print(f"Estimated diversity: {result.diversity:.3f} ± {result.std:.3f}")
            >>> print(f"Method: {result.method}, Model: {result.model}")
        """
        corpus_size = len(corpus)

        if verbose:
            print(f"      Estimating diversity for corpus of {corpus_size} documents...")

        # For small corpora, compute directly
        if corpus_size <= max_sample_size:
            if verbose:
                print(f"      Corpus small enough for direct measurement (n={corpus_size})")
            try:
                diversity = self(corpus)
                if verbose:
                    print(f"      ✓ Direct measurement: {diversity:.4f}")
                return ScaledEstimationResult(
                    diversity=diversity,
                    std=0.0,
                    projected_uncertainty_95=(diversity, diversity),
                    method="direct",
                    corpus_size=corpus_size,
                )
            except Exception as e:
                if verbose:
                    print(f"      ✗ Direct measurement failed: {e}")
                return ScaledEstimationResult(
                    diversity=0.0,
                    method="error",
                    corpus_size=corpus_size,
                )

        # Generate sample sizes: base, base*2, base*4, ...
        sample_sizes = []
        size = base_sample_size
        while size <= min(max_sample_size, corpus_size):
            sample_sizes.append(size)
            size = int(size * growth_factor)

        if len(sample_sizes) < 2:
            sample_sizes = [base_sample_size, min(max_sample_size, corpus_size)]

        total_measurements = len(sample_sizes) * num_trials
        if verbose:
            print(f"      Sample sizes: {sample_sizes}")
            print(f"      Trials per size: {num_trials} ({total_measurements} total measurements)")

        # Measure diversity at each sample size with multiple trials
        measurements: dict[int, list[float]] = {size: [] for size in sample_sizes}
        measurement_count = 0

        for trial in range(num_trials):
            # Shuffle indices for this trial
            rng = np.random.default_rng(random_seed + trial)
            indices = rng.permutation(corpus_size)

            for size in sample_sizes:
                measurement_count += 1
                if verbose:
                    print(f"      [{measurement_count}/{total_measurements}] "
                          f"Trial {trial+1}, n={size}...", end=" ", flush=True)

                sample_indices = indices[:size]
                sample = [corpus[i] for i in sample_indices]

                try:
                    diversity = self(sample)
                    if diversity is not None and np.isfinite(diversity):
                        measurements[size].append(diversity)
                        if verbose:
                            print(f"✓ {diversity:.4f}")
                    else:
                        if verbose:
                            print(f"✗ invalid result")
                except Exception as e:
                    if verbose:
                        print(f"✗ {str(e)[:30]}")

                # Clean up
                del sample
                gc.collect()

        # Calculate mean and std for each sample size
        valid_sizes = []
        size_means = []
        size_stds = []

        for size in sample_sizes:
            if measurements[size]:
                valid_sizes.append(size)
                size_means.append(np.mean(measurements[size]))
                size_stds.append(np.std(measurements[size]) if len(measurements[size]) > 1 else 0.0)

        if verbose:
            print(f"      Summary by sample size:")
            for size, mean, std in zip(valid_sizes, size_means, size_stds):
                print(f"         n={size}: {mean:.4f} ± {std:.4f}")

        if len(valid_sizes) < 2:
            # Not enough data for extrapolation, return last measurement
            if verbose:
                print(f"      ⚠ Not enough valid measurements for curve fitting")
            if valid_sizes and size_means:
                return ScaledEstimationResult(
                    diversity=size_means[-1],
                    std=size_stds[-1] if size_stds else 0.0,
                    method="partial",
                    sample_sizes=valid_sizes,
                    sample_means=[float(m) for m in size_means],
                    sample_stds=[float(s) for s in size_stds],
                    corpus_size=corpus_size,
                )
            return ScaledEstimationResult(diversity=0.0, method="error", corpus_size=corpus_size)

        # Check if metric is normalized (intensive) - prefer asymptotic model
        # Normalized metrics converge rather than grow with sample size
        prefer_asymptotic = getattr(self.config, 'normalize', False)

        # Fit growth curve and extrapolate
        if verbose:
            print(f"      Fitting growth curves...")
        model_name, predict_func, fit_rmse, model_params = self._fit_growth_curve(
            valid_sizes, size_means, prefer_asymptotic=prefer_asymptotic
        )

        # Extrapolate to full corpus size
        estimated_diversity = predict_func(corpus_size)

        # Estimate projected uncertainty (heuristic, not a true statistical CI)
        # Based on sampling variance scaled by extrapolation distance
        avg_std = np.mean(size_stds) if size_stds else 0.0
        # Uncertainty grows with extrapolation distance (log scale)
        extrapolation_factor = np.log(corpus_size / valid_sizes[-1]) / np.log(2) + 1
        estimated_std = avg_std * extrapolation_factor

        if verbose:
            print(f"      Best model: {model_name} (RMSE={fit_rmse:.6f})")
            print(f"      Extrapolated to n={corpus_size}: {estimated_diversity:.4f} ± {estimated_std:.4f}")

        return ScaledEstimationResult(
            diversity=float(estimated_diversity),
            std=float(estimated_std),
            projected_uncertainty_95=(
                float(estimated_diversity - 1.96 * estimated_std),
                float(estimated_diversity + 1.96 * estimated_std),
            ),
            method="extrapolation",
            model=model_name,
            model_params=model_params,
            sample_sizes=valid_sizes,
            sample_means=[float(m) for m in size_means],
            sample_stds=[float(s) for s in size_stds],
            corpus_size=corpus_size,
            fit_rmse=float(fit_rmse),
        )

    @staticmethod
    def _fit_growth_curve(
        sizes: list[int],
        diversities: list[float],
        prefer_asymptotic: bool = False,
    ) -> tuple[str, Callable[[float], float], float, Optional[List[float]]]:
        """Fit a growth curve to diversity measurements.

        Tries multiple models and returns the best fit:
        - Logarithmic: y = a * log(n + b) + c (accumulating metrics)
        - Power law: y = a * n^b + c (Heaps' law, exponent typically 0.4-0.8)
        - Asymptotic: y = a * (1 - exp(-n/b)) + c (converging/normalized metrics)

        Args:
            sizes: Sample sizes.
            diversities: Mean diversity at each size.
            prefer_asymptotic: If True, prioritize asymptotic model (useful for
                normalized metrics that converge rather than grow).

        Returns:
            Tuple of (model_name, prediction_function, rmse, params).
        """
        sizes_arr = np.array(sizes, dtype=float)
        div_arr = np.array(diversities, dtype=float)

        # Define growth curve models
        def logarithmic(n, a, b, c):
            return a * np.log(n + b) + c

        def power_law(n, a, b, c):
            return a * np.power(n, b) + c

        def asymptotic(n, a, b, c):
            return a * (1 - np.exp(-n / b)) + c

        # Model definitions: (function, initial_params, bounds)
        # Power law exponent bounds relaxed to [0.01, 1.0] to accommodate
        # Heaps' Law range (typically 0.4-0.8) and rich-get-richer distributions
        models = {
            'logarithmic': (logarithmic, [0.1, 1.0, 0.5], ([0, 0.1, -10], [1, 1000, 10])),
            'power_law': (power_law, [0.5, 0.5, 0.3], ([0, 0.01, -10], [10, 1.0, 10])),
            'asymptotic': (asymptotic, [0.8, 500, 0.2], ([0, 10, -10], [10, 10000, 10])),
        }

        best_model = None
        best_rmse = float('inf')
        best_func = None
        best_params = None

        if SCIPY_AVAILABLE:
            for name, (func, p0, bounds) in models.items():
                try:
                    params, _ = optimize.curve_fit(
                        func, sizes_arr, div_arr,
                        p0=p0, bounds=bounds,
                        maxfev=5000
                    )

                    predicted = func(sizes_arr, *params)
                    rmse = float(np.sqrt(np.mean((div_arr - predicted) ** 2)))

                    # Apply preference for asymptotic model when requested
                    # (useful for normalized/intensive metrics that converge)
                    adjusted_rmse = rmse
                    if prefer_asymptotic and name == 'asymptotic':
                        adjusted_rmse *= 0.9  # 10% bonus for asymptotic

                    if adjusted_rmse < best_rmse:
                        best_rmse = rmse  # Store actual RMSE, not adjusted
                        best_model = name
                        best_params = params.tolist()
                        # Capture params in closure
                        best_func = lambda n, f=func, p=params: f(n, *p)

                except Exception:
                    continue

        # Fallback to simple linear extrapolation if scipy unavailable or all fits fail
        if best_func is None:
            slope = (div_arr[-1] - div_arr[0]) / (sizes_arr[-1] - sizes_arr[0])
            intercept = div_arr[0] - slope * sizes_arr[0]
            return 'linear', lambda n: slope * n + intercept, 0.0, [slope, intercept]

        return best_model, best_func, best_rmse, best_params


class SimilarityMetric(Metric):
    """Base class for similarity metrics."""

    @abstractmethod
    def __call__(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score.
        """
        ...


class TextDiversity(DiversityMetric):
    """Base class for text diversity metrics using Hill numbers.

    This implements the core diversity calculation using similarity-sensitive
    Hill numbers, based on the framework from ecological diversity studies.
    """

    def __call__(self, corpus: list[str]) -> float:
        """Compute diversity for a corpus."""
        return self.diversity(corpus)

    def diversity(self, corpus: list[str]) -> float:
        """Calculate diversity using similarity-sensitive Hill numbers.

        Args:
            corpus: List of text documents.

        Returns:
            Diversity score (effective number of species).
        """
        # Validate inputs
        if not all(isinstance(d, str) and d.strip() for d in corpus):
            if self.config.verbose:
                print(f"Warning: corpus contains invalid inputs, returning 0")
            return 0.0

        # Extract features and species
        features, species = self.extract_features(corpus)

        # If no features, diversity is 0
        if len(features) == 0:
            return 0.0

        # Calculate similarity matrix Z
        Z = self.calculate_similarities(features)

        # Validate similarity matrix
        if np.any(np.isnan(Z)) or np.any(np.isinf(Z)):
            if self.config.verbose:
                print(f"Warning: similarity matrix contains invalid values, returning 1.0")
            return 1.0  # Minimum diversity (all identical)

        # Check if Z is all zeros (no similarity signal)
        if np.allclose(Z, 0.0):
            if self.config.verbose:
                print(f"Warning: similarity matrix is all zeros, returning {float(len(features))}")
            return float(len(features))  # Maximum diversity (all distinct)

        # Calculate abundance vector p
        p = self.calculate_abundance(species)

        # Calculate diversity
        D = self._calc_diversity(p, Z, self.config.q)

        # Validate result
        if np.isnan(D) or np.isinf(D):
            if self.config.verbose:
                print(f"Warning: diversity calculation returned invalid value, returning {float(len(features))}")
            return float(len(features))  # Maximum diversity as fallback

        # Optionally normalize by number of species
        if self.config.normalize:
            D /= len(p)

        return float(D)

    def similarity(self, corpus: list[str]) -> float:
        """Calculate average similarity across corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Average similarity score.
        """
        if not all(isinstance(d, str) and d.strip() for d in corpus):
            if self.config.verbose:
                print(f"Warning: corpus contains invalid inputs, returning 0")
            return 0.0

        # Extract features
        features, _ = self.extract_features(corpus)

        if len(features) == 0:
            return 0.0

        # Get similarity matrix
        Z = self.calculate_similarities(features)

        # Calculate mean similarity
        return float(Z.sum() / (len(Z) ** 2))

    def rank_similarity(
        self,
        query: list[str],
        corpus: list[str],
        top_n: int = 1,
    ) -> tuple[list[str], npt.NDArray[np.float64]]:
        """Rank corpus documents by similarity to query.

        Args:
            query: Query document(s) (typically single element list).
            corpus: Corpus documents to rank.
            top_n: Number of top results to return (-1 for all).

        Returns:
            Tuple of (ranked_documents, scores).
        """
        if top_n == -1:
            top_n = len(corpus)

        # Extract features for query and corpus
        all_feats, all_docs = self.extract_features(query + corpus)
        q_feats, q_corpus = all_feats[0], all_docs[0]
        c_feats, c_corpus = all_feats[1:], all_docs[1:]

        # Handle empty features
        if len(q_feats) == 0 or len(c_feats) == 0:
            return [], np.array([])

        # Calculate similarity vector
        z = self.calculate_similarity_vector(q_feats, c_feats)

        # Rank by similarity (descending)
        rank_idx = np.argsort(z)[::-1]

        ranking = np.array(c_corpus)[rank_idx].tolist()
        scores = z[rank_idx]

        return ranking[:top_n], scores[:top_n]

    @abstractmethod
    def extract_features(
        self, corpus: list[str]
    ) -> tuple[npt.NDArray[Any], list[Any]]:
        """Extract features and species from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (features, species).
        """
        ...

    @abstractmethod
    def calculate_similarities(
        self, features: npt.NDArray[Any]
    ) -> npt.NDArray[np.float64]:
        """Calculate pairwise similarities between features.

        Args:
            features: Feature matrix.

        Returns:
            Similarity matrix Z (n x n).
        """
        ...

    @abstractmethod
    def calculate_abundance(self, species: list[Any]) -> npt.NDArray[np.float64]:
        """Calculate abundance distribution over species.

        Args:
            species: List of species identifiers.

        Returns:
            Abundance vector p (sums to 1).
        """
        ...

    def calculate_similarity_vector(
        self,
        query_features: Any,
        corpus_features: Any,
    ) -> npt.NDArray[np.float64]:
        """Calculate similarity between query and corpus features.

        Args:
            query_features: Query feature(s).
            corpus_features: Corpus features.

        Returns:
            Similarity vector.
        """
        raise NotImplementedError(
            "Ranking requires document-level metrics. "
            "Override this method to support ranking."
        )

    @staticmethod
    def _calc_diversity(
        p: npt.NDArray[np.float64],
        Z: npt.NDArray[np.float64],
        q: float = 1.0,
    ) -> float:
        """Calculate diversity using Hill number formula.

        Args:
            p: Abundance distribution (n-vector, sums to 1).
            Z: Similarity matrix (n x n).
            q: Diversity order parameter.

        Returns:
            Diversity value.
        """
        Zp = Z @ p

        if q == 1:
            # Shannon diversity case: D = 1 / prod(Zp^p)
            # = exp(-sum(p * log(Zp)))
            # Use log-space calculation to avoid numerical underflow
            with np.errstate(divide="ignore", invalid="ignore"):
                # Add small epsilon to avoid log(0)
                log_D = -np.sum(p * np.log(Zp + 1e-10))
                D = np.exp(log_D)
        elif q == float("inf"):
            # Simpson diversity case: D = 1 / max(Zp)
            D = 1.0 / Zp.max()
        else:
            # General case: D = (sum(p * Zp^(q-1)))^(1/(1-q))
            D = np.power((p * np.power(Zp, q - 1)).sum(), 1 / (1 - q))

        return float(D)
