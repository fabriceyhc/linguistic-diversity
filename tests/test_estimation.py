"""Tests for scaled diversity estimation.

`estimate_diversity` is the documented answer for corpora too large for an exact
O(n^2) similarity matrix, so its sampling, curve fitting and extrapolation are
exercised here with a deterministic stand-in metric rather than a real encoder.
"""

import numpy as np
import pytest

# CountingDiversity treats every document as its own species, so its similarity
# matrix really is the identity. The library warns about that because it usually
# signals a misconfigured distance metric; here it is the intended setup.
pytestmark = pytest.mark.filterwarnings("ignore:.*off-diagonal similarities.*:RuntimeWarning")

from linguistic_diversity.metric import (
    DiversityMetric,
    ScaledEstimationResult,
    TextDiversity,
)


class CountingDiversity(TextDiversity):
    """Diversity equal to the number of distinct documents.

    Deterministic and instant, which lets the estimation machinery be tested
    without loading a model. Species are the documents themselves and the
    similarity matrix is the identity, so diversity is exactly the distinct count.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.calls: list[int] = []

    def extract_features(self, corpus):
        unique = sorted(set(corpus))
        self.calls.append(len(corpus))
        return np.arange(len(unique), dtype=np.float64).reshape(-1, 1), unique

    def calculate_similarities(self, features):
        return np.eye(len(features), dtype=np.float64)

    def calculate_abundance(self, species):
        n = len(species)
        return np.full(n, 1.0 / n, dtype=np.float64)


@pytest.fixture
def corpus_1000():
    return [f"document number {i}" for i in range(1000)]


class TestDirectMeasurement:
    """Corpora small enough to measure exactly must not be extrapolated."""

    def test_small_corpus_is_measured_directly(self):
        metric = CountingDiversity()
        result = metric.estimate_diversity(
            [f"doc {i}" for i in range(10)], max_sample_size=200, verbose=False
        )

        assert result.method == "direct"
        assert result.diversity == pytest.approx(10.0)
        assert result.std == 0.0

    def test_direct_result_has_no_fitted_model(self):
        metric = CountingDiversity()
        result = metric.estimate_diversity([f"doc {i}" for i in range(5)], verbose=False)

        assert result.model is None
        assert result.sample_sizes == []

    def test_boundary_uses_direct_measurement(self):
        """A corpus exactly at max_sample_size still qualifies as measurable."""
        metric = CountingDiversity()
        result = metric.estimate_diversity(
            [f"doc {i}" for i in range(50)], max_sample_size=50, verbose=False
        )
        assert result.method == "direct"


class TestExtrapolation:
    """Corpora above the sampling ceiling are fitted and extrapolated."""

    def test_large_corpus_is_extrapolated(self, corpus_1000):
        metric = CountingDiversity()
        result = metric.estimate_diversity(
            corpus_1000, base_sample_size=50, max_sample_size=200, verbose=False
        )

        assert result.method == "extrapolation"
        assert result.corpus_size == 1000
        assert result.model in {"logarithmic", "power_law", "asymptotic", "linear"}

    def test_never_measures_more_than_max_sample_size(self, corpus_1000):
        """The point is to avoid the O(n^2) matrix; sampling must respect the cap."""
        metric = CountingDiversity()
        metric.estimate_diversity(
            corpus_1000, base_sample_size=50, max_sample_size=200, verbose=False
        )

        assert metric.calls, "metric was never invoked"
        assert max(metric.calls) <= 200

    def test_samples_at_increasing_sizes(self, corpus_1000):
        metric = CountingDiversity()
        result = metric.estimate_diversity(
            corpus_1000, base_sample_size=50, max_sample_size=200, verbose=False
        )

        assert result.sample_sizes == sorted(result.sample_sizes)
        assert len(result.sample_sizes) >= 2

    def test_is_deterministic_for_a_fixed_seed(self, corpus_1000):
        a = CountingDiversity().estimate_diversity(corpus_1000, random_seed=7, verbose=False)
        b = CountingDiversity().estimate_diversity(corpus_1000, random_seed=7, verbose=False)

        assert a.diversity == pytest.approx(b.diversity)
        assert a.model == b.model

    def test_uncertainty_widens_with_extrapolation_distance(self):
        """Projecting further from the measured range should be less certain."""
        near = CountingDiversity().estimate_diversity(
            [f"doc {i}" for i in range(400)], max_sample_size=200, num_trials=3, verbose=False
        )
        far = CountingDiversity().estimate_diversity(
            [f"doc {i}" for i in range(20000)], max_sample_size=200, num_trials=3, verbose=False
        )

        assert far.std >= near.std

    def test_uncertainty_bounds_bracket_the_estimate(self, corpus_1000):
        result = CountingDiversity().estimate_diversity(corpus_1000, num_trials=3, verbose=False)
        low, high = result.projected_uncertainty_95

        assert low <= result.diversity <= high


class TestGrowthCurveFitting:
    """The curve fitter should recover the shape it is given."""

    @staticmethod
    def _fit(sizes, values, **kwargs):
        return DiversityMetric._fit_growth_curve(sizes, values, **kwargs)

    def test_recovers_a_logarithmic_shape(self):
        sizes = [50, 100, 200, 400, 800]
        values = [2.0 * np.log(n + 1.0) + 0.5 for n in sizes]

        name, predict, rmse, _ = self._fit(sizes, values)

        assert rmse < 0.1
        assert predict(1600) > predict(800)

    def test_fit_is_monotonic_for_growing_data(self):
        sizes = [50, 100, 200, 400]
        values = [1.0, 1.6, 2.1, 2.5]

        _, predict, _, _ = self._fit(sizes, values)

        predictions = [predict(n) for n in (500, 1000, 2000)]
        assert predictions == sorted(predictions)

    def test_asymptotic_preference_is_honoured(self):
        """Normalized metrics converge, so callers can ask for a saturating fit."""
        sizes = [50, 100, 200, 400]
        values = [0.5, 0.7, 0.8, 0.85]

        name, _, _, _ = self._fit(sizes, values, prefer_asymptotic=True)

        assert name in {"asymptotic", "logarithmic", "power_law"}

    def test_returns_named_model_and_params(self):
        name, predict, rmse, params = self._fit([50, 100, 200], [1.0, 1.5, 2.0])

        assert isinstance(name, str) and name
        assert callable(predict)
        assert rmse >= 0.0
        assert params is None or len(params) >= 2


class TestScaledEstimationResult:
    """The result object is part of the public surface."""

    def test_round_trips_through_to_dict(self):
        result = ScaledEstimationResult(
            diversity=3.5, std=0.25, method="extrapolation", model="power_law", corpus_size=900
        )
        data = result.to_dict()

        assert data["diversity"] == 3.5
        assert data["model"] == "power_law"
        assert data["corpus_size"] == 900
        assert isinstance(data["projected_uncertainty_95"], list)

    def test_plot_without_matplotlib_does_not_raise(self, monkeypatch):
        """Plotting is optional; its absence must not break a completed estimate."""
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name.startswith("matplotlib"):
                raise ImportError("matplotlib is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        ScaledEstimationResult(diversity=1.0).plot(show=False)
