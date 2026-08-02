"""Tests for diversity-based subset selection.

These algorithms are pure numpy over a (n_items, n_metrics) embedding matrix, so
they are tested directly on constructed embeddings rather than on real corpora.
"""

import numpy as np
import pytest

from linguistic_diversity.selection import (
    BalancedCoverageSelector,
    FacilityLocationSelector,
    MaxMinDiversitySelector,
    SelectionResult,
    select_diverse_texts,
)

SELECTORS = [FacilityLocationSelector, MaxMinDiversitySelector, BalancedCoverageSelector]
SELECTOR_IDS = ["facility_location", "max_min", "balanced_coverage"]


@pytest.fixture
def clustered() -> np.ndarray:
    """Three tight clusters of four points each, in four metric dimensions.

    A selector that ignores diversity will take several points from one cluster;
    a working one spreads across clusters.
    """
    rng = np.random.default_rng(0)
    centres = np.array(
        [[5.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 5.0, 0.0]], dtype=np.float64
    )
    return np.vstack([c + rng.normal(0, 0.05, size=(4, 4)) for c in centres])


@pytest.mark.parametrize("selector_cls", SELECTORS, ids=SELECTOR_IDS)
class TestSelectorContract:
    """Behaviour every selector must share."""

    def test_selects_the_requested_count(self, selector_cls, clustered):
        result = selector_cls().select(clustered, n_select=3)

        assert result.n_selected == 3
        assert len(result.indices) == 3

    def test_indices_are_unique_and_in_range(self, selector_cls, clustered):
        result = selector_cls().select(clustered, n_select=5)

        assert len(set(result.indices.tolist())) == 5
        assert all(0 <= i < len(clustered) for i in result.indices)

    def test_is_deterministic_for_a_fixed_seed(self, selector_cls, clustered):
        a = selector_cls().select(clustered, n_select=4, seed=3)
        b = selector_cls().select(clustered, n_select=4, seed=3)

        assert a.indices.tolist() == b.indices.tolist()

    def test_selecting_everything_returns_every_item(self, selector_cls, clustered):
        result = selector_cls().select(clustered, n_select=len(clustered))

        assert sorted(result.indices.tolist()) == list(range(len(clustered)))

    def test_result_reports_its_method(self, selector_cls, clustered):
        result = selector_cls().select(clustered, n_select=2)

        assert isinstance(result, SelectionResult)
        assert isinstance(result.method, str) and result.method

    def test_spreads_across_clusters(self, selector_cls, clustered):
        """With 3 well-separated clusters, picking 3 should take one from each."""
        result = selector_cls().select(clustered, n_select=3)
        cluster_of = [int(i) // 4 for i in result.indices]

        assert len(set(cluster_of)) == 3, f"all from clusters {cluster_of}"


class TestFacilityLocation:
    """Options specific to the facility location objective."""

    @pytest.mark.parametrize("similarity_fn", ["cosine", "rbf", "linear"])
    def test_supports_each_similarity_function(self, similarity_fn, clustered):
        selector = FacilityLocationSelector(similarity_fn=similarity_fn)
        result = selector.select(clustered, n_select=3)

        assert result.n_selected == 3

    def test_metric_weights_change_the_selection(self, clustered):
        """Down-weighting a dimension should be able to change what is chosen."""
        uniform = FacilityLocationSelector().select(clustered, n_select=3)
        skewed = FacilityLocationSelector(metric_weights=np.array([10.0, 0.01, 0.01, 0.01])).select(
            clustered, n_select=3
        )

        assert uniform.n_selected == skewed.n_selected == 3


class TestSelectDiverseTexts:
    """The convenience wrapper over the selector classes."""

    @pytest.mark.parametrize("method", ["facility_location", "max_min", "balanced"])
    def test_dispatches_to_each_method(self, method, clustered):
        result = select_diverse_texts(clustered, n_select=3, method=method)

        assert result.n_selected == 3

    def test_rejects_an_unknown_method(self, clustered):
        with pytest.raises(ValueError, match="method|Unknown"):
            select_diverse_texts(clustered, n_select=2, method="nonexistent")

    def test_matches_the_class_it_wraps(self, clustered):
        direct = FacilityLocationSelector().select(clustered, n_select=3, seed=11)
        wrapped = select_diverse_texts(clustered, n_select=3, method="facility_location", seed=11)

        assert direct.indices.tolist() == wrapped.indices.tolist()
