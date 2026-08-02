"""Tests for the universal (combined) diversity metric.

The aggregation strategies are arithmetic over a dict of per-metric scores, so
they are tested directly. Tests that construct the underlying metrics are slow
and marked as such.
"""

import numpy as np
import pytest

from linguistic_diversity import UniversalLinguisticDiversity, get_preset_config
from linguistic_diversity.diversities.universal import (
    PRESET_CONFIGS,
    AggregationStrategy,
    UniversalDiversityConfig,
)


@pytest.fixture
def metric():
    """A universal metric with no sub-metrics constructed.

    Aggregation is independent of how the scores were produced, so disabling
    every branch keeps these tests fast while exercising the real code paths.
    """
    instance = UniversalLinguisticDiversity(
        {
            "use_semantic": False,
            "use_syntactic": False,
            "use_morphological": False,
            "use_phonological": False,
            "verbose": False,
        }
    )
    # Aggregation only reads the config, so re-enable the branches afterwards to
    # exercise the real weighting without paying for model loads.
    for branch in ("semantic", "syntactic", "morphological", "phonological"):
        setattr(instance.config, f"use_{branch}", True)
    return instance


@pytest.fixture
def scores():
    return {
        "token_semantics": 8.0,
        "document_semantics": 2.0,
        "dependency_parse": 4.0,
        "pos_sequence": 2.0,
        "rhythmic": 2.0,
        "phonemic": 2.0,
    }


class TestAggregationStrategies:
    """Each strategy combines the same scores differently."""

    def test_geometric_mean_lies_within_the_score_range(self, metric, scores):
        value = metric._weighted_geometric_mean(scores)

        assert min(scores.values()) <= value <= max(scores.values())

    def test_arithmetic_mean_lies_within_the_score_range(self, metric, scores):
        value = metric._weighted_arithmetic_mean(scores)

        assert min(scores.values()) <= value <= max(scores.values())

    def test_harmonic_mean_is_the_most_conservative(self, metric, scores):
        """Harmonic <= geometric <= arithmetic is the standard means inequality."""
        harmonic = metric._harmonic_mean(scores)
        geometric = metric._weighted_geometric_mean(scores)
        arithmetic = metric._weighted_arithmetic_mean(scores)

        assert harmonic <= geometric + 1e-9
        assert geometric <= arithmetic + 1e-9

    def test_identical_scores_aggregate_to_that_score(self, metric):
        uniform = dict.fromkeys(["token_semantics", "document_semantics", "dependency_parse"], 3.0)

        assert metric._weighted_geometric_mean(uniform) == pytest.approx(3.0)
        assert metric._weighted_arithmetic_mean(uniform) == pytest.approx(3.0)
        assert metric._harmonic_mean(uniform) == pytest.approx(3.0)

    def test_hierarchical_aggregation_returns_a_positive_score(self, metric, scores):
        assert metric._hierarchical_aggregation(scores) > 0

    def test_aggregation_is_monotonic(self, metric, scores):
        """Raising every component must not lower the combined score."""
        higher = {k: v * 2 for k, v in scores.items()}

        assert metric._hierarchical_aggregation(higher) >= metric._hierarchical_aggregation(scores)
        assert metric._weighted_geometric_mean(higher) >= metric._weighted_geometric_mean(scores)

    def test_empty_scores_do_not_raise(self, metric):
        """A configuration with nothing enabled must degrade, not crash."""
        for aggregate in (
            metric._weighted_geometric_mean,
            metric._weighted_arithmetic_mean,
            metric._harmonic_mean,
            metric._hierarchical_aggregation,
        ):
            assert aggregate({}) >= 0.0

    def test_harmonic_mean_excludes_zero_scores(self, metric):
        """Zeros are filtered rather than propagating, to avoid a division by zero.

        Worth knowing: a metric that legitimately scores 0 is dropped from the
        combination instead of pulling it down, so the conservative preset is not
        conservative about zeros specifically.
        """
        with_zero = {"token_semantics": 8.0, "document_semantics": 0.0}

        assert metric._harmonic_mean(with_zero) == pytest.approx(8.0)
        assert metric._harmonic_mean({"a": 0.0, "b": 0.0}) == 0.0


class TestPresets:
    """Preset configurations are part of the documented API."""

    @pytest.mark.parametrize("name", sorted(PRESET_CONFIGS))
    def test_each_preset_builds_a_valid_config(self, name):
        config = get_preset_config(name)

        assert isinstance(config, dict) and config
        UniversalDiversityConfig(**config)

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("does-not-exist")

    def test_returns_a_copy_not_the_shared_dict(self):
        """Mutating a returned preset must not corrupt it for later callers."""
        config = get_preset_config("balanced")
        config["semantic_weight"] = 999.0

        assert get_preset_config("balanced")["semantic_weight"] != 999.0

    def test_conservative_preset_uses_a_conservative_strategy(self):
        strategy = get_preset_config("conservative").get("strategy")

        assert strategy in {"harmonic", "minimum"}

    def test_semantic_focus_weights_semantics_highest(self):
        config = get_preset_config("semantic_focus")
        weights = {k: v for k, v in config.items() if k.endswith("_weight") and "_" in k}
        branch_weights = {
            k: v
            for k, v in weights.items()
            for branch in ("semantic", "syntactic", "morphological", "phonological")
            if k == f"{branch}_weight"
        }

        assert branch_weights
        assert max(branch_weights, key=branch_weights.get) == "semantic_weight"


class TestConfiguration:
    """Branch toggles and defaults."""

    def test_constituency_parse_is_opt_in(self):
        """It needs benepar, so it must not be enabled by default."""
        assert UniversalDiversityConfig().use_constituency_parse is False

    def test_disabling_a_branch_removes_its_metrics(self):
        metric = UniversalLinguisticDiversity(
            {
                "use_semantic": False,
                "use_syntactic": False,
                "use_morphological": False,
                "use_phonological": False,
                "verbose": False,
            }
        )

        assert metric._metrics == {}

    def test_strategy_enum_values_are_stable(self):
        """The strings appear in preset configs and user code."""
        assert AggregationStrategy.HIERARCHICAL.value == "hierarchical"
        assert AggregationStrategy.HARMONIC.value == "harmonic"


class TestEmbeddingNormalization:
    """Diversity embeddings feed the selection algorithms, so their range matters."""

    def test_normalized_embedding_is_bounded(self, metric):
        raw = np.array([0.0, 1.0, 5.0, 50.0, 500.0, 2.0, 9.0], dtype=np.float64)
        normalized = metric._normalize_embedding(raw)

        assert normalized.shape == raw.shape
        assert np.all(normalized >= 0.0) and np.all(normalized <= 1.0)

    def test_normalization_is_monotonic(self, metric):
        raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
        normalized = metric._normalize_embedding(raw)

        # Each dimension has its own scale, so compare one dimension across inputs
        larger = metric._normalize_embedding(raw * 2)
        assert np.all(larger >= normalized - 1e-9)

    def test_negative_values_are_clamped(self, metric):
        normalized = metric._normalize_embedding(
            np.array([-5.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
        )

        assert np.all(normalized >= 0.0)


@pytest.mark.slow
class TestUniversalScoring:
    """End-to-end scoring, which constructs the underlying metrics."""

    CORPUS = [
        "a violent tempest wrecked our village",
        "she went for a morning run",
        "the program failed to run correctly",
    ]

    def test_detailed_scores_expose_branches_and_metrics(self):
        detailed = UniversalLinguisticDiversity({"verbose": False}).get_detailed_scores(self.CORPUS)

        assert set(detailed) >= {"universal", "branches", "metrics"}
        assert detailed["universal"] > 0
        assert detailed["branches"]

    def test_call_matches_detailed_universal_score(self):
        metric = UniversalLinguisticDiversity({"verbose": False})

        assert metric(self.CORPUS) == pytest.approx(
            metric.get_detailed_scores(self.CORPUS)["universal"], rel=1e-6
        )

    def test_constituency_absent_from_default_breakdown(self):
        """It is opt-in, so the default run aggregates six of the seven metrics."""
        detailed = UniversalLinguisticDiversity({"verbose": False}).get_detailed_scores(self.CORPUS)

        assert "constituency_parse" not in detailed["metrics"]
