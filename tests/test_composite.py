"""Tests for the composite diversity metric.

The weighting strategies are pure arithmetic over the module's stored effect-size
table, so they are tested without constructing the underlying metrics. The tests
that do need real metrics are marked slow.
"""

import pytest

from linguistic_diversity.composite import CompositeDiversity


@pytest.fixture
def corpus():
    return [
        "a violent tempest wrecked our village",
        "she went for a morning run",
        "the program failed to run correctly",
    ]


class TestWeightingStrategies:
    """Each strategy derives weights differently from the same evidence table."""

    def test_equal_strategy_weights_every_metric_the_same(self):
        weights = CompositeDiversity(strategy="equal").get_weights()

        assert len(set(weights.values())) == 1, weights

    def test_custom_strategy_uses_the_supplied_weights(self):
        custom = {"doc_semantic": 0.7, "token_semantic": 0.3}
        weights = CompositeDiversity(strategy="custom", custom_weights=custom).get_weights()

        assert weights == custom

    def test_effect_size_strategy_follows_cohens_d(self):
        """Document semantics has the largest d in the table, so it should lead."""
        weights = CompositeDiversity(strategy="effect_size").get_weights()

        assert weights
        assert max(weights, key=weights.get) == "doc_semantic"

    def test_effect_size_threshold_excludes_weak_metrics(self):
        permissive = CompositeDiversity(strategy="effect_size", min_effect_size=0.0)
        strict = CompositeDiversity(strategy="effect_size", min_effect_size=0.5)

        assert len(strict.get_weights()) < len(permissive.get_weights())

    def test_significance_strategy_excludes_insignificant_metrics(self):
        permissive = CompositeDiversity(strategy="significance", significance_threshold=0.5)
        strict = CompositeDiversity(strategy="significance", significance_threshold=0.001)

        assert len(strict.get_weights()) < len(permissive.get_weights())

    def test_hybrid_strategy_produces_weights(self):
        weights = CompositeDiversity(strategy="hybrid").get_weights()

        assert weights
        assert all(w > 0 for w in weights.values())

    @pytest.mark.parametrize("strategy", ["equal", "effect_size", "significance", "hybrid"])
    def test_weights_are_positive(self, strategy):
        weights = CompositeDiversity(strategy=strategy).get_weights()

        assert all(w > 0 for w in weights.values()), weights

    def test_get_weights_returns_a_copy(self):
        """Callers must not be able to mutate the metric's internal weights."""
        metric = CompositeDiversity(strategy="equal")
        weights = metric.get_weights()
        weights["doc_semantic"] = 999.0

        assert metric.get_weights().get("doc_semantic") != 999.0


class TestSelectionFailsLoudly:
    """A configuration that selects nothing must not silently score zero."""

    def test_impossible_threshold_raises_on_use(self, corpus):
        metric = CompositeDiversity(strategy="effect_size", min_effect_size=99.0)

        with pytest.raises(ValueError, match="No metrics selected"):
            metric(corpus)

    def test_empty_custom_weights_raise_on_use(self, corpus):
        metric = CompositeDiversity(strategy="custom", custom_weights={})

        with pytest.raises(ValueError, match="No metrics selected"):
            metric(corpus)


class TestEvidenceTable:
    """The published effect sizes back the default weighting."""

    def test_every_entry_has_d_p_and_direction(self):
        for name, entry in CompositeDiversity.DEMENTIA_EVIDENCE.items():
            assert len(entry) == 3, name
            d, p, direction = entry
            assert isinstance(d, float) and isinstance(p, float)
            assert isinstance(direction, str) and direction

    def test_p_values_are_probabilities(self):
        for name, (_, p, _) in CompositeDiversity.DEMENTIA_EVIDENCE.items():
            assert 0.0 <= p <= 1.0, f"{name} has p={p}"


@pytest.mark.slow
class TestCompositeScoring:
    """End-to-end scoring, which loads the underlying metrics."""

    def test_returns_a_positive_score(self, corpus):
        score = CompositeDiversity(strategy="custom", custom_weights={"doc_semantic": 1.0})(corpus)

        assert score > 0

    def test_reports_a_score_per_selected_metric(self, corpus):
        metric = CompositeDiversity(
            strategy="custom", custom_weights={"doc_semantic": 0.5, "token_semantic": 0.5}
        )
        scores = metric.get_metric_scores(corpus)

        assert set(scores) == {"doc_semantic", "token_semantic"}
        assert all(v > 0 for v in scores.values())

    def test_single_metric_composite_tracks_that_metric(self, corpus):
        """With one metric at full weight the composite should follow it."""
        from linguistic_diversity import DocumentSemantics

        composite = CompositeDiversity(
            strategy="custom", custom_weights={"doc_semantic": 1.0}, normalize_scores=False
        )
        assert composite(corpus) == pytest.approx(DocumentSemantics()(corpus), rel=0.01)
