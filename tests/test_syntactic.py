"""Tests for syntactic diversity metrics."""

import pytest

from linguistic_diversity.diversities.syntactic import (
    ConstituencyParse,
    DependencyParse,
)


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "A fast red dog runs quickly",
        "The cat sleeps",
    ]


class TestDependencyParse:
    """Tests for DependencyParse metric."""

    @pytest.mark.slow
    def test_basic_diversity_ldp(self, sample_corpus):
        """Test basic dependency parse diversity with LDP."""
        metric = DependencyParse({"similarity_type": "ldp"})
        diversity = metric(sample_corpus)

        # Should return a positive diversity score
        assert diversity > 0
        assert diversity < 10  # Reasonable range

    @pytest.mark.slow
    def test_basic_diversity_feather(self, sample_corpus):
        """Test basic dependency parse diversity with Feather."""
        metric = DependencyParse({"similarity_type": "feather"})
        diversity = metric(sample_corpus)

        assert diversity > 0
        assert diversity < 10

    @pytest.mark.slow
    def test_tree_edit_distance(self):
        """Test tree edit distance similarity."""
        # Small corpus for edit distance (slow)
        corpus = ["The cat sat", "A dog ran"]

        metric = DependencyParse({"similarity_type": "tree_edit_distance"})
        diversity = metric(corpus)

        assert diversity > 0

    def test_config_override(self):
        """Test configuration override."""
        config = {
            "similarity_type": "ldp",
            "split_sentences": True,
        }
        metric = DependencyParse(config)

        assert metric.config.similarity_type == "ldp"
        assert metric.config.split_sentences is True

    @pytest.mark.slow
    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        metric = DependencyParse()
        diversity = metric([])
        assert diversity == 0.0


class TestConstituencyParse:
    """Tests for ConstituencyParse metric."""

    @pytest.mark.slow
    def test_basic_diversity(self, sample_corpus):
        """Test basic constituency parse diversity."""
        metric = ConstituencyParse({"similarity_type": "ldp"})
        diversity = metric(sample_corpus)

        # Should return a positive diversity score
        assert diversity > 0
        assert diversity < 10

    @pytest.mark.slow
    def test_similarity(self, sample_corpus):
        """Test similarity calculation."""
        metric = ConstituencyParse({"similarity_type": "feather"})
        similarity = metric.similarity(sample_corpus)

        # Similarity should be between 0 and 1
        assert 0 <= similarity <= 1
