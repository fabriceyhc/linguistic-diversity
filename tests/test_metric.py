"""Tests for core metric classes."""

import numpy as np
import pytest

from linguistic_diversity.metric import TextDiversity


class DummyDiversity(TextDiversity):
    """Dummy diversity metric for testing."""

    def extract_features(self, corpus):
        """Extract dummy features (just word counts)."""
        features = np.array([[len(doc.split())] for doc in corpus], dtype=np.float64)
        return features, corpus

    def calculate_similarities(self, features):
        """Calculate dummy similarities (identity matrix)."""
        n = len(features)
        return np.eye(n, dtype=np.float64)

    def calculate_abundance(self, species):
        """Uniform abundance."""
        n = len(species)
        return np.full(n, 1.0 / n, dtype=np.float64)


class TestTextDiversity:
    """Tests for TextDiversity base class."""

    def test_basic_diversity(self):
        """Test basic diversity calculation."""
        metric = DummyDiversity()
        corpus = ["hello world", "foo bar", "test example"]
        diversity = metric(corpus)

        # With identity similarity matrix and uniform abundance,
        # diversity should equal number of documents
        assert diversity == pytest.approx(3.0, rel=1e-6)

    def test_normalized_diversity(self):
        """Test normalized diversity."""
        metric = DummyDiversity({"normalize": True})
        corpus = ["hello world", "foo bar"]
        diversity = metric(corpus)

        # Normalized diversity should be 1.0
        assert diversity == pytest.approx(1.0, rel=1e-6)

    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        metric = DummyDiversity()
        diversity = metric([])
        assert diversity == 0.0

    def test_invalid_corpus(self):
        """Test handling of invalid corpus."""
        metric = DummyDiversity()
        diversity = metric(["", "  ", "valid"])
        # Should handle invalid inputs gracefully
        assert diversity >= 0.0

    def test_diversity_order_q(self):
        """Test different diversity orders."""
        corpus = ["doc1", "doc2", "doc3"]

        # q = 0 (richness)
        metric_q0 = DummyDiversity({"q": 0})
        div_q0 = metric_q0(corpus)

        # q = 1 (Shannon)
        metric_q1 = DummyDiversity({"q": 1})
        div_q1 = metric_q1(corpus)

        # q = 2 (Simpson)
        metric_q2 = DummyDiversity({"q": 2})
        div_q2 = metric_q2(corpus)

        # Hill number property: diversity decreases (or stays same) as q increases
        # Use approximate comparison for floating point precision
        assert div_q0 >= div_q1 - 1e-6  # Allow small numerical error
        assert div_q1 >= div_q2 - 1e-6  # Allow small numerical error

        # All should be close to 3 for uniform distribution
        assert abs(div_q0 - 3.0) < 0.1
        assert abs(div_q1 - 3.0) < 0.01
        assert abs(div_q2 - 3.0) < 0.01

    def test_similarity_method(self):
        """Test similarity calculation."""
        metric = DummyDiversity()
        corpus = ["doc1", "doc2"]
        similarity = metric.similarity(corpus)

        # With identity matrix, average similarity should be 0.5
        # (diagonal is 1, off-diagonal is 0)
        assert 0.0 <= similarity <= 1.0
