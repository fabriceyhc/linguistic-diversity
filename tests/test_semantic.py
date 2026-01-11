"""Tests for semantic diversity metrics."""

import pytest

from linguistic_diversity.diversities.semantic import DocumentSemantics, TokenSemantics


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "A fast auburn fox leaps above a sleepy canine",
        "The cat sits on the mat",
    ]


class TestTokenSemantics:
    """Tests for TokenSemantics metric."""

    @pytest.mark.slow
    def test_basic_diversity(self, sample_corpus):
        """Test basic token semantic diversity."""
        metric = TokenSemantics({"use_cuda": False, "model_name": "bert-base-uncased"})
        diversity = metric(sample_corpus)

        # Should return a positive diversity score
        assert diversity > 0
        # Should be reasonable (not infinity or NaN)
        assert diversity < 1000

    def test_config_override(self):
        """Test configuration override."""
        config = {
            "model_name": "bert-base-uncased",
            "batch_size": 8,
            "remove_stopwords": True,
            "use_cuda": False,
        }
        metric = TokenSemantics(config)

        assert metric.config.model_name == "bert-base-uncased"
        assert metric.config.batch_size == 8
        assert metric.config.remove_stopwords is True

    @pytest.mark.slow
    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        metric = TokenSemantics({"use_cuda": False})
        diversity = metric([])
        assert diversity == 0.0


class TestDocumentSemantics:
    """Tests for DocumentSemantics metric."""

    @pytest.mark.slow
    def test_basic_diversity(self, sample_corpus):
        """Test basic document semantic diversity."""
        metric = DocumentSemantics({
            "use_cuda": False,
            "model_name": "all-MiniLM-L6-v2"
        })
        diversity = metric(sample_corpus)

        # Should return a positive diversity score
        assert diversity > 0
        # First two docs are paraphrases, so diversity should be closer to 2 than 3
        assert 1.5 < diversity < 3.0

    @pytest.mark.slow
    def test_ranking(self, sample_corpus):
        """Test document ranking."""
        metric = DocumentSemantics({
            "use_cuda": False,
            "model_name": "all-MiniLM-L6-v2"
        })
        query = ["A fox jumping"]

        ranking, scores = metric.rank_similarity(query, sample_corpus, top_n=2)

        # Should return 2 results
        assert len(ranking) == 2
        assert len(scores) == 2

        # Scores should be in descending order
        assert scores[0] >= scores[1]

        # First result should be fox-related
        assert "fox" in ranking[0].lower()

    @pytest.mark.slow
    def test_similarity(self, sample_corpus):
        """Test similarity calculation."""
        metric = DocumentSemantics({
            "use_cuda": False,
            "model_name": "all-MiniLM-L6-v2"
        })

        # High similarity corpus (paraphrases)
        high_sim = sample_corpus[:2]
        # Mixed corpus
        mixed_sim = sample_corpus

        sim_high = metric.similarity(high_sim)
        sim_mixed = metric.similarity(mixed_sim)

        # Paraphrases should have higher average similarity
        assert sim_high > sim_mixed
