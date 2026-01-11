"""Tests for morphological diversity metrics."""

import pytest

from linguistic_diversity.diversities.morphological import PartOfSpeechSequence


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "A fast red dog runs quickly today",
        "Birds fly high in the sky",
    ]


class TestPartOfSpeechSequence:
    """Tests for PartOfSpeechSequence metric."""

    @pytest.mark.slow
    def test_basic_diversity(self, sample_corpus):
        """Test basic POS sequence diversity."""
        metric = PartOfSpeechSequence()
        diversity = metric(sample_corpus)

        # Should return a positive diversity score
        assert diversity > 0
        # With 3 documents, diversity should be <= 3
        assert diversity <= 3.5

    @pytest.mark.slow
    def test_identical_pos_patterns(self):
        """Test corpus with identical POS patterns."""
        corpus = [
            "The big dog",
            "A small cat",
            "The quick fox",
        ]

        metric = PartOfSpeechSequence()
        diversity = metric(corpus)

        # All have same POS pattern (DET ADJ NOUN), so diversity should be low
        assert diversity < 1.5

    @pytest.mark.slow
    def test_diverse_pos_patterns(self):
        """Test corpus with diverse POS patterns."""
        corpus = [
            "Run quickly",  # VERB ADV
            "The cat sleeps",  # DET NOUN VERB
            "Very interesting indeed",  # ADV ADJ ADV
        ]

        metric = PartOfSpeechSequence()
        diversity = metric(corpus)

        # Different POS patterns should give higher diversity
        assert diversity > 1.5

    def test_config_override(self):
        """Test configuration override."""
        config = {
            "pad_to_max_len": True,
            "split_sentences": True,
        }
        metric = PartOfSpeechSequence(config)

        assert metric.config.pad_to_max_len is True
        assert metric.config.split_sentences is True

    @pytest.mark.slow
    def test_ranking(self, sample_corpus):
        """Test POS sequence ranking."""
        metric = PartOfSpeechSequence()
        query = ["The small bird flies"]

        ranking, scores = metric.rank_similarity(query, sample_corpus, top_n=2)

        # Should return 2 results
        assert len(ranking) == 2
        assert len(scores) == 2

        # Scores should be in descending order
        assert scores[0] >= scores[1]

    @pytest.mark.slow
    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        metric = PartOfSpeechSequence()
        diversity = metric([])
        assert diversity == 0.0

    @pytest.mark.slow
    def test_similarity(self, sample_corpus):
        """Test similarity calculation."""
        metric = PartOfSpeechSequence()
        similarity = metric.similarity(sample_corpus)

        # Similarity should be between 0 and 1
        assert 0 <= similarity <= 1
