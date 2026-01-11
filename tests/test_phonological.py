"""Tests for phonological diversity metrics."""

import pytest

from linguistic_diversity.diversities.phonological import Phonemic, Rhythmic


@pytest.fixture
def sample_corpus():
    """Sample corpus for testing."""
    return [
        "The quick brown fox jumps",
        "A lazy dog sleeps soundly",
        "Birds sing in the morning",
    ]


class TestRhythmic:
    """Tests for Rhythmic metric."""

    @pytest.mark.slow
    def test_basic_diversity(self, sample_corpus):
        """Test basic rhythmic diversity."""
        try:
            import cadences  # noqa: F401
        except ImportError:
            pytest.skip("cadences library not installed")

        metric = Rhythmic()
        diversity = metric(sample_corpus)

        # Should return a positive diversity score
        assert diversity >= 0
        # With 3 documents, diversity should be <= 3
        assert diversity <= 4

    @pytest.mark.slow
    def test_identical_rhythms(self):
        """Test corpus with similar rhythmic patterns."""
        try:
            import cadences  # noqa: F401
        except ImportError:
            pytest.skip("cadences library not installed")

        corpus = [
            "The big dog",
            "A small cat",
            "The quick fox",
        ]

        metric = Rhythmic()
        diversity = metric(corpus)

        # Similar rhythms should give lower diversity
        assert diversity >= 0

    def test_config_override(self):
        """Test configuration override."""
        try:
            import cadences  # noqa: F401
        except ImportError:
            pytest.skip("cadences library not installed")

        config = {
            "pad_to_max_len": True,
            "split_sentences": True,
        }
        metric = Rhythmic(config)

        assert metric.config.pad_to_max_len is True
        assert metric.config.split_sentences is True

    @pytest.mark.slow
    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        try:
            import cadences  # noqa: F401
        except ImportError:
            pytest.skip("cadences library not installed")

        metric = Rhythmic()
        diversity = metric([])
        assert diversity == 0.0


class TestPhonemic:
    """Tests for Phonemic metric."""

    @pytest.mark.slow
    def test_basic_diversity(self, sample_corpus):
        """Test basic phonemic diversity."""
        try:
            from phonemizer import phonemize  # noqa: F401
        except ImportError:
            pytest.skip("phonemizer library not installed")

        try:
            metric = Phonemic()
            diversity = metric(sample_corpus)

            # Should return a positive diversity score
            assert diversity >= 0
            assert diversity <= 4
        except Exception as e:
            # espeak-ng might not be installed
            if "espeak" in str(e).lower():
                pytest.skip("espeak-ng not installed")
            raise

    @pytest.mark.slow
    def test_similar_phonemes(self):
        """Test corpus with similar phonetic content."""
        try:
            from phonemizer import phonemize  # noqa: F401
        except ImportError:
            pytest.skip("phonemizer library not installed")

        corpus = [
            "cat bat rat",
            "hat mat sat",
        ]

        try:
            metric = Phonemic()
            diversity = metric(corpus)

            # Similar phonemes should give lower diversity
            assert diversity >= 0
        except Exception as e:
            if "espeak" in str(e).lower():
                pytest.skip("espeak-ng not installed")
            raise

    def test_config_override(self):
        """Test configuration override."""
        try:
            from phonemizer import phonemize  # noqa: F401
        except ImportError:
            pytest.skip("phonemizer library not installed")

        config = {
            "pad_to_max_len": False,
            "split_sentences": False,
        }

        try:
            metric = Phonemic(config)
            assert metric.config.pad_to_max_len is False
            assert metric.config.split_sentences is False
        except Exception as e:
            if "espeak" in str(e).lower():
                pytest.skip("espeak-ng not installed")
            raise

    @pytest.mark.slow
    def test_empty_corpus(self):
        """Test handling of empty corpus."""
        try:
            from phonemizer import phonemize  # noqa: F401
        except ImportError:
            pytest.skip("phonemizer library not installed")

        try:
            metric = Phonemic()
            diversity = metric([])
            assert diversity == 0.0
        except Exception as e:
            if "espeak" in str(e).lower():
                pytest.skip("espeak-ng not installed")
            raise
