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
        metric = DocumentSemantics({"use_cuda": False, "model_name": "all-MiniLM-L6-v2"})
        diversity = metric(sample_corpus)

        # Should return a positive diversity score
        assert diversity > 0
        # First two docs are paraphrases, so diversity should be closer to 2 than 3
        assert 1.5 < diversity < 3.0

    @pytest.mark.slow
    def test_ranking(self, sample_corpus):
        """Test document ranking."""
        metric = DocumentSemantics({"use_cuda": False, "model_name": "all-MiniLM-L6-v2"})
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
        metric = DocumentSemantics({"use_cuda": False, "model_name": "all-MiniLM-L6-v2"})

        # High similarity corpus (paraphrases)
        high_sim = sample_corpus[:2]
        # Mixed corpus
        mixed_sim = sample_corpus

        sim_high = metric.similarity(high_sim)
        sim_mixed = metric.similarity(mixed_sim)

        # Paraphrases should have higher average similarity
        assert sim_high > sim_mixed


class TestSemanticDefaults:
    """Pin the default similarity configuration for the semantic metrics.

    These defaults were selected by sweeping the distance/scaling options and
    validating on 68 corpora with known ground-truth diversity plus 600
    human-scored McDiv sets (see benchmarks/embedder_selection/). A silent change
    here would move every documented score, so assert them explicitly.
    """

    def test_token_semantics_uses_squared_cosine(self):
        """TokenSemantics defaults to cosine + power_reg, without mean_adj.

        mean_adj was on until v1.0.3. Re-ablated on all 600 human-scored McDiv
        sets, turning it off improved agreement (rho +0.348 -> +0.388) *and*
        restored replication invariance, which it had been breaking: subtracting
        the off-diagonal mean left two identical occurrences at 0.78 while the
        diagonal stayed at 1.0. Both criteria pointed the same way, so it is off.
        See benchmarks/embedder_selection/ablate_similarity.py.
        """
        import faiss

        config = TokenSemantics._default_config()
        assert config["distance_fn"] == faiss.METRIC_INNER_PRODUCT
        assert config["scale_dist"] is None
        assert config["power_reg"] is True
        assert config["mean_adj"] is False

    def test_document_semantics_uses_plain_cosine(self):
        """DocumentSemantics defaults to cosine with no squaring or adjustment."""
        import faiss

        config = DocumentSemantics._default_config()
        assert config["distance_fn"] == faiss.METRIC_INNER_PRODUCT
        assert config["scale_dist"] is None
        assert config["mean_adj"] is False

    def test_document_semantics_default_encoder_is_calibration_optimised(self):
        """The encoder moves every score, so pin it rather than let it drift.

        The two halves of benchmarks/embedder_selection/ disagree about which
        encoder is best, and the default follows the calibration half: mpnet
        reports 0.97 of true k against bge-large's 0.58. bge-large wins on human
        agreement (+0.779 vs +0.581) and is the documented recommendation for
        *comparing* corpora, but it is not the default -- a score read as a
        quantity should mean what it says, and mpnet is a third of the size.

        Changing this is a legitimate decision; it just has to be a deliberate
        one, because it moves every documented DocumentSemantics number.
        """
        config = DocumentSemantics._default_config()
        assert config["model_name"] == "all-mpnet-base-v2"


class TestSimilarityRangeAndInvariance:
    """Z must be a similarity matrix, and identical items must score 1.0.

    Cosine is defined on [-1, 1] while a similarity-sensitive Hill number needs
    [0, 1]; with all-mpnet-base-v2 about 1.35% of McDiv response pairs land
    below zero. Separately, mean_adj subtracted the off-diagonal mean from every
    off-diagonal entry, so an occurrence and its exact replica scored 0.78 while
    the diagonal stayed 1.0, and diversity was not invariant to replication.
    """

    def test_cosine_similarities_are_clamped_to_unit_interval(self):
        import numpy as np

        from linguistic_diversity import DocumentSemantics

        metric = DocumentSemantics({"verbose": False})
        corpus = [
            "The tall boy kicked the ball.",
            "Quantum chromodynamics predicts asymptotic freedom.",
            "She believes that the plan is sound.",
            "There was a crack in the ceiling.",
        ]
        features, _ = metric.extract_features(corpus)
        Z = metric.calculate_similarities(features)

        assert Z.min() >= 0.0, f"negative similarity {Z.min()} reached the Hill number"
        assert Z.max() <= 1.0, f"similarity above 1: {Z.max()}"
        assert np.allclose(np.diag(Z), 1.0)

    def test_token_semantics_is_replication_invariant(self):
        """Duplicating a corpus leaves relative abundance, so diversity, unchanged."""
        from linguistic_diversity import TokenSemantics

        metric = TokenSemantics({"verbose": False})
        corpus = [
            "The tall boy kicked the ball.",
            "When the rain stopped, the children played.",
            "There was a crack in the ceiling.",
        ]
        once, thrice = float(metric(corpus)), float(metric(corpus * 3))

        assert (
            abs(once - thrice) < 1e-4
        ), f"diversity moved from {once} to {thrice} when the corpus was tripled"

    def test_identical_tokens_score_one(self):
        """An occurrence and its exact replica are the same species."""
        import numpy as np

        from linguistic_diversity import TokenSemantics

        metric = TokenSemantics({"verbose": False})
        corpus = ["The tall boy kicked the ball.", "Birds fly."]
        features, species = metric.extract_features(corpus * 2)
        Z = metric.calculate_similarities(features)
        half = len(species) // 2

        replicas = np.array([Z[i, i + half] for i in range(half)])
        assert np.allclose(replicas, 1.0, atol=1e-4), (
            f"identical token occurrences scored {replicas.min():.4f}..{replicas.max():.4f}, "
            "not 1.0"
        )
