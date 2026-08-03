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


class TestAlignmentSimilarityBounds:
    """The similarity matrix must be a similarity matrix.

    Biopython's PairwiseAligner defaults to open/extend gap scores of -1, so the
    raw alignment score of two sequences of different length is negative. Those
    negatives were being used directly as Z in a similarity-sensitive Hill
    number, which requires Z in [0, 1]. Scores were also divided by the longest
    sequence in the *corpus*, so every pairwise similarity moved when an
    unrelated long document was added.
    """

    def test_similarities_stay_in_unit_interval(self):
        """Sequences of very unequal length must not produce negative similarity."""
        import numpy as np

        metric = PartOfSpeechSequence({"verbose": False})
        corpus = [
            "When the rain stopped, the children played happily outside today.",
            "Birds fly.",
            "The tall boy kicked the ball.",
        ]
        features, _ = metric.extract_features(corpus)
        Z = metric.calculate_similarities(features)

        assert Z.min() >= 0.0, f"negative similarity: {Z.min()}"
        assert Z.max() <= 1.0, f"similarity above 1: {Z.max()}"
        assert np.allclose(Z, Z.T), "similarity matrix is not symmetric"
        assert np.allclose(np.diag(Z), 1.0), "self-similarity is not 1.0"

    def test_identical_pos_sequences_collapse(self):
        """One POS template, several lexicalisations, is one species."""
        metric = PartOfSpeechSequence({"verbose": False})
        one_template = [
            "The tall boy kicked the ball.",
            "A red car struck the fence.",
            "The old woman watered the garden.",
            "That young girl painted the wall.",
        ]
        assert metric(one_template) == pytest.approx(1.0, abs=0.01)

    def test_does_not_saturate_at_species_count(self):
        """Many distinct documents must not each count as a whole species.

        Corpus-wide normalisation drove every off-diagonal similarity to ~0 as
        the corpus grew, so diversity converged on the document count and the
        metric stopped measuring anything.
        """
        metric = PartOfSpeechSequence({"verbose": False})
        corpus = [
            f"The {adj} {noun} {verb} the {obj}."
            for adj in ("tall", "red", "old", "young")
            for noun in ("boy", "car", "woman", "girl")
            for verb, obj in (("kicked", "ball"), ("struck", "fence"))
        ]
        diversity = metric(corpus)
        assert (
            diversity < len(corpus) / 2
        ), f"diversity {diversity} approaches the species count {len(corpus)}"

    def test_similarity_is_pair_local(self):
        """Adding an unrelated long document must not change an existing pair."""
        import numpy as np

        metric = PartOfSpeechSequence({"verbose": False})
        pair = ["The tall boy kicked the ball.", "Birds fly."]
        with_extra = pair + [
            "When the rain stopped yesterday, the tired children finally played outside."
        ]

        z_pair = metric.calculate_similarities(metric.extract_features(pair)[0])
        z_extra = metric.calculate_similarities(metric.extract_features(with_extra)[0])

        assert np.isclose(z_pair[0, 1], z_extra[0, 1]), (
            f"pair similarity changed from {z_pair[0, 1]} to {z_extra[0, 1]} "
            "when an unrelated document joined the corpus"
        )


class TestTagsetGranularity:
    """The tagset decides whether this metric is morphological at all.

    UPOS collapses walks/walked/walking onto VERB, so English inflectional
    morphology is invisible to it: a corpus varying only in tense scores exactly
    1.0. The fine-grained PTB tagset (VBZ/VBD/VBG) is what makes the "species"
    morphological rather than coarse syntactic categories.
    """

    INFLECTION_ONLY = [
        "The workers walk to the site.",
        "The workers walked to the site.",
        "The workers walking to the site.",
        "The workers walks to the site.",
    ]
    ONE_FRAME = [
        "The tall boy kicked the ball.",
        "A red car struck the fence.",
        "The old woman watered the garden.",
        "That young girl painted the wall.",
    ]

    def test_fine_tags_see_inflection(self):
        metric = PartOfSpeechSequence({"tagset": "fine", "verbose": False})
        assert metric(self.INFLECTION_ONLY) > 1.0, (
            "a corpus differing only in verb inflection scored 1.0: the tagset "
            "cannot see morphology"
        )

    def test_upos_is_blind_to_inflection(self):
        """Documents the limitation the default exists to avoid."""
        metric = PartOfSpeechSequence({"tagset": "upos", "verbose": False})
        assert metric(self.INFLECTION_ONLY) == pytest.approx(1.0, abs=1e-6)

    def test_fine_tags_still_collapse_one_frame(self):
        """The converse must survive: same structure, different words, is one species."""
        metric = PartOfSpeechSequence({"tagset": "fine", "verbose": False})
        assert metric(self.ONE_FRAME) == pytest.approx(1.0, abs=0.01)

    def test_default_is_fine(self):
        assert PartOfSpeechSequence._default_config()["tagset"] == "fine"

    def test_unknown_tagset_is_rejected(self):
        metric = PartOfSpeechSequence({"tagset": "penn", "verbose": False})
        with pytest.raises(ValueError, match="Unknown tagset"):
            metric(["A sentence."])

    def test_alignment_alphabet_covers_the_fine_tagset(self):
        """PTB has ~50 tags; chr(65 + i) left the letters at 27."""
        from linguistic_diversity.utils import tag_to_alpha

        many = [[f"TAG{i}" for i in range(60)]]
        mapped = tag_to_alpha(many)[0]
        assert len(set(mapped)) == 60
        assert all(c.isalnum() for c in mapped), f"non-alphanumeric symbols: {set(mapped)}"
