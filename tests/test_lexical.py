"""Tests for the lexical diversity baselines."""

from linguistic_diversity import DistinctN, SelfBLEU, TypeTokenRatio


class TestTypeTokenRatio:
    def test_all_unique_scores_one(self):
        assert TypeTokenRatio()(["alpha beta", "gamma delta"]) == 1.0

    def test_all_identical_scores_low(self):
        assert TypeTokenRatio()(["word word", "word word"]) == 0.25

    def test_empty_corpus(self):
        assert TypeTokenRatio()([]) == 0.0

    def test_punctuation_is_not_a_token(self):
        assert TypeTokenRatio()(["hello, world!"]) == 1.0


class TestDistinctN:
    def test_unigrams_match_type_token_ratio(self):
        corpus = ["the cat sat", "the dog ran"]
        assert DistinctN({"n": 1})(corpus) == TypeTokenRatio()(corpus)

    def test_bigrams_detect_repeated_phrases(self):
        # "the cat" appears in both, so 3 of 4 bigrams are unique
        assert DistinctN({"n": 2})(["the cat sat", "the cat ran"]) == 0.75

    def test_n_larger_than_document(self):
        assert DistinctN({"n": 5})(["short text"]) == 0.0


class TestSelfBLEU:
    def test_identical_documents_score_high(self):
        """Self-BLEU measures overlap, so identical text approaches 1.0."""
        score = SelfBLEU()(["the cat sat down", "the cat sat down"])
        assert score > 0.9

    def test_disjoint_documents_score_low(self):
        """No shared n-grams means near-zero overlap."""
        score = SelfBLEU()(["alpha beta gamma delta", "one two three four"])
        assert score < 0.1

    def test_single_document_returns_zero(self):
        assert SelfBLEU()(["only one document here"]) == 0.0

    def test_direction_is_inverted(self):
        """Lower self-BLEU means MORE diverse -- the opposite of every other metric."""
        varied = SelfBLEU()(["alpha beta gamma", "one two three", "red green blue"])
        repetitive = SelfBLEU()(["the cat sat", "the cat sat", "the cat ran"])
        assert varied < repetitive
