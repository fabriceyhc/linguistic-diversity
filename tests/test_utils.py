"""Tests for subword merging utilities."""

import numpy as np
import pytest

from linguistic_diversity.utils import detect_subword_scheme, merge_bpe


class TestDetectSubwordScheme:
    """Tests for tokenizer convention detection."""

    def test_detects_wordpiece(self):
        """BERT-style continuations are detected."""
        tokens = np.array(["col", "##oss", "##al", "world"])
        assert detect_subword_scheme(tokens) == "wordpiece"

    def test_detects_byte_level_word_start(self):
        """GPT-2/RoBERTa/Qwen-style word starts are detected."""
        tokens = np.array(["one", "Ġmassive", "Ġearth"])
        assert detect_subword_scheme(tokens) == "word_start"

    def test_detects_sentencepiece_word_start(self):
        """SentencePiece marks word starts with a different glyph."""
        tokens = np.array(["▁one", "▁massive", "▁earth"])
        assert detect_subword_scheme(tokens) == "word_start"

    def test_returns_none_for_plain_tokens(self):
        """Whole-word tokens use neither convention."""
        assert detect_subword_scheme(np.array(["one", "massive"])) is None


class TestMergeBpe:
    """Tests for merging subwords back into whole words."""

    @staticmethod
    def _embeddings(n, dim=3):
        return np.arange(n * dim, dtype=np.float64).reshape(n, dim)

    def test_wordpiece_merges_continuations(self):
        """ "##" pieces attach to the token before them."""
        tokens = np.array(["col", "##oss", "##al", "world"])
        merged, embeddings = merge_bpe(tokens, self._embeddings(4), scheme="wordpiece")
        assert merged.tolist() == ["colossal", "world"]
        assert len(embeddings) == 2

    def test_word_start_merges_unmarked_continuations(self):
        """Tokens without a word-start marker attach to the token before them."""
        tokens = np.array(["Ġcol", "oss", "al", "Ġworld"])
        merged, embeddings = merge_bpe(tokens, self._embeddings(4), scheme="word_start")
        assert merged.tolist() == ["colossal", "world"]
        assert len(embeddings) == 2

    def test_merged_embedding_is_mean_of_pieces(self):
        """A merged word takes the mean of its subword embeddings."""
        tokens = np.array(["Ġbig", "ness"])
        embeddings = np.array([[0.0, 0.0], [2.0, 4.0]])
        _, merged = merge_bpe(tokens, embeddings, scheme="word_start")
        np.testing.assert_allclose(merged[0], [1.0, 2.0])

    def test_word_start_does_not_merge_across_documents(self):
        """The first token of a document carries no marker but still starts a word."""
        # "earth" ends doc 0 and "an" opens doc 1; without group_ids they merge
        tokens = np.array(["one", "Ġmassive", "Ġearth", "an", "Ġenormous"])
        group_ids = np.array([0, 0, 0, 1, 1])
        merged, embeddings = merge_bpe(
            tokens, self._embeddings(5), scheme="word_start", group_ids=group_ids
        )
        assert merged.tolist() == ["one", "massive", "earth", "an", "enormous"]
        assert len(embeddings) == 5

    def test_word_start_merges_across_documents_without_group_ids(self):
        """Without document ids the boundary is invisible (regression guard)."""
        tokens = np.array(["Ġearth", "an"])
        merged, _ = merge_bpe(tokens, self._embeddings(2), scheme="word_start")
        assert merged.tolist() == ["earthan"]

    def test_leading_continuation_run_is_kept(self):
        """A continuation run with no preceding word start still forms one word."""
        tokens = np.array(["oss", "al"])
        merged, embeddings = merge_bpe(tokens, self._embeddings(2), scheme="word_start")
        assert merged.tolist() == ["ossal"]
        assert len(embeddings) == 1

    def test_rejects_unknown_scheme(self):
        """An unsupported scheme is an error rather than a silent no-op."""
        with pytest.raises(ValueError, match="Unknown subword scheme"):
            merge_bpe(np.array(["a"]), self._embeddings(1), scheme="nonsense")
