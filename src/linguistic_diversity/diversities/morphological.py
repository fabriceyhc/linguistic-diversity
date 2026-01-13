"""Morphological diversity metrics based on part-of-speech sequences.

This module provides metrics for measuring diversity in the morphological structure
of text using part-of-speech (POS) tag sequences and alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
import spacy
from Bio import Align  # type: ignore

from ..metric import MetricConfig, TextDiversity
from ..utils import (
    clean_text,
    compute_similarity_matrix_pairwise,
    split_sentences,
    tag_to_alpha,
)


@dataclass
class MorphologicalConfig(MetricConfig):
    """Configuration for morphological diversity metrics."""

    # Sequence processing
    pad_to_max_len: bool = False
    split_sentences: bool = False


# Model caching
_SPACY_MODEL_CACHE: dict[str, Any] = {}


@lru_cache(maxsize=1)
def _get_spacy_model(model_name: str = "en_core_web_sm") -> Any:
    """Get or load a cached spaCy model.

    Args:
        model_name: Name of the spaCy model.

    Returns:
        Loaded spaCy model.
    """
    if model_name not in _SPACY_MODEL_CACHE:
        _SPACY_MODEL_CACHE[model_name] = spacy.load(model_name)
    return _SPACY_MODEL_CACHE[model_name]


class PartOfSpeechSequence(TextDiversity):
    """Part-of-speech sequence diversity.

    This metric computes diversity based on the sequences of part-of-speech tags.
    It uses biological sequence alignment (similar to DNA/protein alignment) to
    measure similarity between POS tag sequences.

    Example:
        >>> metric = PartOfSpeechSequence()
        >>> corpus = [
        ...     'The quick brown fox jumps',
        ...     'A fast red dog runs',
        ...     'Birds fly high'
        ... ]
        >>> diversity = metric(corpus)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize POS sequence diversity metric.

        Args:
            config: Optional configuration dict.
        """
        super().__init__(config)
        self.model = _get_spacy_model()
        self.aligner = Align.PairwiseAligner()
        self.max_len = 0

    @classmethod
    def _config_class(cls) -> type[MorphologicalConfig]:
        return MorphologicalConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        return {
            "pad_to_max_len": False,
            "split_sentences": False,
        }

    def _align_and_score(self, seq1: str, seq2: str) -> float:
        """Align two sequences and return alignment score.

        Args:
            seq1: First sequence (string of characters).
            seq2: Second sequence (string of characters).

        Returns:
            Alignment score.
        """
        # Handle empty sequences
        if not seq1 or not seq2:
            return 0.0

        try:
            alignments = self.aligner.align(seq1, seq2)
            return float(alignments.score)
        except (ValueError, IndexError):
            # Alignment failed (e.g., empty sequences after processing)
            return 0.0

    def extract_features(
        self, corpus: list[str]
    ) -> tuple[list[list[str]], list[str]]:
        """Extract POS tag sequences from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (pos_sequences, documents).
        """
        # Clean corpus
        corpus = clean_text(corpus)

        # Optionally split into sentences
        if self.config.split_sentences:
            corpus = split_sentences(corpus)

        # Extract POS tags
        pos_sequences = []
        for text in corpus:
            doc = self.model(text)
            pos_tags = [token.pos_ for token in doc]
            pos_sequences.append(pos_tags)

        # Store max length for normalization
        self.max_len = max(len(seq) for seq in pos_sequences) if pos_sequences else 0

        # Optionally pad to max length
        if self.config.pad_to_max_len:
            pos_sequences = [
                seq + ["NULL"] * (self.max_len - len(seq))
                for seq in pos_sequences
            ]

        return pos_sequences, corpus

    def calculate_similarities(
        self, features: list[list[str]]
    ) -> npt.NDArray[np.float64]:
        """Calculate pairwise similarities using sequence alignment.

        Args:
            features: List of POS tag sequences.

        Returns:
            Similarity matrix (n x n).
        """
        # Convert POS tags to alphabetic characters for alignment
        alpha_features = tag_to_alpha(features)
        string_features = ["".join(seq) for seq in alpha_features]

        # Compute pairwise alignment scores
        Z = compute_similarity_matrix_pairwise(
            string_features,
            self._align_and_score,
            diagonal_val=float(self.max_len),  # Perfect self-alignment
            verbose=self.config.verbose,
        )

        # Normalize by max sequence length
        if self.max_len > 0:
            Z = Z / self.max_len

        return Z

    def calculate_similarity_vector(
        self,
        query_features: list[str],
        corpus_features: list[list[str]],
    ) -> npt.NDArray[np.float64]:
        """Calculate similarity between query and corpus sequences.

        Args:
            query_features: Query POS sequence.
            corpus_features: Corpus POS sequences.

        Returns:
            Similarity scores (n,).
        """
        # Convert to alpha and string format
        all_features = [query_features] + corpus_features
        alpha_features = tag_to_alpha(all_features)
        string_features = ["".join(seq) for seq in alpha_features]

        query_str = string_features[0]
        corpus_strs = string_features[1:]

        # Compute similarity scores
        query_len = len(query_str)
        scores = []

        for corpus_str in corpus_strs:
            score = self._align_and_score(query_str, corpus_str)
            # Normalize by max of the two lengths
            max_len = max(len(corpus_str), query_len)
            if max_len > 0:
                score /= max_len
            scores.append(score)

        return np.array(scores, dtype=np.float64)

    def calculate_abundance(self, species: list[str]) -> npt.NDArray[np.float64]:
        """Calculate uniform abundance distribution.

        Args:
            species: List of documents.

        Returns:
            Uniform distribution over documents.
        """
        n = len(species)
        return np.full(n, 1.0 / n, dtype=np.float64)
