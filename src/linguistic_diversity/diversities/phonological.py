"""Phonological diversity metrics based on rhythmic and phonemic patterns.

This module provides metrics for measuring diversity in the phonological structure
of text using rhythmic patterns (stress and weight) and phonemic representations.
"""

from __future__ import annotations

import string
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
class PhonologicalConfig(MetricConfig):
    """Configuration for phonological diversity metrics."""

    # Sequence processing
    pad_to_max_len: bool = False
    split_sentences: bool = False

    # Backend selection (for Phonemic)
    backend: str = "g2p_en"  # "g2p_en" (pure Python) or "phonemizer" (needs espeak-ng)


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


class Rhythmic(TextDiversity):
    """Rhythmic diversity based on syllable stress and weight patterns.

    This metric computes diversity based on the rhythmic patterns of text,
    using sequences of stressed/unstressed and weighted/unweighted syllables.
    It uses pyphen for syllabification and pronouncing for stress patterns.

    Example:
        >>> metric = Rhythmic()
        >>> corpus = [
        ...     'The quick brown fox',
        ...     'A lazy dog sleeps',
        ...     'Birds sing loudly'
        ... ]
        >>> diversity = metric(corpus)

    Note:
        Requires pyphen and pronouncing libraries (pure Python, no system dependencies).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize rhythmic diversity metric.

        Args:
            config: Optional configuration dict.
        """
        super().__init__(config)
        self.aligner = Align.PairwiseAligner()
        self.max_len = 0

        # Import rhythmic analyzer
        try:
            from .rhythmic_analyzer import RhythmicAnalyzer

            self.analyzer = RhythmicAnalyzer()
        except ImportError as e:
            raise ImportError(
                f"Rhythmic analysis dependencies not installed: {e}. "
                "Install with: pip install pyphen pronouncing"
            )

    @classmethod
    def _config_class(cls) -> type[PhonologicalConfig]:
        return PhonologicalConfig

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
        """Extract rhythmic patterns from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (rhythm_sequences, documents).
        """
        # Clean corpus
        corpus = clean_text(corpus)

        # Optionally split into sentences
        if self.config.split_sentences:
            corpus = split_sentences(corpus)

        # Strip punctuation for rhythmic analysis
        corpus_no_punct = [
            text.translate(str.maketrans("", "", string.punctuation))
            for text in corpus
        ]

        # Extract rhythmic patterns using custom analyzer
        rhythms = []
        for text in corpus_no_punct:
            try:
                # Get rhythm pattern as list of stress+weight codes
                rhythm = self.analyzer.extract_rhythm_pattern(text)
                if not rhythm:
                    rhythm = [""]
            except Exception:
                # If parsing fails, use empty rhythm
                rhythm = [""]

            rhythms.append(rhythm)

        # Store max length for normalization
        self.max_len = max(len(seq) for seq in rhythms) if rhythms else 0
        if self.max_len == 0:
            self.max_len = 1  # Avoid division by zero

        # Optionally pad to max length
        if self.config.pad_to_max_len:
            rhythms = [
                seq + ["N"] * (self.max_len - len(seq))
                for seq in rhythms
            ]

        return rhythms, corpus

    def calculate_similarities(
        self, features: list[list[str]]
    ) -> npt.NDArray[np.float64]:
        """Calculate pairwise similarities using sequence alignment.

        Args:
            features: List of rhythm sequences.

        Returns:
            Similarity matrix (n x n).
        """
        # Convert rhythm tags to alphabetic characters for alignment
        alpha_features = tag_to_alpha(features)
        string_features = ["".join(seq) for seq in alpha_features]

        # Compute pairwise alignment scores
        Z = compute_similarity_matrix_pairwise(
            string_features,
            self._align_and_score,
            diagonal_val=float(self.max_len),
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
        """Calculate similarity between query and corpus rhythms.

        Args:
            query_features: Query rhythm sequence.
            corpus_features: Corpus rhythm sequences.

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


class Phonemic(TextDiversity):
    """Phonemic diversity based on phoneme sequences.

    This metric computes diversity based on phonemic representations of text.
    Supports two backends:
    1. g2p_en (default) - Pure Python, no system dependencies
    2. phonemizer - Requires espeak-ng system library

    Example:
        >>> # Using g2p_en (pure Python, default)
        >>> metric = Phonemic()
        >>> corpus = ['hello world', 'goodbye moon']
        >>> diversity = metric(corpus)

        >>> # Using phonemizer (requires espeak-ng)
        >>> metric = Phonemic({'backend': 'phonemizer'})
        >>> diversity = metric(corpus)

    Note:
        - g2p_en (default): Pure Python, works out of the box
        - phonemizer: Requires espeak-ng system library
          - Linux: sudo apt-get install espeak-ng
          - macOS: brew install espeak-ng
          - Windows: See https://github.com/espeak-ng/espeak-ng/releases
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize phonemic diversity metric.

        Args:
            config: Optional configuration dict.
                   Use {'backend': 'phonemizer'} for phonemizer backend.
        """
        super().__init__(config)
        self.aligner = Align.PairwiseAligner()
        self.max_len = 0

        # Get backend preference (default to g2p_en)
        backend = self.config.backend

        # Try g2p_en first (pure Python, no system deps)
        if backend == 'g2p_en':
            try:
                from g2p_en import G2p
                self.g2p = G2p()
                self.backend = 'g2p_en'
                return
            except ImportError:
                if self.config.verbose if hasattr(self.config, 'verbose') else False:
                    print("g2p_en not found, trying phonemizer...")

        # Fall back to phonemizer
        try:
            from phonemizer import phonemize
            from phonemizer.backend import EspeakBackend
            from phonemizer.punctuation import Punctuation
            from phonemizer.separator import Separator

            self.phonemize = phonemize
            self.EspeakBackend = EspeakBackend
            self.Punctuation = Punctuation
            self.Separator = Separator
            self.backend = 'phonemizer'
        except ImportError:
            raise ImportError(
                "No phoneme conversion library found. Install one of:\n"
                "  1. g2p_en (pure Python, recommended): pip install g2p-en\n"
                "  2. phonemizer (requires espeak-ng): pip install phonemizer"
            )

    @classmethod
    def _config_class(cls) -> type[PhonologicalConfig]:
        return PhonologicalConfig

    @classmethod
    def _default_config(cls) -> dict[str, Any]:
        return {
            "pad_to_max_len": False,
            "split_sentences": False,
        }

    def _align_and_score(self, seq1: str, seq2: str) -> float:
        """Align two sequences and return alignment score.

        Args:
            seq1: First sequence.
            seq2: Second sequence.

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
    ) -> tuple[list[str], list[str]]:
        """Extract phoneme sequences from corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Tuple of (phoneme_sequences, documents).
        """
        # Clean corpus
        corpus = clean_text(corpus)

        # Optionally split into sentences
        if self.config.split_sentences:
            corpus = split_sentences(corpus)

        # Convert to phonemes based on backend
        try:
            if self.backend == 'g2p_en':
                # Use g2p_en (pure Python)
                phonemes = []
                for text in corpus:
                    # g2p returns list of phonemes
                    phoneme_list = self.g2p(text)
                    # Join into string with spaces
                    phoneme_str = " ".join(phoneme_list)
                    phonemes.append(phoneme_str)
            else:
                # Use phonemizer (requires espeak-ng)
                phonemes = self.phonemize(
                    corpus,
                    language="en-us",
                    backend="espeak",
                    strip=True,
                    preserve_punctuation=False,
                    with_stress=False,
                )

                # Convert to list if single string
                if isinstance(phonemes, str):
                    phonemes = [phonemes]

        except Exception as e:
            if self.config.verbose:
                print(f"Phonemization failed: {e}")
            # Fallback to empty phonemes
            phonemes = [""] * len(corpus)

        # Store max length
        self.max_len = max(len(p) for p in phonemes) if phonemes else 0
        if self.max_len == 0:
            self.max_len = 1

        # Optionally pad
        if self.config.pad_to_max_len:
            phonemes = [
                p + " " * (self.max_len - len(p))
                for p in phonemes
            ]

        return phonemes, corpus

    def calculate_similarities(
        self, features: list[str]
    ) -> npt.NDArray[np.float64]:
        """Calculate pairwise similarities using sequence alignment.

        Args:
            features: List of phoneme sequences.

        Returns:
            Similarity matrix (n x n).
        """
        # Compute pairwise alignment scores
        Z = compute_similarity_matrix_pairwise(
            features,
            self._align_and_score,
            diagonal_val=float(self.max_len),
            verbose=self.config.verbose,
        )

        # Normalize by max sequence length
        if self.max_len > 0:
            Z = Z / self.max_len

        return Z

    def calculate_similarity_vector(
        self,
        query_features: str,
        corpus_features: list[str],
    ) -> npt.NDArray[np.float64]:
        """Calculate similarity between query and corpus phonemes.

        Args:
            query_features: Query phoneme sequence.
            corpus_features: Corpus phoneme sequences.

        Returns:
            Similarity scores (n,).
        """
        query_len = len(query_features)
        scores = []

        for corpus_feat in corpus_features:
            score = self._align_and_score(query_features, corpus_feat)
            # Normalize by max length
            max_len = max(len(corpus_feat), query_len)
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
