"""Lexical diversity baselines.

These are the conventional surface-form measures — type-token ratio, distinct-n, and
self-BLEU — included so they can be compared directly against the similarity-sensitive
metrics in this library.

They are **not** Hill numbers. They count or score surface forms and are bounded in
[0, 1] rather than expressed as an effective number of species. They are provided as
baselines, not as recommended diversity measures: a corpus of unique words restating one
idea scores perfectly on all three while carrying a single meaning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..metric import DiversityMetric, MetricConfig

_TOKEN_RE = re.compile(r"\w+")


@dataclass
class LexicalConfig(MetricConfig):
    """Configuration for lexical baselines."""

    n: int = 1  # n-gram order for DistinctN
    max_n: int = 4  # maximum n-gram order for SelfBLEU
    lowercase: bool = True


def _tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Split text into word tokens, dropping punctuation.

    Args:
        text: Input string.
        lowercase: Whether to casefold first.

    Returns:
        List of word tokens.
    """
    return _TOKEN_RE.findall(text.lower() if lowercase else text)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Return the n-grams of a token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


class TypeTokenRatio(DiversityMetric):
    """Ratio of unique tokens to total tokens.

    Ranges from near 0 (one word repeated) to 1 (every token unique). Higher is
    conventionally read as more diverse.

    Example:
        >>> TypeTokenRatio()(['the cat sat', 'a dog ran'])
        1.0
    """

    # Narrow the base annotation so attribute access type-checks
    config: LexicalConfig

    @classmethod
    def _config_class(cls) -> type[LexicalConfig]:
        return LexicalConfig

    def __call__(self, corpus: list[str]) -> float:
        """Compute type-token ratio over the whole corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Unique tokens divided by total tokens, or 0.0 for an empty corpus.
        """
        tokens = [t for doc in corpus for t in _tokenize(doc, self.config.lowercase)]
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)


class DistinctN(DiversityMetric):
    """Fraction of n-grams in the corpus that are unique (Li et al., 2016).

    ``distinct-1`` is the type-token ratio over unigrams; ``distinct-2`` uses bigrams
    and is the more commonly reported variant. Higher is conventionally read as more
    diverse.

    Example:
        >>> DistinctN({'n': 2})(['the cat sat', 'the cat ran'])
        0.75
    """

    # Narrow the base annotation so attribute access type-checks
    config: LexicalConfig

    @classmethod
    def _config_class(cls) -> type[LexicalConfig]:
        return LexicalConfig

    def __call__(self, corpus: list[str]) -> float:
        """Compute distinct-n over the whole corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Unique n-grams divided by total n-grams, or 0.0 if none exist.
        """
        grams = [
            g
            for doc in corpus
            for g in _ngrams(_tokenize(doc, self.config.lowercase), self.config.n)
        ]
        if not grams:
            return 0.0
        return len(set(grams)) / len(grams)


class SelfBLEU(DiversityMetric):
    """Mean BLEU of each document scored against all the others (Zhu et al., 2018).

    Note the direction is **inverted** relative to every other metric here: self-BLEU
    measures how much documents overlap, so **lower means more diverse**. A corpus of
    identical sentences approaches 1.0.

    Example:
        >>> round(SelfBLEU()(['the cat sat', 'a dog ran', 'birds fly']), 3)
        0.0

    Note:
        Requires ``nltk`` (already a dependency). Uses smoothing so that short
        documents with no higher-order n-gram matches do not collapse to exactly 0.
    """

    # Narrow the base annotation so attribute access type-checks
    config: LexicalConfig

    @classmethod
    def _config_class(cls) -> type[LexicalConfig]:
        return LexicalConfig

    def __call__(self, corpus: list[str]) -> float:
        """Compute mean self-BLEU across the corpus.

        Args:
            corpus: List of text documents.

        Returns:
            Mean BLEU against the other documents. Lower means more diverse.
            Returns 0.0 when fewer than two documents are supplied.
        """
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        docs = [_tokenize(d, self.config.lowercase) for d in corpus]
        docs = [d for d in docs if d]
        if len(docs) < 2:
            return 0.0

        weights = tuple([1.0 / self.config.max_n] * self.config.max_n)
        smoothing = SmoothingFunction().method1

        scores = []
        for i, hypothesis in enumerate(docs):
            references = [d for j, d in enumerate(docs) if j != i]
            scores.append(
                sentence_bleu(references, hypothesis, weights=weights, smoothing_function=smoothing)
            )
        return float(sum(scores) / len(scores))
