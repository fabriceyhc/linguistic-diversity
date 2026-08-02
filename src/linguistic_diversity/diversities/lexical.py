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

import math
import re
from collections import Counter
from dataclasses import dataclass

from ..metric import DiversityMetric, MetricConfig

_TOKEN_RE = re.compile(r"\w+")

# Stands in for a zero match count so one missing n-gram order does not zero out
# the whole geometric mean (Chen & Cherry 2014, smoothing method 1).
_SMOOTH_EPSILON = 0.1


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


def _modified_precision(
    references: list[list[str]], hypothesis: list[str], n: int
) -> tuple[int, int]:
    """Clipped n-gram overlap between a hypothesis and its references.

    Each hypothesis n-gram counts at most as many times as it appears in the
    reference that contains it most often, which is what stops a sentence from
    scoring well by repeating one common phrase.

    Args:
        references: Tokenized reference documents.
        hypothesis: Tokenized hypothesis document.
        n: n-gram order.

    Returns:
        Tuple of (clipped matches, total hypothesis n-grams).
    """
    hyp_counts = Counter(_ngrams(hypothesis, n))
    if not hyp_counts:
        return 0, 0

    max_ref_counts: Counter[tuple[str, ...]] = Counter()
    for reference in references:
        ref_counts = Counter(_ngrams(reference, n))
        for gram, count in ref_counts.items():
            if count > max_ref_counts[gram]:
                max_ref_counts[gram] = count

    clipped = sum(min(count, max_ref_counts[gram]) for gram, count in hyp_counts.items())
    return clipped, sum(hyp_counts.values())


def _sentence_bleu(references: list[list[str]], hypothesis: list[str], max_n: int) -> float:
    """Sentence-level BLEU with uniform weights and add-epsilon smoothing.

    Implemented here rather than via nltk: importing ``nltk.translate`` pulls in
    ``nltk.corpus`` and wordnet, which is both heavy for an n-gram count and
    fragile (nltk's import guard rejects defusedxml in some environments).

    Args:
        references: Tokenized reference documents.
        hypothesis: Tokenized hypothesis document.
        max_n: Maximum n-gram order.

    Returns:
        BLEU score in [0, 1].
    """
    if not hypothesis or not references:
        return 0.0

    # No shared words at all means no overlap to measure. Smoothing exists to
    # rescue missing higher-order n-grams, not a total absence of agreement, so
    # this short-circuits rather than reporting a small smoothed value.
    unigram_matches, _ = _modified_precision(references, hypothesis, 1)
    if unigram_matches == 0:
        return 0.0

    log_precision = 0.0
    for n in range(1, max_n + 1):
        clipped, total = _modified_precision(references, hypothesis, n)
        # A hypothesis shorter than n has no n-grams at all. Clamping the
        # denominator to 1 lets smoothing handle it, instead of zeroing a score
        # that lower orders already supported -- documents shorter than max_n
        # would otherwise always score 0.
        denominator = max(1, total)
        # Smoothing: a zero numerator would send the geometric mean to 0 and hide
        # all lower-order agreement, so treat it as a small non-zero count.
        precision = (clipped if clipped > 0 else _SMOOTH_EPSILON) / denominator
        log_precision += math.log(precision) / max_n

    # Brevity penalty against the reference length closest to the hypothesis
    hyp_len = len(hypothesis)
    ref_len = min((len(r) for r in references), key=lambda rl: (abs(rl - hyp_len), rl))
    brevity = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)

    return brevity * math.exp(log_precision)


class SelfBLEU(DiversityMetric):
    """Mean BLEU of each document scored against all the others (Zhu et al., 2018).

    Note the direction is **inverted** relative to every other metric here: self-BLEU
    measures how much documents overlap, so **lower means more diverse**. A corpus of
    identical sentences approaches 1.0.

    Example:
        >>> round(SelfBLEU()(['the cat sat', 'a dog ran', 'birds fly']), 3)
        0.0
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
        docs = [_tokenize(d, self.config.lowercase) for d in corpus]
        docs = [d for d in docs if d]
        if len(docs) < 2:
            return 0.0

        scores = [
            _sentence_bleu([d for j, d in enumerate(docs) if j != i], hypothesis, self.config.max_n)
            for i, hypothesis in enumerate(docs)
        ]
        return float(sum(scores) / len(scores))
