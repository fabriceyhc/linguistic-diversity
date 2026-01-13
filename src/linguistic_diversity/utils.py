"""Utility functions for linguistic diversity calculations."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Callable, Iterator

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm


def chunker(seq: Any, size: int) -> Iterator[Any]:
    """Split sequence into chunks of specified size.

    Args:
        seq: Sequence to chunk.
        size: Chunk size.

    Yields:
        Chunks of the sequence.
    """
    for pos in range(0, len(seq), size):
        yield seq[pos : pos + size]


def merge_bpe(
    tokens: npt.NDArray[Any],
    embeddings: npt.NDArray[np.float64],
    prefix: str = "##",
) -> tuple[npt.NDArray[Any], npt.NDArray[np.float64]]:
    """Merge BPE subword tokens and their embeddings.

    Args:
        tokens: Array of tokens (may include BPE subwords).
        embeddings: Corresponding embeddings.
        prefix: BPE prefix marker (default: "##" for BERT).

    Returns:
        Tuple of (merged_tokens, merged_embeddings).
    """
    merged_tokens = []
    merged_embeddings = []

    current_emb = []
    current_token = ""

    # Process in reverse to handle BPE prefix
    for token, emb in zip(reversed(tokens), reversed(embeddings)):
        token_str = str(token)
        current_emb.append(emb)

        if token_str.startswith(prefix):
            # This is a subword continuation
            current_token = token_str.replace(prefix, "") + current_token
        else:
            # This is a full word or start of word
            current_token = token_str + current_token
            merged_tokens.append(current_token)
            merged_embeddings.append(np.stack(current_emb).mean(axis=0))
            current_emb = []
            current_token = ""

    # Reverse back to original order
    return (
        np.array(merged_tokens[::-1]),
        np.array(merged_embeddings[::-1]),
    )


def cos_sim(
    a: torch.Tensor | npt.NDArray[np.float64],
    b: torch.Tensor | npt.NDArray[np.float64],
) -> torch.Tensor:
    """Compute cosine similarity between all pairs of vectors.

    Args:
        a: First set of vectors (m x d).
        b: Second set of vectors (n x d).

    Returns:
        Similarity matrix (m x n).
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)

    return torch.mm(a_norm, b_norm.transpose(0, 1))


def compute_similarity_matrix_pairwise(
    inputs: list[Any],
    similarity_fn: Callable[[Any, Any], float],
    diagonal_val: float | None = 1.0,
    verbose: bool = False,
) -> npt.NDArray[np.float64]:
    """Compute pairwise similarity matrix using a custom function.

    This function is O(n²) but necessary when vectorization isn't possible.
    Uses upper triangular computation for efficiency.

    Args:
        inputs: List of items to compare.
        similarity_fn: Function computing similarity between two items.
        diagonal_val: Value for diagonal elements (None to compute).
        verbose: Show progress bar.

    Returns:
        Symmetric similarity matrix (n x n).
    """
    n = len(inputs)
    Z = np.zeros((n, n), dtype=np.float64)

    # Get upper triangular indices
    iu = np.triu_indices(n, k=1)

    # Compute upper triangle
    iterator = range(n)
    if verbose:
        iterator = tqdm(iterator, desc="Computing similarities")

    for i in iterator:
        for j in range(i + 1, n):
            Z[i, j] = similarity_fn(inputs[i], inputs[j])

    # Mirror to lower triangle
    Z = Z + Z.T

    # Set diagonal
    if diagonal_val is not None:
        np.fill_diagonal(Z, diagonal_val)

    return Z


def compute_similarity_matrix_faiss(
    features: npt.NDArray[np.float64],
    distance_metric: int,
    postprocess: str | None = None,
) -> npt.NDArray[np.float64]:
    """Compute similarity matrix using FAISS for efficiency.

    Args:
        features: Feature matrix (n x d).
        distance_metric: FAISS distance metric (e.g., faiss.METRIC_INNER_PRODUCT).
        postprocess: Post-processing ("exp" or "invert" or None).

    Returns:
        Similarity matrix (n x n).
    """
    import faiss

    features = np.ascontiguousarray(features.astype(np.float32))
    n, d = features.shape

    # Normalize for inner product if needed
    if distance_metric == faiss.METRIC_INNER_PRODUCT:
        faiss.normalize_L2(features)

    # Build index and search
    index = faiss.IndexFlat(int(d), distance_metric)
    index.add(features)
    distances, indices = index.search(features, n)

    # Construct similarity matrix using vectorized indexing
    Z = np.zeros((n, n), dtype=np.float64)
    row_indices = np.repeat(np.arange(n), n)
    Z[row_indices, indices.ravel()] = distances.ravel()

    # Post-process distances
    if postprocess == "exp":
        Z = np.exp(-Z)
    elif postprocess == "invert":
        Z = 1.0 - Z

    # Ensure diagonal is 1
    np.fill_diagonal(Z, 1.0)

    return Z


def similarity_search_faiss(
    query_features: npt.NDArray[np.float64],
    corpus_features: npt.NDArray[np.float64],
    distance_metric: int,
    postprocess: str | None = None,
) -> npt.NDArray[np.float64]:
    """Search corpus for items similar to query using FAISS.

    Args:
        query_features: Query features (1 x d or m x d).
        corpus_features: Corpus features (n x d).
        distance_metric: FAISS distance metric.
        postprocess: Post-processing ("exp" or "invert" or None).

    Returns:
        Similarity scores (m x n or just n if query is 1D).
    """
    import faiss

    # Ensure contiguous float32
    query_features = np.ascontiguousarray(query_features.astype(np.float32))
    corpus_features = np.ascontiguousarray(corpus_features.astype(np.float32))

    # Handle 1D query
    if len(query_features.shape) == 1:
        query_features = query_features.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False

    n, d = corpus_features.shape

    # Normalize for inner product if needed
    if distance_metric == faiss.METRIC_INNER_PRODUCT:
        faiss.normalize_L2(corpus_features)
        faiss.normalize_L2(query_features)

    # Build index and search
    index = faiss.IndexFlat(int(d), distance_metric)
    index.add(corpus_features)
    distances, _ = index.search(query_features, n)

    # Post-process
    if postprocess == "exp":
        distances = np.exp(-distances)
    elif postprocess == "invert":
        distances = 1.0 - distances

    return distances[0] if squeeze_output else distances


# Text cleaning utilities


_CLEAN_PATTERNS = [
    (r"<br\s*/?>", " "),
    (r"\.{2,}", ". "),
    (r"([.!?])", r"\1 "),
]
_CLEAN_REGEX = [(re.compile(pattern), repl) for pattern, repl in _CLEAN_PATTERNS]


def clean_text(texts: list[str] | str) -> list[str]:
    """Clean text by removing HTML and normalizing punctuation.

    Args:
        texts: Text or list of texts to clean.

    Returns:
        List of cleaned texts.
    """
    if isinstance(texts, str):
        texts = [texts]

    cleaned = []
    for text in texts:
        # Apply regex patterns
        for pattern, repl in _CLEAN_REGEX:
            text = pattern.sub(repl, text)
        # Strip whitespace
        text = " ".join(text.split())
        cleaned.append(text)

    return cleaned


@lru_cache(maxsize=1)
def _get_sentence_splitter() -> Any:
    """Get sentence splitter (cached to avoid repeated loading)."""
    from spacy.lang.en import English

    nlp = English()
    nlp.add_pipe("sentencizer")
    return nlp


def split_sentences(
    texts: list[str] | str,
    return_ids: bool = False,
) -> list[str] | tuple[list[str], list[int], list[int]]:
    """Split texts into sentences.

    Args:
        texts: Text or list of texts.
        return_ids: If True, return (sentences, text_ids, sentence_ids).

    Returns:
        List of sentences, or tuple with IDs if return_ids=True.
    """
    if isinstance(texts, str):
        texts = [texts]

    nlp = _get_sentence_splitter()

    sentences = []
    text_ids = []
    sentence_ids = []

    for text_idx, text in enumerate(texts):
        doc = nlp(text)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]

        sentences.extend(sents)
        text_ids.extend([text_idx] * len(sents))
        sentence_ids.extend(range(len(sents)))

    if return_ids:
        return sentences, text_ids, sentence_ids
    return sentences


def tag_to_alpha(tags: list[list[str]]) -> list[list[str]]:
    """Convert tag sequences to alphabetic sequences.

    Useful for sequence alignment when tags contain special characters.

    Args:
        tags: List of tag sequences.

    Returns:
        List of alphabetic tag sequences.
    """
    # Build unique tag mapping
    unique_tags = sorted(set(tag for seq in tags for tag in seq))
    tag_map = {tag: chr(65 + i) for i, tag in enumerate(unique_tags)}

    # Apply mapping
    return [[tag_map[tag] for tag in seq] for seq in tags]


def hamming_similarity(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute pairwise Hamming similarity matrix.

    Args:
        a: Binary feature matrix (n x d).

    Returns:
        Similarity matrix (n x n) with values in [0, 1].
    """
    dims = a.shape[1]
    # Efficient Hamming distance using matrix operations
    similarity = (2 * np.inner(a - 0.5, 0.5 - a) + dims / 2) / dims
    return similarity
