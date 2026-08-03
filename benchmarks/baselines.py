"""External diversity metrics, implemented here for comparison.

These are not part of the library. They live in the benchmarks because a claim
that similarity-sensitive Hill numbers are worth using has to be measured against
the alternatives, on the same corpora, with the same similarity matrix.

Currently: the Vendi Score (Friedman & Dieng, TMLR 2023).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt


def vendi_score(K: npt.NDArray[np.float64], q: float = 1.0) -> float:
    """Vendi Score: the exponential Shannon entropy of the eigenvalues of K/n.

    Friedman & Dieng, *The Vendi Score: A Diversity Evaluation Metric for Machine
    Learning* (TMLR 2023), arXiv:2210.02410.

        VS(x_1..x_n) = exp(-sum_i lambda_i log lambda_i)

    where lambda are the eigenvalues of K/n. K must be symmetric, positive
    semi-definite, and have unit diagonal, which makes the eigenvalues sum to 1
    and lets them be read as a probability distribution.

    It is the closest published relative of what this library computes, and the
    comparison is worth stating precisely. Both return an effective number of
    elements from a similarity matrix, and both agree at the extremes -- K = I
    gives n, K all-ones gives 1. They differ in between:

    - Vendi reads diversity off the **eigenvalue spectrum** of K/n, so abundance
      enters only through which items are in the matrix. Leinster-Cobbold carries
      an explicit abundance vector p, and D_q = (sum_i p_i (Zp)_i^(q-1))^(1/(1-q))
      responds to how often each species occurs.
    - Vendi requires K positive semi-definite, because a negative eigenvalue makes
      log(lambda) undefined. Leinster-Cobbold requires only Z in [0, 1] with unit
      diagonal, which is a weaker condition -- clamping or rescaling a similarity
      matrix can break positive semi-definiteness while leaving the Hill number
      perfectly well defined.

    Args:
        K: Similarity matrix (n x n), symmetric with unit diagonal.
        q: Order parameter. q=1 is the published Vendi Score; other orders follow
            the Renyi generalisation used in later work.

    Returns:
        Effective number of distinct elements.

    Raises:
        ValueError: If K has a materially negative eigenvalue, which means it is
            not a valid kernel and the score is undefined rather than merely
            inaccurate.
    """
    n = K.shape[0]
    if n == 0:
        return 0.0
    sym = (np.asarray(K, dtype=np.float64) + np.asarray(K, dtype=np.float64).T) / 2
    eigenvalues = np.linalg.eigvalsh(sym / n)

    if eigenvalues.min() < -1e-6:
        raise ValueError(
            f"K is not positive semi-definite (smallest eigenvalue "
            f"{eigenvalues.min():.3e}); the Vendi Score is undefined for it."
        )
    # Clip the numerical dust that eigvalsh leaves around zero.
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    eigenvalues = eigenvalues / total
    nonzero = eigenvalues[eigenvalues > 0]

    if q == 1.0:
        return float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    if np.isinf(q):
        return float(1.0 / nonzero.max())
    return float(np.power(np.sum(np.power(nonzero, q)), 1.0 / (1.0 - q)))


class VendiWrapper:
    """Score a corpus with Vendi, reusing another metric's similarity matrix.

    Holding the similarity function fixed is the point: it isolates the choice of
    diversity index from the choice of representation, which is the only way the
    comparison says anything about the index.
    """

    def __init__(self, base_metric: Any, q: float = 1.0) -> None:
        self.base = base_metric
        self.q = q

    def __call__(self, corpus: list[str]) -> float:
        if not corpus:
            return 0.0
        if len(corpus) == 1:
            return 1.0
        features, _species = self.base.extract_features(corpus)
        K = np.asarray(self.base.calculate_similarities(features), dtype=np.float64)
        return vendi_score(K, q=self.q)

    def extract_features(self, corpus: list[str]) -> Any:
        return self.base.extract_features(corpus)

    def calculate_similarities(self, features: Any) -> Any:
        return self.base.calculate_similarities(features)
