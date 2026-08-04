"""Candidate similarity corrections, each fitted on background text only.

Every transform here takes a set's embeddings and returns a similarity matrix. The
constraint that matters: a transform may depend on the *encoder* and on fixed
background text, never on the other documents in the corpus being scored. Anything
that violates that loses replication invariance, which is what sank `mean_adj`.

An identity worth stating, because it collapses two entries in the reading list into
one: Chao et al.'s tau-truncation of a distance matrix, d_ij(tau) = min(tau, d_ij)
followed by the linear similarity 1 - d_ij(tau)/tau, is *algebraically identical* to
this library's similarity floor with z0 = 1 - tau. So tau does not add a new transform.
What it adds is the ecological default: tau = mean pairwise distance, which is far more
aggressive than the median-of-unrelated-pairs floor we shipped. That is the part worth
testing.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def _finish(Z: Array) -> Array:
    Z = np.clip(Z, 0.0, 1.0)
    np.fill_diagonal(Z, 1.0)
    return Z


# ------------------------------------------------------------------ base geometry


def cosine(E: Array, _ctx: Array | None = None) -> Array:
    return _finish(E @ E.T)


def floored(z0: float):
    """Affine floor. Equivalently Chao tau-truncation with tau = 1 - z0."""

    def f(E: Array, _ctx: Array | None = None) -> Array:
        Z = E @ E.T
        if z0 > 0:
            Z = (Z - z0) / (1.0 - z0)
        return _finish(Z)

    return f


def tau_truncated(tau: float):
    """Chao et al. (2019) tau-truncation, written in its own terms."""

    def f(E: Array, _ctx: Array | None = None) -> Array:
        d = np.clip(1.0 - E @ E.T, 0.0, None)
        return _finish(1.0 - np.minimum(d, tau) / tau)

    return f


# ------------------------------------------------------------ global linear fixes


def whitener(background: Array, shrink: float = 0.0):
    """ZCA whitening fitted on background text. shrink=1 is the identity."""
    mu = background.mean(0, keepdims=True)
    cov = np.cov((background - mu).T)
    cov = (1 - shrink) * cov + shrink * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    W = evecs @ np.diag(np.maximum(evals, 1e-8) ** -0.5) @ evecs.T
    return mu, W


def whitened(mu: Array, W: Array, z0: float = 0.0):
    def f(E: Array, _ctx: Array | None = None) -> Array:
        X = (E - mu) @ W
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        Z = X @ X.T
        if z0 > 0:
            Z = (Z - z0) / (1.0 - z0)
        return _finish(Z)

    return f


# -------------------------------------------------------- local / hubness-aware


def local_scaling(background: Array, k: int = 50, z0: float = 0.0):
    """Zelnik-Manor & Perona local scaling, with sigma_i read off background text.

    sigma_i is document i's distance to its k-th nearest background neighbour, so a
    document sitting in a dense region of the encoder's space gets its similarities
    shrunk. Because sigma depends only on i and on fixed background text, two corpora
    containing the same document give it the same sigma -- replication survives.
    """

    def sigma(E: Array) -> Array:
        d = np.sqrt(np.maximum(2.0 - 2.0 * (E @ background.T), 0.0))
        return np.partition(d, k, axis=1)[:, k]

    def f(E: Array, _ctx: Array | None = None) -> Array:
        s = sigma(E)
        d2 = np.maximum(2.0 - 2.0 * (E @ E.T), 0.0)
        Z = np.exp(-d2 / np.maximum(np.outer(s, s), 1e-12))
        if z0 > 0:
            Z = (Z - z0) / (1.0 - z0)
        return _finish(Z)

    return f


def mutual_proximity(background: Array, z0: float = 0.0):
    """Mutual proximity: how mutually close a pair is *relative to the background*.

    MP(i,j) = P(X_i > d_ij) * P(X_j > d_ij), estimated empirically against each
    document's distance distribution to the background. Hubs -- documents close to
    everything -- get their similarities discounted, which is precisely the pathology
    that inflates the similarity floor.
    """

    def f(E: Array, _ctx: Array | None = None) -> Array:
        d_bg = 1.0 - E @ background.T  # (n, n_bg)
        d = 1.0 - E @ E.T
        # P(random background doc is farther from i than j is)
        frac = (d_bg[:, None, :] > d[:, :, None]).mean(axis=2)  # (n, n)
        Z = frac * frac.T
        if z0 > 0:
            Z = (Z - z0) / (1.0 - z0)
        return _finish(Z)

    return f


# ----------------------------------------------------------- prompt-conditional


def context_projected(z0: float = 0.0, strength: float = 1.0):
    """Remove the prompt's direction from every response before comparing.

    Conditional-Vendi in spirit: the quantity of interest is diversity *given* the
    context, and every response in a set shares the context, so part of their mutual
    similarity is the prompt rather than the response. Unlike earlier attempts to
    remove a shared component, the direction removed here is known rather than
    estimated from the corpus itself.
    """

    def f(E: Array, ctx: Array | None = None) -> Array:
        if ctx is None or not np.any(ctx):
            return floored(z0)(E, None)
        c = ctx / max(float(np.linalg.norm(ctx)), 1e-12)
        X = E - strength * np.outer(E @ c, c)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.maximum(norms, 1e-12)
        Z = X @ X.T
        if z0 > 0:
            Z = (Z - z0) / (1.0 - z0)
        return _finish(Z)

    return f


# ------------------------------------------------------------------- composition


def compose_context_then(inner):
    """Project out the context, then apply another transform to the residuals."""

    def f(E: Array, ctx: Array | None = None) -> Array:
        if ctx is None or not np.any(ctx):
            return inner(E, None)
        c = ctx / max(float(np.linalg.norm(ctx)), 1e-12)
        X = E - np.outer(E @ c, c)
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        return inner(X, None)

    return f
