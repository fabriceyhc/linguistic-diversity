"""Cross-validation against the authors' own Vendi implementation.

What ``index="vendi"`` computes is not novel: it is the probability-weighted Vendi
Score, ``diag(sqrt p) K diag(sqrt p)``, introduced by Friedman & Dieng (TMLR 2023)
alongside the unweighted score, at the Renyi order q of Pasarkar & Dieng (2024).
Both are in the reference ``vendi-score`` package, so agreement with it is a real
external check rather than a self-consistency one.

Skipped when the package is absent so the suite still runs without it.
"""

from __future__ import annotations

import numpy as np
import pytest

from linguistic_diversity.metric import TextDiversity

vendi = pytest.importorskip("vendi_score.vendi", reason="pip install vendi-score to cross-validate")
calc = TextDiversity._calc_diversity
ORDERS = (0.0, 0.5, 1.0, 2.0, 3.0)


def gram(rng, n, dim=16):
    """A full-rank Gram matrix: PSD by construction, which the reference requires."""
    X = rng.random((n, dim))
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Z = np.clip(X @ X.T, 0.0, 1.0)
    np.fill_diagonal(Z, 1.0)
    return Z


class TestAgreesWithReference:
    @pytest.mark.parametrize("seed", range(8))
    def test_matches_across_orders_and_weights(self, seed):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(2, 8))
        Z = gram(rng, n)
        p = rng.random(n)
        p /= p.sum()
        for q in ORDERS:
            expected = float(vendi.score_K(Z, q=q, p=p))
            assert calc(p, Z, q, "vendi") == pytest.approx(expected, rel=1e-6, abs=1e-9)

    def test_matches_at_uniform_abundance(self):
        """p=None in the reference is the published unweighted Vendi Score."""
        rng = np.random.default_rng(42)
        Z = gram(rng, 6)
        for q in ORDERS:
            expected = float(vendi.score_K(Z, q=q, p=None))
            assert calc(np.full(6, 1 / 6), Z, q, "vendi") == pytest.approx(expected, rel=1e-6)

    def test_deliberate_difference_at_q0_on_rank_deficient_input(self):
        """At q=0 we report the numerical rank; the reference counts dust.

        Every eigenvalue contributes a whole unit at q=0, so an eigenvalue of
        1e-17 -- what eigendecomposition leaves behind on a rank-deficient matrix
        -- shifts the reference by 1. Two identical species are one species, and
        the rank says so.
        """
        Z = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        p = np.full(3, 1 / 3)
        assert calc(p, Z, 0.0, "vendi") == pytest.approx(2.0, abs=1e-6)
        assert calc(p, Z, 0.0, "hill") == pytest.approx(2.0, abs=1e-6)


class TestDiversityAxioms:
    """The properties that make an index an effective number at all.

    Leinster & Cobbold (2012) characterise their family by these; the Vendi paper
    does not prove them for the spectral form, so they are checked here for both.
    """

    @staticmethod
    def _pool(block, k):
        """k copies of one community, mutually dissimilar."""
        n = block.shape[0]
        Z = np.zeros((n * k, n * k))
        for i in range(k):
            Z[i * n : (i + 1) * n, i * n : (i + 1) * n] = block
        np.fill_diagonal(Z, 1.0)
        return Z

    @pytest.mark.parametrize("index", ["hill", "vendi"])
    @pytest.mark.parametrize("k", [2, 3, 5])
    def test_replication_principle(self, index, k):
        """Pooling k equally diverse, mutually dissimilar communities gives k x D.

        This is the axiom that makes the number *effective*: without it the score
        is an index but not a count of anything.
        """
        rng = np.random.default_rng(11)
        block = rng.random((4, 4)) * 0.6
        block = (block + block.T) / 2
        np.fill_diagonal(block, 1.0)

        single = calc(np.full(4, 0.25), block, 1.0, index)
        pooled = calc(np.full(4 * k, 1 / (4 * k)), self._pool(block, k), 1.0, index)
        assert pooled == pytest.approx(k * single, rel=1e-6)

    @pytest.mark.parametrize("index", ["hill", "vendi"])
    def test_absent_species_do_not_count(self, index):
        rng = np.random.default_rng(3)
        Z = gram(rng, 5)
        with_absent = calc(np.array([0.25, 0.25, 0.25, 0.25, 0.0]), Z, 1.0, index)
        without = calc(np.full(4, 0.25), Z[:4, :4], 1.0, index)
        assert with_absent == pytest.approx(without, rel=1e-6)

    @pytest.mark.parametrize("index", ["hill", "vendi"])
    @pytest.mark.parametrize("seed", range(6))
    def test_more_similarity_never_raises_diversity(self, index, seed):
        rng = np.random.default_rng(seed + 500)
        n = int(rng.integers(3, 7))
        Z = gram(rng, n)
        p = np.full(n, 1 / n)
        before = calc(p, Z, 1.0, index)
        raised = Z.copy()
        raised[0, 1] = raised[1, 0] = min(Z[0, 1] + 0.3, 1.0)
        assert calc(p, raised, 1.0, index) <= before + 1e-6

    @pytest.mark.parametrize("index", ["hill", "vendi"])
    @pytest.mark.parametrize("seed", range(6))
    def test_bounded_by_one_and_species_count(self, index, seed):
        rng = np.random.default_rng(seed + 700)
        n = int(rng.integers(2, 9))
        Z = gram(rng, n)
        p = rng.random(n)
        p /= p.sum()
        for q in ORDERS:
            assert 1.0 - 1e-6 <= calc(p, Z, q, index) <= n + 1e-6

    @pytest.mark.parametrize("index", ["hill", "vendi"])
    @pytest.mark.parametrize("seed", range(6))
    def test_non_increasing_in_q(self, index, seed):
        rng = np.random.default_rng(seed + 900)
        n = int(rng.integers(3, 8))
        Z = gram(rng, n)
        p = rng.random(n)
        p /= p.sum()
        values = [calc(p, Z, q, index) for q in ORDERS]
        for lower, higher in zip(values, values[1:], strict=False):
            assert lower >= higher - 1e-6, f"{index} rose with q: {values}"
