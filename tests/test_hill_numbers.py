"""Closed-form checks on the Hill number itself.

Every other test in this suite exercises a metric end to end, which means each
one assumes this formula is right. These check it directly, against cases where
the answer can be derived by hand, with no text and no model involved.

    D_q = (sum_i p_i (Zp)_i^(q-1))^(1/(1-q))

References: Leinster & Cobbold (2012), Ecology 93(3); Chao et al. (2014).
"""

from __future__ import annotations

import numpy as np
import pytest

from linguistic_diversity.metric import TextDiversity

calc = TextDiversity._calc_diversity


def uniform(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n, dtype=np.float64)


class TestClosedForms:
    """Cases with a hand-derivable answer."""

    @pytest.mark.parametrize("n", [2, 3, 5, 8, 20])
    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 2.0, 3.0])
    def test_distinct_species_give_the_species_count(self, n, q):
        """Z = I, uniform abundance: D_q = n at every order.

        (Zp)_i = 1/n, so D_q = (n . (1/n) . (1/n)^(q-1))^(1/(1-q)) = n.
        This is the ceiling every metric is measured against.
        """
        assert calc(uniform(n), np.eye(n), q) == pytest.approx(n, rel=1e-6)

    @pytest.mark.parametrize("n", [2, 4, 10])
    @pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 2.0])
    def test_identical_species_give_one(self, n, q):
        """Z = all ones: every species is the same species, so D_q = 1.

        (Zp)_i = 1 for all i, so the sum is 1 whatever q is.
        """
        assert calc(uniform(n), np.ones((n, n)), q) == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("z", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_two_species_closed_form(self, z):
        """Two species with similarity z and equal abundance: D_1 = 2/(1+z).

        p = [1/2, 1/2] gives Zp = [(1+z)/2, (1+z)/2], so
        D_1 = exp(-sum p_i ln (Zp)_i) = 2/(1+z).
        Interpolates the two cases above: z=0 -> 2, z=1 -> 1.
        """
        Z = np.array([[1.0, z], [z, 1.0]])
        assert calc(uniform(2), Z, 1.0) == pytest.approx(2.0 / (1.0 + z), rel=1e-5)

    @pytest.mark.parametrize("z", [0.0, 0.3, 0.6, 1.0])
    @pytest.mark.parametrize("q", [0.0, 2.0, 3.0])
    def test_two_species_closed_form_general_q(self, z, q):
        """The same pair at other orders is also 2/(1+z).

        With equal abundance both (Zp)_i are equal, so the order parameter
        cancels: D_q = ((Zp)^(q-1))^(1/(1-q)) = 1/(Zp) = 2/(1+z).
        A q-dependence here would mean the general branch disagrees with the
        q=1 branch.
        """
        Z = np.array([[1.0, z], [z, 1.0]])
        assert calc(uniform(2), Z, q) == pytest.approx(2.0 / (1.0 + z), rel=1e-5)

    def test_berger_parker_limit(self):
        """q = infinity is 1/max(Zp), the Berger-Parker index."""
        Z = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.1], [0.0, 0.1, 1.0]])
        p = np.array([0.6, 0.3, 0.1])
        assert calc(p, Z, float("inf")) == pytest.approx(1.0 / (Z @ p).max(), rel=1e-9)


class TestOrderingLaws:
    """Properties that must hold for any valid Z and p."""

    @pytest.mark.parametrize("seed", range(12))
    def test_non_increasing_in_q(self, seed):
        """Hill numbers are non-increasing in the order q.

        Higher q weights abundant species more heavily, so it cannot report more
        effective species than a lower order does.
        """
        rng = np.random.default_rng(seed)
        n = int(rng.integers(3, 9))
        A = rng.random((n, n))
        Z = (A + A.T) / 2
        np.fill_diagonal(Z, 1.0)
        p = rng.random(n)
        p /= p.sum()

        values = [calc(p, Z, q) for q in (0.0, 0.5, 1.0, 2.0, 3.0)]
        # Consecutive pairs, so the second argument is one shorter by design.
        for lower, higher in zip(values, values[1:], strict=False):
            assert lower >= higher - 1e-6, f"diversity rose with q: {values}"

    @pytest.mark.parametrize("seed", range(12))
    def test_more_similarity_never_means_more_diversity(self, seed):
        """Raising an off-diagonal entry cannot increase diversity.

        Making two species more alike can only reduce the effective number of
        them. This is the relation a sign error in the formula would break, and
        nothing else in the suite tests it.
        """
        rng = np.random.default_rng(seed + 100)
        n = int(rng.integers(3, 8))
        A = rng.random((n, n)) * 0.5
        Z = (A + A.T) / 2
        np.fill_diagonal(Z, 1.0)
        p = uniform(n)
        before = calc(p, Z, 1.0)

        i, j = 0, 1
        raised = Z.copy()
        raised[i, j] = raised[j, i] = min(Z[i, j] + 0.4, 1.0)
        after = calc(p, raised, 1.0)

        assert after <= before + 1e-6, (
            f"raising Z[{i},{j}] from {Z[i, j]:.3f} to {raised[i, j]:.3f} "
            f"raised diversity {before:.6f} -> {after:.6f}"
        )

    @pytest.mark.parametrize("seed", range(10))
    def test_bounded_by_one_and_species_count(self, seed):
        """1 <= D_q <= n for any similarity matrix and abundance."""
        rng = np.random.default_rng(seed + 200)
        n = int(rng.integers(2, 10))
        A = rng.random((n, n))
        Z = (A + A.T) / 2
        np.fill_diagonal(Z, 1.0)
        p = rng.random(n)
        p /= p.sum()

        for q in (0.0, 1.0, 2.0):
            d = calc(p, Z, q)
            assert 1.0 - 1e-6 <= d <= n + 1e-6, f"D_{q} = {d} outside [1, {n}]"

    @pytest.mark.parametrize("seed", range(8))
    def test_invariant_to_species_relabelling(self, seed):
        """Permuting species leaves diversity unchanged."""
        rng = np.random.default_rng(seed + 300)
        n = int(rng.integers(3, 9))
        A = rng.random((n, n))
        Z = (A + A.T) / 2
        np.fill_diagonal(Z, 1.0)
        p = rng.random(n)
        p /= p.sum()

        order = rng.permutation(n)
        assert calc(p, Z, 1.0) == pytest.approx(
            calc(p[order], Z[np.ix_(order, order)], 1.0), rel=1e-9
        )


class TestNumericalEdges:
    """Where the implementation could quietly disagree with the mathematics."""

    def test_q_just_above_one_approaches_the_q1_branch(self):
        """The general branch must converge on the q=1 special case.

        q=1 is a removable singularity handled by a separate code path, so the
        two branches can drift apart without anything failing loudly.
        """
        rng = np.random.default_rng(7)
        n = 6
        A = rng.random((n, n))
        Z = (A + A.T) / 2
        np.fill_diagonal(Z, 1.0)
        p = rng.random(n)
        p /= p.sum()

        exact = calc(p, Z, 1.0)
        near = calc(p, Z, 1.0 + 1e-6)
        assert near == pytest.approx(
            exact, rel=1e-3
        ), f"q=1 branch gives {exact}, q=1+1e-6 gives {near}"

    def test_near_zero_similarity_does_not_blow_up(self):
        """A very small (Zp)_i must not produce inf or nan.

        The q=1 path adds 1e-10 inside the log to avoid log(0); that guard is
        load-bearing and worth pinning.
        """
        n = 4
        Z = np.eye(n) * 1.0
        p = uniform(n)
        d = calc(p, Z, 1.0)
        assert np.isfinite(d), f"non-finite diversity {d}"
        assert d == pytest.approx(n, rel=1e-3)


class TestMaximumDiversity:
    """Leinster & Meckes (2016): max over p of D_q is independent of q.

    That makes it a property of the similarity structure alone, and the correct
    denominator for a relative measure -- the species count is a ceiling only
    reachable when every species is mutually dissimilar.
    """

    @staticmethod
    def _max(Z):
        from linguistic_diversity.utils import maximum_diversity

        return maximum_diversity(np.asarray(Z, dtype=float))

    @pytest.mark.parametrize("n", [2, 3, 5, 8])
    def test_distinct_species_ceiling_is_the_species_count(self, n):
        """Only when Z = I does the species count become reachable."""
        value, p = self._max(np.eye(n))
        assert value == pytest.approx(n, rel=1e-6)
        assert p == pytest.approx(np.full(n, 1.0 / n), abs=1e-6)

    @pytest.mark.parametrize("n", [2, 4, 7])
    def test_identical_species_ceiling_is_one(self, n):
        value, _ = self._max(np.ones((n, n)))
        assert value == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.parametrize("z", [0.0, 0.25, 0.5, 0.9])
    def test_two_species_ceiling_matches_the_closed_form(self, z):
        """For a symmetric pair, uniform abundance is already optimal: 2/(1+z)."""
        value, _ = self._max([[1.0, z], [z, 1.0]])
        assert value == pytest.approx(2.0 / (1.0 + z), rel=1e-6)

    @pytest.mark.parametrize("seed", range(10))
    def test_maximum_is_independent_of_q(self, seed):
        """The theorem itself: one ceiling serves every order."""
        rng = np.random.default_rng(seed + 900)
        n = int(rng.integers(3, 8))
        A = rng.random((n, n)) * 0.7
        Z = (A + A.T) / 2
        np.fill_diagonal(Z, 1.0)

        value, p = self._max(Z)
        for q in (0.0, 0.5, 1.0, 2.0, 3.0):
            assert calc(p, Z, q) == pytest.approx(
                value, rel=1e-5
            ), f"D_{q} at the optimising p is {calc(p, Z, q)}, not {value}"

    @pytest.mark.parametrize("seed", range(10))
    def test_no_abundance_beats_the_maximum(self, seed):
        rng = np.random.default_rng(seed + 950)
        n = int(rng.integers(3, 7))
        A = rng.random((n, n)) * 0.8
        Z = (A + A.T) / 2
        np.fill_diagonal(Z, 1.0)
        ceiling, _ = self._max(Z)

        for _ in range(60):
            p = rng.random(n)
            p /= p.sum()
            assert calc(p, Z, 1.0) <= ceiling + 1e-6

    def test_ceiling_never_exceeds_the_species_count(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            n = int(rng.integers(2, 9))
            A = rng.random((n, n))
            Z = (A + A.T) / 2
            np.fill_diagonal(Z, 1.0)
            value, _ = self._max(Z)
            assert 1.0 - 1e-6 <= value <= n + 1e-6
