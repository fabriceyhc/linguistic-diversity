"""The selectable diversity index.

Two indices consume the same similarity matrix. They agree at both extremes --
Z = I gives n, Z all-ones gives 1 -- and differ in between, because a uniform
baseline similarity contributes a rank-one component that the spectral form
confines to one eigenvalue while the Hill form spreads through every (Zp)_i.
"""

from __future__ import annotations

import numpy as np
import pytest

from linguistic_diversity import DocumentSemantics
from linguistic_diversity.metric import TextDiversity, _nearest_psd

calc = TextDiversity._calc_diversity
CORPUS = [
    "The tall boy kicked the ball.",
    "When the rain stopped, the children played.",
    "She believes that the plan is sound.",
    "There was a crack in the ceiling.",
]


class TestIndexAgreement:
    @pytest.mark.parametrize("n", [2, 4, 7])
    def test_both_return_n_for_distinct_species(self, n):
        p = np.full(n, 1.0 / n)
        for index in ("hill", "vendi"):
            assert calc(p, np.eye(n), 1.0, index) == pytest.approx(n, rel=1e-6)

    @pytest.mark.parametrize("n", [3, 5])
    def test_both_return_one_for_identical_species(self, n):
        p = np.full(n, 1.0 / n)
        for index in ("hill", "vendi"):
            assert calc(p, np.ones((n, n)), 1.0, index) == pytest.approx(1.0, rel=1e-6)

    def test_vendi_reduces_to_hill_when_z_is_identity(self):
        """With no similarity to redistribute, the spectrum lands on the abundance."""
        p = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
        assert calc(p, np.eye(5), 1.0, "vendi") == pytest.approx(
            calc(p, np.eye(5), 1.0, "hill"), rel=1e-6
        )

    def test_vendi_resists_a_similarity_floor(self):
        """The reason to offer it: Hill degrades toward 1/z, the spectral form does not."""
        n, z = 50, 0.3
        Z = np.full((n, n), z)
        np.fill_diagonal(Z, 1.0)
        p = np.full(n, 1.0 / n)
        hill, vendi = calc(p, Z, 1.0, "hill"), calc(p, Z, 1.0, "vendi")
        assert hill < 5, f"Hill should collapse under a 0.3 floor, got {hill}"
        assert vendi > 20, f"Vendi should resist it, got {vendi}"

    def test_rejects_an_unknown_index(self):
        with pytest.raises(ValueError, match="'hill' or 'vendi'"):
            calc(np.full(3, 1 / 3), np.eye(3), 1.0, "spectral")


class TestNearestPSD:
    """Alignment and tree-edit similarities are not kernels, so this is load-bearing."""

    def test_projection_makes_a_matrix_psd(self):
        Z = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.9], [0.0, 0.9, 1.0]])
        assert np.linalg.eigvalsh(Z).min() < 0, "fixture is already PSD"
        P = _nearest_psd(Z)
        assert np.linalg.eigvalsh(P).min() >= -1e-9
        assert np.allclose(np.diag(P), 1.0)
        assert P.min() >= 0.0 and P.max() <= 1.0

    def test_projection_leaves_a_psd_matrix_alone(self):
        rng = np.random.default_rng(0)
        X = rng.random((6, 12))
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        Z = np.clip(X @ X.T, 0, 1)
        np.fill_diagonal(Z, 1.0)
        assert np.allclose(_nearest_psd(Z), Z, atol=1e-8)

    def test_vendi_is_defined_on_a_non_psd_matrix(self):
        """Without the projection this raises; the Hill number never needed it."""
        Z = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.9], [0.0, 0.9, 1.0]])
        value = calc(np.full(3, 1 / 3), Z, 1.0, "vendi")
        assert np.isfinite(value) and 1.0 <= value <= 3.0


class TestVendiThroughTheMetric:
    @pytest.fixture(scope="class")
    def vendi(self):
        return DocumentSemantics({"index": "vendi", "verbose": False})

    def test_default_index_is_vendi(self):
        """Default from v1.1.0.

        Better on both criteria at every level measured -- rank agreement against
        known ground truth and calibration ratio -- and on human agreement, while
        preserving discriminant behaviour and every metamorphic law. "hill" stays
        available for large corpora, where the O(n^3) eigendecomposition costs,
        and for the exact Leinster-Cobbold quantity.
        """
        assert DocumentSemantics({}).config.index == "vendi"

    def test_identical_corpus_is_one(self, vendi):
        assert vendi(["One sentence."] * 5) == pytest.approx(1.0, abs=1e-3)

    def test_replication_invariant(self, vendi):
        assert vendi(CORPUS) == pytest.approx(vendi(CORPUS * 3), rel=1e-4)

    def test_permutation_invariant(self, vendi):
        assert vendi(CORPUS) == pytest.approx(vendi(list(reversed(CORPUS))), rel=1e-9)

    def test_honours_abundance(self, vendi):
        """The published Vendi Score cannot do this; the weighted form can."""
        skewed = vendi(CORPUS, abundance=[0.97, 0.01, 0.01, 0.01])
        assert skewed < vendi(CORPUS)
        assert skewed == pytest.approx(1.0, abs=0.3)

    def test_profile_is_non_increasing_in_q(self, vendi):
        profile = vendi.diversity_profile(CORPUS, q_values=[0.0, 1.0, 2.0])
        values = [profile[q] for q in (0.0, 1.0, 2.0)]
        for lower, higher in zip(values, values[1:], strict=False):
            assert lower >= higher - 1e-6
