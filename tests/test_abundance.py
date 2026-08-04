"""Abundance weighting and the diversity profile.

Every metric passed uniform abundance until v1.0.3, which left the one capability
a spectral index cannot match entirely unused. A similarity-sensitive Hill number
takes abundance as an explicit vector, so corpus frequencies, sampling weights and
duplicate counts can be expressed without materialising them as repeated rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from linguistic_diversity import DependencyParse, DocumentSemantics, TokenSemantics

CORPUS = [
    "A violent storm wrecked our village.",
    "A fierce gale devastated our village.",
    "She sold her bicycle on Tuesday.",
    "Neutron stars spin hundreds of times per second.",
]


@pytest.fixture(scope="module")
def metric():
    return DocumentSemantics({"verbose": False})


class TestAbundance:
    def test_default_is_unchanged(self, metric):
        """Omitting abundance must behave exactly as before."""
        assert metric(CORPUS) == pytest.approx(metric.diversity(CORPUS))

    def test_concentrating_abundance_lowers_diversity(self, metric):
        """One document holding almost the whole corpus is effectively one thing."""
        even = metric(CORPUS, abundance=[0.25, 0.25, 0.25, 0.25])
        skewed = metric(CORPUS, abundance=[0.97, 0.01, 0.01, 0.01])
        assert skewed < even
        assert skewed == pytest.approx(1.0, abs=0.2)

    def test_weights_need_not_be_normalised(self, metric):
        """Counts are the natural input, so they are normalised internally."""
        assert metric(CORPUS, abundance=[97, 1, 1, 1]) == pytest.approx(
            metric(CORPUS, abundance=[0.97, 0.01, 0.01, 0.01])
        )

    def test_deduplication_matches_materialised_duplicates(self, metric):
        """The whole point: same answer, far smaller matrix.

        A corpus of 20,000 documents over 500 distinct texts needs a 500x500
        matrix this way and a 20,000x20,000 matrix otherwise, and the
        eigendecomposition of the second is ~64,000x the work.
        """
        duplicated = [CORPUS[0]] * 30 + CORPUS[1:]
        assert metric(duplicated, deduplicate=True) == pytest.approx(metric(duplicated), rel=1e-6)

    def test_deduplication_shrinks_the_problem(self, metric):
        duplicated = [CORPUS[0]] * 30 + CORPUS[1:]
        _features, species = metric.extract_features(duplicated)
        assert len(species) == 33
        p, keep = metric._resolve_abundance(duplicated, species, None, True)
        assert len(p) == 4, "duplicates were not collapsed"
        assert keep is not None and len(keep) == 4
        assert p[0] == pytest.approx(30 / 33)

    def test_rejects_misaligned_abundance(self, metric):
        with pytest.raises(ValueError, match="one entry per document"):
            metric(CORPUS, abundance=[0.5, 0.5])

    def test_rejects_negative_abundance(self, metric):
        with pytest.raises(ValueError, match="non-negative"):
            metric(CORPUS, abundance=[-1.0, 1.0, 1.0, 1.0])

    def test_rejects_abundance_on_token_unit_metrics(self):
        """TokenSemantics species are tokens, so per-document weights cannot align."""
        token = TokenSemantics({"verbose": False})
        with pytest.raises(ValueError, match="species are not documents"):
            token(CORPUS, abundance=[0.25, 0.25, 0.25, 0.25])


class TestDiversityProfile:
    def test_profile_is_non_increasing_in_q(self, metric):
        """Hill numbers fall with q; the profile is that curve."""
        profile = metric.diversity_profile(CORPUS)
        values = [profile[q] for q in sorted(profile, key=lambda x: (np.isinf(x), x))]
        for lower, higher in zip(values, values[1:], strict=False):
            assert lower >= higher - 1e-6, f"profile rose with q: {profile}"

    def test_profile_agrees_with_diversity_at_each_q(self):
        """The shared similarity matrix must not change the answer."""
        for q in (0.0, 1.0, 2.0):
            metric = DocumentSemantics({"q": q, "verbose": False})
            assert metric.diversity_profile(CORPUS, q_values=[q])[q] == pytest.approx(
                metric(CORPUS), rel=1e-9
            )

    def test_skew_widens_the_profile(self, metric):
        """An even corpus gives a flat profile; a skewed one a steep drop."""
        even = metric.diversity_profile(CORPUS, abundance=[0.25] * 4)
        skewed = metric.diversity_profile(CORPUS, abundance=[0.97, 0.01, 0.01, 0.01])
        spread = lambda p: p[0.0] - p[float("inf")]  # noqa: E731
        assert spread(skewed) > spread(
            even
        ), "concentrating abundance did not widen the gap between richness and dominance"

    def test_profile_carries_abundance_and_deduplication(self, metric):
        duplicated = [CORPUS[0]] * 30 + CORPUS[1:]
        assert metric.diversity_profile(duplicated, deduplicate=True)[1.0] == pytest.approx(
            metric.diversity_profile(duplicated)[1.0], rel=1e-6
        )

    def test_empty_corpus_gives_an_empty_profile(self, metric):
        assert metric.diversity_profile([]) == {}

    def test_available_on_structural_metrics_too(self):
        profile = DependencyParse({"verbose": False}).diversity_profile(
            CORPUS, q_values=[0.0, 1.0, 2.0]
        )
        assert set(profile) == {0.0, 1.0, 2.0}
        assert all(v >= 1.0 for v in profile.values())
