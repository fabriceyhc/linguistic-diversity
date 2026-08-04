"""Evenness, sample coverage and the alpha/beta/gamma partition.

Formulas are transcribed from the reference implementations (`iNEXT.4steps`,
`rdiversity`), which are R, so these tests check them against worked cases, limiting
behaviour and -- for the rarefaction estimator -- a brute-force simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from linguistic_diversity.ecology import (
    EVENNESS_CLASSES,
    coverage_deficit,
    evenness,
    expected_coverage,
    partition_diversity,
    power_mean,
    sample_coverage,
    size_for_coverage,
)
from linguistic_diversity.metric import TextDiversity

# ------------------------------------------------------------------------- evenness


@pytest.mark.parametrize("measure", EVENNESS_CLASSES)
def test_perfect_evenness_is_one(measure: str) -> None:
    """When diversity equals richness every abundance is equal, so evenness is 1."""
    assert evenness(8.0, 8.0, q=1.0, measure=measure) == pytest.approx(1.0)
    assert evenness(8.0, 8.0, q=2.0, measure=measure) == pytest.approx(1.0)


@pytest.mark.parametrize("measure", EVENNESS_CLASSES)
def test_maximal_unevenness_approaches_zero(measure: str) -> None:
    """One species holding essentially everything drives every class toward 0."""
    assert evenness(1.0 + 1e-9, 1000.0, q=2.0, measure=measure) < 1e-3


@pytest.mark.parametrize("measure", EVENNESS_CLASSES)
@pytest.mark.parametrize("q", [0.5, 1.0, 2.0, 4.0])
def test_evenness_is_bounded(measure: str, q: float) -> None:
    rng = np.random.default_rng(11)
    for _ in range(200):
        richness = float(rng.integers(2, 60))
        diversity = float(rng.uniform(1.0, richness))
        assert 0.0 <= evenness(diversity, richness, q=q, measure=measure) <= 1.0


def test_evenness_worked_values() -> None:
    """Chao & Ricotta Table 1, evaluated by hand at D=3, S=5."""
    assert evenness(3.0, 5.0, q=2.0, measure="E3") == pytest.approx((3 - 1) / (5 - 1))
    assert evenness(3.0, 5.0, q=2.0, measure="E4") == pytest.approx((1 - 1 / 3) / (1 - 1 / 5))
    assert evenness(3.0, 5.0, q=2.0, measure="E5") == pytest.approx(np.log(3) / np.log(5))
    assert evenness(3.0, 5.0, q=2.0, measure="E1") == pytest.approx((1 - 3.0**-1) / (1 - 5.0**-1))
    assert evenness(3.0, 5.0, q=2.0, measure="E2") == pytest.approx((1 - 3.0**1) / (1 - 5.0**1))


def test_e1_and_e2_converge_at_q_one() -> None:
    """Both classes tend to log(D)/log(S) as q -> 1; the code special-cases it."""
    expected = evenness(3.0, 5.0, q=1.0, measure="E5")
    assert evenness(3.0, 5.0, q=1.0, measure="E1") == pytest.approx(expected)
    assert evenness(3.0, 5.0, q=1.0, measure="E2") == pytest.approx(expected)
    near = evenness(3.0, 5.0, q=1.0001, measure="E1")
    assert near == pytest.approx(expected, abs=1e-3)


def test_single_species_is_trivially_even() -> None:
    assert evenness(1.0, 1.0, q=1.0) == 1.0


def test_evenness_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="q > 0"):
        evenness(2.0, 4.0, q=0.0)
    with pytest.raises(ValueError, match="one of"):
        evenness(2.0, 4.0, measure="E9")
    with pytest.raises(ValueError, match="at least 1"):
        evenness(0.5, 4.0)


# ------------------------------------------------------------------------- coverage


def test_coverage_worked_example() -> None:
    """counts [5,3,2,1,1]: n=12, f1=2, f2=1, so C = 1 - (2/12)(22/24)."""
    counts = [5, 3, 2, 1, 1]
    expected = 1 - (2 / 12) * ((11 * 2) / ((11 * 2) + 2 * 1))
    assert sample_coverage(counts) == pytest.approx(expected)
    assert coverage_deficit(counts) == pytest.approx(1 - expected)


def test_all_singletons_gives_zero_coverage() -> None:
    """The normal case for distinct documents: nothing repeats, so nothing is known."""
    assert sample_coverage([1] * 40) == pytest.approx(0.0)


def test_no_singletons_gives_full_coverage() -> None:
    assert sample_coverage([4, 6, 10]) == pytest.approx(1.0)


def test_coverage_rejects_fractional_counts() -> None:
    with pytest.raises(ValueError, match="integer counts"):
        sample_coverage([1.5, 2.5])


def test_expected_coverage_matches_estimator_at_full_size() -> None:
    """Extrapolating by zero must reproduce eq. 4a exactly."""
    counts = [7, 5, 3, 2, 1, 1, 1]
    n = int(sum(counts))
    assert expected_coverage(counts, n) == pytest.approx(sample_coverage(counts))


def test_expected_coverage_is_monotone_and_bounded() -> None:
    counts = [12, 8, 5, 3, 2, 2, 1, 1, 1]
    values = [expected_coverage(counts, m) for m in range(1, 80)]
    # Deliberately ragged: pairs of consecutive values, so strict is False.
    assert all(b >= a - 1e-12 for a, b in zip(values, values[1:], strict=False))
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values[-1] > values[0]


def test_rarefied_coverage_matches_simulation() -> None:
    """Chao & Jost eq. 4b against the unbiased resampling algorithm they describe.

    Draw m+1 individuals without replacement; one minus the expected proportion of
    singletons in that subsample equals C_m.
    """
    counts = np.array([9, 6, 4, 3, 2, 2, 1, 1, 1, 1])
    population = np.repeat(np.arange(len(counts)), counts)
    rng = np.random.default_rng(4)
    for m in (5, 10, 15):
        singleton_share = []
        for _ in range(6000):
            draw = rng.permutation(population)[: m + 1]
            _, c = np.unique(draw, return_counts=True)
            singleton_share.append(np.sum(c == 1) / (m + 1))
        simulated = 1.0 - float(np.mean(singleton_share))
        assert expected_coverage(counts, m) == pytest.approx(simulated, abs=0.01)


def test_size_for_coverage_round_trips() -> None:
    counts = [20, 14, 9, 6, 4, 3, 2, 2, 1, 1, 1]
    for target in (0.5, 0.8, 0.95):
        m = size_for_coverage(counts, target)
        assert expected_coverage(counts, m) >= target
        if m > 1:
            assert expected_coverage(counts, m - 1) < target


def test_size_for_coverage_rejects_impossible_targets() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        size_for_coverage([3, 2, 1], 1.0)


def test_coverage_standardisation_beats_size_standardisation() -> None:
    """The point of the method: equal size misranks, equal coverage does not.

    A rich assemblage and a dull one, sampled to the same *size*, are not equally
    complete -- the rich one is always further from its own asymptote.
    """
    rich = [1] * 60 + [2] * 20
    dull = [40, 30, 20, 10]
    assert sample_coverage(dull) > sample_coverage(rich)
    n = min(int(sum(rich)), int(sum(dull)))
    assert expected_coverage(dull, n) > expected_coverage(rich, n)


# ------------------------------------------------------------------------ power mean


def test_power_mean_special_cases() -> None:
    x = np.array([1.0, 2.0, 4.0])
    w = np.full(3, 1 / 3)
    assert power_mean(x, 1.0, w) == pytest.approx(7 / 3)
    assert power_mean(x, 0.0, w) == pytest.approx(2.0)  # geometric
    assert power_mean(x, -1.0, w) == pytest.approx(3 / (1 / 1 + 1 / 2 + 1 / 4))
    assert power_mean(x, np.inf, w) == pytest.approx(4.0)
    assert power_mean(x, -np.inf, w) == pytest.approx(1.0)


def test_power_mean_ignores_zero_weights() -> None:
    x = np.array([1.0, 2.0, np.nan])
    w = np.array([0.5, 0.5, 0.0])
    assert power_mean(x, 1.0, w) == pytest.approx(1.5)


# ------------------------------------------------------------------------ partition


def _blocks(n_sub: int, per_sub: int, within: float, between: float) -> np.ndarray:
    """Block-structured similarity: `within` inside a block, `between` across."""
    n = n_sub * per_sub
    Z = np.full((n, n), between)
    for j in range(n_sub):
        s = slice(j * per_sub, (j + 1) * per_sub)
        Z[s, s] = within
    np.fill_diagonal(Z, 1.0)
    return Z


@pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 2.0])
def test_gamma_equals_pooled_hill_number(q: float) -> None:
    """Gamma must be exactly the ordinary similarity-sensitive diversity."""
    rng = np.random.default_rng(3)
    P = rng.random((7, 3))
    Z = _blocks(1, 7, 0.4, 0.4)
    result = partition_diversity(P, Z, q=q)
    p = P.sum(axis=1) / P.sum()
    expected = TextDiversity._calc_diversity(p, Z, q=q, index="hill")
    assert result.gamma == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("q", [0.0, 1.0, 2.0])
def test_identical_subcommunities_have_beta_one(q: float) -> None:
    """Duplicating a community adds no distinctiveness."""
    column = np.array([0.5, 0.3, 0.2])
    P = np.column_stack([column, column, column])
    Z = np.eye(3)
    assert partition_diversity(P, Z, q=q).beta == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("n_sub", [2, 3, 5])
def test_disjoint_dissimilar_subcommunities_give_beta_n(n_sub: int) -> None:
    """N sources sharing nothing and resembling nothing in each other read as N."""
    per_sub = 4
    P = np.zeros((n_sub * per_sub, n_sub))
    for j in range(n_sub):
        P[j * per_sub : (j + 1) * per_sub, j] = 1.0
    Z = _blocks(n_sub, per_sub, within=1.0, between=0.0)
    result = partition_diversity(P, Z, q=1.0)
    assert result.beta == pytest.approx(float(n_sub), rel=1e-9)
    # Each source collapses to one effective item, so the pool holds exactly n_sub.
    assert result.alpha == pytest.approx(1.0, rel=1e-9)
    assert result.gamma == pytest.approx(float(n_sub), rel=1e-9)


@pytest.mark.parametrize("q", [0.0, 0.5, 1.0, 2.0, 4.0])
def test_beta_is_bounded_by_the_number_of_subcommunities(q: float) -> None:
    rng = np.random.default_rng(19)
    for _ in range(60):
        n_sub = int(rng.integers(2, 5))
        P = rng.random((9, n_sub)) * (rng.random((9, n_sub)) > 0.3)
        if np.any(P.sum(axis=0) == 0):
            continue
        A = rng.random((9, 9)) * 0.7
        Z = (A + A.T) / 2
        np.fill_diagonal(Z, 1.0)
        result = partition_diversity(P, Z, q=q)
        assert 1.0 - 1e-9 <= result.beta <= n_sub + 1e-9
        assert result.alpha <= result.gamma + 1e-9


def test_beta_rises_as_sources_diverge() -> None:
    """Beta must be monotone in how unlike the sources are."""
    P = np.zeros((8, 2))
    P[:4, 0] = 1.0
    P[4:, 1] = 1.0
    # Cross-block similarity falls, so the two sources become more distinct.
    betas = [
        partition_diversity(P, _blocks(2, 4, within=1.0, between=b), q=1.0).beta
        for b in (0.9, 0.6, 0.3, 0.0)
    ]
    assert all(a < b for a, b in zip(betas, betas[1:], strict=False)), betas
    assert betas[0] == pytest.approx(1.0, abs=0.15)  # nearly interchangeable
    assert betas[-1] == pytest.approx(2.0, rel=1e-9)  # fully distinct


def test_partition_reports_per_subcommunity_values() -> None:
    P = np.zeros((6, 2))
    P[:3, 0] = 1.0
    P[3:, 1] = 1.0
    result = partition_diversity(P, _blocks(2, 3, 0.5, 0.05), q=1.0, names=["a", "b"])
    assert result.subcommunities == ["a", "b"]
    assert len(result.subcommunity_alpha) == 2
    assert result.weights == pytest.approx([0.5, 0.5])
    assert set(result.to_dict()["subcommunities"]) == {"a", "b"}


def test_partition_validates_shapes() -> None:
    with pytest.raises(ValueError, match="2-D"):
        partition_diversity(np.ones(4), np.eye(4))
    with pytest.raises(ValueError, match="Z must be"):
        partition_diversity(np.ones((4, 2)), np.eye(3))
    with pytest.raises(ValueError, match="non-negative"):
        partition_diversity(-np.ones((4, 2)), np.eye(4))
    with pytest.raises(ValueError, match="expected 2 names"):
        partition_diversity(np.ones((4, 2)), np.eye(4), names=["only-one"])


# ------------------------------------------------------- integration with a metric


@pytest.mark.slow
def test_metric_evenness_separates_from_diversity() -> None:
    """Two corpora with the same distinct content but different balance.

    Diversity falls when abundance concentrates; evenness is what names *why*.
    """
    from linguistic_diversity import DocumentSemantics

    metric = DocumentSemantics({"verbose": False})
    corpus = [
        "The stock market closed higher on Tuesday.",
        "She baked a loaf of sourdough bread.",
        "The telescope detected a distant galaxy.",
        "He repaired the bicycle's rear brake.",
    ]
    even = metric.evenness(corpus, q=1.0)
    skewed = metric.evenness(corpus, q=1.0, abundance=[97.0, 1.0, 1.0, 1.0])
    assert 0.0 <= skewed < even <= 1.0
    assert even > 0.9


@pytest.mark.slow
def test_metric_sample_coverage_reflects_feature_repetition() -> None:
    """Coverage is vacuous for distinct documents and informative where features repeat."""
    from linguistic_diversity import DocumentSemantics, PartOfSpeechSequence

    corpus = [
        "The chef prepared the meal.",
        "The driver parked the car.",
        "The teacher graded the exam.",
        "A distant galaxy drifted quietly beyond every charted boundary.",
    ]
    # Embeddings are unique per document, so every species is a singleton.
    assert DocumentSemantics({"verbose": False}).sample_coverage(corpus) == pytest.approx(0.0)
    # The first three share a POS skeleton, so that level has something to cover.
    assert PartOfSpeechSequence({"verbose": False}).sample_coverage(corpus) > 0.0


@pytest.mark.slow
def test_metric_partition_recovers_distinct_sources() -> None:
    """Two topically separate sources should read as ~2 distinct subcommunities."""
    from linguistic_diversity import DocumentSemantics

    metric = DocumentSemantics({"verbose": False})
    result = metric.partition(
        {
            "finance": [
                "The stock market closed higher on Tuesday.",
                "Investors bought bank shares in early trading.",
                "Bond yields fell after the central bank spoke.",
            ],
            "baking": [
                "She baked a loaf of sourdough bread.",
                "The dough rose overnight in the warm kitchen.",
                "He kneaded the rye starter before dawn.",
            ],
        },
        q=1.0,
    )
    assert 1.5 < result.beta <= 2.0
    assert result.alpha <= result.gamma
    assert result.subcommunities == ["finance", "baking"]


@pytest.mark.slow
def test_metric_partition_sees_through_shared_wording() -> None:
    """Sources with no shared text but the same content are *not* distinct."""
    from linguistic_diversity import DocumentSemantics

    metric = DocumentSemantics({"verbose": False})
    result = metric.partition(
        [
            ["The cat sat on the mat.", "A feline rested upon the rug."],
            ["The mat was sat on by the cat.", "Upon the rug, a cat reclined."],
        ],
        q=1.0,
    )
    assert result.beta < 1.4


@pytest.mark.slow
def test_metric_partition_rejects_non_document_species() -> None:
    from linguistic_diversity import TokenSemantics

    with pytest.raises(ValueError, match="species are not documents"):
        TokenSemantics({"verbose": False}).partition([["one two three"], ["four five six"]])


@pytest.mark.slow
def test_metric_partition_requires_two_subcommunities() -> None:
    from linguistic_diversity import DocumentSemantics

    with pytest.raises(ValueError, match="at least two"):
        DocumentSemantics({"verbose": False}).partition([["only one source"]])
