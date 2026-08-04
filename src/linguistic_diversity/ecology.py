"""Ecological measures that a single diversity number leaves out.

An effective number answers "how many distinct things are there". Three questions it
cannot answer on its own, each with a settled answer in the ecology literature:

    evenness      Is the corpus diverse because it holds many things, or because
                  they are evenly balanced? Chao & Ricotta (2019).
    coverage      Corpus A has 10,000 documents and corpus B has 500. Are they
                  comparably *complete*, and if not, at what size are they?
                  Chao & Jost (2012).
    partition     Is the corpus diverse because each source is diverse, or because
                  the sources differ from one another? Reeve et al. (2016), which
                  is the similarity-sensitive continuation of the Leinster-Cobbold
                  measure this library is built on.

Everything here operates on abundance vectors and similarity matrices, so it composes
with either index. Formulas are transcribed from the reference implementations --
`iNEXT.4steps` for evenness, `rdiversity` for the partition -- and the tests check
against worked cases rather than against those packages, which are R.

References:
    Chao, A. & Ricotta, C. (2019). Quantifying evenness and linking it to diversity,
        beta diversity, and similarity. Ecology 100(12), e02852.
    Chao, A. & Jost, L. (2012). Coverage-based rarefaction and extrapolation:
        standardizing samples by completeness rather than size. Ecology 93, 2533-2547.
    Reeve, R., Leinster, T., Cobbold, C. et al. (2016). How to partition diversity.
        arXiv:1404.6520.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

__all__ = [
    "EVENNESS_CLASSES",
    "PartitionResult",
    "coverage_deficit",
    "evenness",
    "expected_coverage",
    "partition_diversity",
    "power_mean",
    "sample_coverage",
    "size_for_coverage",
]

Array = npt.NDArray[np.float64]

EVENNESS_CLASSES = ("E1", "E2", "E3", "E4", "E5")


# --------------------------------------------------------------------------- evenness


def evenness(
    diversity: float,
    richness: float,
    q: float = 1.0,
    measure: str = "E3",
) -> float:
    """Normalised evenness from a diversity value and a richness.

    A diversity of 3.0 means something different in a corpus of 4 documents than in a
    corpus of 400. Evenness divides that out: each measure is 1 when abundances are
    perfectly even and approaches 0 when one species holds essentially everything.

    The five classes are Chao & Ricotta (2019) Table 1. They differ in how they
    normalise, not in what they detect, and E3 -- the normalised slope of the diversity
    profile -- is the one those authors single out.

    Args:
        diversity: Hill number of order q. May be similarity-sensitive; see below.
        richness: Diversity at q = 0, i.e. the effective species count.
        q: The order the diversity was computed at. Must be > 0.
        measure: One of E1..E5.

    Returns:
        Evenness in [0, 1]. Returns 1.0 when richness is 1, since a single species is
        trivially even and every formula is 0/0 there.

    Note:
        Chao & Ricotta define these for classical Hill numbers, where richness is the
        raw species count. Passing similarity-sensitive values for both arguments is a
        generalisation of ours, not theirs: it stays in [0, 1] because D_q is
        non-increasing in q, so D_q <= D_0 either way, but the reading becomes "even
        across *distinct* content" rather than "even across species".
    """
    if measure not in EVENNESS_CLASSES:
        raise ValueError(f"measure must be one of {EVENNESS_CLASSES}, got {measure!r}")
    if q <= 0:
        raise ValueError(f"evenness is defined for q > 0, got q={q}")
    if diversity < 1.0 - 1e-9 or richness < 1.0 - 1e-9:
        raise ValueError(
            f"diversity and richness must be at least 1; got {diversity} and {richness}"
        )
    if richness <= 1.0 + 1e-12:
        return 1.0
    # Guard against a similarity-sensitive D_q that exceeds D_0 by floating-point dust.
    d = min(float(diversity), float(richness))
    s = float(richness)

    if measure == "E3":
        return float((d - 1.0) / (s - 1.0))
    if measure == "E4":
        return float((1.0 - 1.0 / d) / (1.0 - 1.0 / s))
    if measure == "E5" or abs(q - 1.0) < 1e-12:
        # E1 and E2 both tend to log(D)/log(S) as q -> 1.
        return float(np.log(d) / np.log(s))
    if measure == "E1":
        return float((1.0 - d ** (1.0 - q)) / (1.0 - s ** (1.0 - q)))
    return float((1.0 - d ** (q - 1.0)) / (1.0 - s ** (q - 1.0)))


# --------------------------------------------------------------------------- coverage


def _counts(abundance: npt.ArrayLike) -> Array:
    counts = np.asarray(abundance, dtype=np.float64).ravel()
    if np.any(counts < 0):
        raise ValueError("abundance counts must be non-negative")
    return counts[counts > 0]


def sample_coverage(abundance: npt.ArrayLike) -> float:
    """Estimated fraction of the population belonging to species we have seen.

    Chao & Jost (2012) eq. 4a, the Chao & Shen refinement of Good-Turing:

        C = 1 - (f1/n) * [(n-1) f1 / ((n-1) f1 + 2 f2)]

    where f1 and f2 are the numbers of species seen exactly once and exactly twice.
    Counts must be integers -- singletons and doubletons are the whole signal, and
    fractional weights do not have them.

    A corpus of entirely distinct documents has f1 = n, f2 = 0 and therefore coverage
    **0**. That is not a defect: with every species seen once, the sample carries no
    evidence about how much of the population remains unseen. Coverage is informative
    for the levels where features repeat -- `ConstituencyParse`, `Rhythmic`,
    `PartOfSpeechSequence`, token-level metrics -- and vacuous for distinct documents.
    """
    counts = _counts(abundance)
    if counts.size == 0:
        return 0.0
    if not np.allclose(counts, np.round(counts)):
        raise ValueError(
            "sample coverage needs integer counts: it is estimated from the number of "
            "species seen exactly once and exactly twice, which fractional weights do "
            "not define."
        )
    n = float(counts.sum())
    if n <= 1:
        return 0.0
    f1 = float(np.sum(np.round(counts) == 1))
    f2 = float(np.sum(np.round(counts) == 2))
    if f1 == 0:
        return 1.0
    denominator = (n - 1.0) * f1 + 2.0 * f2
    return float(1.0 - (f1 / n) * ((n - 1.0) * f1 / denominator))


def coverage_deficit(abundance: npt.ArrayLike) -> float:
    """1 - sample_coverage: the share of the population still unseen."""
    return 1.0 - sample_coverage(abundance)


def expected_coverage(abundance: npt.ArrayLike, m: int) -> float:
    """Expected coverage of a sub- or super-sample of size m.

    For m below the observed size this is the minimum-variance unbiased estimator,
    Chao & Jost eq. 4b:

        C_m = 1 - sum_i (X_i/n) * C(n - X_i, m) / C(n - 1, m)

    For m above it, the extrapolation of eq. 9a. Computed in log space, since the
    binomial coefficients overflow well before the corpus sizes this library targets.
    """
    counts = _counts(abundance)
    if counts.size == 0 or m <= 0:
        return 0.0
    n = float(counts.sum())
    if m >= n:
        # Extrapolate: coverage deficit decays geometrically in the extra sample.
        f1 = float(np.sum(np.round(counts) == 1))
        f2 = float(np.sum(np.round(counts) == 2))
        if f1 == 0:
            return 1.0
        ratio = (n - 1.0) * f1 / ((n - 1.0) * f1 + 2.0 * f2)
        return float(1.0 - (f1 / n) * ratio ** (m - n + 1.0))

    # Rarefaction. Terms with n - X_i < m contribute nothing.
    usable = counts[(n - counts) >= m]
    if usable.size == 0:
        return 1.0
    log_ratio = (
        gammaln(n - usable + 1.0) - gammaln(n - usable - m + 1.0) - (gammaln(n) - gammaln(n - m))
    )
    return float(1.0 - np.sum((usable / n) * np.exp(log_ratio)))


def size_for_coverage(abundance: npt.ArrayLike, coverage: float) -> int:
    """Smallest sample size reaching the target coverage.

    This is what makes two corpora comparable: rarefy each to the size at which it is
    equally *complete*, rather than to a common document count. Chao & Jost's central
    point is that equal-size comparison is biased against the more diverse corpus,
    because a size sufficient to characterise a dull corpus is too small for a rich one.
    """
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie strictly between 0 and 1, got {coverage}")
    counts = _counts(abundance)
    n = int(counts.sum())
    if n == 0:
        return 0
    lo, hi = 1, max(n, 2)
    # Expand until the target is reachable; extrapolated coverage tends to 1.
    while expected_coverage(counts, hi) < coverage:
        hi *= 2
        if hi > 10_000_000:
            raise ValueError(f"coverage {coverage} is not reachable from this sample")
    while lo < hi:
        mid = (lo + hi) // 2
        if expected_coverage(counts, mid) >= coverage:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


# -------------------------------------------------------------------------- partition


def power_mean(values: Array, order: float, weights: Array) -> float:
    """Weighted power mean of the given order, with zero-weight terms dropped.

    Order 0 is the geometric mean, and +/- infinity the weighted max and min. This is
    the operation every measure in Reeve et al. is built from: subcommunity values are
    power means over species, metacommunity values are power means over subcommunities.
    """
    w = np.asarray(weights, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64)
    mask = (w > 0) & np.isfinite(x)
    if not np.any(mask):
        return float("nan")
    w, x = w[mask], x[mask]
    w = w / w.sum()
    if np.isinf(order):
        return float(x.max() if order > 0 else x.min())
    if abs(order) < 1e-12:
        return float(np.exp(np.sum(w * np.log(x))))
    return float(np.sum(w * x**order) ** (1.0 / order))


@dataclass
class PartitionResult:
    """Similarity-sensitive alpha, beta and gamma for a partitioned corpus.

    Reeve et al. distinguish *raw* from *normalised* measures. Normalised measures
    control for subcommunity size and describe the subcommunity itself; raw measures
    describe the average contribution of one document in it. The normalised ones are
    what you usually want, and are what `beta` and `alpha` below report.
    """

    q: float
    gamma: float
    alpha: float
    beta: float
    raw_alpha: float
    raw_beta: float
    representativeness: float
    redundancy: float
    subcommunities: list[str] = field(default_factory=list)
    subcommunity_alpha: list[float] = field(default_factory=list)
    subcommunity_beta: list[float] = field(default_factory=list)
    subcommunity_representativeness: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "q": self.q,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta,
            "raw_alpha": self.raw_alpha,
            "raw_beta": self.raw_beta,
            "representativeness": self.representativeness,
            "redundancy": self.redundancy,
            "subcommunities": {
                name: {
                    "alpha": a,
                    "beta": b,
                    "representativeness": r,
                    "weight": w,
                }
                for name, a, b, r, w in zip(
                    self.subcommunities,
                    self.subcommunity_alpha,
                    self.subcommunity_beta,
                    self.subcommunity_representativeness,
                    self.weights,
                    strict=True,
                )
            },
        }

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"PartitionResult(q={self.q:g}, gamma={self.gamma:.4f}, "
            f"alpha={self.alpha:.4f}, beta={self.beta:.4f})"
        )


def partition_diversity(
    abundances: Array,
    Z: Array,
    q: float = 1.0,
    names: list[str] | None = None,
) -> PartitionResult:
    """Partition diversity into within- and between-subcommunity components.

    Args:
        abundances: S x N matrix. Entry (i, j) is the abundance of species i in
            subcommunity j, on any common scale -- it is normalised to sum to 1 over
            the whole matrix, and the column sums become the subcommunity weights.
        Z: S x S similarity matrix, unit diagonal.
        q: Order. Low q weights rare species, high q weights common ones.
        names: Optional subcommunity labels.

    Returns:
        A `PartitionResult`. The headline numbers:

        gamma  diversity of the pooled corpus. Identical to the ordinary
               similarity-sensitive Hill number of the pooled abundance vector.
        alpha  average diversity within a subcommunity.
        beta   effective number of *distinct* subcommunities: 1 when they are
               interchangeable, N when they share nothing and resemble nothing in
               each other.

    The partition is multiplicative in the sense Reeve et al. establish, but note it is
    not the naive `gamma = alpha * beta` at every q -- that identity holds exactly at
    q = 1 and is an approximation elsewhere, which is why all three are reported rather
    than two and a derivation.
    """
    P = np.asarray(abundances, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"abundances must be a 2-D species-by-subcommunity matrix, got {P.shape}")
    if np.any(P < 0):
        raise ValueError("abundances must be non-negative")
    total = P.sum()
    if total <= 0:
        raise ValueError("abundances sum to zero")
    P = P / total

    Zm = np.asarray(Z, dtype=np.float64)
    if Zm.shape != (P.shape[0], P.shape[0]):
        raise ValueError(f"Z must be {P.shape[0]}x{P.shape[0]}, got {Zm.shape}")

    n_sub = P.shape[1]
    if names is None:
        names = [f"subcommunity_{j}" for j in range(n_sub)]
    elif len(names) != n_sub:
        raise ValueError(f"expected {n_sub} names, got {len(names)}")

    w = P.sum(axis=0)  # subcommunity weights
    p = P.sum(axis=1)  # pooled abundance
    with np.errstate(divide="ignore", invalid="ignore"):
        type_weights = np.where(w > 0, P / np.where(w > 0, w, 1.0), 0.0)

    ZP = Zm @ P  # ordinariness, per subcommunity
    Zp = Zm @ p  # ordinariness, pooled
    ZP = np.where(ZP > 0, ZP, np.nan)
    Zp_safe = np.where(Zp > 0, Zp, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_alpha_i = 1.0 / ZP
        norm_alpha_i = w[None, :] / ZP
        raw_rho_i = Zp_safe[:, None] / ZP
        norm_rho_i = (Zp_safe[:, None] * w[None, :]) / ZP
        gamma_i = np.broadcast_to((1.0 / Zp_safe)[:, None], P.shape).copy()
    gamma_i[P <= 0] = np.nan  # a species absent from j contributes nothing there

    def per_sub(values: Array, order: float) -> list[float]:
        return [power_mean(values[:, j], order, type_weights[:, j]) for j in range(n_sub)]

    sub_gamma = per_sub(gamma_i, 1.0 - q)
    sub_alpha = per_sub(norm_alpha_i, 1.0 - q)
    sub_raw_alpha = per_sub(raw_alpha_i, 1.0 - q)
    sub_representativeness = per_sub(norm_rho_i, 1.0 - q)
    sub_redundancy = per_sub(raw_rho_i, 1.0 - q)
    # Beta is a relative entropy, so its power mean takes the conjugate order.
    sub_beta = per_sub(1.0 / norm_rho_i, q - 1.0)
    sub_raw_beta = per_sub(1.0 / raw_rho_i, q - 1.0)

    def meta(values: list[float]) -> float:
        return power_mean(np.asarray(values, dtype=np.float64), 1.0 - q, w)

    return PartitionResult(
        q=float(q),
        gamma=meta(sub_gamma),
        alpha=meta(sub_alpha),
        beta=meta(sub_beta),
        raw_alpha=meta(sub_raw_alpha),
        raw_beta=meta(sub_raw_beta),
        representativeness=meta(sub_representativeness),
        redundancy=meta(sub_redundancy),
        subcommunities=list(names),
        subcommunity_alpha=sub_alpha,
        subcommunity_beta=sub_beta,
        subcommunity_representativeness=sub_representativeness,
        weights=[float(x) for x in w],
    )
