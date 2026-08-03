# Skewed Abundance

Every other benchmark here uses uniform abundance — which is precisely the regime
where a similarity-sensitive Hill number and a purely spectral index behave most
alike. This one varies abundance on purpose, because that is the axis where they
differ.

```bash
python run_study.py
```

## Construction

Eight themes, each a set of paraphrases of one proposition, so documents within a
theme are near-identical and documents across themes are unrelated. Five weight
profiles are laid over them, from uniform to near-degenerate, including a Zipfian
shape — the usual distribution of topic frequency in a real corpus.

**The corpus text is identical across all five profiles. Only the weights change.**
That is what isolates abundance from everything else.

| profile | shape |
|---|---|
| `uniform` | even coverage; the flat-profile control |
| `zipf` | 1, ½, ⅓, ¼ … the usual shape of a real corpus |
| `heavy_head` | one boilerplate response dominating, as in scraped or templated data |
| `long_tail` | two common cases and a tail of rare ones |
| `near_degenerate` | a single case is effectively the whole corpus |

## Result

| profile | true D₁ | weighted | uniform | Vendi |
|---|---:|---:|---:|---:|
| `uniform` | 8.000 | 4.753 | 4.753 | 12.541 |
| `zipf` | 6.147 | 3.886 | 4.753 | 12.541 |
| `long_tail` | 3.531 | 2.746 | 4.753 | 12.541 |
| `heavy_head` | 2.236 | 1.857 | 4.753 | 12.541 |
| `near_degenerate` | 1.238 | 1.288 | 4.753 | 12.541 |

The truth spans 1.24 to 8.00. Uniform abundance returns 4.753 for every profile and
**Vendi returns 12.541 for every profile** — both are blind to the distinction,
because the only thing that changed is a weight vector neither of them receives.

Rank recovery across the five profiles, at fixed *q*:

| q | ρ(weighted, true) | ρ(uniform, true) |
|---|---:|---|
| 0 | *n/a* | constant |
| 1 | **+1.000** | constant |
| 2 | **+1.000** | constant |
| ∞ | **+1.000** | constant |

Perfect ordering at every order that abundance affects. At q = 0 the true value is
itself constant at 8.000 — richness counts every theme that is present at all,
whatever its weight — so there is nothing to correlate, which is the correct
behaviour rather than a gap.

## Two honest qualifications

**The closed form is an idealisation.** Under a perfectly block-diagonal similarity
matrix the true diversity is exactly the Hill number of the weight vector. The actual
matrix is not that clean: within-theme similarity averages **0.746** rather than 1.0,
and across-theme **0.126** rather than 0. So measured values sit systematically below
the closed form — the ordering is recovered, the magnitude is compressed. That
compression is the encoder's, not the index's.

**An earlier revision of this analysis reported a meaningless statistic.** It
correlated the measured profile against the true profile *across q within a single
weighting*, and reported ρ = +1.000 for both weighted and uniform. Since D_q is
non-increasing in q by construction, any profile correlates +1 with any other — the
test could not fail. The comparison that carries information is across *profiles* at
fixed *q*, which is what the table above reports.

## What it establishes

Vendi beat this library's index on calibration and human agreement in
[`../vendi_comparison/`](../vendi_comparison/). Both of those benchmarks are
uniform-abundance. This is the complementary case, and here the spectral formulation
cannot compete at all — not because it scores worse, but because it has no way to
receive the input that distinguishes the corpora.

Neither result subsumes the other. The honest summary is that the two indices are
strongest in different regimes, and which one to reach for depends on whether the
abundance of your species is known and uneven.
