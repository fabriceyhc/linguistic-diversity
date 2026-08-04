# Ecology: evenness, coverage, and the alpha/beta/gamma partition

Three questions a single effective number cannot answer, each with a settled answer in
the ecology literature, each now in `linguistic_diversity.ecology`.

```bash
python run_ecology.py
```

The unit tests in `tests/test_ecology.py` establish that the implementations are
correct — including a brute-force simulation of Chao & Jost's rarefaction estimator.
This benchmark establishes that they say something *useful about text*.

## Q1 Evenness — separating "many things" from "balanced things"

Same 24 documents over 8 themes; only the abundance weights change.

| profile | D₁ | richness | E3 | E5 |
|---|---:|---:|---:|---:|
| uniform | 12.541 | 24.000 | 0.502 | 0.796 |
| zipf | 9.362 | 24.000 | 0.364 | 0.704 |
| long_tail | 5.415 | 24.000 | 0.192 | 0.532 |
| heavy_head | 3.544 | 24.000 | 0.111 | 0.398 |
| near_degenerate | 2.006 | 24.000 | 0.044 | 0.219 |

Richness is fixed by the documents. Diversity falls sixfold and **evenness is what
names the reason** — a single number cannot tell "fewer distinct things" from "the same
things, less balanced".

## Q2 Coverage — which levels repeat enough to be estimable

150 real generated sentences. Species are equivalence classes under `Z = 1`, i.e.
documents the metric cannot tell apart, which is the library's own identical-species
axiom rather than a new convention.

| metric | coverage | deficit |
|---|---:|---:|
| DocumentSemantics | 0.360 | 0.640 |
| Phonemic | 0.374 | 0.626 |
| DependencyParse | 0.407 | 0.593 |
| PartOfSpeechSequence | 0.407 | 0.593 |
| Rhythmic | 0.454 | 0.546 |
| ConstituencyParse | 0.501 | 0.499 |

The ordering matches the duplicate-rate study: coverage tracks **alphabet size against
sequence length**, not linguistic level. `ConstituencyParse` — four labels over short
skeletons — repeats most; `Phonemic` is discrete yet nearly unique.

Every level is above zero here only because this corpus contains genuinely repeated
generations. A corpus of distinct documents scores **exactly 0** at the semantic level,
and that is the honest answer: with every species seen once, the sample carries no
evidence about what it has missed.

## Q3 Partition — is diversity within sources or between them?

Two contrasting cases, `DocumentSemantics`, 60 documents per source.

**(a) Three genuinely different decTest generation tasks**

| q | gamma | alpha | beta |
|---|---:|---:|---:|
| 0 | 19.165 | 10.391 | 1.966 |
| 1 | 17.151 | 8.532 | **2.010** |
| 2 | 15.678 | 7.252 | 2.051 |

**(b) One task split arbitrarily three ways — same source**

| q | gamma | alpha | beta |
|---|---:|---:|---:|
| 0 | 7.022 | 5.625 | 1.256 |
| 1 | 6.475 | 5.089 | **1.272** |
| 2 | 6.149 | 4.771 | 1.289 |

`beta` is the effective number of **distinct** sources: 1 when interchangeable, N when
they share nothing. Three different tasks read as 2.0 distinct sources out of a possible
3 — they differ substantially but share a language and a generator. Three arbitrary
slices of one task read as 1.27, close to the floor of 1.

The contrast between (a) and (b) is the test, and it passes: the measure separates
"several genuinely different sources" from "one source cut into pieces", which no
unpartitioned diversity number can do.

## References

- Chao & Ricotta (2019), *Quantifying evenness*, Ecology 100(12) e02852.
- Chao & Jost (2012), *Coverage-based rarefaction and extrapolation*, Ecology 93, 2533.
- Reeve, Leinster, Cobbold et al. (2016), *How to partition diversity*, arXiv:1404.6520.
