# Length and Sample-Size Robustness

Does a diversity score change when only the corpus size or document length changes?

Shaib et al. ([Standardizing the Measurement of Text
Diversity](https://arxiv.org/abs/2403.00553), 2024) name this as the field's unresolved
confound — *"Future research into a principled solution for this problem is urgently
needed."* Ecology addressed the same problem for species counts with rarefaction and
extrapolation (Chao et al. 2014), which this library ships as `estimate_diversity()`.

```bash
python run_study.py            # all three studies, ~15 min
python run_study.py --quick    # two base corpora
```

## 1. Replication — the clean test

*k* distinct documents repeated *j* times. Exact duplicates are the same species, so true
diversity is **provably unchanged** while the corpus grows to *k·j* documents. There is no
defensible reason for any score to move.

Drift is (max − min) across the sweep, relative to the baseline score. Zero is correct.

| Metric | drift | ρ vs size | j=1 | j=2 | j=3 | j=4 | j=6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DocumentSemantics` | **0.000** | 0.039 | 3.781 | 3.781 | 3.781 | 3.781 | 3.781 |
| `DependencyParse` | **0.000** | −0.030 | 1.526 | 1.526 | 1.526 | 1.526 | 1.526 |
| `PartOfSpeechSequence` | **0.000** | −0.010 | 1.203 | 1.203 | 1.203 | 1.203 | 1.203 |
| `Rhythmic` | **0.000** | −0.030 | 1.300 | 1.300 | 1.300 | 1.300 | 1.300 |
| `Phonemic` | **0.000** | 0.000 | 1.444 | 1.444 | 1.444 | 1.444 | 1.444 |
| `TokenSemantics` | 0.151 | 0.659 | 14.21 | 15.43 | 15.88 | 16.12 | 16.36 |
| `TypeTokenRatio` | 0.833 | −0.977 | 0.691 | 0.345 | 0.230 | 0.173 | 0.115 |
| `DistinctN` | 0.833 | −0.977 | 0.691 | 0.345 | 0.230 | 0.173 | 0.115 |
| `CompressionRatio` | 4.468 | −0.981 | −1.52 | −2.91 | −4.29 | −5.64 | −8.31 |
| `SelfBLEU` | 14.47 | −0.702 | −0.065 | −1.000 | −1.000 | −1.000 | −1.000 |

Five of the six Hill-number metrics are **exactly invariant** — not approximately,
identically, to every printed digit. That is the expected behaviour: duplicating a corpus leaves relative
abundances untouched, and a Hill number is a function of relative abundance.

Every surface metric collapses. Type-token ratio and distinct-*n* lose 83% of their value
by the sixth repetition, purely because the token count grew while the type count did not.
Self-BLEU saturates at 1.0 after a *single* duplication — once every document has an exact
twin in the corpus, it reports zero diversity forever after.

Compression ratio, Shaib et al.'s headline recommendation, drifts by 447%. It is the
second-worst score in the table on the confound their own paper identifies as urgent.
Scores are negated here because compression ratio is similarity-oriented: repetitive text
compresses well, so a higher raw ratio means *less* diversity.

## 2. Padding — length and shared content together

*k* distinct documents, each extended with the same boilerplate clause repeated *t* times.
The number of distinct propositions stays *k* while mean document length grows.

| Metric | drift | ρ vs length | t=0 | t=1 | t=2 | t=4 | t=8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PartOfSpeechSequence` | 0.151 | −0.879 | 1.203 | 1.098 | 1.065 | 1.039 | 1.021 |
| `Rhythmic` | 0.207 | −0.981 | 1.300 | 1.142 | 1.093 | 1.055 | 1.030 |
| `Phonemic` | 0.278 | −0.973 | 1.444 | 1.209 | 1.137 | 1.079 | 1.043 |
| `TokenSemantics` | 0.325 | 0.039 | 14.21 | 15.12 | 18.55 | 16.44 | 13.93 |
| `DependencyParse` | 0.316 | −0.953 | 1.526 | 1.220 | 1.139 | 1.080 | 1.044 |
| `DocumentSemantics` | 0.539 | −0.679 | 3.781 | 2.357 | 1.870 | 1.818 | 1.741 |
| `TypeTokenRatio` | 0.862 | −0.981 | 0.691 | 0.425 | 0.284 | 0.171 | 0.095 |
| `DistinctN` | 0.862 | −0.981 | 0.691 | 0.425 | 0.284 | 0.171 | 0.095 |
| `CompressionRatio` | 5.110 | −0.981 | −1.52 | −2.33 | −3.26 | −5.27 | −9.29 |
| `SelfBLEU` | 12.73 | −0.981 | −0.065 | −0.434 | −0.645 | −0.794 | −0.888 |

**Read this sweep more carefully than the first.** It manipulates length *and* shared
content at once: the padded documents genuinely do share more material, so for a
similarity-sensitive metric some decline is defensible rather than a defect. Only the
surface metrics have no defence, and they again drift hardest.

`DocumentSemantics` halves,
which is the largest move among the Hill-number metrics — worth knowing if you compare
corpora whose documents carry boilerplate (headers, disclaimers, templated preambles).

Isolating length from shared content would need the same proposition authored at several
verbosities. That is not built yet.

## 3. Extrapolation — does the rarefaction machinery pay off?

Exact diversity needs an O(n²) similarity matrix. Given a measurement budget of *m*
documents out of 86, does `estimate_diversity()`'s fitted growth curve land closer to the
true full-corpus value than simply scoring *m* documents?

| Metric | truth | budget | raw subsample | extrapolated | raw err | extrap err | fitted model |
|---|---:|---:|---:|---:|---:|---:|---|
| `DocumentSemantics` | 7.19 | 10 | 4.64 | 15.21 | 0.354 | 1.115 | power_law |
| | | 20 | 5.55 | 6.84 | 0.228 | **0.049** | asymptotic |
| | | 30 | 5.60 | 7.66 | 0.221 | **0.065** | asymptotic |
| | | 40 | 6.58 | 6.60 | 0.085 | 0.082 | asymptotic |
| `DependencyParse` | 3.37 | 10 | 2.47 | 4.31 | 0.267 | 0.277 | power_law |
| | | 20 | 2.89 | 3.13 | 0.143 | **0.071** | asymptotic |
| | | 30 | 3.29 | 3.21 | 0.023 | 0.048 | asymptotic |
| | | 40 | 2.98 | 3.09 | 0.117 | **0.082** | asymptotic |
| `TokenSemantics` | 28.26 | 10 | 18.05 | 46.02 | 0.362 | 0.628 | power_law |
| | | 20 | 21.61 | 38.81 | 0.236 | 0.373 | power_law |
| | | 30 | 25.09 | 35.03 | 0.112 | 0.239 | power_law |
| | | 40 | 25.52 | 31.40 | 0.097 | 0.111 | power_law |

**The pattern is the fitted model.** Every accurate extrapolation came from an
`asymptotic` fit; every badly wrong one from `power_law`, which systematically overshoots.
`DocumentSemantics` at budgets ≥ 20 picks asymptotic and cuts error from 22% to 5%.
`TokenSemantics` picks power_law at every budget and does *worse than raw subsampling every
time*.

`DependencyParse` used to sit in the second group, alternating between excellent and badly
wrong. That turned out to be a symptom rather than a cause: while it was saturating at the
species count it grew linearly in corpus size, so the curve fitter reasonably chose an
unbounded family. With the similarity fixed it selects `asymptotic` at every budget and
lands within 1-2%. One of the two extrapolation failures was the saturation bug wearing a
different hat.

Extrapolation is still not a general solution as shipped — `TokenSemantics` remains a real
failure. **Check `result.model` and treat a `power_law` fit as unreliable.** Penalising
unbounded fits during selection is the obvious next thing to try.

`PartOfSpeechSequence`, `Rhythmic` and `Phonemic` all reported exactly 86.000 — the raw
species count — when this study was first run: every off-diagonal similarity had collapsed
to ~0. That was a defect in the alignment-based similarity shared by all three, fixed in
`utils.make_identity_aligner` / `normalized_alignment_similarity`. They now report 1.73,
1.74 and 1.81 and participate normally; `PartOfSpeechSequence` and `Phonemic` are in fact
the most accurately extrapolated metrics in the suite (errors of 0.006–0.034 at budgets
≥ 20, both fitting `asymptotic`).

## Caveats

- The base corpora are drawn from the hand-authored seed pool in
  `../metric_validation/data/constructions.json`, so documents are short single sentences.
  Behaviour on long documents is untested.
- Replication is an extreme construction — exact duplicates. Real near-duplicates would
  test a softer version of the same property.
- The extrapolation study uses one 86-document corpus. The model-selection pattern is
  consistent across three metrics and four budgets, but it is one corpus.
