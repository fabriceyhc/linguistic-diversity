# Research log

Findings, decisions and deferred work that do not belong in a commit message or a
benchmark README — the reasoning behind choices, the things measured and set aside,
and the claims that turned out to be wrong.

The paper-level plan lives in
[linguistic-diversity-experiments/RESEARCH_PLAN.md](https://github.com/fabriceyhc/linguistic-diversity-experiments).
This file is the engineering and measurement record.

Newest first.

---

## 2026-08-04 — Same kernel for everyone: the index contributes nothing here

`run_same_kernel.py`. Every comparison so far gave our configuration a cross-encoder
and the alternatives bi-encoder cosine, which confounds index with representation.
This holds the matrix fixed.

| index | McDiv | conTest | decTest | mean | calib ρ | ratio |
|---|---:|---:|---:|---:|---:|---:|
| Hill (Leinster–Cobbold) | +0.8126 | +0.7495 | +0.8134 | 0.7918 | +0.9721 | 0.9716 |
| Vendi Score (reference impl) | +0.8406 | +0.7788 | +0.8333 | **0.8176** | +0.9746 | 0.9998 |
| pVS_q (this library) | +0.8406 | +0.7788 | +0.8333 | **0.8176** | +0.9746 | 0.9998 |

**max |pVS_q − Vendi| = 4.5e-14.** They are the same number. At uniform abundance
`diag(√p) Z diag(√p) = Z/n`, so pVS_q *is* the Vendi Score, and on uniformly-weighted
benchmarks the two cannot differ by construction. Every gap previously reported between
"ours" and "Vendi" was a difference of **kernel**, not of index.

So the honest accounting on these benchmarks: the entire improvement is representation.
The index contributes zero, and the one index that does differ — the Leinster–Cobbold
Hill number the library is named for — is *worse* by 0.026.

**Two places the index does contribute, neither visible in these benchmarks:**

1. **Abundance.** Same matrices, Zipfian weights instead of uniform: ours moves to 4.178
   mean, the spectral form stays at 5.143 because it has no way to accept weights. Mean
   absolute divergence 0.971. Every human-rating benchmark here is uniformly weighted,
   so none of them can see this.
2. **Non-kernel matrices.** **21/679** cross-encoder matrices are not positive
   semi-definite (down to −9e-4), and the published Vendi Score raises on them — it is
   *undefined*, not merely inaccurate. The reference implementation only appears in the
   table above because it was handed the PSD projection first.

**Baselines that cannot participate at all**, which is a finding rather than an omission:
distinct-n, Self-BLEU and Decan have no notion of a similarity matrix, so "same kernel"
is undefined for them. PRDC needs a metric space and hundreds of samples; via classical
MDS on 1 − Z it degenerates on five-item sets and returns nan.

**Consequence for the paper.** The claim cannot be "a better diversity index". It has to
be one of: a better *representation* for diversity measurement (the cross-encoder
result), the *abundance* generalisation, or the surrounding apparatus — partition,
evenness, coverage, non-PSD handling, validated axioms.

---

## 2026-08-04 — The NLI kernel worked by accident; four alternatives measured

`run_nli_models.py`. Five cross-encoders, each on a four-pair diagnostic probe and on
the real criteria.

**Better NLI models do fix the labelling defect.** Probability assigned to *neutral* for
an unrelated pair, where neutral is the correct label:

| model | P(neutral), unrelated | argmax |
|---|---:|---|
| `cross-encoder/nli-deberta-v3-base` *(shipped)* | 0.0002 | **contradiction** |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | **0.959** | neutral |
| `MoritzLaurer/DeBERTa-v3-large-…-ling-wanli` | **0.932** | neutral |
| `tasksource/deberta-base-long-nli` | 0.143 | contradiction |

**But swapping the model alone makes the kernel worse, because the formula was fitted to
the defect.** `Z = 0.5(1 + e − c)` sends unrelated text to 0 *only if* unrelated is
labelled contradiction. Give it a correctly-calibrated model and unrelated lands on
neutral, so `e ≈ 0.01`, `c ≈ 0.03`, and **Z ≈ 0.49** — a 0.49 similarity floor, which is
precisely the pathology the NLI kernel was introduced to remove. Calibration ratio falls
0.979 → 0.836.

**The fix is to match the formula to the model** (150-set subsample, NLI kernel alone):

| model | formula | mean ρ | calib ratio |
|---|---|---:|---:|
| shipped (mislabels) | ent−contra | +0.7121 | 0.9787 |
| shipped (mislabels) | entailment only | +0.6636 | 1.0000 |
| MoritzLaurer large | ent−contra | +0.6905 | **0.8364** |
| **MoritzLaurer large** | **entailment only** | **+0.7156** | **0.9999** |

Each model needs the formula that maps *its* "unrelated" verdict to zero. Net accuracy
between the two correct pairings is a wash. So the case for switching is not accuracy —
it is that the shipped pairing works by an artefact of SNLI/MNLI and would break
unpredictably wherever the contradiction class absorbs something different.

**The better answer is to stop using NLI at all.** `cross-encoder/stsb-roberta-large` is
trained directly for *graded* similarity, which is what a similarity matrix wants; using
a 3-way classifier to approximate one was the wrong tool. On the full held-out split:

| configuration | McDiv | conTest | decTest | mean | calib ratio |
|---|---:|---:|---:|---:|---:|
| NLI-ec + ctx + LS + τ *(previous best)* | +0.8319 | +0.7973 | +0.8441 | **0.8245** | 1.0000 |
| **STS alone** | +0.8406 | +0.7778 | +0.8308 | 0.8164 | 0.9998 |
| STS + ctx + LS + τ | +0.8175 | +0.7812 | +0.8437 | 0.8141 | 1.0000 |

0.008 behind the four-component pipeline — inside the bootstrap CI, and almost exactly
the +0.018 that context projection was worth, which is the component STS does without.
**One model, one matrix, and no prompt required**, against NLI + context projection +
local scaling + τ. Blending it with the embedding kernel *hurts*, so the simplification
is real rather than a trade.

That also disposes of the fairness objection to the headline number: the prompt was the
input the baselines never received, and this configuration does not use it.

**Recommendations.** Keep NLI only with `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-
ling-wanli` *and* the entailment-only formula — never one without the other. Prefer
`cross-encoder/stsb-roberta-large` alone. Neither is shipped yet; both are opt-in kernels
at O(n²) cross-encoder cost.

---

## 2026-08-03 — What the NLI kernel is actually doing (and a retracted mechanism)

Two things I had been running together under the word "prompt". They are independent and
only one of them is an extra input.

**The cross-encoder needs no query.** An NLI cross-encoder consumes a *sentence pair*,
premise and hypothesis — not a query and a document. The pairs are just the documents in
the set: for n documents, all C(n,2) pairs scored in both orderings, n(n−1) forward
passes, symmetrised. Nothing external is required, which is why it can be a drop-in
kernel for `DocumentSemantics`.

Model: **`cross-encoder/nli-deberta-v3-base`**, 184M parameters, 3-way softmax over
`{contradiction, entailment, neutral}` (id2label 0/1/2).

**Context projection is the separate thing, and it *is* an extra input.** It embeds the
dataset's `context` column and projects that direction out of each response. Worth
+0.018 of the +0.117 total, and unavailable whenever there is no prompt — which is most
real use.

**Retraction.** I documented the kernel as working because "unrelated sentences are
neutral, and neutral means entailment near zero". The conclusion holds; the mechanism is
wrong. Measured:

| pair | entail | contra | Z (ent-contra) |
|---|---:|---:|---:|
| paraphrase | 0.993 | 0.000 | **0.994** |
| same topic, not entailed | 0.001 | 0.000 | **0.500** |
| different topic | 0.000 | 1.000 | **0.000** |
| contradiction | 0.000 | 1.000 | **0.000** |

Unrelated pairs are classified as **contradiction with p ≈ 1.0**, not neutral. That is an
artefact of MNLI-style training, where the neutral class means "same topic, not entailed"
rather than "unrelated". Consequences worth being explicit about:

1. The floor-free property is genuine — entailment for unrelated text is 0.000, so
   nothing accumulates. That part of the claim survives.
2. `ent-contra` is really a **three-level topical scale** (paraphrase 1.0 / same-topic
   0.5 / off-topic 0.0), which is a sensible similarity structure and probably explains
   why it beat raw entailment.
3. It **cannot distinguish an off-topic sentence from a contradictory one**. Defensible
   for diversity, since both are maximally dissimilar, but it is not a property of NLI —
   it is a property of this training data.

So the kernel should be described as domain-dependent, not universal. Anything that
shifts what MNLI's contradiction class absorbs will move the scores, and nothing in the
benchmarks tests for that.

---

## 2026-08-03 — The lead is real, significant, and mostly a bigger model

`run_significance.py`. 2000 paired bootstrap resamples over held-out sets, same
resample indices for every configuration.

| configuration | mean ρ | 95% CI |
|---|---:|---|
| Self-BLEU | 0.5714 | [0.5149, 0.6213] |
| Vendi (raw cosine) | 0.7078 | [0.6630, 0.7444] |
| ours, current default | 0.7092 | [0.6641, 0.7457] |
| ours, no prompt no NLI | 0.7320 | [0.6884, 0.7658] |
| ours, no prompt (+NLI) | 0.8062 | [0.7744, 0.8299] |
| **ours, full** | **0.8245** | [0.7954, 0.8454] |

Every pairwise gap excludes zero. But the ablation is the finding, and it is not
flattering:

| against Vendi | Δ | 95% CI |
|---|---:|---|
| transform only (no prompt, no NLI) | **+0.024** | [+0.009, +0.039] |
| + NLI cross-encoder | **+0.099** | [+0.074, +0.125] |
| + prompt | +0.117 | [+0.088, +0.148] |

**Roughly 85% of the lead is the NLI cross-encoder.** Strip it and the similarity
transforms beat Vendi by 0.024 — real, but a rounding error next to the headline. So
what has been demonstrated is *not* a better diversity index. It is that a cross-encoder
makes a better similarity function than a bi-encoder, which is unsurprising, costs 45× at
n = 40, and would presumably help every competing metric equally if they were given one.

A further 0.018 comes from the prompt, which the baselines are not given at all. Any
comparison table that keeps both advantages and calls the result state of the art is
comparing a bigger model with more inputs against smaller models with fewer.

**What this does license**, stated narrowly: on these three datasets, at these sizes,
with a cross-encoder kernel and prompt conditioning, the similarity-sensitive index
tracks human diversity judgement better than the published alternatives measured the same
way, and the gap is not noise. That is a much smaller claim than "best metric" and it is
the one the evidence supports.

---

## 2026-08-03 — "We win small and lose large" was mostly my measurement error

`benchmarks/similarity_transforms/run_scale.py`. Prompted by the obvious objection that
large-scale application is where a diversity metric earns its keep.

**Retraction first.** The claim that Vendi beats us at distribution level is wrong. Two
errors compounded: I compared *different tasks* (per-set human ratings at n=5 against
temperature bins at n=200) and I carried the small-set-tuned kernel to a setting two of
its three components do not apply in. Measured properly, within one task across sizes:

| kernel | n=5 | n=25 | n=100 | n=200 | n=400 |
|---|---:|---:|---:|---:|---:|
| raw cosine (= Vendi) | +0.649 | +0.952 | +0.927 | +0.976 | +0.988 |
| **τ=0.70 only** | +0.682 | +0.976 | +0.952 | **+1.000** | **+1.000** |
| local scaling k=10 | +0.515 | +0.879 | +0.721 | +0.818 | +0.733 |
| LS + τ *(small-set winner)* | +0.576 | +0.939 | +0.927 | +0.952 | +0.927 |

**With the right kernel we beat Vendi at every size**, including the largest. The
distribution-level table now reports both kernels rather than the one that flattered the
wrong conclusion.

**What actually fails to transfer, and it is not scale as such:**

- **Context projection** — structurally unavailable. It removes a *shared prompt*, and a
  pooled corpus has no single prompt. This is the biggest single small-set correction
  (+0.105 on McDiv), and it is simply not defined at scale.
- **Local scaling** — a *task* effect, not a size effect. It helps human ratings at every
  size and hurts temperature at every size. It was selected on human ratings, so
  carrying it elsewhere was the mistake.
- **NLI kernel** — a genuine scale barrier, but of cost, not accuracy: 40,000
  cross-encoder pairs per 200-document bin, quadratic thereafter.
- **τ-truncation** — transfers cleanly, and is the component doing the work.

**The real scale problem is the opposite of the one I claimed, and it is about the
absolute number rather than the ranking.** Saturation, mean D / n:

| floor | n=5 | n=50 | n=200 | n=400 |
|---|---:|---:|---:|---:|
| z₀ = 0 | 0.884 | 0.621 | 0.422 | **0.306** |
| z₀ = 0.053 (shipped) | 0.901 | 0.674 | 0.491 | 0.374 |
| z₀ = 0.30 (small-set best) | 0.950 | 0.867 | 0.849 | **0.829** |

At n = 400 the aggressive floor reports **332 effective documents out of 400 generated
sentences**. That is not a credible effective count, and the effective count is the whole
selling point. Meanwhile ranking is nearly *insensitive* to the floor — every value from
0 to 0.5 scores ≥ 0.95 at n ≥ 25.

So the floor should be chosen on the calibration criterion, not the ranking one, and the
choice is size-dependent: **z₀ = 0.30 is a five-item-set constant that must not be
shipped as a global default.** This follows from the motivation itself — the cap
`D_max = n/(1+(n−1)z)` is a function of n, so its correction should be too. Worth noting
that the uniform-z model also *overstates* the problem at scale: it predicts a cap near
1/z ≈ 14 at z = 0.07, but real heterogeneous Z gives 122 at n = 400.

**Latent defect found and fixed: `_nearest_psd` un-did its own projection.** It projected
onto the PSD cone, restored the unit diagonal, and then clipped off-diagonal entries back
into [0, 1] — a second unconstrained perturbation straight back out of the cone, whose
damage `_calc_vendi` then silently absorbed by clipping negative eigenvalues. On a
rank-deficient input this is severe: 200 unit vectors in 32 dimensions went from rank 32
to rank 200, and the score from 29.5 to **85.0**.

After removing the clip, our index reproduces the reference `vendi-score` **exactly** at
uniform abundance for every size and rank tested — the identity pVS(uniform) = VS, which
was silently violated before. Impact on shipped results: **none measurable**. The library
clamps cosine to [0, 1] and every transform ends in [0, 1], so the projection rarely
produces negatives. On real text with genuinely non-PSD matrices (`DependencyParse` min
eigenvalue −0.22, 0.47% negative mass) the change is ≤ 0.06%, and on the validation
corpora it is exactly zero. Every previously reported number stands.

---

## 2026-08-03 — Against the published alternatives

`benchmarks/similarity_transforms/run_competitors.py`. Decan and PRDC added to
`baselines.py`, then everything scored on the same fixed subsample as the hybrid table.

**Per response set (5–10 documents):**

| metric | McDiv | conTest | decTest | mean | calib ρ | ratio |
|---|---:|---:|---:|---:|---:|---:|
| distinct-n / TTR | +0.389 | +0.489 | +0.815 | 0.564 | +0.530 | — |
| Self-BLEU | +0.418 | +0.525 | +0.819 | 0.587 | +0.542 | — |
| Decan (LM surprise) | +0.440 | +0.504 | +0.708 | 0.550 | +0.598 | — |
| PRDC coverage | +0.207 | +0.283 | +0.366 | 0.285 | +0.805 | — |
| Vendi | +0.576 | +0.670 | +0.811 | 0.686 | +0.974 | 0.973 |
| ours, current default | +0.576 | +0.670 | +0.813 | 0.686 | +0.974 | 0.986 |
| **ours, best kernel** | **+0.808** | **+0.811** | **+0.819** | **0.812** | **+0.975** | **1.000** |

The gap that matters for the paper is not against the lexical baselines — it is that
**calibration separates the families completely**. Every surface metric sits at ρ ≈ 0.53
against a known concept count while the similarity-sensitive ones sit at ρ ≈ 0.97. A
Self-BLEU of 0.4 is not an answer to "how many distinct ideas are in here"; distinct-n
and TTR are also literally identical to four decimal places on all three datasets,
which is worth a footnote.

**Two results I would have got wrong by assuming.**

*PRDC is not weak; it was being asked the wrong question.* On five-item sets its recall
is 0 for **every** set — correlation undefined, not poor. Run where it was designed to
run (10 temperature bins, ≤200 documents each, held-out reference) it recovers
temperature at **+0.927**. Reporting the per-set number alone would have been a rigged
comparison.

*Decan is genuinely different, and mid-table.* No embedder, no similarity matrix, no
reference corpus — diversity read off GPT-2's per-token log-probabilities. It lands
between the lexical metrics and the similarity-sensitive ones (0.550 per-set, +0.721
distribution-level). Sanity check on constructed sets confirms the reimplementation
behaves as the paper describes, including that the coherence term keeps noise (0.171)
below genuinely diverse text (0.555) — without it, gibberish reads as maximally diverse.

**Distribution level (10 temperature bins, ≤200 documents each):**

| metric | ρ vs temperature |
|---|---:|
| Vendi | **+0.988** |
| ours, best kernel *(no NLI half)* | +0.952 |
| PRDC recall | +0.927 |
| Decan | +0.721 |
| PRDC coverage | −0.049 |
| PRDC density | **−0.891** |

Two honest notes. **Vendi beats us here**, on raw cosine at n=200 and uniform abundance
— the regime where a spectral index is strongest and where our kernel is running without
its NLI component, because 40,000 cross-encoder pairs per bin is exactly the practical
limit the cost table predicted. And PRDC density going *strongly negative* is correct
behaviour, not a bug: density is the fidelity half, and raising temperature lowers
fidelity. That it comes out at −0.891 is the best evidence the PRDC implementation is
sound.

**Fixed while running this:** `vendi_score` was being handed a clipped cosine matrix,
which is not PSD at n=200 and which it rightly refuses. It now gets the raw Gram matrix,
PSD by construction. The clip was harmless at n=5 and would have gone unnoticed.

---

## 2026-08-03 — Evenness, coverage and the alpha/beta/gamma partition

`src/linguistic_diversity/ecology.py`, `benchmarks/ecology/`. The three remaining items
from the literature review — capability additions rather than score improvements, since
none of them touches the agreement or calibration numbers.

Formulas were transcribed from the reference implementations rather than reconstructed:
`iNEXT.4steps` (R) for the evenness classes, `rdiversity` (R) for the partition, the
Chao & Jost paper text for the coverage estimators. Worth the effort — three of them are
easy to get subtly wrong, and the rarefaction estimator is now checked against a
brute-force simulation of the resampling algorithm those authors describe.

**Evenness** (Chao & Ricotta 2019, E1–E5). Same 24 documents, only the weights change:

| profile | D₁ | richness | E3 |
|---|---:|---:|---:|
| uniform | 12.541 | 24.000 | 0.502 |
| zipf | 9.362 | 24.000 | 0.364 |
| near_degenerate | 2.006 | 24.000 | 0.044 |

Richness is fixed by the documents, so evenness is exactly the part diversity conflates:
"fewer distinct things" versus "the same things, less balanced". Note the generalisation
is ours — Chao & Ricotta define these over classical Hill numbers, and passing
similarity-sensitive values for both terms changes the reading to "even across *distinct
content*". It stays in [0,1] because D_q is non-increasing in q either way.

**Coverage** (Chao & Jost 2012). The interesting design question was what counts as a
species. Documents are the wrong answer — `PartOfSpeechSequence` returns raw documents as
species, so counting duplicates there measures nothing about POS. The right answer is
**equivalence classes under Z = 1**, which is the library's own identical-species axiom
rather than a new convention, and makes the count correctly metric-relative.

| metric | coverage |
|---|---:|
| DocumentSemantics | 0.360 |
| Phonemic | 0.374 |
| DependencyParse / PartOfSpeechSequence | 0.407 |
| Rhythmic | 0.454 |
| ConstituencyParse | **0.501** |

Ordering reproduces the duplicate-rate study exactly: alphabet size against sequence
length, not linguistic level. **Coverage is 0 for a corpus of distinct documents**, and
that is the honest answer rather than a defect — with every species seen once there is no
evidence about what was missed. The measure is informative for the colliding levels and
vacuous for semantics on distinct text, which is worth saying plainly in the docs so
nobody reads 0.0 as a bug.

**Partition** (Reeve et al. 2016). The similarity-sensitive continuation of the
Leinster–Cobbold measure this library already computes, so it is the natural extension
rather than a bolt-on. Verified: gamma equals the ordinary similarity-sensitive Hill
number of the pooled abundance vector, to 1e-9, at every q.

| corpus | beta (q=1) |
|---|---:|
| three different decTest generation tasks | **2.010** |
| one task split arbitrarily three ways | **1.272** |

That contrast is the test. Three genuinely different sources read as 2.0 distinct out of
a possible 3 — they differ substantially but share a language and a generator; three
arbitrary slices of one source read near the floor of 1. No unpartitioned number can
make that distinction.

**One thing to keep in mind for the paper.** Reeve et al.'s partition is *not*
`gamma = alpha × beta` at every q; that identity is exact at q = 1 and approximate
elsewhere. All three are reported rather than two and a derivation.

---

## 2026-08-03 — Three corrections, each attacking a different failure, and they compose

`benchmarks/similarity_transforms/`. Every angle from the literature review that could
move a score, tested on identical cached embeddings so the only difference between rows
is the transform. ~35 configurations.

**Headline, on sets no configuration was selected on:**

| | McDiv | conTest | decTest | mean | calib ratio | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| current default | +0.635 | +0.663 | +0.829 | 0.709 | 0.986 | 2.33 |
| **NLI-ec + ctx + LS + τ (geometric)** | **+0.832** | **+0.797** | **+0.844** | **0.824** | **1.000** | **3.08** |

**+0.115 mean agreement, and the calibration ratio reaches 1.000 rather than trading
against it.** As a fraction of the annotator noise ceiling, on this same split:
McDiv 65→85%, conTest 71→85%, decTest 86→87%. (The 60/70/81% quoted in the entry below
are the full-dataset figures; these are the held-out split, so they differ slightly.)

### The three wins are independent because they fix different things

| correction | fixes | effect alone |
|---|---|---|
| **context projection** | shared prompt inflates every pair | McDiv +0.596 → **+0.701** |
| **local scaling** (hubness, k=10) | a few documents are near-neighbours of everything | McDiv +0.596 → **+0.641** |
| **τ-truncation** at τ=0.70 | the residual similarity floor | calib ratio 0.986 → **1.000**, ceiling 2.32 → 3.28 |
| **NLI entailment kernel** | cosine is not calibrated at all | McDiv +0.596 → **+0.808** |

Context projection removes the *known* context direction rather than one estimated from
the corpus, which is why it succeeds where common-component removal failed. It is
Conditional-Vendi in spirit and it needs an API that accepts the prompt.

**τ-truncation is not a new transform.** Chao's `d_ij(τ) = min(τ, d_ij)` with the linear
similarity `1 − d_ij(τ)/τ` is algebraically *identical* to this library's floor with
z₀ = 1 − τ. What the ecology literature contributes is the default: τ should be far more
aggressive than the median-of-unrelated-pairs value we shipped. Sweeping it, 0.053 →
0.30 gains on both criteria at once. Their own τ = d_mean recommendation (0.917 here)
is close to what we already had and is *not* where the optimum sits.

### The entailment kernel is the single biggest lever

Unrelated sentences are *neutral*, and neutral means entailment near zero — so there is
no floor to remove. It beats every transform of the cosine matrix, which is the point:
three years of corrections applied to an uncalibrated matrix, against one calibrated
matrix. `P(entail) − P(contradict)` beats raw `P(entail)` on agreement (+0.808 vs
+0.776 on McDiv) but has a much worse ceiling (2.05 vs 3.88).

**Cost is the catch.** O(n²) cross-encoder passes: 1.6× a cosine encode at n=5, **45× at
n=40**, quadratic after that. This has to be opt-in, not the default.

### Axioms hold, and replication is *better* than the current default

| kernel | k=2 | k=3 | k=5 | D(5 identical) | monotonicity |
|---|---:|---:|---:|---:|---:|
| cosine + floor 0.053 | 1.903 | 2.752 | 4.613 | 1.0000 | ok |
| ctx + LS + τ | 1.986 | 2.913 | 4.911 | 1.0000 | ok |
| hybrid geometric | 1.974 | 2.903 | 4.923 | 1.0057 | 0/200 |

The current default is the *furthest* from exact replication, because its residual floor
accumulates with corpus size — exactly the mechanism the floor was introduced to fix.
Correcting it properly improves the axiom as a side effect.

### Rejected here

- **Mutual proximity.** Agreement +0.658, ratio 0.633, ceiling 1.11. Over-corrects.
- **Global ZCA whitening.** Confirmed again: best ceiling of any linear fix, worst
  agreement. Fourth member of the flatten-the-dominant-directions family.
- **Order parameter.** q = 0.5 is marginally better than q = 1 on agreement (+0.6995 vs
  +0.6968) and clearly worse on calibration rank (0.941 vs 0.975). q = 1 stays.
  Note vendi at q = 0 returns the numerical rank, which is constant at n for
  general-position embeddings, so agreement is undefined — expected, not a defect.

### Caveat

The winner was chosen after ~35 configurations against these datasets, so the selection
split is optimistic by construction. The table above is the held-out complement, which
is why it is the one quoted; the two agree to within 0.01 on ranking. What is *not*
established is generalisation beyond these three datasets and one encoder.

---

## 2026-08-03 — The human ceiling is high, so the remaining gap is ours

Before hunting for better indices, it is worth knowing how much of the agreement gap
is annotator noise. Tevet & Berant collected **10 ratings per set** (§6.2), and ship
`metric_abs_hds_std` alongside the mean, so the reliability of the averaged label can
be computed directly:

```
Var(true) = Var(observed mean) − E[σ²_within] / k       ceiling ρ = √(Var(true)/Var(obs))
```

| dataset | reliability | ceiling ρ | ours | % of ceiling |
|---|---:|---:|---:|---:|
| McDiv_nuggets | 0.951 | 0.975 | 0.581 | **60%** |
| conTest | 0.870 | 0.933 | 0.650 | **70%** |
| decTest | 0.932 | 0.966 | 0.778 | **81%** |

**The labels are not the bottleneck.** Averaging ten ratings gives a very reliable
target, and attenuation correction buys almost nothing. There is 20–40% real headroom
and no excuse available.

Context from the paper worth carrying: on conTest, *absHDS itself* correlates
0.70–0.84 with the binary content-diversity parameter, and the best automatic metric
they tested (sent-BERT) reaches 0.60–0.75. On decTest, plain distinct-n reaches
0.86–0.91 against temperature — which is why `TypeTokenRatio` beats every semantic
metric on that dataset here (0.817 against 0.778). decTest rewards form; conTest and
McDiv reward content. A metric should not be tuned to one of those.

---

## 2026-08-03 — Whitening removes the similarity floor completely, and costs agreement

The anisotropy literature (WhiteningBERT and successors) says the encoder's baseline
inflation is *directional*, not scalar: a handful of dominant directions carry most of
the covariance. The shipped floor is a rank-one fix for a full-covariance problem, so
ZCA whitening should be strictly better. Fitted mean and covariance on 6,000 held-out
decTest responses, applied unchanged to every scored set — so replication invariance
survives, exactly as with the floor.

| variant | McDiv | conTest | mean D | ceiling |
|---|---:|---:|---:|---:|
| raw cosine | +0.5952 | +0.6663 | 3.659 | 2.158 |
| scalar floor (shipped, 0.053) | +0.5959 | +0.6668 | 3.769 | 2.320 |
| scalar floor (background-fit, 0.071) | +0.5958 | +0.6675 | 3.807 | 2.383 |
| **whitened** | **+0.5297** | **+0.6275** | **4.292** | **3.015** |

**It works on the criterion it was aimed at and fails on the other.** Background median
cosine goes 0.0714 → −0.0014, so the `1/z` cap disappears outright (14.0 → unbounded),
the achievable ceiling rises 40% and reported diversity rises 17%. Human agreement
drops 0.066 and 0.039.

This is the *third* time a correction that flattens the dominant directions has traded
ranking for calibration — after common-component removal and top-eigencomponent
deflation. The pattern is now firm enough to state as a finding rather than a
coincidence: **in sentence-encoder space the concentrated directions carry semantic
signal, not only baseline inflation.** Any future fix has to be non-linear or local
(hubness/local-scaling), not a global linear flattening. Kept at the scalar floor.

---

## 2026-08-03 — The "new" index was already published, and validates against its authors

**Naming question, answered by the literature rather than by invention.** What
`index="vendi"` computes is the **probability-weighted Vendi Score at Rényi order
q**. Neither half is novel:

- `diag(√p) K diag(√p)` — Friedman & Dieng, *The Vendi Score* (TMLR 2023). Defined
  in the same paper as the unweighted score, which I had not registered when I
  derived it independently.
- the order parameter — Pasarkar & Dieng (2024), where low q is more sensitive to
  rare features and high q to common ones.

So it gets attribution, not a new name. Corrected in `metric.py`, the README and
the tests.

**Cross-validated against the authors' `vendi-score` package** (now a dev
dependency, skipped if absent):

| comparison | result |
|---|---|
| 200 full-rank Gram matrices × 5 orders | **1000/1000 exact** |
| 400 random (Z, p, q) triples, q ≠ 0 | **0 mismatches** |
| same, q = 0 | 7 mismatches, all rank-deficient |

The q = 0 difference is deliberate and now enforced with a relative rank tolerance
(`n · eps · λ_max`, numpy's `matrix_rank` convention). At q = 0 every surviving
eigenvalue contributes a whole unit, so an eigenvalue of **1.18e-17** — what
eigendecomposition leaves on a rank-deficient matrix — moves the reference by 1.
Two identical species are one species; the numerical rank says so and a bare
`> 0` test does not.

**Axioms verified for both indices**, since the Vendi paper does not prove them
for the spectral form:

| axiom | hill | vendi |
|---|---|---|
| replication (pool k dissimilar communities → k·D) | exact | **exact** |
| absent species do not count | ok | ok |
| more similarity never raises diversity (300 trials) | 0 violations | 0 violations |
| bounded by 1 and the species count (300 trials) | 0 violations | 0 violations |
| non-increasing in q | ok | ok |

Replication is the axiom that makes the number *effective* rather than merely an
index, and it is the one the original paper leaves unproved. It holds to floating
point: 2.0000×, 3.0000×, 5.0000×.

---

## 2026-08-03 — Weighted Vendi becomes the default index *(v1.1.0)*

`K_p = diag(√p) Z diag(√p)`. Reduces to the published Vendi Score at uniform **p**
and to the classical Hill number at Z = I, so it is the common generalisation of
the two rather than a third thing.

**The PSD blocker was softer than it looked.** Alignment and tree-edit
similarities are not kernels, and `DependencyParse` reaches −1.6. But negative
eigenvalues hold only 0.7% (`PartOfSpeechSequence`) to 4.2% (`DependencyParse`) of
total spectral magnitude, and projecting onto the nearest PSD matrix moves it
1.3–13.6% in Frobenius norm. A modest correction, not surgery — so the index is
available at every level, not only the two that were PSD to begin with.

**Result on `metric_validation`,** ρ and calibration ratio, Hill → wVendi:

| metric | ρ | ratio |
|---|---|---|
| `DocumentSemantics` | +0.922 → **+0.939** | 0.706 → **0.953** |
| `DependencyParse` | +0.911 → **+0.960** | 0.667 → **0.930** |
| `ConstituencyParse` | +0.885 → **+0.915** | 0.506 → **0.727** |
| `PartOfSpeechSequence` | +0.772 → **+0.869** | 0.437 → **0.714** |
| `Rhythmic` | +0.771 → **+0.811** | 0.492 → **0.848** |
| `Phonemic` | +0.874 → **+0.927** | 0.372 → **0.664** |

Every metric improves on both criteria. Discriminant behaviour preserved
(`PartOfSpeechSequence` gets *more* correct, 0.056 → 0.000 on the inverse pair).
Human agreement +0.5813 → +0.5959 and +0.6503 → +0.6668. All metamorphic laws hold.

**Consequence caught during the change:** `relative_diversity` broke. Its
denominator is the Leinster–Meckes magnitude, which is the ceiling for the *Hill*
quantity; the spectral index exceeds it, so headroom came out at 1.28–2.71 instead
of ≤1. It now raises for `index="vendi"` rather than returning a meaningless
ratio, and the benchmark measures headroom on a Hill-configured twin, since it
diagnoses the similarity structure rather than the index.

**Kept `"hill"`** for large corpora — O(n³) eigendecomposition against O(n²)
matrix-vector — and for the exact Leinster–Cobbold quantity.

---

## 2026-08-03 — Feature-level duplication is already handled *(deferred: an optimisation only)*

**Question.** Discrete levels can produce identical species from different
sentences — two sentences sharing a constituency skeleton. Does the library handle
that, and does `deduplicate=True` catch it?

**Answer.** The maths already handles it; `deduplicate=True` does not, and does not
need to. Merging three sentences with identical POS sequences into one species of
weight 3/5 gives *exactly* the same score as leaving them as three species of 1/5:

| q | 5 species | merged to 3 |
|---|---:|---:|
| 0 | 1.533333 | 1.533333 |
| 1 | 1.499875 | 1.499875 |
| ∞ | 1.285714 | 1.285714 |

This is the identical-species axiom — two species at Z = 1 with weights *w₁*, *w₂*
are indistinguishable from one of weight *w₁+w₂*. `deduplicate=True` merges
byte-identical *documents*, which is a different (and much rarer) event; it would
have caught none of those three sentences.

**Duplicate rate by level**, on 150 textually distinct real sentences, counting
species the metric itself cannot tell apart (Z = 1):

| metric | duplicate species |
|---|---:|
| `DocumentSemantics` | 0% |
| `Phonemic` | 0% |
| `PartOfSpeechSequence` | 2.7% |
| `DependencyParse` | 3.3% |
| `Rhythmic` | 10.7% |
| `ConstituencyParse` | **18.7%** |

The operative variable is **alphabet size relative to sequence length**, not the
linguistic level. `Phonemic` is discrete yet has zero duplicates because sentence-long
phoneme strings are as unique as text; `ConstituencyParse` collides constantly because
its label alphabet is tiny (S, NP, VP, PP).

**Deferred.** Hashing discrete features *before* building Z would cut
`ConstituencyParse` to a 122×122 matrix instead of 150×150. Worth doing only if those
metrics get run at scale — and only if the hash comes before the O(n²) matrix, since
discovering the duplicates from Z afterwards saves nothing that matters. Caveat: 150
short generated sentences from one source. Longer documents push every rate toward
zero; templated corpora push them up. The ordering should be stable, the magnitudes
will not be.

---

## 2026-08-03 — Abundance is where this index beats a spectral one

Added `abundance=` and `deduplicate=` to `diversity()`, plus `diversity_profile()`.

**The efficiency argument.** Weighted 4×4 and materialised 33×33 agree to 1e-6. For
20,000 documents over 500 distinct texts that is 1,600× fewer entries and, at O(n³),
some 64,000× less work. Vendi has no way to accept counts, so abundance must be
materialised for it — the concrete cost of the spectral formulation.

**The benchmark.** `benchmarks/abundance/` holds eight themes under five weight
profiles. *The corpus text is identical across all five; only the weights change.*
True D₁ ranges 1.24 to 8.00. Weighted diversity recovers the ordering perfectly
(ρ = +1.000 at q ≥ 1). Uniform returns 4.753 five times; **Vendi returns 12.541 five
times**. Neither can see the distinction, because the only thing that changed is an
input they do not receive.

**Qualification.** The closed form assumes a block-diagonal Z. Measured: within-theme
0.746, across-theme 0.126. So magnitudes are compressed relative to the closed form
even though ordering is exact — and that compression is the encoder's.

---

## 2026-08-03 — Vendi beats this library's index at uniform abundance

`benchmarks/vendi_comparison/`. Same encoder, same similarity matrix, same floor
correction; only the index differs.

| index | ρ vs known *k* | median ratio | McDiv | conTest |
|---|---:|---:|---:|---:|
| Hill (this library) | +0.9615 | 0.839 | +0.5813 | +0.6503 |
| **Vendi** | **+0.9743** | **0.986** | **+0.5959** | **+0.6668** |

**Why.** A uniform baseline similarity makes `K = (1−z)I + zJ`, and **J is rank one**.
Vendi's spectral decomposition confines it to a single eigenvalue and leaves the other
n−1 untouched; the Hill number spreads it through every `(Zp)ᵢ = (1 + (n−1)z)/n`, where
it accumulates linearly and pulls the score toward `1/z` regardless of n. At z = 0.3,
n = 50: Hill 3.18, Vendi 26.90, truth 50.

**Things tried that do not close the gap:**

- **The order parameter.** At uniform abundance every `(Zp)ᵢ` is equal, so q cancels
  entirely — q = 0, 0.5, 1, 2 all return 11.36 at n=25, z=0.05. Cannot help by
  construction.
- **Common-component removal** (shared embedding direction from the reference corpus,
  projected out). Improves rank agreement 0.957 → 0.967, leaves calibration at 0.800 —
  below the scalar floor's 0.839.
- **Deflating the top eigencomponent** of the corpus's own K. Median ratio hits 1.000,
  but it is an artefact: it returns ≈ n for *everything*. Five paraphrases of one idea
  come back as 4.68 distinct concepts against a truth of 1. Rank agreement falls to
  0.81, human agreement to +0.32/+0.28. The top eigencomponent carries real signal, not
  only the shared floor.

**Standing.** Vendi wins on uniform-abundance benchmarks and cannot compete at all on
skewed abundance. Neither result subsumes the other. **Resolved** by the weighted form
(entry above): `diag(√p) Z diag(√p)` takes abundance *and* keeps the spectral
robustness, and became the default in v1.1.0.

---

## 2026-08-03 — The similarity floor caps diversity independently of corpus size

Encoders do not send unrelated text to orthogonal vectors. Because every document is
slightly similar to every *other* one, the baseline accumulates:

```
D_max = n / (1 + (n−1)z)  →  1/z   as n → ∞
```

| z | n=5 | n=100 | limit |
|---:|---:|---:|---:|
| 0.32 | 2.19 | 3.06 | **3.1** |
| 0.08 | 3.79 | 11.21 | **12.5** |

Measured floors on `REFERENCE_CORPUS`: mpnet 0.053, MiniLM 0.058, bge-large 0.351,
bge-base 0.377, TokenSemantics/bert 0.117.

**This explains the embedder benchmark's central puzzle.** bge-large wins on human
agreement while reporting ~58% of true *k* — one number accounts for both columns:
ranking is unaffected by a floor, counting is capped by it at ~2.9 effective species.

**Fix.** `z' = max(0, (z − z₀)/(1 − z₀))`, `z₀` a per-encoder constant, on by default
(`similarity_floor="auto"`). Validated on 1,270 human-scored sets before enabling:
agreement moves ≤ 0.0014 while reported diversity rises 7–38%.

**Why a constant and not a corpus statistic.** `mean_adj` subtracted the corpus's own
mean, which made each pair's similarity depend on unrelated documents — it cost
replication invariance and left two identical items at 0.777. Every attempt to estimate
this per corpus hits the same wall.

---

## 2026-08-03 — No species aggregation, and why

Ecology assigns individuals to species before counting. This library does not, and
should not.

**Documents.** Merging a 9-document corpus into clusters moves D by 12% at the most
aggressive threshold (9 → 3 species, 3.148 → 2.771), and merging only near-identical
items (Z ≥ 0.95) changes it by nothing. Similarity-sensitivity dissolves the
species-boundary problem that made taxonomic aggregation necessary — you merge cows
because Z = I cannot say "these two are alike".

**Tokens.** Grouping the five senses of *run* by surface form collapses the score
5.12 → 1.77. Their contextual embeddings sit at pairwise similarity 0.09–0.24; they are
already separate species. Type-level aggregation would destroy precisely what the
README's showcase argues for.

---

## 2026-08-03 — Claims made and retracted

Kept because the reasoning that produced them was plausible and will recur.

1. **"Headroom shows the index is extracting everything its similarity structure
   allows, so the shortfall is the embedder's."** Wrong twice over. Headroom is
   near-tautological at uniform abundance — for `(1−z)I + zJ` the magnitude equals the
   Hill number at uniform **p** *exactly*, so it reads 1.0000 by construction. And Vendi
   reads 0.986 against Hill's 0.706 on the *same* matrix, so the representation cannot
   carry the whole explanation.

2. **"Vendi never sees p."** Too strong. It cannot accept a weight *vector*, but it does
   respond to abundance expressed as repeated rows. The real distinction is weights
   versus multiplicity, and the cost of the latter is O(n³) in the materialised size.

3. **Correlating the diversity profile across q within one weighting.** Meaningless:
   D_q is non-increasing in q by construction, so any profile correlates +1 with any
   other. The informative comparison is across *profiles* at fixed q.

4. **CommonGen (Zhang, Peng & Bollegala 2025) described as human-annotated.** It is
   LLM-annotated. That matters — validating an embedding-based metric against
   LLM-generated diversity ratings risks circularity.

5. **`metric_validation` reporting `DependencyParse` calibration at 0.986.** That was
   the saturation bug flattering it: a similarity matrix near the identity reports near
   the species count, which happened to sit close to true *k* on small constructed
   corpora and was catastrophically wrong on real text.

---

## 2026-08-02/03 — Defects found, all fixed in the unreleased 1.0.3

Every one was found by a benchmark built during this work, and every one is now
covered by a regression test.

| defect | effect |
|---|---|
| ZSS nodes labelled by **token index** | parses compared by tree *shape*; `TED("She sings beautifully", "Dogs eat bones") = 0` |
| `exp(−edit_distance)` not scale-free | Z → identity on ordinary text; `DependencyParse` read 4.94 of a ceiling of 5 |
| Biopython gap defaults (−1) | **negative** similarities in Z, invalid for a Hill number |
| alignment normalised corpus-wide | POS/Rhythmic/Phonemic all returned exactly 86.000 on 86 documents — the raw species count |
| cosine ∈ [−1, 1] fed to a Hill number | 1.35% of McDiv pairs negative |
| `mean_adj` | identical items scored 0.777 while the diagonal stayed 1.0 |
| `PartOfSpeechSequence` read UPOS | *walks/walked/walking* all `VERB`; not a morphological metric |

**Pattern worth noting:** four of the seven are the same failure — a similarity that is
not scale-free or not bounded, producing a Z that saturates toward the identity and a
score that silently converges on the species count. The metamorphic suite now tests for
exactly that shape.

---

## Open items

**Decisions pending**

- When to merge PR #1 and tag. *(Version settled: 1.1.0, since the scope is well past a
  patch and the index default now changes too.)*
- Whether an independent validation pass should confirm the index change before release.
  The evidence is strong but comes from benchmarks built in the same session as the
  index.

**Deferred work**

- Feature-level deduplication by hashing discrete features before building Z
  (`ConstituencyParse`, `Rhythmic`; performance only).
- Reliability: split-half, seed stability, cross-encoder rank stability. Named in the
  research plan, never built.
- α/β/γ decomposition (Chao et al.) — would let "is this corpus diverse because each
  source is, or because sources differ?" be answered directly.
- `TokenSemantics` extrapolation still fits `power_law` at every budget and does worse
  than raw subsampling. The one genuine remaining `estimate_diversity` failure.
- Cross-lingual scope: spaCy and benepar are English-only, which bounds what
  "linguistic diversity" can currently mean here.
