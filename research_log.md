# Research log

Findings, decisions and deferred work that do not belong in a commit message or a
benchmark README — the reasoning behind choices, the things measured and set aside,
and the claims that turned out to be wrong.

The paper-level plan lives in
[linguistic-diversity-experiments/RESEARCH_PLAN.md](https://github.com/fabriceyhc/linguistic-diversity-experiments).
This file is the engineering and measurement record.

Newest first.

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
skewed abundance. Neither result subsumes the other. **Open decision:** whether to offer
Vendi as a selectable aggregator, and whether Hill should remain the default index.

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

- Release as 1.0.3 or 1.1.0. Six metrics change behaviour; `~=1.0.2` picks a patch up
  with no signal. Was chosen as a patch on the grounds that the previous behaviour was
  defective, but the scope has roughly tripled since.
- When to merge PR #1 and tag.
- Whether Hill remains the default index, given the Vendi result.

**Deferred work**

- Feature-level deduplication by hashing discrete features before building Z
  (`ConstituencyParse`, `Rhythmic`; performance only).
- Vendi as a selectable aggregator across the existing extractors.
- Reliability: split-half, seed stability, cross-encoder rank stability. Named in the
  research plan, never built.
- α/β/γ decomposition (Chao et al.) — would let "is this corpus diverse because each
  source is, or because sources differ?" be answered directly.
- `TokenSemantics` extrapolation still fits `power_law` at every budget and does worse
  than raw subsampling. The one genuine remaining `estimate_diversity` failure.
- Cross-lingual scope: spaCy and benepar are English-only, which bounds what
  "linguistic diversity" can currently mean here.
