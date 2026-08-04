# Linguistic Diversity

[![PyPI version](https://img.shields.io/pypi/v/linguistic-diversity.svg)](https://pypi.org/project/linguistic-diversity/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fabriceyhc/linguistic-diversity/blob/main/examples/demo.ipynb)

**Linguistic diversity metrics using similarity-sensitive Hill numbers.**

Hill numbers come from ecology, where they measure the *effective number of species* in a
population. Here "species" are linguistic units — words, parse trees, phoneme sequences —
and the population is a corpus. A token semantic diversity of 9 means the corpus carries
roughly 9 distinct semantic concepts. Unlike lexical measures, these are
**similarity-sensitive**: near-duplicates count as fractional species, not whole ones.

```bash
pip install linguistic-diversity
```

Optional extras: `[syntactic]` (constituency parsing via benepar), `[phonological]`,
`[viz]`, `[dev]`. All metrics run on pure-Python dependencies; only the optional
`phonemizer` backend needs a system library (`espeak-ng`).

For development, after `pip install -e ".[dev]"` the test suite also needs the spaCy
pipeline and NLTK corpora:

```bash
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords cmudict averaged_perceptron_tagger_eng punkt
```

`make check-all` runs everything CI runs.

## Why not lexical diversity?

Most diversity work in NLP counts *surface forms* — type-token ratio, distinct-n,
self-BLEU. This library measures whether the *meanings* differ. The two can point in
opposite directions:

```python
from linguistic_diversity import DocumentSemantics

# Set A: 30 words, every one unique. One idea, restated five times.
lexically_diverse = [
    'a violent tempest wrecked our village',
    'the fierce gale devastated their settlement',
    'that savage hurricane destroyed this community',
    'an intense cyclone flattened every township',
    'some brutal windstorm ruined nearby neighborhoods',
]

# Set B: 30 words, "run" five times. Five unrelated meanings.
semantically_diverse = [
    'she went for a morning run',            # jogging
    'he will run the entire company',        # to manage
    'a run appeared in her stocking',        # a tear
    'the program failed to run correctly',   # to execute
    'they scored the winning run today',     # a baseball point
]

metric = DocumentSemantics()
print(f"{metric(lexically_diverse):.2f}")     # 3.10
print(f"{metric(semantically_diverse):.2f}")  # 4.60
```

Set A is **perfect on every standard lexical measure** — type-token ratio 1.000,
distinct-1 1.000, distinct-2 1.000, and self-BLEU 0.000, meaning literally zero n-gram
overlap between its sentences — yet it states one proposition five times. Set B looks
repetitive to those measures because *run* recurs, but each use is a different sense, so
it carries ~4.6 distinct meanings out of 5.

All four lexical baselines ship with the library (`TypeTokenRatio`, `DistinctN`,
`SelfBLEU`) so you can reproduce the comparison rather than take it on faith. If you are
selecting training data, deduplicating, or scoring generation diversity, they will accept
set A as maximally diverse. It isn't.

## Metrics

Every metric shares one interface — `metric(corpus) -> float` — and takes an optional
config dict. The last two columns show each metric on the two sets above (matched at 5
documents / 30 words / 30 token species, so ceilings are identical):

| Class | Dimension | Measures | Set A | Set B | Ceiling |
|---|---|---|---:|---:|---:|
| `TypeTokenRatio` | Lexical *(baseline)* | unique tokens / total | **1.000** | 0.767 | 1 |
| `DistinctN` (n=1) | Lexical *(baseline)* | unique unigrams / total | **1.000** | 0.767 | 1 |
| `DistinctN` (n=2) | Lexical *(baseline)* | unique bigrams / total | 1.000 | 1.000 | 1 |
| `SelfBLEU` | Lexical *(baseline)* | n-gram overlap — *lower* is diverse | **0.000** | 0.049 | 0 |
| `TokenSemantics` | Semantic | contextualized token embeddings | 15.41 | **21.11** | 30 |
| `DocumentSemantics` | Semantic | sentence embeddings | 3.10 | **4.60** | 5 |
| `DependencyParse` | Syntactic | dependency tree structure | 2.13 | **4.36** | 5 |
| `ConstituencyParse` | Syntactic | phrase structure *(needs benepar)* | 1.33 | **1.86** | 5 |
| `PartOfSpeechSequence` | Morphological | POS sequences, aligned biologically | 2.43 | **3.34** | 5 |
| `Rhythmic` | Phonological | stress and syllable weight | 2.59 | 2.72 | 5 |
| `Phonemic` | Phonological | phoneme sequences | 3.25 | 3.07 | 5 |
| `UniversalLinguisticDiversity` | Combined | all branches, hierarchically | 3.49 | **5.13** | — |

```python
from linguistic_diversity import DependencyParse, UniversalLinguisticDiversity

DependencyParse()(corpus)

# The species count is the wrong ceiling: n effective species needs n mutually
# dissimilar documents. These give the one that actually applies.
metric = DependencyParse()
metric.max_diversity(corpus)       # largest value any abundance could reach here
metric.relative_diversity(corpus)  # diversity as a fraction of that, in (0, 1]

universal = UniversalLinguisticDiversity()
detailed = universal.get_detailed_scores(corpus)   # {'universal': ..., 'branches': {...}}
```

Reading the results:

- **Document semantics separates the sets** (3.10 vs 4.60). Reach
  for it when you care how many distinct *things* a corpus says.
- **Syntax is an independent signal** (2.13 vs 4.36). Set A leans on one frame without
  repeating it exactly — four distinct POS sequences across five sentences, three of them
  `DET ADJ NOUN VERB · NOUN` — so it is syntactically narrow rather than monotonous. A
  corpus can be semantically varied yet syntactically narrow, and only measuring both will
  tell you which.
- **Several metrics barely separate these two sets** — `ConstituencyParse` 1.33 vs 1.86,
  `Rhythmic` 2.59 vs 2.72, `Phonemic` 3.25 vs 3.07 the wrong way round. They measure
  something real; these two sets just do not differ much in it. Stated rather than left to
  be inferred from which rows are unbolded, because a table that only shows its winners is
  not much use for picking a metric.

### Reading a score as a quantity

*n* effective species is only reachable when all *n* documents are mutually dissimilar,
and encoders do not make unrelated text dissimilar — they place it at cosine ~0.05 to
~0.35. Because every document is slightly like every *other* one, that floor accumulates:
the largest diversity any corpus can reach is `n / (1 + (n-1)z)`, tending to **1/z**. An
uncorrected floor of 0.35 caps a corpus at about 2.9 effective species however large it is.

So the semantic metrics rescale it away by default, `z' = max(0, (z - z₀) / (1 - z₀))`
with `z₀` a per-encoder constant — looked up for known encoders, otherwise calibrated once
against a fixed corpus of mutually unrelated sentences and cached. It is a constant, never
estimated from the corpus being measured, which is what keeps identical documents at
similarity 1 and diversity invariant to replication. Pass `similarity_floor=None` for the
pre-1.0.3 behaviour, or a float to set it yourself.

`max_diversity(corpus)` reports that ceiling and `relative_diversity(corpus)` the score
as a fraction of it. Read the second with care: at uniform abundance it is close to 1 by
construction, so it says the abundance is optimal for this index rather than that the
index is extracting all the data holds. On the same similarity matrix the Vendi Score
recovers 0.99 of a known concept count where these metrics recover 0.71 — see
[`benchmarks/vendi_comparison/`](benchmarks/vendi_comparison/) for why, and for what
follows from it.

`UniversalLinguisticDiversity` aggregates by geometric mean within a branch, then weighted
across branches. It enables six of the seven metrics by default; `ConstituencyParse` is
opt-in via `use_constituency_parse: True`. Presets: `balanced`, `semantic_focus`,
`structural_focus`, `minimal`, `conservative` via `get_preset_config(name)` — see
[docs/universal-metric.md](docs/universal-metric.md). Reproduce the table with
[`examples/all_metrics.py`](examples/all_metrics.py), or work through it interactively in
[`examples/demo.ipynb`](examples/demo.ipynb) ([open in Colab](https://colab.research.google.com/github/fabriceyhc/linguistic-diversity/blob/main/examples/demo.ipynb)).

Each metric's defaults were chosen by measurement, not assumption, and each benchmark
answers a different question:

- [`benchmarks/embedder_selection/`](benchmarks/embedder_selection/) — which encoder should
  back `DocumentSemantics`, scored against 600 human-judged response sets.
- [`benchmarks/metric_validation/`](benchmarks/metric_validation/) — does each metric
  respond to the linguistic level it claims and stay flat on the others?
- [`benchmarks/length_robustness/`](benchmarks/length_robustness/) — does a score move when
  only corpus size or document length changes? (Hill numbers: exactly invariant to
  replication. Type-token ratio, distinct-*n*, Self-BLEU and compression ratio: not.)
- [`benchmarks/metamorphic/`](benchmarks/metamorphic/) — properties that must hold for any
  corpus, checked without ground truth.
- [`benchmarks/human_agreement/`](benchmarks/human_agreement/) — agreement with graded human
  diversity judgments on three datasets, and what sampling temperature actually moves
  (form 0.66–0.72, structure 0.57–0.61, content 0.22–0.54).
- [`benchmarks/vendi_comparison/`](benchmarks/vendi_comparison/) — this library's index
  against the Vendi Score on one shared similarity matrix. **Vendi wins on both human
  agreement and calibration**; what survives the comparison is the multi-level
  instrumentation, not the choice of index.

## Choosing the index

Two indices consume the same similarity matrix, and the default changed in v1.1.0.

```python
DocumentSemantics()                    # index="vendi", the default
DocumentSemantics({"index": "hill"})   # Leinster-Cobbold
```

**`"vendi"`** — the **probability-weighted Vendi Score** at Rényi order *q*: `exp` of the
entropy of the eigenvalues of `diag(√p) Z diag(√p)`. Not novel here — the weighting is
Friedman & Dieng's own ([TMLR 2023](https://arxiv.org/abs/2210.02410), defined alongside
the unweighted score) and the order parameter is Pasarkar & Dieng (2024). Agreement with
the authors' `vendi-score` package is asserted in the test suite.
**`"hill"`** — `D_q = (Σᵢ pᵢ (Zp)ᵢ^(q−1))^(1/(1−q))`, Leinster–Cobbold.

They agree exactly at both extremes — Z = I gives *n*, Z all-ones gives 1 — and differ in
between. A uniform baseline similarity contributes a **rank-one** component: the spectral
form confines it to a single eigenvalue, while the Hill form spreads it through every
`(Zp)ᵢ`, where it accumulates linearly in *n* and pulls the score toward `1/z` whatever
the corpus size. At z = 0.3 and n = 50 the Hill number reads 3.18 against a truth of 50;
the spectral form reads 26.90.

Vendi is the default because it is better on both criteria at **every** level measured —
rank agreement against known ground truth and calibration ratio — and on human agreement,
while preserving the discriminant behaviour and every metamorphic law. See
[`benchmarks/vendi_comparison/`](benchmarks/vendi_comparison/).

Keep `"hill"` for very large corpora, where an O(n³) eigendecomposition costs more than an
O(n²) matrix-vector product, or when the exact Leinster–Cobbold quantity is wanted.
`relative_diversity()` is Hill-only: its ceiling comes from a theorem about that quantity,
and the spectral index routinely exceeds it.

## Abundance and the diversity profile

Species are rarely equally common. A Hill number takes that as an explicit abundance
vector — corpus frequencies, sampling weights, duplicate counts — which is the one thing
a purely spectral index cannot express.

```python
from linguistic_diversity import DocumentSemantics

metric = DocumentSemantics()

metric(corpus)                             # uniform, the default
metric(corpus, abundance=[97, 1, 1, 1])    # counts, normalised internally
metric(corpus, deduplicate=True)           # merge identical docs, weight by count
```

`deduplicate=True` returns the same value as leaving the duplicates in, on a matrix the
size of the distinct set. For 20,000 documents over 500 distinct texts that is 500×500
rather than 20,000×20,000 — 1,600× fewer entries, and since the work is O(n³), some
64,000× less of it.

One number hides the shape, so report the profile:

```python
metric.diversity_profile(corpus)
# {0.0: 2.85, 0.5: 2.79, 1.0: 2.72, 2.0: 2.60, 4.0: 2.49, inf: 1.98}
```

Low *q* asks how many distinct things are **present**; high *q* asks how many
**dominate**. A flat profile is an even corpus; a steep one means a few items carry it.
Weighted `[0.97, 0.01, 0.01, 0.01]` the same corpus runs 2.08 → 1.02: four things are
there, and one of them is effectively all of it. Both readings are true, and neither
number alone tells you which case you are in. The similarity matrix is computed once and
reused across every *q*.

## Cross-encoder kernel

A bi-encoder reads each document once and compares vectors, so unrelated text lands on
the encoder's floor rather than on 0 — which is why `similarity_floor` exists. A
cross-encoder reads both documents together and outputs the comparison directly, so
unrelated text scores ~0.01 and there is no floor to correct.

```python
metric = DocumentSemantics({"cross_encoder": "cross-encoder/stsb-roberta-large"})
```

Measured on 1,270 human-scored sets, it matches the best embedding pipeline while
needing no similarity floor, no hubness correction and no prompt. NLI checkpoints are
auto-detected from their label map and scored on entailment; prefer one whose *neutral*
class is calibrated, such as
`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, since SNLI/MNLI-only models
label unrelated text "contradiction".

**Opt-in because of cost**: O(n²) forward passes against O(n) encodes — roughly 45× a
bi-encoder at 40 documents, quadratic thereafter. `cross_encoder_max_docs` (default 512)
refuses rather than hangs.

## Beyond one number

Three questions an effective number cannot answer on its own. Each has a settled answer
in ecology, and each is implemented here from the reference implementations.

### Evenness — many things, or balanced things?

A diversity of 3.0 means something different out of 4 documents than out of 400.
Evenness divides the richness out, so the two can be reported separately.

```python
metric.evenness(corpus)                                # 0.99  balanced
metric.evenness(corpus, abundance=[97, 1, 1, 1])       # 0.06  one item dominates
```

Five classes, `E1`–`E5` (Chao & Ricotta 2019); `E3`, the normalised slope of the
diversity profile, is their headline choice and the default here. Passing
similarity-sensitive values for both terms is a generalisation of ours, not theirs: it
reads as "even across *distinct content*" rather than "even across species".

### Coverage — is this sample complete enough to compare?

Comparing two corpora at equal **size** is biased against the more diverse one: a size
sufficient to characterise a dull corpus is too small for a rich one. Coverage says how
complete each sample is, so they can be compared at equal completeness instead
(Chao & Jost 2012).

```python
metric.sample_coverage(corpus)   # 0.0 for wholly distinct documents -- see below
```

Species here are equivalence classes under `Z = 1`: documents *this metric* cannot tell
apart. Three sentences sharing a POS skeleton are one species to `PartOfSpeechSequence`
and three to `DocumentSemantics`.

**Coverage is 0 when every species occurs exactly once**, the normal case for whole
documents. That is the honest answer, not a bug — with nothing repeated, the sample
carries no evidence about what it has missed. The measure is informative for the levels
whose features collide (`ConstituencyParse` 0.50, `Rhythmic` 0.45 on real text) and
vacuous for semantics on distinct documents.

### Partition — within sources, or between them?

```python
metric.partition({"finance": docs_a, "baking": docs_b})
# PartitionResult(q=1, gamma=3.5077, alpha=1.9593, beta=1.7903)
```

`gamma` is the pooled diversity, `alpha` the average within a source, and `beta` the
effective number of **distinct** sources — 1 when interchangeable, N when they share
nothing. One similarity matrix is built over the pooled documents, so cross-source
similarity is *measured*: two sources with no shared wording but the same content come
back as one. Reeve et al. (2016), the similarity-sensitive continuation of the
Leinster–Cobbold measure this library is built on.

Measured on real corpora: three different generation tasks read as **2.01** distinct
sources; one task cut arbitrarily into three reads as **1.27**.

### Why there is no species aggregation

Ecology assigns individuals to species before counting — two cows are one species despite
differing DNA. This library does not, and does not need to: similarity-sensitivity
dissolves the species-boundary problem that made taxonomic aggregation necessary.
Merging a 9-document corpus down to 3 clustered species moves the score by 12%, and
merging only near-identical items (Z ≥ 0.95) changes it by nothing at all — the
similarity matrix has already done the discounting.

For tokens it would be actively wrong. Grouping the five senses of *run* in the example
above by surface form collapses the score from 5.12 to 1.77, destroying the distinction
the metric exists to detect. Their contextual embeddings sit at pairwise similarity
0.09–0.24; they are already separate species and should stay that way.

## Large corpora

Exact diversity needs an O(n²) similarity matrix. Every metric offers
`estimate_diversity()`, which samples at increasing sizes, fits a growth curve
(logarithmic, power-law, or asymptotic), and extrapolates:

```python
from linguistic_diversity import TokenSemantics

result = TokenSemantics().estimate_diversity(large_corpus, max_sample_size=200)

print(f"{result.diversity:.3f} ± {result.std:.3f}")  # extrapolated estimate
print(result.model, result.fit_rmse)                 # best-fit curve, goodness of fit
result.plot()                                        # observed samples + fitted curve
```

## Configuration

```python
from linguistic_diversity import TokenSemantics

TokenSemantics({
    'model_name': 'roberta-base',  # any HF encoder
    'q': 2.0,                      # diversity order: 0=richness, 1=Shannon, 2=Simpson
    'normalize': True,             # divide by species count
    'use_cuda': True,
    'remove_stopwords': True,
    'trust_remote_code': True,     # for checkpoints shipping custom code
    'encode_kwargs': {},           # extra args for task-conditioned embedders
})
```

## Development notes

[`research_log.md`](research_log.md) records what was measured and set aside — defects
found and their shape, theoretical results applied, options tested and rejected, and
claims made here that later turned out to be wrong.

## Theory

Hill numbers unify **richness** (how many types) and **similarity** (how alike they are):

```
D = (Σ p_i (Σ Z_ij p_j)^(q-1))^(1/(1-q))
```

`p` is the abundance distribution, `Z` the similarity matrix, and `q` the order parameter
(0 = richness, 1 = Shannon, 2 = Simpson, ∞ = Berger-Parker). At `q=1`, the default, this
is the effective number of species weighted by similarity.

## Citation

```bibtex
@software{linguistic_diversity_2026,
  title={Linguistic Diversity: Modernized Implementation of Similarity-Sensitive Hill Numbers for NLP},
  author={Harel-Canada, Fabrice},
  year={2026},
  url={https://github.com/fabriceyhc/linguistic-diversity}
}
```

Supersedes [TextDiversity](https://github.com/fabriceyhc/TextDiversity) (2022), which it
reimplements with FAISS-accelerated similarity, model caching, vectorized operations, type
hints throughout, and a pytest suite.

## Links

- **PyPI**: [pypi.org/project/linguistic-diversity](https://pypi.org/project/linguistic-diversity)
- **Issues**: [GitHub Issues](https://github.com/fabriceyhc/linguistic-diversity/issues)
- **Docs**: [universal metric](docs/universal-metric.md) · [troubleshooting](docs/troubleshooting.md)
- **Experiments**: [linguistic-diversity-experiments](https://github.com/fabriceyhc/linguistic-diversity-experiments) — authorship verification, dementia detection, diversity-based data selection
- **License**: MIT, see [LICENSE](LICENSE)
- Ecological diversity theory from Chao et al. (2014)
