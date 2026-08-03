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
print(f"{metric(lexically_diverse):.2f}")     # 1.51
print(f"{metric(semantically_diverse):.2f}")  # 3.75
```

Set A is **perfect on every standard lexical measure** — type-token ratio 1.000,
distinct-1 1.000, distinct-2 1.000, and self-BLEU 0.000, meaning literally zero n-gram
overlap between its sentences — yet it states one proposition five times. Set B looks
repetitive to those measures because *run* recurs, but each use is a different sense, so
it carries ~3.8 distinct meanings out of 5.

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
| `TokenSemantics` | Semantic | contextualized token embeddings | 3.46 | **5.12** | 30 |
| `DocumentSemantics` | Semantic | sentence embeddings | 1.56 | **3.75** | 5 |
| `DependencyParse` | Syntactic | dependency tree structure | 1.38 | **2.91** | 5 |
| `ConstituencyParse` | Syntactic | phrase structure *(needs benepar)* | 1.10 | **1.24** | 5 |
| `PartOfSpeechSequence` | Morphological | POS sequences, aligned biologically | 1.46 | **1.75** | 5 |
| `Rhythmic` | Phonological | stress and syllable weight | 1.39 | 1.48 | 5 |
| `Phonemic` | Phonological | phoneme sequences | 1.60 | 1.53 | 5 |
| `UniversalLinguisticDiversity` | Combined | all branches, hierarchically | 1.70 | **2.73** | — |

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

- **Document semantics separates the sets most sharply** (1.56 vs 3.75, a 2.4x gap). Reach
  for it when you care how many distinct *things* a corpus says.
- **Syntax is an independent signal** (1.38 vs 2.91). Set A leans on one frame without
  repeating it exactly — four distinct POS sequences across five sentences, three of them
  `DET ADJ NOUN VERB · NOUN` — so it is syntactically narrow rather than monotonous. A
  corpus can be semantically varied yet syntactically narrow, and only measuring both will
  tell you which.
- **Several metrics barely separate these two sets** — `ConstituencyParse` 1.10 vs 1.24,
  `Rhythmic` 1.39 vs 1.48, `Phonemic` 1.60 vs 1.53 the wrong way round. They measure
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
