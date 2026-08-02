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
print(f"{metric(semantically_diverse):.2f}")  # 3.41
```

Set A is **perfect on every standard lexical measure** — type-token ratio 1.000,
distinct-1 1.000, distinct-2 1.000, and self-BLEU 0.000, meaning literally zero n-gram
overlap between its sentences — yet it states one proposition five times. Set B looks
repetitive to those measures because *run* recurs, but each use is a different sense, so
it carries ~3.4 distinct meanings out of 5.

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
| `TokenSemantics` | Semantic | contextualized token embeddings | 12.33 | **14.68** | 30 |
| `DocumentSemantics` | Semantic | sentence embeddings | 1.51 | **3.41** | 5 |
| `DependencyParse` | Syntactic | dependency tree structure | 1.47 | **4.66** | 5 |
| `ConstituencyParse` | Syntactic | phrase structure *(needs benepar)* | 1.65 | **2.85** | 5 |
| `PartOfSpeechSequence` | Morphological | POS sequences, aligned biologically | 1.28 | **2.36** | 5 |
| `Rhythmic` | Phonological | stress and syllable weight | 1.75 | 2.00 | 5 |
| `Phonemic` | Phonological | phoneme sequences | 2.20 | 2.16 | 5 |
| `UniversalLinguisticDiversity` | Combined | all branches, hierarchically | 2.22 | **4.14** | — |

```python
from linguistic_diversity import DependencyParse, UniversalLinguisticDiversity

DependencyParse()(corpus)

universal = UniversalLinguisticDiversity()
detailed = universal.get_detailed_scores(corpus)   # {'universal': ..., 'branches': {...}}
```

Reading the results:

- **Document semantics separates the sets most sharply** (1.51 vs 3.41). Reach for it when
  you care how many distinct *things* a corpus says.
- **Syntax is an independent signal**, and here the strongest one (1.47 vs 4.66). Set A
  repeats one frame (`DET ADJ NOUN VERB DET NOUN`) five times — a corpus can be
  semantically varied yet syntactically monotonous, and only measuring both will tell you.

`UniversalLinguisticDiversity` aggregates by geometric mean within a branch, then weighted
across branches. It enables six of the seven metrics by default; `ConstituencyParse` is
opt-in via `use_constituency_parse: True`. Presets: `balanced`, `semantic_focus`,
`structural_focus`, `minimal`, `conservative` via `get_preset_config(name)` — see
[docs/universal-metric.md](docs/universal-metric.md). Reproduce the table with
[`examples/all_metrics.py`](examples/all_metrics.py), or work through it interactively in
[`examples/demo.ipynb`](examples/demo.ipynb) ([open in Colab](https://colab.research.google.com/github/fabriceyhc/linguistic-diversity/blob/main/examples/demo.ipynb)).

Each metric's default distance and similarity settings were chosen by measurement, not
assumption — see [`benchmarks/embedder_selection/`](benchmarks/embedder_selection/),
which scores embedders against corpora with known ground-truth diversity and 600
human-judged response sets.

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
