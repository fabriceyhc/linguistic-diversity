# Embedder Selection for Semantic Diversity

Which sentence embedder should back `DocumentSemantics`? This use case answers that
against human diversity judgments, with a synthetic calibration benchmark as a
secondary probe.

**Recommendation: `BAAI/bge-large-en-v1.5`.**

## Headline result

Scored against **McDiv**, 600 response sets carrying graded human diversity scores
(Tevet & Berant, [*Evaluating the Evaluation of Diversity in Natural Language
Generation*](https://arxiv.org/abs/2004.02990), EACL 2021):

| Model | ρ vs. human | pair acc | params |
|---|---:|---:|---|
| infgrad/Jasper-Token-Compression-600M | +0.781 | 0.939 | 607M |
| **BAAI/bge-large-en-v1.5** | **+0.779** | **0.957** | 335M |
| mixedbread-ai/mxbai-embed-large-v1 | +0.767 | 0.957 | 335M |
| BAAI/bge-base-en-v1.5 | +0.760 | 0.942 | 109M |
| WhereIsAI/UAE-Large-V1 | +0.752 | 0.958 | 335M |
| avsolatorio/GIST-large-Embedding-v0 | +0.749 | 0.957 | 335M |
| google/embeddinggemma-300m | +0.730 | 0.938 | 308M |
| Qwen/Qwen3-Embedding-0.6B | +0.716 | 0.943 | 596M |
| sentence-transformers/all-mpnet-base-v2 | +0.581 | 0.939 | 110M |
| sentence-transformers/all-MiniLM-L6-v2 | +0.530 | 0.926 | 22M |

Baselines shipped with the dataset: `bert_score` +0.410,
`averaged_cosine_similarity` +0.297, `averaged_distinct_ngrams` +0.239.
`DocumentSemantics` with a good embedder roughly **doubles** the best of them.

Jasper edges out bge-large by +0.002 — noise at n=600 — while scoring lower on pair
accuracy, being ~2× the size, and requiring `trust_remote_code` plus a custom
`AutoModel` path. bge-large is the better operational choice.

## MTEB scores do predict this — use PairClassification

Over these 10 models, MTEB rank correlates positively with McDiv agreement:

| predictor | Pearson r | Spearman ρ |
|---|---:|---:|
| **MTEB PairClassification** | **+0.879** | +0.709 |
| MTEB STS | +0.729 | +0.467 |

PairClassification is the better screen. Both are computed on **sentence pairs**, which
is the same operation a similarity-sensitive Hill number performs across a corpus, so
the transfer is unsurprising in hindsight.

> **Correction.** Earlier revisions of this document claimed the opposite — that MTEB
> was *anti*-correlated with usefulness here (r ≈ −0.65), using Jasper as the showcase
> failure. That conclusion came from a single hand-built 3-sentence corpus pair and did
> not survive contact with 600 human-scored sets. It was wrong. Screen with
> PairClassification.

## The two benchmarks measure different things

`build_benchmark.py` / `evaluate_embedders.py` construct corpora with known
ground-truth diversity and score **absolute calibration**:

| axis | construction | ground truth | tests |
|---|---|---|---|
| **Synonymy** | *k* concepts × *m* paraphrases | *k* (not *k×m*) | does it **collapse** paraphrases? |
| **Polysemy** | *n* senses of one surface form | *n* | does it **separate** meaning despite shared form? |

Calibration and human agreement rank models almost inversely:

| Model | calibration (1.0 = perfect) | McDiv ρ | mean cosine |
|---|---:|---:|---:|
| all-MiniLM-L6-v2 | 0.976 | +0.530 | +0.298 |
| all-mpnet-base-v2 | 0.973 | +0.581 | +0.302 |
| GIST-large-Embedding-v0 | 0.700 | +0.749 | +0.480 |
| bge-large-en-v1.5 | 0.577 | +0.779 | +0.640 |
| Jasper-600M | 0.521 | +0.781 | +0.750 |

This is not a contradiction — the two ask different questions:

- **Calibration** asks whether the reported effective number of concepts is *numerically
  correct*. Models with a wide cosine range (mpnet, ~0.30 mean) land near 1.0.
- **Human agreement** asks whether corpora are *ordered* correctly. Spearman is
  rank-based, so cosine compression costs it little.

Compressed embeddings systematically under-report absolute diversity (bge-large reports
~58% of true *k*) while still ranking corpora well. **Pick by intended use:** if you read
the score as a quantity, calibration matters and mpnet wins; if you compare corpora
against each other — the common case — human agreement matters and bge-large wins.

Where the two disagree, prefer McDiv: it is human ground truth at 600 sets, whereas the
synthetic ground truth is an authored construct.

## Running it

```bash
# Synthetic calibration benchmark (deterministic; 68 corpora, 533 sentences)
python build_benchmark.py
python evaluate_embedders.py --preset default

# Human-judgment validation
curl -o data.zip http://diversity-eval.s3-us-west-2.amazonaws.com/data.zip
unzip data.zip
python evaluate_mcdiv.py --data-dir ./data --pair-limit 0
```

Both sweeps call `clear_model_cache()` between checkpoints. The model cache is
unbounded, and ten models will otherwise exhaust VRAM — this run peaked at 5.8 GB of
6 GB on an RTX 2060.

Note the dataset's similarity-derived baseline columns are **already
diversity-oriented** (negated by `Similarity2DiversityMetric`, hence their negative
value ranges) — do not flip the sign when correlating.

## Caveats

- Pair accuracy is near-saturated (0.926–0.958 across all 10 models), so the graded
  correlation is the discriminating measure. The binary task is too easy to rank by.
- The synthetic seed data is hand-authored for this repository and not validated by
  external annotators. It is held out from the worked example in the main README, which
  was used to tune metric settings.
- `NovaSearch/stella_en_400M_v5` is unmeasured: its custom code requires xformers, which
  currently mismatches the installed torch.

## Files

| path | purpose |
|---|---|
| `data/concepts.json` | Hand-authored concept clusters and polysemy sets |
| `data/mteb_eng_v2_summary.csv` | MTEB English v2 leaderboard snapshot, used to pick candidates |
| `build_benchmark.py` | Constructs corpora with known ground-truth diversity |
| `evaluate_embedders.py` | Scores embedders on absolute calibration |
| `evaluate_mcdiv.py` | Scores embedders against human judgments (McDiv) |
| `output/` | Generated benchmark and recorded results |
