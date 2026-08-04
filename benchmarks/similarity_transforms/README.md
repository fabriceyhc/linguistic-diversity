# Similarity transforms

Every candidate correction from the extended literature review, tested on identical
cached embeddings so that the only difference between two rows is the transform.

## Why

`DocumentSemantics` agreed with human diversity ratings at ρ ≈ 0.60–0.82 and could only
ever report 2.3 effective documents out of 5, because sentence encoders place unrelated
text at cosine ≈ 0.07 rather than 0 and the baseline accumulates: `D_max → 1/z`.

Computing the annotator noise ceiling from the dispersion Tevet & Berant ship (10
ratings per set) puts the achievable correlation at 0.93–0.98, so almost none of the
gap was label noise.

## Scripts

| script | what it does |
|---|---|
| `common.py` | loading, embedding cache, the two evaluation criteria |
| `transforms.py` | candidate corrections, each fitted on background text only |
| `run_sweep.py` | stage 1: each angle alone. stage 2: combinations, and a q sweep |
| `run_nli.py` | the entailment kernel, which replaces cosine rather than correcting it |
| `run_hybrid.py` | head-to-head on one subsample, plus hybrid kernels. `--holdout` scores the complement |
| `verify_winner.py` | axioms and wall-clock cost for the survivors |
| `run_competitors.py` | this library against the published alternatives |

```bash
python run_sweep.py
python run_hybrid.py --limit 400            # selection split
python run_hybrid.py --limit 400 --holdout  # the honest numbers
python verify_winner.py
python run_competitors.py --limit 400
```

Embeddings and NLI matrices are cached in `cache/` by content hash; re-runs are cheap.

## The rule every transform obeys

A transform may depend on the encoder, on fixed background text, and on the prompt if
one is given — **never on the other documents in the corpus being scored**. Violating
that costs replication invariance, which is what sank `mean_adj`: it left two identical
documents at similarity 0.777.

## Result

On held-out sets (no configuration was selected on them):

| | McDiv | conTest | decTest | mean | calib ratio | ceiling |
|---|---:|---:|---:|---:|---:|---:|
| current default | +0.635 | +0.663 | +0.829 | 0.709 | 0.986 | 2.33 |
| ctx + LS + τ | +0.742 | +0.751 | +0.836 | 0.776 | 1.000 | 3.40 |
| NLI ent−contra | +0.799 | +0.708 | +0.719 | 0.742 | 0.979 | 2.03 |
| **hybrid (geometric)** | **+0.832** | **+0.797** | **+0.844** | **0.824** | **1.000** | **3.08** |
| hybrid (arithmetic) | +0.850 | +0.812 | +0.817 | 0.826 | 0.992 | 2.43 |

Four independent corrections, each fixing a different defect:

- **context projection** — responses to one prompt share the prompt, and that shared
  content is not diversity. Removes a *known* direction, unlike the estimated one that
  failed before.
- **local scaling** — hubness: a few documents sit close to everything in encoder space.
- **τ-truncation** — the residual floor. Note τ-truncation is algebraically identical to
  the existing floor with z₀ = 1 − τ; the contribution is that τ should be *far* more
  aggressive (0.70, i.e. z₀ = 0.30) than the value we shipped.
- **NLI kernel** — the only one that fixes the cause rather than the symptom, since
  unrelated sentences are neutral and neutral entails nothing. Costs 45× a cosine encode
  at n = 40 and grows quadratically, so it is opt-in.

All axioms hold. Replication is *closer* to exact than the current default, whose
residual floor accumulates with corpus size.

## Against the published alternatives

`run_competitors.py`, same fixed subsample, so this reads together with the table above.

| metric | McDiv | conTest | decTest | mean | calib ρ | ratio |
|---|---:|---:|---:|---:|---:|---:|
| distinct-n / TTR | +0.389 | +0.489 | +0.815 | 0.564 | +0.530 | — |
| Self-BLEU | +0.418 | +0.525 | +0.819 | 0.587 | +0.542 | — |
| Decan (LM surprise) | +0.440 | +0.504 | +0.708 | 0.550 | +0.598 | — |
| PRDC coverage | +0.207 | +0.283 | +0.366 | 0.285 | +0.805 | — |
| Vendi | +0.576 | +0.670 | +0.811 | 0.686 | +0.974 | 0.973 |
| ours, current default | +0.576 | +0.670 | +0.813 | 0.686 | +0.974 | 0.986 |
| **ours, best kernel** | **+0.808** | **+0.811** | **+0.819** | **0.812** | **+0.975** | **1.000** |

**Calibration separates the families completely.** Every surface metric sits at ρ ≈ 0.53
against a *known* number of concepts while every similarity-sensitive one sits at ρ ≈
0.97. A Self-BLEU of 0.4 is not an answer to "how many distinct ideas are in here".
(distinct-n and TTR are also identical to four decimals on all three datasets.)

`calib_ratio` is blank where a metric does not claim to be an effective count — there is
no ratio between 0.4 and "5 concepts".

### PRDC needed a fair test

On five-item sets PRDC recall is 0 for **every** set, so its correlation is undefined
rather than poor. It is a distribution-level statistic built for hundreds of samples.
Run where it belongs — 10 temperature bins, ≤200 documents each, held-out reference:

| metric | ρ vs temperature |
|---|---:|
| Vendi | **+0.988** |
| ours, best kernel *(no NLI half)* | +0.952 |
| PRDC recall | +0.927 |
| Decan | +0.721 |
| PRDC coverage | −0.049 |
| PRDC density | −0.891 |

| **ours, scale kernel (τ only)** | **+1.000** |

Two things worth stating plainly. **The small-set winner is not the scale winner** — its
local-scaling component is tuned to human ratings and costs accuracy here, and its
context projection is undefined once corpora are pooled. Swap it for τ-truncation alone
and we lead at every size (see `run_scale.py`). And **PRDC density going strongly
negative is correct**: density is the *fidelity* half, and raising temperature lowers
fidelity. That it lands at −0.891 is the best evidence the implementation is sound.

## Does any of this survive at scale? (`run_scale.py`)

Discrimination against temperature, one task, sizes matched:

| kernel | n=5 | n=25 | n=100 | n=200 | n=400 |
|---|---:|---:|---:|---:|---:|
| raw cosine (= Vendi) | +0.649 | +0.952 | +0.927 | +0.976 | +0.988 |
| **τ=0.70 only** | +0.682 | +0.976 | +0.952 | **+1.000** | **+1.000** |
| local scaling k=10 | +0.515 | +0.879 | +0.721 | +0.818 | +0.733 |
| LS + τ *(small-set winner)* | +0.576 | +0.939 | +0.927 | +0.952 | +0.927 |

τ-truncation transfers; the other two corrections do not, for reasons that are not about
size. Context projection removes a *shared prompt* and a pooled corpus has none. Local
scaling helps human ratings and hurts temperature **at every size** — a task effect that
was selected on one task and carried to another.

### The genuine scale problem is saturation, not ranking

Mean D / n. Approaching 1 means Z has become the identity and the score is just the
corpus size:

| floor | n=5 | n=50 | n=200 | n=400 |
|---|---:|---:|---:|---:|
| z₀ = 0 | 0.884 | 0.621 | 0.422 | **0.306** |
| z₀ = 0.053 (shipped) | 0.901 | 0.674 | 0.491 | 0.374 |
| z₀ = 0.30 (small-set best) | 0.950 | 0.867 | 0.849 | **0.829** |

At n = 400 the aggressive floor reports **332 effective documents out of 400 generated
sentences**, which is not a credible effective count. Ranking, meanwhile, is nearly
insensitive to the floor — every value from 0 to 0.5 scores ≥ 0.95 at n ≥ 25.

**So the floor must be chosen on calibration, not ranking, and it is size-dependent.**
That follows from its own motivation: the cap `D_max = n/(1+(n−1)z)` is a function of n,
so the correction for it should be too. `z₀ = 0.30` is a five-item-set constant and must
not ship as a global default.
