# Hill Numbers vs the Vendi Score

The Vendi Score (Friedman & Dieng, [TMLR 2023](https://arxiv.org/abs/2210.02410))
is the closest published relative of what this library computes. Anyone evaluating
this work will ask why not just use it. This measures the answer, and the answer
is not the comfortable one.

```bash
python run_comparison.py --data-dir ../embedder_selection/data
```

Both indices are computed from the **same similarity matrix**, produced by the same
encoder with the same floor correction. Nothing separates them except the index.

## The two definitions

| | Vendi | Leinster–Cobbold (this library) |
|---|---|---|
| formula | `exp(H(λ))`, λ = eigenvalues of `K/n` | `(Σᵢ pᵢ (Zp)ᵢ^(q−1))^(1/(1−q))` |
| abundance | implicit, via which items are in K | explicit vector **p** |
| order *q* | none in the original (Rényi extension later) | native |
| requires | K symmetric, unit diagonal, **positive semi-definite** | Z ∈ [0,1], unit diagonal |

They agree exactly at both extremes — `K = I` gives *n*, `K` all-ones gives 1 — and
diverge in between. For two species at similarity *z*:

| z | Vendi | Hill, 2/(1+z) |
|---:|---:|---:|
| 0.00 | 2.000 | 2.000 |
| 0.25 | 1.938 | 1.600 |
| 0.50 | 1.755 | 1.333 |
| 0.90 | 1.220 | 1.053 |

Vendi discounts similarity far less aggressively.

## Result: Vendi wins on both criteria

**Agreement with graded human diversity judgments** (1,270 sets):

| index | McDiv_nuggets | conTest |
|---|---:|---:|
| Hill (this library) | +0.5813 | +0.6503 |
| **Vendi** | **+0.5959** | **+0.6668** |

**Recovering a known number of concepts** (69 corpora with authored ground truth):

| index | ρ vs known *k* | median ratio |
|---|---:|---:|
| Hill (this library) | +0.9615 | 0.839 |
| **Vendi** | **+0.9743** | **0.986** |

A median ratio of 0.986 means Vendi recovers the true concept count almost exactly.
The Hill number reads 84% of it.

**And the similarity floor does not explain it:**

| floor | index | ρ | median ratio |
|---|---|---:|---:|
| off | Hill | +0.9570 | 0.775 |
| off | Vendi | +0.9741 | 0.973 |
| auto | Hill | +0.9615 | 0.839 |
| auto | Vendi | +0.9743 | 0.986 |

The floor correction helps the Hill number substantially (0.775 → 0.839) and moves
Vendi barely at all. Vendi is well calibrated with or without it, which makes this a
property of the index rather than an artefact of the representation.

## What this means for the library

Stated plainly: **on this evidence the Vendi Score is the better index**, and the
case for similarity-sensitive Hill numbers cannot rest on accuracy.

What survives the comparison is not the index but the instrumentation around it:

- **Level resolution.** Vendi is one number over one similarity matrix. The
  contribution here is six linguistic levels with a validated claim about what each
  responds to — and note that Vendi can be computed per-level too, exactly as it is
  in this script. That makes the levels a property of the *library*, not of the Hill
  number, and the honest framing follows: this project's contribution is validated
  multi-level instrumentation, not the choice of index.
- **Explicit abundance.** Hill numbers weight species by how often they occur; Vendi
  reads the spectrum. Every metric here currently uses uniform abundance, so the
  library does not yet cash this in — it would matter for weighted or deduplicated
  corpora.
- **A weaker requirement on Z.** Vendi needs positive semi-definiteness, because a
  negative eigenvalue makes `log λ` undefined; the Hill number needs only
  Z ∈ [0,1]. Clamping and rescaling can break PSD while leaving the Hill number well
  defined. In practice this rarely binds — 0 of 60 matrices sampled here had a
  negative eigenvalue — so it is a robustness argument, not an accuracy one.
- **A known maximum.** Leinster & Meckes give the achievable ceiling for a
  similarity matrix, which `max_diversity()` uses. No comparable standard result was
  applied to Vendi here.

## What to do about it

Unresolved, and deliberately left so rather than argued away:

1. **Offer Vendi as an aggregator** across the existing feature extractors. Most of
   the work — parses, POS sequences, phoneme sequences, embeddings — is unaffected by
   which index consumes the similarity matrix.
2. **Understand the gap** rather than adopt on one benchmark. Vendi's advantage is
   consistent with it under-penalising residual similarity, which is closer to right
   when that residual is representational noise and further from right when it is
   real. The constructions here may favour the first case.
3. **Report both.** For a paper whose subject is measurement validity, publishing a
   comparison that the library loses is better evidence of method than publishing one
   it wins.

## Caveats

- 69 corpora with authored ground truth, all hand-built for this repository, all
  English, all short documents.
- Only `DocumentSemantics` similarity matrices were compared. The structural and
  phonological levels are untested under Vendi.
- The Vendi implementation here is written from the paper's definition
  ([`../baselines.py`](../baselines.py)), not the authors' package, and has been
  checked against the two cases with known answers (`K = I` → *n*, `K` all-ones → 1).
