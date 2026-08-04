#!/usr/bin/env python
"""Do the winning kernels still satisfy the diversity axioms, and what do they cost?

A better correlation is worthless if the number stops being an effective count. Each
candidate is checked for the properties the library guarantees today, and timed, since
the entailment kernel is O(n^2) cross-encoder passes against O(n) encodes.

Replication is the one that matters: pooling k mutually dissimilar corpora must give
exactly k times the diversity. It is also the property most at risk here, because two
of the three corrections (local scaling, context projection) are per-document maps and
the third (NLI) is pairwise.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import transforms as T  # noqa: E402
from common import encode, load_background  # noqa: E402
from run_nli import nli_matrices  # noqa: E402

from linguistic_diversity.metric import TextDiversity  # noqa: E402

THEMES = [
    ["The stock market closed higher on Tuesday.", "Investors bought bank shares."],
    ["She baked a loaf of sourdough bread.", "The dough rose overnight in the kitchen."],
    ["The telescope detected a distant galaxy.", "Astronomers measured its redshift."],
    ["He repaired the bicycle's rear brake.", "The gear cable had frayed badly."],
    ["The court dismissed the appeal.", "Lawyers filed a motion the next morning."],
]


def build(kernel: str, docs: list[str], E: np.ndarray, ctx, bg_small, model: str):
    ctx_ls = T.compose_context_then(T.local_scaling(bg_small, k=10, z0=0.30))
    if kernel == "cosine+floor":
        return T.floored(0.053)(E, None)
    if kernel == "ctx+LS+tau":
        return ctx_ls(E, ctx)
    nli = nli_matrices([docs], model, progress=False)[0]
    Z_nli = nli[1]
    if kernel == "NLI-ec":
        return Z_nli
    Z_emb = ctx_ls(E, ctx)
    if kernel == "hybrid-geometric":
        Z = np.sqrt(np.clip(Z_nli, 0, 1) * np.clip(Z_emb, 0, 1))
    else:
        Z = 0.5 * (Z_nli + Z_emb)
    np.fill_diagonal(Z, 1.0)
    return Z


def main() -> None:
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    bg = encode(load_background(), "background", st)
    bg_small = bg[np.random.default_rng(0).choice(len(bg), 2000, replace=False)]
    model = "cross-encoder/nli-deberta-v3-base"

    kernels = ["cosine+floor", "ctx+LS+tau", "NLI-ec", "hybrid-geometric", "hybrid-arithmetic"]

    print("=" * 100)
    print("REPLICATION   pooling k mutually dissimilar 2-document corpora must give k x D")
    print("=" * 100)
    print(f"  {'kernel':22s} " + " ".join(f"{f'k={k}':>12s}" for k in (2, 3, 5)))
    for kernel in kernels:
        cells = []
        base = None
        for k in (1, 2, 3, 5):
            docs = [d for theme in THEMES[:k] for d in theme]
            E = np.asarray(st.encode(docs, normalize_embeddings=True), dtype=np.float64)
            Z = build(kernel, docs, E, None, bg_small, model)
            p = np.full(len(docs), 1.0 / len(docs))
            D = TextDiversity._calc_diversity(p, Z, q=1.0, index="vendi")
            if k == 1:
                base = D
            else:
                cells.append(f"{D / base:12.4f}")
        print(f"  {kernel:22s} " + " ".join(cells))

    print(f"\n{'=' * 100}")
    print("BOUNDS AND IDENTITY")
    print("=" * 100)
    print(f"  {'kernel':22s} {'D(5 identical)':>16s} {'D(5 distinct)':>16s} {'in [1,n]':>10s}")
    for kernel in kernels:
        same = ["The cat sat on the mat."] * 5
        diff = [t[0] for t in THEMES]
        vals = []
        for docs in (same, diff):
            E = np.asarray(st.encode(docs, normalize_embeddings=True), dtype=np.float64)
            Z = build(kernel, docs, E, None, bg_small, model)
            p = np.full(5, 0.2)
            vals.append(TextDiversity._calc_diversity(p, Z, q=1.0, index="vendi"))
        ok = "yes" if (0.99 <= vals[0] <= 5.01 and 0.99 <= vals[1] <= 5.01) else "NO"
        print(f"  {kernel:22s} {vals[0]:16.4f} {vals[1]:16.4f} {ok:>10s}")

    print(f"\n{'=' * 100}")
    print("MONOTONICITY   raising every similarity must never raise diversity")
    print("=" * 100)
    rng = np.random.default_rng(7)
    violations = 0
    for _ in range(200):
        n = rng.integers(3, 9)
        A = rng.random((n, n)) * 0.6
        A = (A + A.T) / 2
        np.fill_diagonal(A, 1.0)
        B = np.clip(A + rng.random((n, n)) * 0.2, 0, 1)
        B = (B + B.T) / 2
        np.fill_diagonal(B, 1.0)
        p = rng.dirichlet(np.ones(n))
        da = TextDiversity._calc_diversity(p, A, q=1.0, index="vendi")
        db = TextDiversity._calc_diversity(p, B, q=1.0, index="vendi")
        if db > da + 1e-9:
            violations += 1
    print(f"  blended-matrix shape, 200 random trials: {violations} violations")

    print(f"\n{'=' * 100}")
    print("COST   wall-clock to score one corpus")
    print("=" * 100)
    print(f"  {'n docs':>8s} {'cosine':>12s} {'NLI':>12s} {'ratio':>10s} {'pairs':>10s}")
    pool = [d for theme in THEMES for d in theme] * 8
    for n in (5, 10, 20, 40):
        docs = pool[:n]
        t0 = time.perf_counter()
        st.encode(docs, normalize_embeddings=True, show_progress_bar=False)
        t_cos = time.perf_counter() - t0
        t0 = time.perf_counter()
        nli_matrices([docs], model, progress=False)
        t_nli = time.perf_counter() - t0
        print(f"  {n:8d} {t_cos:11.3f}s {t_nli:11.3f}s {t_nli / t_cos:9.1f}x " f"{n * (n - 1):10d}")


if __name__ == "__main__":
    main()
