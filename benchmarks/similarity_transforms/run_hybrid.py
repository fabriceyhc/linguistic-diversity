#!/usr/bin/env python
"""Head-to-head on identical data, then hybrid kernels.

run_sweep.py scored embedding transforms on every set; run_nli.py scored the
entailment kernel on a subsample. Those numbers are not comparable, so everything is
re-scored here on one fixed subsample.

Then the question the two halves raise: the entailment kernel and the (context-
projected, hubness-corrected) cosine kernel are built from different evidence, so a
combination may beat both. Four ways of combining are tried; none of them can break
the axioms, since an elementwise mean, product or minimum of two matrices with unit
diagonal and entries in [0,1] has the same properties.

Usage:
    python run_hybrid.py --limit 400
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import transforms as T  # noqa: E402
from common import (  # noqa: E402
    CACHE,
    HERE,
    encode,
    load_background,
    load_calibration_sets,
    load_human_sets,
)
from run_nli import nli_matrices  # noqa: E402

OUT = HERE / "output" / "hybrid.json"
OUT_HOLD = HERE / "output" / "hybrid_holdout.json"
MODEL = "cross-encoder/nli-deberta-v3-base"
BG_SUB = 2000


def cached_nli(name: str, sets: list[list[str]], model: str) -> list[tuple[np.ndarray, np.ndarray]]:
    path = CACHE / f"nli-{name}-{len(sets)}.npz"  # length distinguishes splits
    if path.exists():
        d = np.load(path, allow_pickle=True)
        # Saved as object arrays when set sizes differ; restore float64 or the
        # elementwise blends below silently produce object-dtype matrices.
        return [
            (np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
            for a, b in zip(d["ent"], d["ec"], strict=True)
        ]
    mats = nli_matrices(sets, model)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ent=np.array([m[0] for m in mats], dtype=object),
        ec=np.array([m[1] for m in mats], dtype=object),
    )
    return mats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="score the COMPLEMENT of the selection subsample -- sets no configuration "
        "was chosen on. Roughly 35 configurations have been tried against the "
        "selection split, so its numbers are optimistic by construction.",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    from linguistic_diversity.metric import TextDiversity
    from linguistic_diversity.utils import maximum_diversity

    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    human = load_human_sets()
    calib = load_calibration_sets()

    # One fixed subsample, shared by every configuration.
    rng = np.random.default_rng(20260803)
    tasks: dict[str, dict[str, Any]] = {}
    for name, d in human.items():
        chosen = rng.choice(
            np.arange(len(d["sets"])), min(args.limit, len(d["sets"])), replace=False
        )
        idx = np.setdiff1d(np.arange(len(d["sets"])), chosen) if args.holdout else chosen
        tasks[name] = {
            "sets": [d["sets"][i] for i in idx],
            "contexts": [d["contexts"][i] for i in idx],
            "target": d["human"][idx],
            "kind": "human",
        }
    tasks["calibration"] = {
        "sets": calib["sets"],
        "contexts": [""] * len(calib["sets"]),
        "target": calib["expected"],
        "kind": "calib",
    }

    bg = encode(load_background(), "background", st)
    bg_small = bg[np.random.default_rng(0).choice(len(bg), BG_SUB, replace=False)]

    # Embeddings and NLI matrices for every set, computed once.
    for name, t in tasks.items():
        flat, bounds = [], []
        for s in t["sets"]:
            bounds.append((len(flat), len(flat) + len(s)))
            flat += s
        E = encode(flat, f"hy-{name}-docs", st)
        t["embeds"] = [E[a:b] for a, b in bounds]
        has_ctx = any(c.strip() for c in t["contexts"])
        if has_ctx:
            C = encode([c if c.strip() else " " for c in t["contexts"]], f"hy-{name}-ctx", st)
            t["ctx"] = [C[i] if t["contexts"][i].strip() else None for i in range(len(C))]
        else:
            t["ctx"] = [None] * len(t["sets"])
        print(f"NLI for {name} ...")
        t["nli"] = cached_nli(name, t["sets"], args.model)

    # ------------------------------------------------------------------ configs
    ctx_ls_tau = T.compose_context_then(T.local_scaling(bg_small, k=10, z0=0.30))
    ctx_tau = T.compose_context_then(T.tau_truncated(0.70))

    def emb(fn: Callable) -> Callable:
        return lambda t, i: fn(t["embeds"][i], t["ctx"][i])

    def nli(pos: int) -> Callable:
        return lambda t, i: t["nli"][i][pos]

    def blend(a: Callable, b: Callable, how: str) -> Callable:
        def f(t: dict, i: int) -> np.ndarray:
            A, B = a(t, i), b(t, i)
            if how == "geometric":
                Z = np.sqrt(np.clip(A, 0, 1) * np.clip(B, 0, 1))
            elif how == "arithmetic":
                Z = 0.5 * (A + B)
            elif how == "product":
                Z = A * B
            else:
                Z = np.minimum(A, B)
            np.fill_diagonal(Z, 1.0)
            return Z

        return f

    configs: dict[str, Callable] = {
        "CURRENT: cosine + floor 0.053": emb(T.floored(0.053)),
        "best embedding: ctx + LS + tau": emb(ctx_ls_tau),
        "ctx + tau": emb(ctx_tau),
        "NLI entailment": nli(0),
        "NLI ent-contra": nli(1),
        "NLI-ec + cosine (geometric)": blend(nli(1), emb(T.floored(0.053)), "geometric"),
        "NLI-ec + ctxLS (geometric)": blend(nli(1), emb(ctx_ls_tau), "geometric"),
        "NLI-ec + ctxLS (arithmetic)": blend(nli(1), emb(ctx_ls_tau), "arithmetic"),
        "NLI-ec + ctxLS (product)": blend(nli(1), emb(ctx_ls_tau), "product"),
        "NLI-ec + ctxLS (min)": blend(nli(1), emb(ctx_ls_tau), "min"),
        "NLI-ent + ctxLS (geometric)": blend(nli(0), emb(ctx_ls_tau), "geometric"),
        "NLI-ent + ctxLS (min)": blend(nli(0), emb(ctx_ls_tau), "min"),
    }

    names = ["McDiv_nuggets", "conTest", "decTest"]
    hdr = names + ["agreement_mean", "calib_rho", "calib_ratio", "ceiling"]
    print("\n" + "=" * 140)
    split = (
        "HELD OUT (never selected on)"
        if args.holdout
        else f"selection split ({args.limit}/dataset)"
    )
    print(f"HEAD TO HEAD  {split}, index=vendi q=1")
    print("=" * 140)
    print(f"  {'configuration':32s} " + " ".join(f"{h:>14s}" for h in hdr))
    print("  " + "-" * 138)

    results: dict[str, Any] = {}
    for label, fn in configs.items():
        row: dict[str, Any] = {}
        for name in names:
            t = tasks[name]
            s = np.array(
                [
                    TextDiversity._calc_diversity(
                        np.full(len(t["sets"][i]), 1 / len(t["sets"][i])),
                        fn(t, i),
                        q=1.0,
                        index="vendi",
                    )
                    for i in range(len(t["sets"]))
                ]
            )
            row[name] = round(float(spearmanr(s, t["target"]).statistic), 4)
        row["agreement_mean"] = round(float(np.mean([row[n] for n in names])), 4)

        t = tasks["calibration"]
        s = np.array(
            [
                TextDiversity._calc_diversity(
                    np.full(len(t["sets"][i]), 1 / len(t["sets"][i])),
                    fn(t, i),
                    q=1.0,
                    index="vendi",
                )
                for i in range(len(t["sets"]))
            ]
        )
        row["calib_rho"] = round(float(spearmanr(s, t["target"]).statistic), 4)
        row["calib_ratio"] = round(float(np.median(s / t["target"])), 4)
        tm = tasks["McDiv_nuggets"]
        row["ceiling"] = round(
            float(np.mean([maximum_diversity(fn(tm, i))[0] for i in range(len(tm["sets"]))])), 3
        )

        results[label] = row
        cells = " ".join(f"{row[h]:+14.4f}" if h != "ceiling" else f"{row[h]:14.3f}" for h in hdr)
        print(f"  {label:32s} {cells}")

    out_path = OUT_HOLD if args.holdout else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"limit": args.limit, "model": args.model, "holdout": args.holdout, "results": results},
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
