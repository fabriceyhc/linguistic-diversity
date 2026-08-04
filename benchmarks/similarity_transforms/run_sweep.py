#!/usr/bin/env python
"""Sweep candidate similarity corrections, then combine whatever survives.

Stage 1 tests each angle from the literature review on its own; stage 2 crosses the
survivors. Everything is scored on the same cached embeddings, so differences between
rows are the transform and nothing else.

Usage:
    python run_sweep.py                 # stages 1 and 2
    python run_sweep.py --stage 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import transforms as T  # noqa: E402
from common import (  # noqa: E402
    HERE,
    encode,
    evaluate,
    load_background,
    load_calibration_sets,
    load_human_sets,
    print_header,
    print_row,
)

OUT = HERE / "output" / "sweep.json"
BG_SUB = 2000  # background subsample for the O(n^2 * n_bg) transforms


def prepare() -> tuple[dict[str, Any], np.ndarray]:
    """Encode every set once, keyed so transforms can be swapped for free."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        T.__dict__.get("ENCODER", "sentence-transformers/all-mpnet-base-v2")
    )

    human = load_human_sets()
    calib = load_calibration_sets()

    def attach(d: dict[str, Any], tag: str) -> None:
        flat: list[str] = []
        bounds = []
        for s in d["sets"]:
            bounds.append((len(flat), len(flat) + len(s)))
            flat += s
        E = encode(flat, f"{tag}-docs", model)
        d["embeds"] = [E[a:b] for a, b in bounds]
        ctx = d.get("contexts") or []
        if any(c.strip() for c in ctx):
            C = encode([c if c.strip() else " " for c in ctx], f"{tag}-ctx", model)
            d["ctx"] = [C[i] if ctx[i].strip() else None for i in range(len(ctx))]
        else:
            d["ctx"] = [None] * len(d["sets"])

    for name, d in human.items():
        print(f"encoding {name} ({len(d['sets'])} sets) ...")
        attach(d, name)
    print(f"encoding calibration ({len(calib['sets'])} sets) ...")
    attach(calib, "calibration")

    print("encoding background ...")
    bg = encode(load_background(), "background", model)

    n_ctx = sum(c is not None for d in human.values() for c in d["ctx"])
    print(
        f"\nready: {sum(len(d['sets']) for d in human.values())} human sets "
        f"({n_ctx} with context), {len(calib['sets'])} calibration sets, "
        f"{len(bg)} background texts\n"
    )
    return {"human": human, "calibration": calib}, bg


def stage1(prepared: dict[str, Any], bg: np.ndarray) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    bg_small = bg[rng.choice(len(bg), BG_SUB, replace=False)]

    # Background baseline statistics, used to set the principled defaults.
    S = bg_small[:1500] @ bg_small[:1500].T
    off = S[~np.eye(1500, dtype=bool)]
    z_med, z_mean = float(np.median(off)), float(off.mean())
    tau_mean = 1.0 - z_mean
    print(f"background similarity: median {z_med:.4f}  mean {z_mean:.4f}")
    print(f"Chao tau = mean distance: {tau_mean:.4f}  (equivalent floor z0 = {z_mean:.4f})\n")

    mu, W = T.whitener(bg)

    configs: dict[str, Any] = {
        # -- where we are today
        "baseline: raw cosine": T.cosine,
        "CURRENT: floor 0.053": T.floored(0.053),
        # -- angle 1: tau-truncation, i.e. how aggressive should the floor be?
        "tau=0.99 (z0=0.01)": T.tau_truncated(0.99),
        "tau=0.95 (z0=0.05)": T.tau_truncated(0.95),
        "tau=0.90 (z0=0.10)": T.tau_truncated(0.90),
        "tau=0.85 (z0=0.15)": T.tau_truncated(0.85),
        "tau=0.80 (z0=0.20)": T.tau_truncated(0.80),
        "tau=0.70 (z0=0.30)": T.tau_truncated(0.70),
        "tau=0.60 (z0=0.40)": T.tau_truncated(0.60),
        "tau=0.50 (z0=0.50)": T.tau_truncated(0.50),
        f"tau=dmean ({tau_mean:.3f})": T.tau_truncated(tau_mean),
        # -- angle 2: global linear (already known to fail; kept as control)
        "whitened": T.whitened(mu, W),
        # -- angle 3: local / hubness-aware
        "local scaling k=10": T.local_scaling(bg_small, k=10),
        "local scaling k=50": T.local_scaling(bg_small, k=50),
        "local scaling k=200": T.local_scaling(bg_small, k=200),
        "mutual proximity": T.mutual_proximity(bg_small[:800]),
        # -- angle 4: prompt-conditional
        "context projected": T.context_projected(),
        "context projected + floor": T.context_projected(z0=0.053),
    }

    print("=" * 132)
    print("STAGE 1  Each angle on its own  (index=vendi, q=1)")
    print("=" * 132)
    print_header()
    results = {}
    for label, fn in configs.items():
        row = evaluate(prepared, fn)
        results[label] = row
        print_row(label, row)
    return results


def stage2(
    prepared: dict[str, Any], bg: np.ndarray, stage1_results: dict[str, Any]
) -> dict[str, Any]:
    """Cross the survivors, and sweep the order parameter on the winner."""
    rng = np.random.default_rng(0)
    bg_small = bg[rng.choice(len(bg), BG_SUB, replace=False)]

    # Best floor found in stage 1, by mean agreement.
    tau_rows = {k: v for k, v in stage1_results.items() if k.startswith("tau=")}
    best_tau_label = max(tau_rows, key=lambda k: tau_rows[k]["agreement_mean"])
    best_tau = float(best_tau_label.split("=")[1].split()[0])
    print(f"\nbest tau from stage 1: {best_tau_label} -> tau={best_tau}")

    ls_rows = {k: v for k, v in stage1_results.items() if k.startswith("local scaling")}
    best_ls = max(ls_rows, key=lambda k: ls_rows[k]["agreement_mean"])
    best_k = int(best_ls.split("k=")[1])
    print(f"best local scaling: {best_ls}\n")

    configs: dict[str, Any] = {
        "context + tau": T.compose_context_then(T.tau_truncated(best_tau)),
        "context + local scaling": T.compose_context_then(T.local_scaling(bg_small, k=best_k)),
        "local scaling + tau": T.local_scaling(bg_small, k=best_k, z0=1.0 - best_tau),
        "context + LS + tau": T.compose_context_then(
            T.local_scaling(bg_small, k=best_k, z0=1.0 - best_tau)
        ),
    }

    print("=" * 132)
    print("STAGE 2a  Combinations")
    print("=" * 132)
    print_header()
    results = {}
    for label, fn in configs.items():
        row = evaluate(prepared, fn)
        results[label] = row
        print_row(label, row)

    print(f"\n{'=' * 132}")
    print("STAGE 2b  Order parameter q, on the best single transform")
    print("=" * 132)
    print_header()
    best_single = T.tau_truncated(best_tau)
    for q in (0.0, 0.5, 1.0, 2.0, 4.0):
        for index in ("vendi", "hill"):
            row = evaluate(prepared, best_single, index=index, q=q)
            label = f"{index} q={q}"
            results[label] = row
            print_row(label, row)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    prepared, bg = prepare()
    out: dict[str, Any] = {}
    s1 = stage1(prepared, bg)
    out["stage1"] = s1
    if args.stage in (0, 2):
        out["stage2"] = stage2(prepared, bg, s1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
