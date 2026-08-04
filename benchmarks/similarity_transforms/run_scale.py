#!/usr/bin/env python
"""Why does the tuned kernel win at n=5 and lose at n=200?

The competitive comparison found our best kernel ahead of everything on five-item
response sets and *behind* plain Vendi on 200-document temperature bins. Since our
index at uniform abundance is the Vendi Score applied to a transformed matrix, the
whole gap has to live in the transform. This finds out which part, and at what size
it turns.

Three candidate mechanisms, all testable:

  saturation   The floor sends every pair below z0 to exactly 0. The larger the
               corpus, the greater the share of unrelated pairs, so Z tends to the
               identity and D tends to n for *every* corpus -- discrimination
               collapses even as the number gets bigger.
  no context   The biggest single correction projects out a shared prompt. Pooled
               corpora have no single prompt, so it is unavailable at scale.
  psd repair   The transforms are not kernels, so the spectral index routes them
               through _nearest_psd. Whatever that does wrong grows with n.

Usage:
    python run_scale.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import transforms as T  # noqa: E402
from common import DATA, HERE, encode, load_background  # noqa: E402

from linguistic_diversity.metric import TextDiversity, _nearest_psd  # noqa: E402

OUT = HERE / "output" / "scale.json"
SIZES = (5, 10, 25, 50, 100, 200, 400)


def temperature_bins(n_bins: int = 10) -> list[tuple[float, list[str]]]:
    files = sorted(glob.glob(str(DATA / "raw/decTest/*no_hds*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    resp_cols = [c for c in df.columns if c.startswith("resp_")]
    temps = df["label_value"].to_numpy(dtype=float)
    edges = np.quantile(temps, np.linspace(0, 1, n_bins + 1))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        rows = df[(temps >= lo) & (temps < hi)]
        docs: list[str] = []
        for _, row in rows.iterrows():
            docs += [
                str(row[c]) for c in resp_cols if isinstance(row[c], str) and str(row[c]).strip()
            ]
        if len(docs) >= max(SIZES):
            out.append((float((lo + hi) / 2), docs[: max(SIZES)]))
    return out


def psd_audit(
    bins: list[tuple[float, list[str]]], embeds: list[np.ndarray], bg: np.ndarray
) -> dict:
    """How much does the PSD repair actually move the matrix, and which way?"""
    print("=" * 104)
    print("MECHANISM 3  What the PSD repair does, by corpus size")
    print("=" * 104)
    ls_tau = T.local_scaling(bg, k=10, z0=0.30)
    print(
        f"  {'n':>5s} {'min eig before':>15s} {'neg mass %':>11s} "
        f"{'rank before':>12s} {'rank after':>11s} {'clip changed':>13s}"
    )
    rows = {}
    for n in SIZES:
        E = embeds[0][:n]
        Z = ls_tau(E, None)
        ev = np.linalg.eigvalsh((Z + Z.T) / 2)
        neg_mass = float(np.abs(ev[ev < 0]).sum() / np.abs(ev).sum())
        # Replicate _nearest_psd but stop before the clip, to isolate its effect.
        evals, vecs = np.linalg.eigh((Z + Z.T) / 2)
        proj = vecs @ np.diag(np.clip(evals, 0.0, None)) @ vecs.T
        scale = np.sqrt(np.clip(np.diag(proj), 1e-12, None))
        proj = proj / np.outer(scale, scale)
        clipped = _nearest_psd(Z)
        moved = float(np.linalg.norm(clipped - proj))
        rows[n] = {
            "min_eig": round(float(ev.min()), 6),
            "neg_mass_frac": round(neg_mass, 6),
            "rank_before": int(np.linalg.matrix_rank(Z)),
            "rank_after": int(np.linalg.matrix_rank(clipped)),
            "clip_frobenius": round(moved, 4),
        }
        print(
            f"  {n:5d} {ev.min():15.6f} {100 * neg_mass:10.3f}% "
            f"{rows[n]['rank_before']:12d} {rows[n]['rank_after']:11d} {moved:13.4f}"
        )
    print(
        "\n  _nearest_psd projects onto the PSD cone and then clips off-diagonal entries\n"
        "  back into [0, 1] -- which un-does the projection. 'clip changed' is how far\n"
        "  that last step moves the matrix away from the PSD matrix just computed.\n"
    )
    return rows


def scale_sweep(
    bins: list[tuple[float, list[str]]], embeds: list[np.ndarray], bg: np.ndarray
) -> dict:
    print("=" * 104)
    print("MECHANISMS 1 & 2  Discrimination against temperature, by corpus size")
    print("=" * 104)
    centres = np.array([c for c, _ in bins])
    variants: dict[str, Any] = {
        "raw cosine (= Vendi)": T.cosine,
        "floor 0.053 (shipped)": T.floored(0.053),
        "tau 0.70 (z0=0.30)": T.tau_truncated(0.70),
        "local scaling k=10": T.local_scaling(bg, k=10),
        "LS + tau (best small-set)": T.local_scaling(bg, k=10, z0=0.30),
    }

    print(f"  {'variant':28s} " + " ".join(f"{f'n={n}':>10s}" for n in SIZES))
    print("  " + "-" * 100)
    rho_rows: dict[str, dict[int, float]] = {}
    sat_rows: dict[str, dict[int, float]] = {}
    for label, fn in variants.items():
        rho_rows[label], sat_rows[label] = {}, {}
        for n in SIZES:
            scores, sat = [], []
            for E in embeds:
                Z = fn(E[:n], None)
                p = np.full(n, 1.0 / n)
                d = TextDiversity._calc_diversity(p, Z, q=1.0, index="vendi")
                scores.append(d)
                sat.append(d / n)
            rho_rows[label][n] = round(float(spearmanr(scores, centres).statistic), 4)
            sat_rows[label][n] = round(float(np.mean(sat)), 4)
        print(f"  {label:28s} " + " ".join(f"{rho_rows[label][n]:+10.4f}" for n in SIZES))

    print(
        f"\n  Saturation: mean D / n. Approaching 1 means Z has become the identity\n"
        f"  and every corpus reports its own size, whatever is in it."
    )
    print(f"  {'variant':28s} " + " ".join(f"{f'n={n}':>10s}" for n in SIZES))
    print("  " + "-" * 100)
    for label in variants:
        print(f"  {label:28s} " + " ".join(f"{sat_rows[label][n]:10.4f}" for n in SIZES))
    return {"rho": rho_rows, "saturation": sat_rows}


def floor_by_size(bins: list[tuple[float, list[str]]], embeds: list[np.ndarray]) -> dict:
    """Which floor is right at which corpus size?

    The floor exists because D_max = n/(1 + (n-1)z) caps diversity at 1/z. That cap is
    a function of n, so the correction for it should be too -- a floor chosen on
    five-item sets has no reason to be right at four hundred. Two criteria, and they
    pull in opposite directions: rank agreement wants an aggressive floor, the
    effective-number reading wants one that does not drive Z to the identity.
    """
    print("=" * 104)
    print("ACTIONABLE  Best floor by corpus size: ranking versus saturation")
    print("=" * 104)
    centres = np.array([c for c, _ in bins])
    floors = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"  {'n':>5s} " + " ".join(f"{f'z0={z:g}':>12s}" for z in floors) + "   best")
    out: dict[str, Any] = {"rho": {}, "saturation": {}, "best": {}}
    for n in SIZES:
        rhos, sats = [], []
        for z0 in floors:
            fn = T.floored(z0)
            scores = []
            for E in embeds:
                Z = fn(E[:n], None)
                p = np.full(n, 1.0 / n)
                scores.append(TextDiversity._calc_diversity(p, Z, q=1.0, index="vendi"))
            rhos.append(float(spearmanr(scores, centres).statistic))
            sats.append(float(np.mean(scores)) / n)
        best = floors[int(np.argmax(rhos))]
        out["rho"][n] = [round(r, 4) for r in rhos]
        out["saturation"][n] = [round(s, 4) for s in sats]
        out["best"][n] = best
        print(f"  {n:5d} " + " ".join(f"{r:+12.4f}" for r in rhos) + f"   z0={best:g}")

    print(f"\n  Saturation (mean D / n) at the same floors -- 1.0 means Z has become the")
    print(f"  identity and the score is just the corpus size.")
    print(f"  {'n':>5s} " + " ".join(f"{f'z0={z:g}':>12s}" for z in floors))
    for n in SIZES:
        print(f"  {n:5d} " + " ".join(f"{s:12.4f}" for s in out["saturation"][n]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    bins = temperature_bins()
    print(f"{len(bins)} temperature bins, {max(SIZES)} documents each\n")
    embeds = [
        np.asarray(
            st.encode(d, show_progress_bar=False, normalize_embeddings=True), dtype=np.float64
        )
        for _, d in bins
    ]
    bg = encode(load_background(), "background", st)
    bg_small = bg[np.random.default_rng(0).choice(len(bg), 2000, replace=False)]

    results = {
        "scale": scale_sweep(bins, embeds, bg_small),
        "psd_audit": psd_audit(bins, embeds, bg_small),
        "floor_by_size": floor_by_size(bins, embeds),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
