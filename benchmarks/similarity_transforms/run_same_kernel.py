#!/usr/bin/env python
"""One similarity matrix, every index: what does the *index* contribute?

The comparison tables so far confound two things. Our best configuration used a
cross-encoder while the published alternatives used bi-encoder cosine, so a lead
could be the index or could be the representation. This holds the representation
fixed -- the identical cross-encoder matrix for everyone -- and varies only the
index.

Not every competitor can take part, and that is itself the finding:

    Hill (Leinster-Cobbold)  takes a similarity matrix        -> included
    Vendi Score              takes a similarity matrix        -> included
    pVS_q (this library)     takes a similarity matrix        -> included
    PRDC                     takes a metric space; 1 - Z      -> included
    distinct-n, Self-BLEU    surface statistics, no matrix    -> cannot participate
    Decan                    LM log-probabilities, no matrix  -> cannot participate

The last row is the honest boundary of this experiment: three of the baselines have
no notion of a similarity matrix, so "same kernel" is undefined for them and the
earlier comparison against them was measuring something else.

Usage:
    python run_same_kernel.py --limit 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from baselines import prdc, vendi_score  # noqa: E402
from common import CACHE, HERE, load_calibration_sets, load_human_sets  # noqa: E402

from linguistic_diversity.diversities.semantic import (  # noqa: E402
    _load_cross_encoder,
    cross_encoder_similarities,
)
from linguistic_diversity.metric import TextDiversity, _nearest_psd  # noqa: E402

OUT = HERE / "output" / "same_kernel.json"
STS = "cross-encoder/stsb-roberta-large"
NAMES = ("McDiv_nuggets", "conTest", "decTest")


def cached_matrices(tag: str, sets: list[list[str]], model: Any) -> list[np.ndarray]:
    path = CACHE / f"sts-{tag}-{len(sets)}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return [np.asarray(z, dtype=np.float64) for z in d["Z"]]
    print(f"  scoring {tag} ({len(sets)} sets) ...")
    out = [cross_encoder_similarities(s, model, batch_size=64) for s in sets]
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, Z=np.array(out, dtype=object))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    human = load_human_sets()
    calib = load_calibration_sets()

    # The held-out split, as everywhere else.
    rng = np.random.default_rng(20260803)
    tasks: dict[str, dict[str, Any]] = {}
    for name, d in human.items():
        chosen = rng.choice(
            np.arange(len(d["sets"])), min(args.limit, len(d["sets"])), replace=False
        )
        idx = np.setdiff1d(np.arange(len(d["sets"])), chosen)
        tasks[name] = {"sets": [d["sets"][i] for i in idx], "target": d["human"][idx]}
    tasks["calibration"] = {"sets": calib["sets"], "target": calib["expected"]}

    model = _load_cross_encoder(STS, "cuda")
    for name, t in tasks.items():
        t["Z"] = cached_matrices(name, t["sets"], model)

    def hill(Z: np.ndarray) -> float:
        p = np.full(len(Z), 1.0 / len(Z))
        return TextDiversity._calc_diversity(p, Z, q=1.0, index="hill")

    def ours(Z: np.ndarray) -> float:
        p = np.full(len(Z), 1.0 / len(Z))
        return TextDiversity._calc_diversity(p, Z, q=1.0, index="vendi")

    def reference_vendi(Z: np.ndarray) -> float:
        # The published Vendi Score is undefined on a matrix that is not a kernel,
        # and a cross-encoder matrix is not one -- these reach -9e-4. So it only
        # runs at all via the PSD projection, which is part of what this library
        # adds. Passing the raw matrix raises, by design.
        return vendi_score(_nearest_psd(Z), q=1.0)

    def prdc_coverage(Z: np.ndarray) -> float:
        # Embed the matrix's own geometry: classical MDS on 1 - Z gives PRDC a metric
        # space consistent with the kernel every other row is using.
        D = 1.0 - Z
        n = len(D)
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D**2) @ J
        w, V = np.linalg.eigh(B)
        keep = w > 1e-9
        X = V[:, keep] * np.sqrt(w[keep])
        if X.shape[1] == 0:
            return 0.0
        half = max(2, n // 2)
        return prdc(X[:half], X[half:], k=2)["coverage"]

    n_nonpsd = sum(
        1
        for name in NAMES
        for Z in tasks[name]["Z"]
        if np.linalg.eigvalsh((Z + Z.T) / 2).min() < -1e-6
    )
    n_total = sum(len(tasks[name]["Z"]) for name in NAMES)
    print(f"\n  {n_nonpsd}/{n_total} cross-encoder matrices are NOT positive semi-definite,")
    print("  so the published Vendi Score is undefined on them without a PSD projection.")

    indices = {
        "Hill (Leinster-Cobbold)": hill,
        "Vendi Score (reference impl)": reference_vendi,
        "pVS_q (this library)": ours,
        "PRDC coverage": prdc_coverage,
    }

    hdr = list(NAMES) + ["mean", "calib_rho", "calib_ratio"]
    print("\n" + "=" * 124)
    print(f"SAME KERNEL FOR EVERY INDEX  ({STS}, held-out split)")
    print("=" * 124)
    print(f"  {'index':30s} " + " ".join(f"{h:>13s}" for h in hdr))
    print("  " + "-" * 122)

    results: dict[str, Any] = {}
    scored: dict[str, dict[str, np.ndarray]] = {}
    for label, fn in indices.items():
        row: dict[str, Any] = {}
        scored[label] = {}
        for name in NAMES:
            s = np.array([fn(Z) for Z in tasks[name]["Z"]], dtype=float)
            scored[label][name] = s
            row[name] = round(float(spearmanr(s, tasks[name]["target"]).statistic), 4)
        row["mean"] = round(float(np.mean([row[n] for n in NAMES])), 4)
        s = np.array([fn(Z) for Z in tasks["calibration"]["Z"]], dtype=float)
        scored[label]["calibration"] = s
        row["calib_rho"] = round(float(spearmanr(s, tasks["calibration"]["target"]).statistic), 4)
        row["calib_ratio"] = round(float(np.median(s / tasks["calibration"]["target"])), 4)
        results[label] = row
        print(f"  {label:30s} " + " ".join(f"{row[h]:+13.4f}" for h in hdr))

    # The identity that explains the table.
    diffs = [
        float(
            np.max(
                np.abs(
                    scored["pVS_q (this library)"][n] - scored["Vendi Score (reference impl)"][n]
                )
            )
        )
        for n in NAMES
    ]
    print(f"\n  max |pVS_q - Vendi| across all sets: {max(diffs):.2e}")
    print(
        "  At uniform abundance diag(sqrt p) Z diag(sqrt p) = Z/n, so pVS_q IS the\n"
        "  Vendi Score. On uniformly-weighted benchmarks the two cannot differ, and\n"
        "  any gap reported between them was a difference of kernel, not of index."
    )
    results["_pvs_vendi_max_abs_diff"] = max(diffs)

    # Where the index *does* matter: abundance. Same matrices, same sets, but the
    # documents are no longer equally frequent -- which is the normal case in a real
    # corpus and the one input the spectral form cannot accept.
    print(f"\n{'=' * 124}")
    print("WHERE THE INDEX DOES DIFFER  identical kernel, non-uniform abundance")
    print("=" * 124)
    rows = []
    for name in NAMES:
        for Z in tasks[name]["Z"][:200]:
            n = len(Z)
            zipf = 1.0 / np.arange(1, n + 1)
            zipf = zipf / zipf.sum()
            rows.append(
                (
                    TextDiversity._calc_diversity(zipf, Z, q=1.0, index="vendi"),
                    vendi_score(_nearest_psd(Z), q=1.0),
                    TextDiversity._calc_diversity(np.full(n, 1 / n), Z, q=1.0, index="vendi"),
                )
            )
    weighted, spectral, uniform = map(np.array, zip(*rows, strict=True))
    print(f"  {len(rows)} sets, Zipfian abundance over the same similarity matrices")
    print(f"  {'quantity':44s} {'mean':>10s} {'sd':>10s}")
    print(f"  {'pVS_q with abundance (ours)':44s} {weighted.mean():10.4f} {weighted.std():10.4f}")
    print(
        f"  {'Vendi Score (cannot accept abundance)':44s} {spectral.mean():10.4f} {spectral.std():10.4f}"
    )
    print(f"  {'pVS_q at uniform abundance':44s} {uniform.mean():10.4f} {uniform.std():10.4f}")
    print(
        f"\n  max |Vendi - pVS_q(uniform)| = {np.max(np.abs(spectral - uniform)):.2e}   (identical)\n"
        f"  mean |Vendi - pVS_q(Zipf)|   = {np.mean(np.abs(spectral - weighted)):.4f}   (diverge)\n"
        "  The spectral form takes only the matrix, so it returns the same number for\n"
        "  every weighting of the same documents. That is the whole algorithmic\n"
        "  difference, and these benchmarks -- all uniformly weighted -- cannot see it."
    )
    results["abundance"] = {
        "weighted_mean": round(float(weighted.mean()), 4),
        "spectral_mean": round(float(spectral.mean()), 4),
        "uniform_mean": round(float(uniform.mean()), 4),
        "mean_abs_diff_zipf": round(float(np.mean(np.abs(spectral - weighted))), 4),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
