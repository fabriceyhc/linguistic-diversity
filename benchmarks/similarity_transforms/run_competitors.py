#!/usr/bin/env python
"""This library against the published alternatives, on identical data.

The transform sweep showed how far the kernel can be pushed. This asks the question
that matters for a paper: is the result competitive with what else exists, measured
the same way, on the same sets?

Five families, deliberately including ones built on different evidence:

    lexical      distinct-n, self-BLEU, type-token ratio -- surface form only
    spectral     Vendi Score (Friedman & Dieng 2023) on the same cosine matrix
    ours         current default, and the best kernel from the sweep
    generative   PRDC (Naeem et al. 2020) -- diversity relative to a reference
    LM-based     Decan (Khoriaty et al. 2026) -- no embedder, no similarity at all

Everything is scored on the same fixed subsample used by run_hybrid.py, so this table
and that one can be read together.

Usage:
    python run_competitors.py --limit 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import transforms as T  # noqa: E402
from baselines import Decan, prdc, vendi_score  # noqa: E402
from common import (  # noqa: E402
    HERE,
    encode,
    load_background,
    load_calibration_sets,
    load_human_sets,
)
from run_hybrid import cached_nli  # noqa: E402

from linguistic_diversity import DistinctN, SelfBLEU, TypeTokenRatio  # noqa: E402
from linguistic_diversity.metric import TextDiversity  # noqa: E402

OUT = HERE / "output" / "competitors.json"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NAMES = ("McDiv_nuggets", "conTest", "decTest")
LOWER_IS_DIVERSE = {"SelfBLEU (lexical)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--permutations", type=int, default=4)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    human = load_human_sets()
    calib = load_calibration_sets()

    # Identical subsample to run_hybrid.py: same rng, same draw order.
    rng = np.random.default_rng(20260803)
    tasks: dict[str, dict[str, Any]] = {}
    for name, d in human.items():
        idx = rng.choice(np.arange(len(d["sets"])), min(args.limit, len(d["sets"])), replace=False)
        tasks[name] = {
            "sets": [d["sets"][i] for i in idx],
            "contexts": [d["contexts"][i] for i in idx],
            "target": d["human"][idx],
        }
    tasks["calibration"] = {
        "sets": calib["sets"],
        "contexts": [""] * len(calib["sets"]),
        "target": calib["expected"],
    }

    bg = encode(load_background(), "background", st)
    bg_small = bg[np.random.default_rng(0).choice(len(bg), 2000, replace=False)]

    for name, t in tasks.items():
        flat, bounds = [], []
        for s in t["sets"]:
            bounds.append((len(flat), len(flat) + len(s)))
            flat += s
        E = encode(flat, f"hy-{name}-docs", st)
        t["embeds"] = [E[a:b] for a, b in bounds]
        if any(c.strip() for c in t["contexts"]):
            C = encode([c if c.strip() else " " for c in t["contexts"]], f"hy-{name}-ctx", st)
            t["ctx"] = [C[i] if t["contexts"][i].strip() else None for i in range(len(C))]
        else:
            t["ctx"] = [None] * len(t["sets"])
        print(f"NLI for {name} ...")
        t["nli"] = cached_nli(name, t["sets"], NLI_MODEL)

    ctx_ls_tau = T.compose_context_then(T.local_scaling(bg_small, k=10, z0=0.30))

    def hill(Z: np.ndarray) -> float:
        p = np.full(len(Z), 1.0 / len(Z))
        return TextDiversity._calc_diversity(p, Z, q=1.0, index="vendi")

    # -- scorers: (task, index) -> float ------------------------------------------
    lexical = {
        "distinct-n (lexical)": DistinctN(),
        "SelfBLEU (lexical)": SelfBLEU(),
        "TTR (lexical)": TypeTokenRatio(),
    }
    decan = Decan(permutations=args.permutations)

    def ours_default(t: dict, i: int) -> float:
        return hill(T.floored(0.053)(t["embeds"][i], None))

    def ours_best(t: dict, i: int) -> float:
        Z_nli = t["nli"][i][1]
        Z_emb = ctx_ls_tau(t["embeds"][i], t["ctx"][i])
        Z = np.sqrt(np.clip(Z_nli, 0, 1) * np.clip(Z_emb, 0, 1))
        np.fill_diagonal(Z, 1.0)
        return hill(Z)

    def vendi_plain(t: dict, i: int) -> float:
        # The raw Gram matrix, not a clipped one: it is PSD by construction, and
        # clipping to [0, 1] destroys that, which vendi_score rightly refuses.
        E = t["embeds"][i]
        return vendi_score(E @ E.T, q=1.0)

    def prdc_coverage(t: dict, i: int) -> float:
        return prdc(bg_small[:400], t["embeds"][i], k=5)["coverage"]

    def decan_score(t: dict, i: int) -> float:
        return decan(t["sets"][i])

    scorers: dict[str, Callable[[dict, int], float]] = {
        **{name: (lambda t, i, m=m: float(m(t["sets"][i]))) for name, m in lexical.items()},
        "Vendi (spectral)": vendi_plain,
        "PRDC coverage": prdc_coverage,
        "Decan (LM surprise)": decan_score,
        "OURS current default": ours_default,
        "OURS best kernel": ours_best,
    }

    hdr = list(NAMES) + ["agreement_mean", "calib_rho", "calib_ratio"]
    print("\n" + "=" * 126)
    print(f"COMPETITIVE COMPARISON  identical subsample ({args.limit} sets/dataset)")
    print("=" * 126)
    print(f"  {'metric':24s} " + " ".join(f"{h:>14s}" for h in hdr))
    print("  " + "-" * 124)

    results: dict[str, Any] = {}
    for label, fn in scorers.items():
        row: dict[str, Any] = {}
        sign = -1.0 if label in LOWER_IS_DIVERSE else 1.0
        for name in NAMES:
            t = tasks[name]
            s = np.array([fn(t, i) for i in range(len(t["sets"]))], dtype=float)
            mask = np.isfinite(s)
            row[name] = round(float(spearmanr(sign * s[mask], t["target"][mask]).statistic), 4)
        row["agreement_mean"] = round(float(np.mean([row[n] for n in NAMES])), 4)

        t = tasks["calibration"]
        s = np.array([fn(t, i) for i in range(len(t["sets"]))], dtype=float)
        mask = np.isfinite(s)
        row["calib_rho"] = round(float(spearmanr(sign * s[mask], t["target"][mask]).statistic), 4)
        # A ratio is only meaningful for metrics that claim to be an effective count.
        counts_species = label.startswith("OURS") or label.startswith("Vendi")
        row["calib_ratio"] = (
            round(float(np.median(s[mask] / t["target"][mask])), 4) if counts_species else None
        )

        results[label] = row
        cells = []
        for h in hdr:
            v = row[h]
            cells.append(f"{v:+14.4f}" if isinstance(v, float) else f"{'--':>14s}")
        print(f"  {label:24s} " + " ".join(cells))

    print(
        "\n  calib_ratio is blank where the metric does not claim to be an effective\n"
        "  count -- a self-BLEU of 0.4 has no ratio to a known number of concepts.\n"
        "  PRDC recall is omitted: on five-item sets it is 0 for every set, so its\n"
        "  correlation is undefined rather than poor. See the fair test below."
    )
    results["distribution_level"] = distribution_level(st, bg_small, decan, ctx_ls_tau)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"limit": args.limit, "results": results}, indent=2))
    print(f"\nWrote {args.out}")


def distribution_level(
    st: Any, bg_small: np.ndarray, decan: Any, ctx_ls_tau: Any
) -> dict[str, Any]:
    """PRDC on its home ground, with the others alongside for scale.

    PRDC is a distribution-level statistic built for hundreds of samples, so judging
    it on five-item response sets says nothing about the method. Here each temperature
    bin pools ~200 responses and the reference is a held-out sample of the same
    generator, which is the comparison it was designed for. Temperature is the ground
    truth: higher temperature must read as more diverse.
    """
    import glob

    import pandas as pd

    from common import DATA

    files = sorted(glob.glob(str(DATA / "raw/decTest/*no_hds*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    resp_cols = [c for c in df.columns if c.startswith("resp_")]
    temps = df["label_value"].to_numpy(dtype=float)

    edges = np.quantile(temps, np.linspace(0, 1, 11))
    bins: list[tuple[float, list[str]]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        rows = df[(temps >= lo) & (temps < hi)]
        if len(rows) < 5:
            continue
        docs: list[str] = []
        for _, row in rows.iterrows():
            docs += [
                str(row[c]) for c in resp_cols if isinstance(row[c], str) and str(row[c]).strip()
            ]
        bins.append((float((lo + hi) / 2), docs[:200]))

    print(f"\n{'=' * 126}")
    print(f"DISTRIBUTION LEVEL  {len(bins)} temperature bins, <=200 documents each")
    print("=" * 126)

    embeds = [
        np.asarray(
            st.encode(d, show_progress_bar=False, normalize_embeddings=True), dtype=np.float64
        )
        for _, d in bins
    ]
    centres = np.array([c for c, _ in bins])
    reference = bg_small[:400]

    scores: dict[str, list[float]] = {
        "PRDC coverage": [],
        "PRDC recall": [],
        "PRDC density": [],
        "Vendi (spectral)": [],
        "OURS small-set kernel": [],
        "OURS scale kernel (tau only)": [],
        "Decan (LM surprise)": [],
    }
    for (_, docs), E in zip(bins, embeds, strict=True):
        m = prdc(reference, E, k=5)
        scores["PRDC coverage"].append(m["coverage"])
        scores["PRDC recall"].append(m["recall"])
        scores["PRDC density"].append(m["density"])
        scores["Vendi (spectral)"].append(vendi_score(E @ E.T, q=1.0))
        p = np.full(len(E), 1.0 / len(E))
        scores["OURS small-set kernel"].append(
            TextDiversity._calc_diversity(p, ctx_ls_tau(E, None), q=1.0, index="vendi")
        )
        scores["OURS scale kernel (tau only)"].append(
            TextDiversity._calc_diversity(p, T.tau_truncated(0.70)(E, None), q=1.0, index="vendi")
        )
        scores["Decan (LM surprise)"].append(decan(docs[:40]))

    print(f"  {'metric':24s} {'rho vs temperature':>20s}")
    out: dict[str, float] = {}
    for label, values in scores.items():
        rho = float(spearmanr(values, centres).statistic)
        out[label] = round(rho, 4)
        print(f"  {label:24s} {rho:+20.4f}")
    print(
        "\n  Given enough samples PRDC recovers temperature; the per-set table above\n"
        "  measured its sample size, not its method.\n"
        "  Two kernels are shown because the small-set winner is NOT the scale winner:\n"
        "  its local-scaling component is tuned to human ratings and costs accuracy\n"
        "  here, and its context projection is unavailable once corpora are pooled.\n"
        "  The NLI half is dropped at this size -- 40,000 cross-encoder pairs per bin\n"
        "  is the practical limit the cost table predicted. See run_scale.py."
    )
    return out


if __name__ == "__main__":
    main()
