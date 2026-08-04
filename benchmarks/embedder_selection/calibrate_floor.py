#!/usr/bin/env python
"""Estimate each encoder's similarity floor, then check whether removing it helps.

Sentence encoders do not send unrelated text to orthogonal vectors. They send it
to some positive baseline, and because every document is slightly similar to
every *other* document that baseline accumulates: the largest diversity any
abundance distribution can reach is n / (1 + (n-1)z), which tends to 1/z. A floor
of 0.46 caps a corpus at about 2.2 effective species however large it is.

Rescaling by  z' = max(0, (z - floor) / (1 - floor))  sends unrelated text to 0
and leaves identical text at 1. Whether that is an improvement is an empirical
question with two halves, and they can disagree:

  calibration   does the reported number of concepts get closer to the truth?
  agreement     do corpora still get ordered the way people order them?

The floor is estimated from text known to be unrelated -- responses to *different*
prompts -- never from the corpus being measured. Estimating it per corpus is what
mean_adj did, and it cost replication invariance.

Usage:
    python calibrate_floor.py --data-dir ./data
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import spearmanr

from linguistic_diversity import DocumentSemantics, clear_model_cache
from linguistic_diversity.utils import maximum_diversity

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "output" / "floor_calibration.json"
RESP_COLS = [f"resp_{i}" for i in range(5)]
HUMAN_COL = "metric_abs_hds_mean"

ENCODERS = [
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-large-en-v1.5",
]

DATASETS = {
    "McDiv_nuggets": "with_metrics/McDiv_nuggets/*with_hds*.csv",
    "conTest": "raw/conTest/*with_hds*.csv",
}


def load(data_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / pattern)))
    if not files:
        raise FileNotFoundError(f"nothing matched {data_dir / pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def estimate_floor(
    metric: DocumentSemantics, df: pd.DataFrame, n_pairs: int = 400, seed: int = 20260803
) -> dict[str, float]:
    """Baseline similarity between responses to *different* prompts.

    Cross-prompt pairs are unrelated by construction, which is what makes this an
    estimate of the floor rather than of the corpus.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(df), size=min(2 * n_pairs, len(df)), replace=False)
    texts, owners = [], []
    for r in rows:
        row = df.iloc[int(r)]
        for c in RESP_COLS:
            if isinstance(row[c], str) and row[c].strip():
                texts.append(str(row[c]))
                owners.append(int(r))
                break
    feats, _ = metric.extract_features(texts)
    Z = np.asarray(metric.calculate_similarities(feats), dtype=float)
    owner = np.array(owners)
    cross = owner[:, None] != owner[None, :]
    vals = Z[cross]
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "n_pairs": int(cross.sum()),
    }


def score_sets(metric: DocumentSemantics, df: pd.DataFrame) -> npt.NDArray[np.float64]:
    out = []
    for _, row in df.iterrows():
        docs = [str(row[c]) for c in RESP_COLS if isinstance(row[c], str) and str(row[c]).strip()]
        out.append(float(metric(docs)) if len(docs) > 1 else float("nan"))
    return np.array(out, dtype=float)


def mean_ceiling(metric: DocumentSemantics, df: pd.DataFrame, limit: int = 120) -> float:
    """Average achievable ceiling, which is what the floor is meant to raise."""
    vals = []
    for _, row in df.head(limit).iterrows():
        docs = [str(row[c]) for c in RESP_COLS if isinstance(row[c], str) and str(row[c]).strip()]
        if len(docs) < 2:
            continue
        feats, _ = metric.extract_features(docs)
        Z = np.asarray(metric.calculate_similarities(feats), dtype=float)
        vals.append(maximum_diversity(Z)[0])
    return float(np.mean(vals)) if vals else float("nan")


def rho(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    return float(spearmanr(x[mask], y[mask]).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--encoders", nargs="*", default=ENCODERS)
    args = parser.parse_args()

    loaded = {k: load(args.data_dir, p) for k, p in DATASETS.items()}
    results: dict[str, Any] = {}

    for encoder in args.encoders:
        print(f"\n{'=' * 78}\n{encoder}\n{'-' * 78}")
        base = DocumentSemantics({"model_name": encoder, "verbose": False})
        floor = estimate_floor(base, loaded["McDiv_nuggets"])
        print(
            f"  cross-prompt similarity: mean {floor['mean']:.3f}  median "
            f"{floor['median']:.3f}  p10 {floor['p10']:.3f}  p90 {floor['p90']:.3f}"
        )
        print(f"  implied cap 1/mean = {1 / floor['mean']:.1f} effective species\n")

        # Sweep candidate floors, including none.
        candidates = [
            None,
            round(floor["p10"], 3),
            round(floor["median"], 3),
            round(floor["mean"], 3),
        ]
        rows = []
        print(f"  {'floor':>8s} {'ceiling':>8s} " + " ".join(f"{k:>14s}" for k in DATASETS))
        for cand in candidates:
            cfg = {"model_name": encoder, "verbose": False}
            if cand is not None:
                cfg["similarity_floor"] = cand
            m = DocumentSemantics(cfg)
            ceiling = mean_ceiling(m, loaded["McDiv_nuggets"])
            agree = {
                k: rho(score_sets(m, df), df[HUMAN_COL].to_numpy(dtype=float))
                for k, df in loaded.items()
            }
            rows.append(
                {
                    "floor": cand,
                    "mean_ceiling": round(ceiling, 3),
                    "agreement": {k: round(v, 4) for k, v in agree.items()},
                }
            )
            label = "none" if cand is None else f"{cand:.3f}"
            print(
                f"  {label:>8s} {ceiling:8.2f} " + " ".join(f"{agree[k]:+14.4f}" for k in DATASETS)
            )
        results[encoder] = {"floor_estimate": floor, "sweep": rows}
        clear_model_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
