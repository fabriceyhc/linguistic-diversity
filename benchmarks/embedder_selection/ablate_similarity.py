#!/usr/bin/env python
"""Two design questions, answered against McDiv human judgments.

1. Should TokenSemantics keep ``mean_adj``? It subtracts the off-diagonal mean
   from every off-diagonal entry, which sharpens contrasts but breaks a law:
   an occurrence and its exact replica score 0.78 instead of 1.0, so the metric
   is not invariant to corpus replication.

2. Cosine is defined on [-1, 1] but a similarity-sensitive Hill number needs
   [0, 1]. Clamping negatives to zero is one option; it is not obviously the
   best one. This sweeps the alternatives on the same 600 human-scored sets.

Both are decided by agreement with human diversity judgments, which is the same
evidence that chose the default encoder -- see README.

Usage:
    python ablate_similarity.py --data-dir ./data
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

from linguistic_diversity import DocumentSemantics, TokenSemantics, clear_model_cache

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "output" / "similarity_ablation.json"
RESP_COLS = [f"resp_{i}" for i in range(5)]
HUMAN_COL = "metric_abs_hds_mean"
ENCODER = "BAAI/bge-large-en-v1.5"  # the benchmark's recommended default

# Counts how many raw cosines fall below zero across the whole sweep, so the
# question "does clamping matter at all?" is answered with a number.
_NEGATIVE_TALLY = {"below_zero": 0, "total": 0}


def _cosine(features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Raw cosine on L2-normalised embeddings, before any range correction."""
    feats = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = feats / norms
    Z = (unit @ unit.T).astype(np.float64)
    off = ~np.eye(Z.shape[0], dtype=bool)
    _NEGATIVE_TALLY["below_zero"] += int((Z[off] < 0).sum())
    _NEGATIVE_TALLY["total"] += int(off.sum())
    return Z


def _finish(Z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    np.fill_diagonal(Z, 1.0)
    return Z


class ClampedCosine(DocumentSemantics):
    """max(cos, 0). Current behaviour. Negative cosine means 'more than
    orthogonal', which for diversity purposes is just maximally dissimilar."""

    def calculate_similarities(self, features: Any) -> npt.NDArray[np.float64]:
        return _finish(np.clip(_cosine(features), 0.0, 1.0))


class RescaledCosine(DocumentSemantics):
    """(1 + cos) / 2. Uses the whole range and stays positive semi-definite,
    but puts orthogonal documents at 0.5, which compresses diversity."""

    def calculate_similarities(self, features: Any) -> npt.NDArray[np.float64]:
        return _finish((_cosine(features) + 1.0) / 2.0)


class AngularSimilarity(DocumentSemantics):
    """1 - arccos(cos)/pi. Linear in the angle rather than its cosine, so it
    spreads the high-similarity region that sentence encoders crowd into."""

    def calculate_similarities(self, features: Any) -> npt.NDArray[np.float64]:
        Z = np.clip(_cosine(features), -1.0, 1.0)
        return _finish(1.0 - np.arccos(Z) / np.pi)


class SquaredCosine(DocumentSemantics):
    """cos^2. In [0, 1] without clamping, but antipodal embeddings map to 1.0 --
    opposite meanings would count as identical. Included to measure that cost."""

    def calculate_similarities(self, features: Any) -> npt.NDArray[np.float64]:
        return _finish(np.square(_cosine(features)))


class ExpEuclidean(DocumentSemantics):
    """exp(-||a-b||). Strictly positive, positive semi-definite, and never needs
    clamping because it is built from a distance rather than an inner product."""

    def calculate_similarities(self, features: Any) -> npt.NDArray[np.float64]:
        feats = np.asarray(features, dtype=np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit = feats / norms
        sq = np.sum(unit**2, axis=1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * (unit @ unit.T), 0.0)
        return _finish(np.exp(-np.sqrt(d2)).astype(np.float64))


DOC_VARIANTS: dict[str, type[DocumentSemantics]] = {
    "clamped_cosine (current)": ClampedCosine,
    "rescaled_(1+cos)/2": RescaledCosine,
    "angular_1-acos/pi": AngularSimilarity,
    "squared_cosine": SquaredCosine,
    "exp_euclidean": ExpEuclidean,
}

TOKEN_VARIANTS: dict[str, dict[str, Any]] = {
    "mean_adj=True, power_reg=True (current)": {"mean_adj": True, "power_reg": True},
    "mean_adj=False, power_reg=True": {"mean_adj": False, "power_reg": True},
    "mean_adj=False, power_reg=False": {"mean_adj": False, "power_reg": False},
    "mean_adj=True, power_reg=False": {"mean_adj": True, "power_reg": False},
}


def load_nuggets(data_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / "with_metrics" / "McDiv_nuggets" / "*with_hds*.csv")))
    if not files:
        raise FileNotFoundError(f"No McDiv_nuggets files under {data_dir}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def score_sets(metric: Any, df: pd.DataFrame) -> npt.NDArray[np.float64]:
    out = []
    for _, row in df.iterrows():
        docs = [str(row[c]) for c in RESP_COLS if isinstance(row[c], str) or not pd.isna(row[c])]
        docs = [d for d in docs if d.strip()]
        out.append(metric(docs) if len(docs) > 1 else float("nan"))
    return np.array(out, dtype=float)


def agreement(scores: npt.NDArray[np.float64], human: npt.NDArray[np.float64]) -> dict[str, Any]:
    mask = ~np.isnan(scores) & ~np.isnan(human)
    rho, p = spearmanr(scores[mask], human[mask])
    return {"spearman": round(float(rho), 4), "p_value": float(p), "n": int(mask.sum())}


def replication_invariant(metric: Any) -> tuple[bool, float, float]:
    """The law mean_adj breaks: duplicating a corpus must not change diversity."""
    corpus = [
        "The tall boy kicked the ball.",
        "When the rain stopped, the children played.",
        "She believes that the plan is sound.",
        "There was a crack in the ceiling.",
    ]
    a, b = float(metric(corpus)), float(metric(corpus * 3))
    return abs(a - b) < 1e-4, a, b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=0, help="Rows to score (0 = all)")
    args = parser.parse_args()

    df = load_nuggets(args.data_dir)
    if args.limit:
        df = df.head(args.limit)
    human = df[HUMAN_COL].to_numpy(dtype=float)
    print(f"{len(df)} human-scored response sets, encoder {ENCODER}\n")

    results: dict[str, Any] = {"_meta": {"n_sets": len(df), "encoder": ENCODER}}

    print("=" * 78)
    print("Q1  DocumentSemantics: mapping cosine into [0, 1]")
    print("-" * 78)
    doc_rows = {}
    for label, cls in DOC_VARIANTS.items():
        metric = cls({"model_name": ENCODER, "verbose": False})
        scores = score_sets(metric, df)
        doc_rows[label] = agreement(scores, human)
        doc_rows[label]["mean_similarity"] = round(float(np.nanmean(scores)), 4)
        print(f"  {label:26s} rho={doc_rows[label]['spearman']:+.4f}  "
              f"mean D={doc_rows[label]['mean_similarity']:7.3f}  n={doc_rows[label]['n']}")
        clear_model_cache()
    results["document_semantics"] = doc_rows

    neg = _NEGATIVE_TALLY
    frac = neg["below_zero"] / neg["total"] if neg["total"] else 0.0
    results["negative_cosines"] = {**neg, "fraction": round(frac, 6)}
    print(f"\n  raw cosines below zero: {neg['below_zero']:,} of {neg['total']:,} "
          f"off-diagonal pairs ({frac:.2%})")

    print("\n" + "=" * 78)
    print("Q2  TokenSemantics: keep mean_adj?")
    print("-" * 78)
    tok_rows = {}
    for label, cfg in TOKEN_VARIANTS.items():
        metric = TokenSemantics({**cfg, "verbose": False})
        scores = score_sets(metric, df)
        ok, a, b = replication_invariant(metric)
        tok_rows[label] = {
            **agreement(scores, human),
            "replication_invariant": ok,
            "D_1x": round(a, 4),
            "D_3x": round(b, 4),
        }
        print(f"  {label:42s} rho={tok_rows[label]['spearman']:+.4f}  "
              f"invariant={'yes' if ok else 'NO ':3s}  ({a:.2f} -> {b:.2f} at 3x)")
        clear_model_cache()
    results["token_semantics"] = tok_rows

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
