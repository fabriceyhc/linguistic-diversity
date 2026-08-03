#!/usr/bin/env python
"""Hill numbers against the Vendi Score, on one similarity matrix.

The Vendi Score (Friedman & Dieng, TMLR 2023) is the closest published relative of
what this library computes: both return an effective number of elements from a
similarity matrix, and both agree exactly at the extremes -- K = I gives n, K
all-ones gives 1.

They disagree in between, and the disagreement is systematic. For two species at
similarity z, Vendi reads 1.75 where the Hill number reads 1.33 (z = 0.5): Vendi
discounts similarity far less aggressively, because it reads diversity off the
eigenvalue spectrum rather than weighting each species by how much of the corpus
looks like it.

Both are computed here from the *same* similarity matrix, so nothing separates
them except the index itself. Two questions:

1. Which orders corpora the way people do?
2. Which recovers a known number of concepts?

Usage:
    python run_comparison.py --data-dir ../embedder_selection/data
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines import VendiWrapper  # noqa: E402

from linguistic_diversity import DocumentSemantics, clear_model_cache  # noqa: E402

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "output" / "results.json"
BENCHMARK = HERE.parent / "metric_validation" / "output" / "benchmark.json"
RESP_COLS = [f"resp_{i}" for i in range(5)]
HUMAN_COL = "metric_abs_hds_mean"

DATASETS = {
    "McDiv_nuggets": "with_metrics/McDiv_nuggets/*with_hds*.csv",
    "conTest": "raw/conTest/*with_hds*.csv",
}


def load(data_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / pattern)))
    if not files:
        raise FileNotFoundError(f"nothing matched {data_dir / pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def score_rows(metric: Any, df: pd.DataFrame) -> npt.NDArray[np.float64]:
    out = []
    for _, row in df.iterrows():
        docs = [str(row[c]) for c in RESP_COLS if isinstance(row[c], str) and str(row[c]).strip()]
        out.append(float(metric(docs)) if len(docs) > 1 else float("nan"))
    return np.array(out, dtype=float)


def rho(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    return float(spearmanr(x[mask], y[mask]).statistic)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=HERE.parent / "embedder_selection" / "data"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    base = DocumentSemantics({"verbose": False})
    indices = {"Hill (this library)": base, "Vendi": VendiWrapper(base)}
    results: dict[str, Any] = {}

    print("=" * 78)
    print("Q1  Agreement with graded human diversity judgments")
    print("-" * 78)
    loaded = {k: load(args.data_dir, p) for k, p in DATASETS.items()}
    agreement: dict[str, dict[str, float]] = {}
    for name, metric in indices.items():
        agreement[name] = {}
        for key, df in loaded.items():
            scores = score_rows(metric, df)
            agreement[name][key] = round(rho(scores, df[HUMAN_COL].to_numpy(dtype=float)), 4)
    print(f"  {'index':22s} " + " ".join(f"{k:>15s}" for k in DATASETS))
    for name, row in agreement.items():
        print(f"  {name:22s} " + " ".join(f"{row[k]:+15.4f}" for k in DATASETS))
    results["human_agreement"] = agreement

    print(f"\n{'=' * 78}")
    print("Q2  Recovering a known number of concepts")
    print("-" * 78)
    benchmark = json.loads(BENCHMARK.read_text())
    targeted = [
        c
        for c in benchmark["corpora"]
        if c["expected"].get("semantic") is not None
        and c["family"] in ("syntactic_alternations", "syntactic_frames", "random_controls")
    ]
    calib: dict[str, dict[str, float]] = {}
    for name, metric in indices.items():
        observed, expected = [], []
        for corpus in targeted:
            try:
                observed.append(float(metric(corpus["documents"])))
                expected.append(float(corpus["expected"]["semantic"]))
            except Exception:
                continue
        ratios = [o / e for o, e in zip(observed, expected, strict=True) if e > 0]
        calib[name] = {
            "spearman_vs_expected": round(float(spearmanr(observed, expected).statistic), 4),
            "median_ratio": round(statistics.median(ratios), 4),
            "n": len(observed),
        }
    print(f"  {'index':22s} {'rho vs known k':>15s} {'median ratio':>13s} {'n':>5s}")
    for name, r in calib.items():
        print(
            f"  {name:22s} {r['spearman_vs_expected']:+15.4f} "
            f"{r['median_ratio']:13.3f} {r['n']:5d}"
        )
    results["calibration"] = calib

    print(f"\n{'=' * 78}")
    print("Q3  Where the two indices disagree")
    print("-" * 78)
    pairs = []
    for corpus in targeted[:80]:
        try:
            h = float(indices["Hill (this library)"](corpus["documents"]))
            v = float(indices["Vendi"](corpus["documents"]))
        except Exception:
            continue
        pairs.append((corpus["family"], len(corpus["documents"]), h, v))
    by_family: dict[str, list[float]] = {}
    for family, _n, h, v in pairs:
        by_family.setdefault(family, []).append(v / h if h else float("nan"))
    print(f"  {'family':28s} {'Vendi / Hill':>13s}")
    for family, ratios in sorted(by_family.items()):
        print(f"  {family:28s} {statistics.median(ratios):13.3f}")
    results["disagreement"] = {k: round(statistics.median(v), 4) for k, v in by_family.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")
    clear_model_cache()


if __name__ == "__main__":
    main()
