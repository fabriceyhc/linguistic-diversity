#!/usr/bin/env python
"""Do the metrics agree with people, and does that agreement generalise?

The embedder-selection benchmark validated `DocumentSemantics` on one dataset --
McDiv_nuggets, 600 sets. Two more human-scored sets ship in the same download and
had never been used, which left the encoder and `mean_adj` choices resting on a
single corpus. This runs all three, and adds the analysis the third one enables.

  McDiv_nuggets  600 sets, 5 responses, graded human diversity scores
  conTest        670 sets, 5 responses, graded human diversity scores
  decTest        609 sets, 10 responses, graded scores *and* the sampling
                 temperature that produced them
  decTest_full   2979 sets, 10 responses, temperature only

Two questions:

1. AGREEMENT. Does every metric rank corpora the way people do, on all three
   datasets? A choice that only holds on one of them is tuned to that one.

2. FORM VERSUS CONTENT. Tevet & Berant (EACL 2021) found that decoding
   parameters "mostly affect form but not meaning". They demonstrated it with one
   content metric and one form metric. decTest carries the temperature that
   generated each set, so the claim can be measured at every linguistic level at
   once: correlate each metric against temperature and see which levels move.

Usage:
    python run_study.py --data-dir ../embedder_selection/data
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

from linguistic_diversity import (
    DependencyParse,
    DistinctN,
    DocumentSemantics,
    PartOfSpeechSequence,
    Phonemic,
    Rhythmic,
    SelfBLEU,
    TokenSemantics,
    TypeTokenRatio,
    clear_model_cache,
)

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "output" / "results.json"
HUMAN_COL = "metric_abs_hds_mean"

# Level each metric claims, so the temperature sweep can be read by level.
METRICS: dict[str, tuple[Any, str]] = {
    "DocumentSemantics": (lambda: DocumentSemantics({"verbose": False}), "semantic"),
    "TokenSemantics": (lambda: TokenSemantics({"verbose": False}), "semantic"),
    "DependencyParse": (lambda: DependencyParse({"verbose": False}), "syntactic"),
    "PartOfSpeechSequence": (lambda: PartOfSpeechSequence({"verbose": False}), "morphological"),
    "Rhythmic": (lambda: Rhythmic({"verbose": False}), "phonological"),
    "Phonemic": (lambda: Phonemic({"verbose": False}), "phonological"),
    "TypeTokenRatio": (lambda: TypeTokenRatio(), "lexical (form)"),
    "DistinctN": (lambda: DistinctN(), "lexical (form)"),
    "SelfBLEU": (lambda: SelfBLEU(), "lexical (form)"),
}
LOWER_IS_DIVERSE = {"SelfBLEU"}

DATASETS = {
    "McDiv_nuggets": "with_metrics/McDiv_nuggets/*with_hds*.csv",
    "conTest": "raw/conTest/*with_hds*.csv",
    "decTest": "raw/decTest/*with_hds*.csv",
}
TEMPERATURE_SET = ("decTest_full", "raw/decTest/*no_hds*.csv")


def load(data_dir: Path, pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(data_dir / pattern)))
    if not files:
        raise FileNotFoundError(f"nothing matched {data_dir / pattern}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["source"] = Path(path).stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def score_rows(metric: Any, df: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Diversity of each row's response set."""
    resp_cols = [c for c in df.columns if c.startswith("resp_")]
    out = []
    for _, row in df.iterrows():
        docs = [str(row[c]) for c in resp_cols if isinstance(row[c], str) and str(row[c]).strip()]
        out.append(float(metric(docs)) if len(docs) > 1 else float("nan"))
    return np.array(out, dtype=float)


def oriented(name: str, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return -values if name in LOWER_IS_DIVERSE else values


def rho(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> tuple[float, int]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 3:
        return float("nan"), int(mask.sum())
    return float(spearmanr(x[mask], y[mask]).statistic), int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=HERE.parent / "embedder_selection" / "data")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--temp-limit", type=int, default=900,
                        help="Rows of decTest_full for the temperature sweep (0 = all)")
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    names = args.only or list(METRICS)
    results: dict[str, Any] = {"_meta": {}}

    # ---- Q1: agreement with human judgments, on all three datasets ----
    loaded = {k: load(args.data_dir, p) for k, p in DATASETS.items()}
    for k, df in loaded.items():
        print(f"{k:15s} {len(df):5d} sets, "
              f"{len([c for c in df.columns if c.startswith('resp_')])} responses each")
    results["_meta"]["dataset_sizes"] = {k: len(v) for k, v in loaded.items()}

    agreement: dict[str, dict[str, Any]] = {}
    for name in names:
        build, _level = METRICS[name]
        print(f"\n  {name} ...", end=" ", flush=True)
        metric = build()
        agreement[name] = {}
        for k, df in loaded.items():
            scores = oriented(name, score_rows(metric, df))
            r, n = rho(scores, df[HUMAN_COL].to_numpy(dtype=float))
            agreement[name][k] = {"spearman": round(r, 4), "n": n}
            print(f"{k}={r:+.3f}", end=" ", flush=True)
        clear_model_cache()
    results["agreement"] = agreement

    # ---- Q2: what does temperature actually move? ----
    temp_df = load(args.data_dir, TEMPERATURE_SET[1])
    if args.temp_limit:
        temp_df = temp_df.sample(n=min(args.temp_limit, len(temp_df)), random_state=20260803)
    temps = temp_df["label_value"].to_numpy(dtype=float)
    print(f"\n\n{TEMPERATURE_SET[0]}: {len(temp_df)} sets, "
          f"temperature {temps.min():.2f}-{temps.max():.2f}")

    temperature: dict[str, dict[str, Any]] = {}
    for name in names:
        build, level = METRICS[name]
        print(f"  {name} ...", end=" ", flush=True)
        metric = build()
        scores = oriented(name, score_rows(metric, temp_df))
        r, n = rho(scores, temps)
        temperature[name] = {"spearman_vs_temperature": round(r, 4), "n": n, "level": level}
        print(f"rho={r:+.4f}")
        clear_model_cache()
    results["temperature"] = temperature

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 78}\nAGREEMENT WITH HUMAN JUDGMENTS\n{'-' * 78}")
    print(f"  {'metric':22s} " + " ".join(f"{k:>14s}" for k in DATASETS))
    for name, row in agreement.items():
        print(f"  {name:22s} " + " ".join(f"{row[k]['spearman']:+14.3f}" for k in DATASETS))

    print(f"\n{'=' * 78}")
    print("WHAT DOES SAMPLING TEMPERATURE MOVE?")
    print("  Tevet & Berant: decoding parameters change form, not meaning.")
    print("-" * 78)
    print(f"  {'metric':22s} {'level':16s} {'rho vs temp':>12s}")
    for name, row in sorted(temperature.items(),
                            key=lambda kv: -abs(kv[1]["spearman_vs_temperature"])):
        print(f"  {name:22s} {row['level']:16s} "
              f"{row['spearman_vs_temperature']:+12.4f}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
