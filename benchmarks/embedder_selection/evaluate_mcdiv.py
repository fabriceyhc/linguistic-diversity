#!/usr/bin/env python
"""Validate embedder choice against human diversity judgments (McDiv).

McDiv comes from Tevet & Berant, "Evaluating the Evaluation of Diversity in
Natural Language Generation" (EACL 2021), https://github.com/GuyTevet/diversity-eval

Two evaluations:
  1. Correlation with graded human scores on the 600 "nuggets" sets.
  2. Pairwise accuracy on the full set: given a high- and a low-diversity set of
     responses to the same context, does the metric rank them correctly?

Get the data first:
    curl -o data.zip http://diversity-eval.s3-us-west-2.amazonaws.com/data.zip
    unzip data.zip          # yields ./data
Then:
    python evaluate_mcdiv.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import glob
import json
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from linguistic_diversity import DocumentSemantics, clear_model_cache

HERE = Path(__file__).parent
RESP_COLS = [f"resp_{i}" for i in range(5)]

MODELS: list[dict[str, Any]] = [
    {"name": "sentence-transformers/all-mpnet-base-v2"},
    {"name": "sentence-transformers/all-MiniLM-L6-v2"},
    {"name": "BAAI/bge-base-en-v1.5"},
    {"name": "BAAI/bge-large-en-v1.5"},
    {"name": "avsolatorio/GIST-large-Embedding-v0"},
    {"name": "mixedbread-ai/mxbai-embed-large-v1"},
    {"name": "WhereIsAI/UAE-Large-V1"},
    {"name": "google/embeddinggemma-300m", "trust_remote_code": True},
    {"name": "Qwen/Qwen3-Embedding-0.6B"},
    {"name": "infgrad/Jasper-Token-Compression-600M", "trust_remote_code": True},
]

# Baselines shipped with the dataset, already computed by the paper's authors.
# These columns are already diversity-oriented: the similarity-derived ones come
# from Similarity2DiversityMetric, which negates the similarity (hence their
# negative value ranges). Higher means more diverse for all of them, so no sign
# flip is applied.
BASELINES = (
    "averaged_cosine_similarity",
    "bert_score",
    "averaged_distinct_ngrams",
)


def load_nuggets(data_dir: Path) -> pd.DataFrame:
    """Load the 600 sets carrying graded human diversity scores."""
    files = sorted(glob.glob(str(data_dir / "with_metrics" / "McDiv_nuggets" / "*with_hds*.csv")))
    if not files:
        raise FileNotFoundError(f"No McDiv_nuggets files under {data_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["domain"] = Path(path).stem.split("_")[-2] + "_" + Path(path).stem.split("_")[-1]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_pairs(data_dir: Path) -> pd.DataFrame:
    """Load the full McDiv set of high/low content-diversity pairs."""
    files = sorted(glob.glob(str(data_dir / "raw" / "McDiv" / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No McDiv files under {data_dir}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["domain"] = Path(path).stem.split("_")[-2] + "_" + Path(path).stem.split("_")[-1]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def score_sets(metric: DocumentSemantics, df: pd.DataFrame) -> np.ndarray:
    """Compute diversity for each row's 5 responses."""
    out = []
    for _, row in df.iterrows():
        docs = [str(row[c]) for c in RESP_COLS if isinstance(row[c], str) or not pd.isna(row[c])]
        docs = [d for d in docs if d.strip()]
        out.append(metric(docs) if len(docs) > 1 else float("nan"))
    return np.array(out, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Unzipped McDiv data dir")
    parser.add_argument("--out", type=Path, default=HERE / "output" / "mcdiv_results.json")
    parser.add_argument("--skip-pairs", action="store_true", help="Only run the graded correlation")
    parser.add_argument("--pair-limit", type=int, default=600, help="Pairs to sample (0 = all)")
    args = parser.parse_args()

    nuggets = load_nuggets(args.data_dir)
    human = nuggets["metric_abs_hds_mean"].to_numpy()
    print(
        f"Loaded {len(nuggets)} nugget sets with human scores "
        f"(human score range {human.min():.2f}-{human.max():.2f})"
    )

    # Paper's own metrics, as a reference point
    print("\n--- baselines shipped with the dataset ---")
    baseline_rows = []
    for col in BASELINES:
        full = f"metric_{col}"
        if full in nuggets.columns:
            rho = spearmanr(nuggets[full].to_numpy(), human, nan_policy="omit").statistic
            baseline_rows.append({"name": col, "spearman": float(rho)})
            print(f"  {col:32s} rho = {rho:+.3f}")

    pairs = None
    if not args.skip_pairs:
        pairs = load_pairs(args.data_dir)
        if args.pair_limit:
            ids = pairs["sample_id"].drop_duplicates().head(args.pair_limit)
            pairs = pairs[pairs["sample_id"].isin(ids)]
        print(f"\nLoaded {pairs['sample_id'].nunique()} high/low pairs for the ranking test")

    results = []
    for spec in MODELS:
        config = {
            "model_name": spec["name"],
            "verbose": False,
            "trust_remote_code": spec.get("trust_remote_code", False),
            "encode_kwargs": spec.get("encode_kwargs", {}),
        }
        try:
            metric = DocumentSemantics(config)
            measured = score_sets(metric, nuggets)
            rho = spearmanr(measured, human, nan_policy="omit").statistic
            rec = {"model": spec["name"], "spearman_vs_human": float(rho)}

            for domain in sorted(nuggets["domain"].unique()):
                mask = (nuggets["domain"] == domain).to_numpy()
                rec[f"rho_{domain}"] = float(
                    spearmanr(measured[mask], human[mask], nan_policy="omit").statistic
                )

            if pairs is not None:
                pm = score_sets(metric, pairs)
                tmp = pairs.assign(_m=pm)
                correct = total = 0
                for _, grp in tmp.groupby("sample_id"):
                    hi = grp[grp["label_value"] == 1.0]["_m"]
                    lo = grp[grp["label_value"] == 0.0]["_m"]
                    if len(hi) and len(lo):
                        total += 1
                        correct += int(hi.mean() > lo.mean())
                rec["pair_accuracy"] = correct / total if total else float("nan")
                rec["n_pairs"] = total

            results.append(rec)
            print(
                f"done {spec['name']}: rho={rho:+.3f}"
                + (f" acc={rec['pair_accuracy']:.3f}" if pairs is not None else ""),
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {spec['name']}: {type(exc).__name__}: {exc}"[:150], flush=True)
        finally:
            # Release before the next checkpoint: the cache is unbounded and ten
            # models will exhaust VRAM (and this box's 10GB of RAM) otherwise.
            clear_model_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"baselines": baseline_rows, "models": results}, indent=2))

    results.sort(key=lambda r: -r["spearman_vs_human"])
    print(f"\n{'model':44s} {'rho(human)':>11s} {'pair acc':>9s}")
    print("-" * 68)
    for r in results:
        acc = f"{r['pair_accuracy']:9.3f}" if "pair_accuracy" in r else "        -"
        print(f"{r['model'][:43]:44s} {r['spearman_vs_human']:+11.3f} {acc}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
