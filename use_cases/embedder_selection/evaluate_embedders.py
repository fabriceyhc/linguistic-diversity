#!/usr/bin/env python
"""Score sentence embedders on the held-out diversity benchmark.

Reports absolute calibration, which is the property MTEB's rank-based metrics do
not measure and which similarity-sensitive Hill numbers depend on.

Usage:
    python evaluate_embedders.py --models all-mpnet-base-v2 BAAI/bge-large-en-v1.5
    python evaluate_embedders.py --preset default
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np

from linguistic_diversity import DocumentSemantics, clear_model_cache

HERE = Path(__file__).parent
DEFAULT_BENCHMARK = HERE / "output" / "benchmark.json"
DEFAULT_RESULTS = HERE / "output" / "results.json"

# Best agreement with human diversity judgments on McDiv (rho +0.779, pair acc 0.957)
# at half the size of the nominal top scorer, with no trust_remote_code requirement.
# See README.md for the full ranking.
RECOMMENDED_MODEL = "BAAI/bge-large-en-v1.5"

# Models worth screening. trust/encode_kwargs cover task-conditioned checkpoints.
# Ordered by McDiv agreement, best first.
PRESETS: dict[str, list[dict[str, Any]]] = {
    "default": [
        {"name": "infgrad/Jasper-Token-Compression-600M", "trust_remote_code": True},
        {"name": RECOMMENDED_MODEL},
        {"name": "mixedbread-ai/mxbai-embed-large-v1"},
        {"name": "BAAI/bge-base-en-v1.5"},
        {"name": "WhereIsAI/UAE-Large-V1"},
        {"name": "avsolatorio/GIST-large-Embedding-v0"},
        {"name": "google/embeddinggemma-300m", "trust_remote_code": True},
        {"name": "Qwen/Qwen3-Embedding-0.6B"},
        {"name": "sentence-transformers/all-mpnet-base-v2"},
        {"name": "sentence-transformers/all-MiniLM-L6-v2"},
    ],
}


def evaluate_model(spec: dict[str, Any], corpora: list[dict[str, Any]]) -> dict[str, Any]:
    """Measure diversity on every corpus and score against ground truth.

    Args:
        spec: Model spec with "name" and optional trust_remote_code / encode_kwargs.
        corpora: Benchmark corpora.

    Returns:
        Result record with per-axis calibration and correlation scores.
    """
    config = {
        "model_name": spec["name"],
        "verbose": False,
        "trust_remote_code": spec.get("trust_remote_code", False),
        "encode_kwargs": spec.get("encode_kwargs", {}),
    }
    metric = DocumentSemantics(config)

    measured, truth, axes, cosines = [], [], [], []
    for corpus in corpora:
        docs = corpus["documents"]
        value = metric(docs)
        measured.append(value)
        truth.append(corpus["true_diversity"])
        axes.append(f"{corpus['axis']}/{corpus['regime']}")

        feats, _ = metric.extract_features(docs)
        norm = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        sim = norm @ norm.T
        cosines.append(float(sim[~np.eye(len(sim), dtype=bool)].mean()))

    measured_arr = np.array(measured)
    truth_arr = np.array(truth)
    axes_arr = np.array(axes)

    result: dict[str, Any] = {
        "model": spec["name"],
        "mean_cosine": float(np.mean(cosines)),
        # Ratio is scale-free: 1.0 is perfect, <1 under-reports, >1 over-reports
        "calibration_ratio": float(np.mean(measured_arr / truth_arr)),
        "mean_abs_error": float(np.mean(np.abs(measured_arr - truth_arr))),
        "per_axis": {},
        "measurements": [
            {"corpus": c["id"], "true": t, "measured": m}
            for c, t, m in zip(corpora, truth, measured)
        ],
    }
    for axis in sorted(set(axes)):
        mask = axes_arr == axis
        result["per_axis"][axis] = {
            "calibration_ratio": float(np.mean(measured_arr[mask] / truth_arr[mask])),
            "mean_abs_error": float(np.mean(np.abs(measured_arr[mask] - truth_arr[mask]))),
            "n": int(mask.sum()),
        }

    # Ordering: does the metric rank corpora by true diversity? (what MTEB-style
    # rank metrics would capture, kept for contrast with calibration)
    syn = axes_arr != "polysemy/shared_surface_form"
    if syn.sum() > 2 and len(set(truth_arr[syn])) > 1:
        result["synonymy_rank_corr"] = float(
            np.corrcoef(
                np.argsort(np.argsort(measured_arr[syn])),
                np.argsort(np.argsort(truth_arr[syn])),
            )[0, 1]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--preset", default="default", choices=sorted(PRESETS))
    args = parser.parse_args()

    corpora = json.loads(args.benchmark.read_text())["corpora"]
    specs = ([{"name": m} for m in args.models] if args.models else PRESETS[args.preset])

    results = []
    for spec in specs:
        try:
            results.append(evaluate_model(spec, corpora))
            print(f"done {spec['name']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report and continue the sweep
            print(f"FAILED {spec['name']}: {type(exc).__name__}: {exc}"[:160], flush=True)
        finally:
            clear_model_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    # Rank by how close the calibration ratio is to 1.0
    results.sort(key=lambda r: abs(r["calibration_ratio"] - 1.0))
    print(f"\n{'model':44s} {'calib':>7s} {'MAE':>6s} {'synonymy':>9s} "
          f"{'polysemy':>9s} {'cos':>6s} {'rank r':>7s}")
    print("-" * 92)
    for r in results:
        syn = np.mean([
            v["calibration_ratio"] for k, v in r["per_axis"].items() if k.startswith("synonymy")
        ])
        poly = r["per_axis"].get("polysemy/shared_surface_form", {}).get("calibration_ratio", float("nan"))
        print(f"{r['model'][:43]:44s} {r['calibration_ratio']:7.3f} {r['mean_abs_error']:6.2f} "
              f"{syn:9.3f} {poly:9.3f} {r['mean_cosine']:+6.3f} "
              f"{r.get('synonymy_rank_corr', float('nan')):7.3f}")
    print("\ncalib/synonymy/polysemy: measured / true. 1.000 is perfectly calibrated.")
    print("Values <1 under-report diversity (over-merging); >1 over-report (under-merging).")


if __name__ == "__main__":
    main()
