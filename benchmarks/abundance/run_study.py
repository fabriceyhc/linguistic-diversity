#!/usr/bin/env python
"""Does the score track abundance, and does the profile recover its shape?

Every other benchmark here uses uniform abundance, which is the regime where a
similarity-sensitive Hill number and a purely spectral index behave most alike.
This one varies abundance deliberately.

The construction gives an exact ground truth rather than an authored guess. Each
theme is a set of paraphrases of one proposition, so documents within a theme are
near-identical and documents across themes are unrelated. Under that structure the
similarity matrix is block-diagonal and the true diversity at order q is exactly
the Hill number of the *theme weight vector*:

    D_q(true) = (sum_i w_i^q)^(1/(1-q))     w normalised, q != 1
    D_1(true) = exp(-sum_i w_i log w_i)

So the whole diversity profile has a closed form, and a metric can be scored on
recovering the curve rather than a single point.

Weight profiles run from uniform to near-degenerate, including a Zipfian shape --
the usual distribution of topic frequency in a real corpus.

Usage:
    python run_study.py [--out output/results.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines import vendi_score  # noqa: E402

from linguistic_diversity import DocumentSemantics, clear_model_cache  # noqa: E402

HERE = Path(__file__).parent
SEED = HERE.parent / "metric_validation" / "data" / "constructions.json"
DEFAULT_OUT = HERE / "output" / "results.json"
Q_VALUES = (0.0, 0.5, 1.0, 2.0, float("inf"))
DOCS_PER_THEME = 3


def true_profile(weights: npt.NDArray[np.float64]) -> dict[float, float]:
    """Closed-form Hill numbers of the weight vector: the exact answer."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    out: dict[float, float] = {}
    for q in Q_VALUES:
        if np.isinf(q):
            out[float(q)] = float(1.0 / w.max())
        elif q == 1.0:
            out[float(q)] = float(np.exp(-np.sum(w * np.log(w))))
        else:
            out[float(q)] = float(np.power(np.sum(np.power(w, q)), 1.0 / (1.0 - q)))
    return out


def build(seed: dict, profile: dict) -> tuple[list[str], npt.NDArray[np.float64]]:
    """One document per theme-paraphrase, weighted by the theme's share."""
    themes = seed["abundance_themes"]["themes"]
    weights = np.asarray(profile["weights"], dtype=np.float64)[: len(themes)]
    docs: list[str] = []
    abundance: list[float] = []
    for theme, w in zip(themes, weights, strict=True):
        chosen = theme["paraphrases"][:DOCS_PER_THEME]
        docs.extend(chosen)
        # The theme's mass is split evenly across its paraphrases.
        abundance.extend([w / len(chosen)] * len(chosen))
    return docs, np.asarray(abundance, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    seed = json.loads(SEED.read_text())
    metric = DocumentSemantics({"verbose": False})
    results: dict[str, Any] = {}

    print(
        f"{len(seed['abundance_themes']['themes'])} themes x {DOCS_PER_THEME} paraphrases; "
        f"true diversity is the Hill number of the theme weights\n"
    )

    for profile in seed["abundance_themes"]["profiles"]:
        docs, abundance = build(seed, profile)
        truth = true_profile(
            np.asarray(profile["weights"][: len(seed["abundance_themes"]["themes"])])
        )
        measured = metric.diversity_profile(docs, q_values=Q_VALUES, abundance=abundance)
        uniform = metric.diversity_profile(docs, q_values=Q_VALUES)

        print(f"{profile['id']}  -- {profile['note']}")
        print(f"  {'q':>5s} {'true':>8s} {'weighted':>9s} {'uniform':>9s}")
        for q in Q_VALUES:
            label = "inf" if np.isinf(q) else f"{q:g}"
            print(
                f"  {label:>5s} {truth[float(q)]:8.3f} {measured[float(q)]:9.3f} "
                f"{uniform[float(q)]:9.3f}"
            )
        results[profile["id"]] = {
            "note": profile["note"],
            "true": {str(k): round(v, 4) for k, v in truth.items()},
            "weighted": {str(k): round(v, 4) for k, v in measured.items()},
            "uniform": {str(k): round(v, 4) for k, v in uniform.items()},
        }
        print()

    # Rank recovery must be measured ACROSS profiles at fixed q. Comparing across q
    # within one profile cannot fail: D_q is non-increasing in q by construction, so
    # any profile correlates +1 with any other.
    print("=" * 74)
    print("Rank recovery across profiles, at fixed q")
    print("-" * 74)
    print(f"  {'q':>5s} {'rho(weighted, true)':>20s} {'rho(uniform, true)':>20s}")
    keys = list(results)
    for q in Q_VALUES:
        t = [results[k]["true"][str(float(q))] for k in keys]
        w = [results[k]["weighted"][str(float(q))] for k in keys]
        u = [results[k]["uniform"][str(float(q))] for k in keys]
        label = "inf" if np.isinf(q) else f"{q:g}"
        if np.std(t) < 1e-9:
            print(f"  {label:>5s} {'n/a - truth is constant':>20s} {'constant':>20s}")
            continue
        uniform_cell = "constant" if np.std(u) < 1e-9 else f"{spearmanr(u, t).statistic:+.3f}"
        print(f"  {label:>5s} {spearmanr(w, t).statistic:+20.3f} {uniform_cell:>20s}")

    # Vendi cannot take weights: it sees the same matrix whatever the abundance.
    print(f"\n{'=' * 74}")
    print("The same corpora under Vendi, which has no abundance to take")
    print("-" * 74)
    for profile in seed["abundance_themes"]["profiles"]:
        docs, _ = build(seed, profile)
        feats, _ = metric.extract_features(docs)
        K = np.asarray(metric.calculate_similarities(feats), dtype=np.float64)
        t = true_profile(np.asarray(profile["weights"][: len(seed["abundance_themes"]["themes"])]))
        print(f"  {profile['id']:16s} Vendi {vendi_score(K):7.3f}   " f"true D_1 {t[1.0]:7.3f}")
    print("\n  The corpus text is identical across profiles -- only the weights differ --")
    print("  so Vendi returns one number for all five while the truth ranges widely.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")
    clear_model_cache()


if __name__ == "__main__":
    main()
