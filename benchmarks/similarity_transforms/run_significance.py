#!/usr/bin/env python
"""Are the differences in the comparison table real, or within noise?

Every table so far reports point estimates of Spearman's rho on a few hundred sets.
A gap of 0.02 between two configurations on 400 sets may be nothing at all, and the
headline claim depends on which gaps survive.

Bootstrap over *sets* (the sampling unit), 2000 resamples, paired -- the same resample
indices are used for every configuration, so the difference distribution accounts for
the fact that they are scored on identical data.

Also isolates the fairness question the comparison raises: the best kernel is given the
prompt and an NLI cross-encoder, neither of which the baselines receive. How much of
the lead survives taking those away?

Usage:
    python run_significance.py --limit 400 --resamples 2000
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
from baselines import vendi_score  # noqa: E402
from common import HERE, encode, load_background, load_human_sets  # noqa: E402
from run_hybrid import cached_nli  # noqa: E402

from linguistic_diversity import SelfBLEU  # noqa: E402
from linguistic_diversity.metric import TextDiversity  # noqa: E402

OUT = HERE / "output" / "significance.json"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NAMES = ("McDiv_nuggets", "conTest", "decTest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    human = load_human_sets()

    # The held-out split: the complement of what every configuration was selected on.
    rng = np.random.default_rng(20260803)
    tasks: dict[str, dict[str, Any]] = {}
    for name, d in human.items():
        chosen = rng.choice(
            np.arange(len(d["sets"])), min(args.limit, len(d["sets"])), replace=False
        )
        idx = np.setdiff1d(np.arange(len(d["sets"])), chosen)
        tasks[name] = {
            "sets": [d["sets"][i] for i in idx],
            "contexts": [d["contexts"][i] for i in idx],
            "target": d["human"][idx],
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
        C = encode([c if c.strip() else " " for c in t["contexts"]], f"hy-{name}-ctx", st)
        t["ctx"] = [C[i] if t["contexts"][i].strip() else None for i in range(len(C))]
        t["nli"] = cached_nli(name, t["sets"], NLI_MODEL)

    ctx_ls_tau = T.compose_context_then(T.local_scaling(bg_small, k=10, z0=0.30))
    ls_tau = T.local_scaling(bg_small, k=10, z0=0.30)
    self_bleu = SelfBLEU()

    def hill(Z: np.ndarray) -> float:
        p = np.full(len(Z), 1.0 / len(Z))
        return TextDiversity._calc_diversity(p, Z, q=1.0, index="vendi")

    def blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        Z = np.sqrt(np.clip(a, 0, 1) * np.clip(b, 0, 1))
        np.fill_diagonal(Z, 1.0)
        return Z

    configs: dict[str, Callable[[dict, int], float]] = {
        "Self-BLEU": lambda t, i: -float(self_bleu(t["sets"][i])),
        "Vendi (raw cosine)": lambda t, i: vendi_score(t["embeds"][i] @ t["embeds"][i].T),
        "ours: current default": lambda t, i: hill(T.floored(0.053)(t["embeds"][i], None)),
        "ours: no prompt, no NLI": lambda t, i: hill(ls_tau(t["embeds"][i], None)),
        "ours: no prompt (+NLI)": lambda t, i: hill(
            blend(t["nli"][i][1], ls_tau(t["embeds"][i], None))
        ),
        "ours: full (prompt+NLI)": lambda t, i: hill(
            blend(t["nli"][i][1], ctx_ls_tau(t["embeds"][i], t["ctx"][i]))
        ),
    }

    # Score once; bootstrap over the stored scores rather than rescoring 2000 times.
    scores: dict[str, dict[str, np.ndarray]] = {}
    for label, fn in configs.items():
        scores[label] = {
            name: np.array([fn(tasks[name], i) for i in range(len(tasks[name]["sets"]))])
            for name in NAMES
        }

    boot = np.random.default_rng(7)
    draws: dict[str, np.ndarray] = {label: np.empty(args.resamples) for label in configs}
    for b in range(args.resamples):
        per_dataset = {}
        for name in NAMES:
            n = len(tasks[name]["sets"])
            per_dataset[name] = boot.integers(0, n, n)
        for label in configs:
            rhos = []
            for name in NAMES:
                sel = per_dataset[name]
                s = scores[label][name][sel]
                y = tasks[name]["target"][sel]
                r = spearmanr(s, y).statistic
                rhos.append(0.0 if not np.isfinite(r) else r)
            draws[label][b] = float(np.mean(rhos))

    print("=" * 96)
    print(f"BOOTSTRAP  {args.resamples} paired resamples over held-out sets")
    print("=" * 96)
    print(f"  {'configuration':26s} {'mean rho':>10s} {'95% CI':>20s}")
    point = {}
    for label in configs:
        obs = float(
            np.mean([spearmanr(scores[label][n], tasks[n]["target"]).statistic for n in NAMES])
        )
        lo, hi = np.percentile(draws[label], [2.5, 97.5])
        point[label] = obs
        print(f"  {label:26s} {obs:10.4f}   [{lo:6.4f}, {hi:6.4f}]")

    print(f"\n{'=' * 96}")
    print("PAIRED DIFFERENCES  does the gap exclude zero?")
    print("=" * 96)
    pairs = [
        ("ours: full (prompt+NLI)", "ours: current default"),
        ("ours: full (prompt+NLI)", "Vendi (raw cosine)"),
        ("ours: no prompt (+NLI)", "Vendi (raw cosine)"),
        ("ours: no prompt, no NLI", "Vendi (raw cosine)"),
        ("ours: full (prompt+NLI)", "ours: no prompt (+NLI)"),
        ("ours: current default", "Self-BLEU"),
    ]
    print(f"  {'comparison':56s} {'diff':>8s} {'95% CI':>20s} {'sig':>5s}")
    out_pairs = {}
    for a, b in pairs:
        d = draws[a] - draws[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = "yes" if lo > 0 or hi < 0 else "NO"
        print(
            f"  {a + '  vs  ' + b:56s} {point[a] - point[b]:+8.4f} "
            f"  [{lo:+6.4f}, {hi:+6.4f}] {sig:>5s}"
        )
        out_pairs[f"{a} vs {b}"] = {
            "diff": round(point[a] - point[b], 4),
            "ci": [round(float(lo), 4), round(float(hi), 4)],
            "significant": sig == "yes",
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {"point": {k: round(v, 4) for k, v in point.items()}, "pairs": out_pairs}, indent=2
        )
    )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
