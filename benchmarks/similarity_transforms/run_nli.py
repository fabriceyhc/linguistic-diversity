#!/usr/bin/env python
"""Does an entailment kernel beat a cosine kernel?

The similarity floor exists because cosine similarity of sentence embeddings is not
calibrated: unrelated text lands near 0.07 rather than 0, which caps diversity at 1/z
however many documents there are. Every correction tried so far is a transform of an
already-uncalibrated matrix.

An NLI classifier's output is bounded and calibrated by construction, and its
entailment probability for unrelated text really is ~0.000 -- so there is no floor to
remove. No query or prompt is involved: an NLI cross-encoder consumes a *sentence
pair*, so the pairs are simply the documents in the set, scored in both orderings and
symmetrised.

    sim(a, b) = mean over both directions of P(entail)                     "entailment"
    sim(a, b) = mean over both directions of (1 + P(entail) - P(contra))/2 "ent-contra"

**How this model actually behaves, which is not what one might assume.** Measured on
`cross-encoder/nli-deberta-v3-base`:

    paraphrase          entail 0.993, contra 0.000   ->  Z = 0.994
    same topic, neutral entail 0.001, contra 0.000   ->  Z = 0.500
    different topic     entail 0.000, contra 1.000   ->  Z = 0.000
    contradiction       entail 0.000, contra 1.000   ->  Z = 0.000

Unrelated sentences are classified as **contradiction**, not neutral -- an artefact of
MNLI-style training, where the neutral class means "same topic, not entailed" rather
than "unrelated". So `ent-contra` is really a three-level topical scale (paraphrase /
same-topic / off-topic) and it cannot distinguish an off-topic sentence from a
contradictory one. For measuring diversity that conflation is defensible, since both are
maximally dissimilar, but it is a property of the training data rather than of NLI, and
it is the main reason to treat this kernel as domain-dependent rather than universal.

Cost: O(n^2) cross-encoder passes against O(n) encodes, which is why this is measured
as an option rather than assumed as a default.

Usage:
    python run_nli.py --limit 400
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from common import HERE, load_calibration_sets, load_human_sets  # noqa: E402

OUT = HERE / "output" / "nli.json"
MODEL = "cross-encoder/nli-deberta-v3-base"


_MODELS: dict[str, Any] = {}


def _cross_encoder(model_name: str) -> Any:
    """One instance per process; reloading it per call exhausts VRAM."""
    from sentence_transformers import CrossEncoder

    if model_name not in _MODELS:
        _MODELS[model_name] = CrossEncoder(
            model_name, device="cuda" if torch.cuda.is_available() else "cpu"
        )
    return _MODELS[model_name]


def nli_matrices(
    sets: list[list[str]], model_name: str, batch_size: int = 64, progress: bool = True
) -> list[np.ndarray]:
    model = _cross_encoder(model_name)
    label2id = {k.lower(): v for v, k in model.config.id2label.items()}
    ent = label2id.get("entailment", 1)
    con = label2id.get("contradiction", 0)

    pairs: list[tuple[str, str]] = []
    index: list[list[tuple[int, int, int]]] = []
    for s in sets:
        entries = []
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                entries.append((i, j, len(pairs)))
                pairs.append((s[i], s[j]))
                pairs.append((s[j], s[i]))
        index.append(entries)

    if progress:
        print(f"  scoring {len(pairs)} ordered pairs ...")
    logits = model.predict(
        pairs,
        batch_size=batch_size,
        show_progress_bar=progress,
        apply_softmax=True,
        convert_to_numpy=True,
    )

    out = []
    for s, entries in zip(sets, index):
        n = len(s)
        Z_ent = np.eye(n)
        Z_ec = np.eye(n)
        for i, j, base in entries:
            ab, ba = logits[base], logits[base + 1]
            e = 0.5 * (ab[ent] + ba[ent])
            c = 0.5 * (ab[con] + ba[con])
            Z_ent[i, j] = Z_ent[j, i] = e
            # entailment minus contradiction, rescaled from [-1,1] to [0,1]
            Z_ec[i, j] = Z_ec[j, i] = np.clip(0.5 * (1.0 + e - c), 0.0, 1.0)
        out.append((np.clip(Z_ent, 0, 1), Z_ec))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=400, help="sets per dataset (0 = all)")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    from linguistic_diversity.metric import TextDiversity
    from linguistic_diversity.utils import maximum_diversity

    human = load_human_sets()
    calib = load_calibration_sets()

    rng = np.random.default_rng(20260803)
    tasks: dict[str, dict[str, Any]] = {}
    for name, d in human.items():
        idx = np.arange(len(d["sets"]))
        if args.limit:
            idx = rng.choice(idx, min(args.limit, len(idx)), replace=False)
        tasks[name] = {"sets": [d["sets"][i] for i in idx], "target": d["human"][idx]}
    tasks["calibration"] = {"sets": calib["sets"], "target": calib["expected"]}

    results: dict[str, Any] = {}
    for name, t in tasks.items():
        print(f"\n{name}: {len(t['sets'])} sets")
        mats = nli_matrices(t["sets"], args.model)
        for variant, pos in (("NLI entailment", 0), ("NLI ent-contra", 1)):
            scores, ceil = [], []
            for Z, _ in [(m[pos], None) for m in mats]:
                p = np.full(len(Z), 1.0 / len(Z))
                scores.append(TextDiversity._calc_diversity(p, Z, q=1.0, index="vendi"))
                ceil.append(maximum_diversity(Z)[0])
            s = np.array(scores, float)
            rho = float(spearmanr(s, t["target"]).statistic)
            entry = {
                "spearman": round(rho, 4),
                "mean_D": round(float(s.mean()), 3),
                "ceiling": round(float(np.mean(ceil)), 3),
            }
            if name == "calibration":
                entry["median_ratio"] = round(float(np.median(s / t["target"])), 4)
            results.setdefault(variant, {})[name] = entry
            print(
                f"  {variant:16s} rho={rho:+.4f}  mean D={s.mean():6.3f}  "
                f"ceiling={np.mean(ceil):.3f}"
                + (f"  ratio={entry.get('median_ratio')}" if name == "calibration" else "")
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"model": args.model, "limit": args.limit, "results": results}, indent=2)
    )
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
