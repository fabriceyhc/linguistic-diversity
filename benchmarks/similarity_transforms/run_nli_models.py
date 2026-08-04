#!/usr/bin/env python
"""Which cross-encoder should the entailment kernel use?

The shipped choice, `cross-encoder/nli-deberta-v3-base`, was the first one tried. It
turns out to classify *unrelated* text as contradiction with p ~ 1.0 rather than
neutral -- an artefact of SNLI/MNLI training, where the neutral class means "same
topic, not entailed". That makes the kernel a three-level topical scale rather than
the calibrated entailment measure it was chosen to be.

So: does a model trained on harder, more varied NLI data separate neutral from
contradiction properly, and does that translate into a better diversity kernel? Those
are different questions and they may not have the same answer -- a metric only needs a
useful ordering, not correct labels.

Also included is a graded STS cross-encoder, which sidesteps the question by predicting
similarity directly instead of a label distribution.

Two evaluations per model:

  probe       four constructed pairs where the correct label is unambiguous
  benchmark   human agreement and calibration, the criteria that decide anything

Usage:
    python run_nli_models.py --limit 200
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
sys.path.insert(0, str(Path(__file__).parent.parent))
import transforms as T  # noqa: E402
from common import (
    HERE,
    encode,
    load_background,
    load_calibration_sets,
    load_human_sets,
)  # noqa: E402

from linguistic_diversity.metric import TextDiversity  # noqa: E402

OUT = HERE / "output" / "nli_models.json"

# (name, kind). "nli" -> 3-way label distribution; "sts" -> single graded score.
MODELS: list[tuple[str, str]] = [
    ("cross-encoder/nli-deberta-v3-base", "nli"),
    ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", "nli"),
    ("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", "nli"),
    ("tasksource/deberta-base-long-nli", "nli"),
    ("cross-encoder/stsb-roberta-large", "sts"),
]

PROBE = [
    ("The cat sat on the mat.", "A feline rested upon the rug.", "entailment"),
    ("The cat sat on the mat.", "The cat was black and white.", "neutral"),
    ("The cat sat on the mat.", "No animal was anywhere near the mat.", "contradiction"),
    ("The cat sat on the mat.", "The telescope detected a distant galaxy.", "neutral"),
]


def load(model_name: str) -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, device="cuda" if torch.cuda.is_available() else "cpu")


def label_ids(model: Any) -> tuple[int, int, int]:
    mapping = {str(v).lower(): k for k, v in model.config.id2label.items()}
    return (
        mapping.get("entailment", 1),
        mapping.get("contradiction", 0),
        mapping.get("neutral", 2),
    )


def score_pairs(model: Any, kind: str, pairs: list[tuple[str, str]], bs: int = 32) -> np.ndarray:
    if kind == "sts":
        out = model.predict(pairs, batch_size=bs, show_progress_bar=False, convert_to_numpy=True)
        return np.asarray(out, dtype=np.float64).reshape(-1, 1)
    out = model.predict(
        pairs, batch_size=bs, show_progress_bar=False, apply_softmax=True, convert_to_numpy=True
    )
    return np.asarray(out, dtype=np.float64)


def matrices(model: Any, kind: str, sets: list[list[str]], bs: int = 32) -> list[np.ndarray]:
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
    scored = score_pairs(model, kind, pairs, bs)

    out = []
    if kind == "sts":
        for s, entries in zip(sets, index, strict=True):
            Z = np.eye(len(s))
            for i, j, base in entries:
                v = float(np.clip(0.5 * (scored[base, 0] + scored[base + 1, 0]), 0.0, 1.0))
                Z[i, j] = Z[j, i] = v
            out.append(Z)
        return out

    ent, con, _neu = label_ids(model)
    for s, entries in zip(sets, index, strict=True):
        Z = np.eye(len(s))
        for i, j, base in entries:
            ab, ba = scored[base], scored[base + 1]
            e = 0.5 * (ab[ent] + ba[ent])
            c = 0.5 * (ab[con] + ba[con])
            Z[i, j] = Z[j, i] = float(np.clip(0.5 * (1.0 + e - c), 0.0, 1.0))
        out.append(Z)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--dec-limit", type=int, default=100)
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    human = load_human_sets()
    calib = load_calibration_sets()
    rng = np.random.default_rng(20260803)

    tasks: dict[str, dict[str, Any]] = {}
    for name, d in human.items():
        cap = args.dec_limit if name == "decTest" else args.limit
        idx = rng.choice(np.arange(len(d["sets"])), min(cap, len(d["sets"])), replace=False)
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
    ctx_ls_tau = T.compose_context_then(T.local_scaling(bg_small, k=10, z0=0.30))
    for name, t in tasks.items():
        flat, bounds = [], []
        for s in t["sets"]:
            bounds.append((len(flat), len(flat) + len(s)))
            flat += s
        E = encode(flat, f"nm-{name}-docs", st)
        t["embeds"] = [E[a:b] for a, b in bounds]
        if any(c.strip() for c in t["contexts"]):
            C = encode([c if c.strip() else " " for c in t["contexts"]], f"nm-{name}-ctx", st)
            t["ctx"] = [C[i] if t["contexts"][i].strip() else None for i in range(len(C))]
        else:
            t["ctx"] = [None] * len(t["sets"])
        t["emb_Z"] = [ctx_ls_tau(t["embeds"][i], t["ctx"][i]) for i in range(len(t["sets"]))]

    names = ["McDiv_nuggets", "conTest", "decTest"]
    results: dict[str, Any] = {}

    print("=" * 118)
    print("PROBE  does the model separate 'unrelated' from 'contradiction'?")
    print("=" * 118)
    print(f"  {'model':56s} {'entail':>9s} {'same-topic':>11s} {'contra':>9s} {'unrelated':>11s}")

    probe_rows: dict[str, Any] = {}
    kernels: dict[str, dict[str, list[np.ndarray]]] = {}
    for model_name, kind in MODELS:
        try:
            model = load(model_name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {model_name:56s} unavailable ({type(exc).__name__})")
            continue

        if kind == "nli":
            ent, con, neu = label_ids(model)
            probs = score_pairs(model, kind, [(a, b) for a, b, _ in PROBE])
            cells, detail = [], {}
            for (a, b, gold), p in zip(PROBE, probs, strict=True):
                predicted = ["contradiction", "entailment", "neutral"][
                    int(np.argmax([p[con], p[ent], p[neu]]))
                ]
                mark = "ok" if predicted == gold else predicted[:6]
                cells.append(f"{p[neu]:.2f}/{mark}")
                detail[gold + ":" + b[:20]] = {
                    "entail": round(float(p[ent]), 4),
                    "contra": round(float(p[con]), 4),
                    "neutral": round(float(p[neu]), 4),
                    "predicted": predicted,
                }
            probe_rows[model_name] = detail
            print(f"  {model_name:56s} " + " ".join(f"{c:>11s}" for c in cells))
        else:
            vals = score_pairs(model, kind, [(a, b) for a, b, _ in PROBE])[:, 0]
            probe_rows[model_name] = {
                g + ":" + b[:20]: round(float(v), 4)
                for (a, b, g), v in zip(PROBE, vals, strict=True)
            }
            print(f"  {model_name:56s} " + " ".join(f"{v:11.3f}" for v in vals))

        kernels[model_name] = {}
        for name, t in tasks.items():
            kernels[model_name][name] = matrices(model, kind, t["sets"])
        del model
        torch.cuda.empty_cache()

    print(
        "\n  Column 2 shows P(neutral) and the argmax label. The last column is the one\n"
        "  that matters: an unrelated pair should be NEUTRAL, not contradiction.\n"
    )

    print("=" * 118)
    print("BENCHMARK  NLI kernel alone, and blended with the embedding kernel")
    print("=" * 118)
    hdr = names + ["mean", "calib_rho", "calib_ratio"]
    print(f"  {'model':56s} {'mode':9s} " + " ".join(f"{h:>11s}" for h in hdr))

    def evaluate(get_Z: Any) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for name in names:
            t = tasks[name]
            s = np.array(
                [
                    TextDiversity._calc_diversity(
                        np.full(len(t["sets"][i]), 1 / len(t["sets"][i])),
                        get_Z(name, i),
                        q=1.0,
                        index="vendi",
                    )
                    for i in range(len(t["sets"]))
                ]
            )
            row[name] = round(float(spearmanr(s, t["target"]).statistic), 4)
        row["mean"] = round(float(np.mean([row[n] for n in names])), 4)
        t = tasks["calibration"]
        s = np.array(
            [
                TextDiversity._calc_diversity(
                    np.full(len(t["sets"][i]), 1 / len(t["sets"][i])),
                    get_Z("calibration", i),
                    q=1.0,
                    index="vendi",
                )
                for i in range(len(t["sets"]))
            ]
        )
        row["calib_rho"] = round(float(spearmanr(s, t["target"]).statistic), 4)
        row["calib_ratio"] = round(float(np.median(s / t["target"])), 4)
        return row

    for model_name in kernels:
        alone = evaluate(lambda n, i, m=model_name: kernels[m][n][i])

        def blended(n: str, i: int, m: str = model_name) -> np.ndarray:
            Z = np.sqrt(np.clip(kernels[m][n][i], 0, 1) * np.clip(tasks[n]["emb_Z"][i], 0, 1))
            np.fill_diagonal(Z, 1.0)
            return Z

        both = evaluate(blended)
        results[model_name] = {"probe": probe_rows[model_name], "alone": alone, "blended": both}
        for mode, row in (("alone", alone), ("blended", both)):
            print(
                f"  {model_name if mode == 'alone' else '':56s} {mode:9s} "
                + " ".join(f"{row[h]:+11.4f}" for h in hdr)
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
