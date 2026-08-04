"""Shared loading, caching and evaluation for the similarity-transform sweep.

Every candidate correction is a function that turns cached embeddings into a
similarity matrix, so all of them are scored on identical data with identical
code and can be composed without re-encoding anything.

Two criteria, and they can disagree -- that disagreement is the whole point:

  agreement    Spearman against averaged human diversity ratings, three datasets.
  calibration  do we recover a *known* number of distinct concepts?

Also reported is the achievable ceiling (Leinster-Meckes magnitude), which is what
the similarity floor exists to raise.
"""

from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).parent
CACHE = HERE / "cache"
DATA = HERE.parent / "embedder_selection" / "data"
VALIDATION = HERE.parent / "metric_validation" / "output" / "benchmark.json"
HUMAN = "metric_abs_hds_mean"
ENCODER = "sentence-transformers/all-mpnet-base-v2"

# Families whose expected semantic count is a genuine concept count.
CALIB_FAMILIES = ("syntactic_alternations", "syntactic_frames", "random_controls")

Array = npt.NDArray[np.float64]
# (embeddings of one set, context embedding or None) -> similarity matrix
Transform = Callable[[Array, Array | None], Array]


# --------------------------------------------------------------------------- data


def _load(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA / pattern)))
    if not files:
        raise FileNotFoundError(f"nothing matched {DATA / pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_human_sets() -> dict[str, dict[str, Any]]:
    """Response sets with human diversity scores, plus the context of each set."""
    specs = {
        "McDiv_nuggets": "with_metrics/McDiv_nuggets/*with_hds*.csv",
        "conTest": "raw/conTest/*with_hds*.csv",
        "decTest": "raw/decTest/*with_hds*.csv",
    }
    out: dict[str, dict[str, Any]] = {}
    for name, pattern in specs.items():
        df = _load(pattern)
        resp_cols = [c for c in df.columns if c.startswith("resp_")]
        sets, contexts, human = [], [], []
        for _, row in df.iterrows():
            docs = [
                str(row[c]) for c in resp_cols if isinstance(row[c], str) and str(row[c]).strip()
            ]
            if len(docs) < 2:
                continue
            sets.append(docs)
            contexts.append(str(row["context"]) if "context" in df.columns else "")
            human.append(float(row[HUMAN]))
        out[name] = {"sets": sets, "contexts": contexts, "human": np.array(human, float)}
    return out


def load_calibration_sets() -> dict[str, Any]:
    """Constructed corpora with a known number of distinct concepts."""
    benchmark = json.loads(VALIDATION.read_text())
    sets, expected = [], []
    for corpus in benchmark["corpora"]:
        k = corpus["expected"].get("semantic")
        if k is None or corpus["family"] not in CALIB_FAMILIES:
            continue
        sets.append([str(d) for d in corpus["documents"]])
        expected.append(float(k))
    return {"sets": sets, "contexts": [""] * len(sets), "expected": np.array(expected, float)}


def load_background(n_texts: int = 6000, seed: int = 20260803) -> list[str]:
    """Text used to fit every correction. Disjoint from everything scored."""
    df = _load("raw/decTest/*no_hds*.csv").sample(n=600, random_state=seed)
    texts: list[str] = []
    for _, row in df.iterrows():
        texts += [
            str(row[c])
            for c in df.columns
            if str(c).startswith("resp_") and isinstance(row[c], str) and str(row[c]).strip()
        ]
    return texts[:n_texts]


# ----------------------------------------------------------------------- encoding


def _cache_path(tag: str, texts: list[str]) -> Path:
    digest = hashlib.sha1("\x00".join(texts).encode()).hexdigest()[:16]
    return CACHE / f"{tag}-{digest}.npy"


def encode(texts: list[str], tag: str, model: Any = None, batch_size: int = 128) -> Array:
    """Encode with caching. Unit-normalised, float64."""
    path = _cache_path(tag, texts)
    if path.exists():
        return np.load(path)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(ENCODER)
    E = model.encode(
        texts, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
    )
    E = np.asarray(E, dtype=np.float64)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(path, E)
    return E


# --------------------------------------------------------------------- evaluation


def _score_all(
    sets: list[list[str]],
    embeds: list[Array],
    ctx: list[Array | None],
    transform: Transform,
    index: str,
    q: float,
) -> Array:
    from linguistic_diversity.metric import TextDiversity

    out = np.empty(len(sets), dtype=float)
    for i, E in enumerate(embeds):
        Z = transform(E, ctx[i])
        p = np.full(len(E), 1.0 / len(E))
        out[i] = TextDiversity._calc_diversity(p, Z, q=q, index=index)
    return out


def ceiling(embeds: list[Array], ctx: list[Array | None], transform: Transform) -> float:
    from linguistic_diversity.utils import maximum_diversity

    return float(np.mean([maximum_diversity(transform(E, c))[0] for E, c in zip(embeds, ctx)]))


def evaluate(
    prepared: dict[str, Any],
    transform: Transform,
    index: str = "vendi",
    q: float = 1.0,
) -> dict[str, Any]:
    """Score one configuration on every criterion."""
    row: dict[str, Any] = {}
    for name, d in prepared["human"].items():
        s = _score_all(d["sets"], d["embeds"], d["ctx"], transform, index, q)
        mask = np.isfinite(s) & np.isfinite(d["human"])
        row[name] = round(float(spearmanr(s[mask], d["human"][mask]).statistic), 4)
    row["agreement_mean"] = round(float(np.mean([row[k] for k in prepared["human"]])), 4)

    c = prepared["calibration"]
    s = _score_all(c["sets"], c["embeds"], c["ctx"], transform, index, q)
    ratios = s / c["expected"]
    row["calib_rho"] = round(float(spearmanr(s, c["expected"]).statistic), 4)
    row["calib_ratio"] = round(float(np.median(ratios)), 4)
    row["ceiling"] = round(
        ceiling(
            prepared["human"]["McDiv_nuggets"]["embeds"],
            prepared["human"]["McDiv_nuggets"]["ctx"],
            transform,
        ),
        3,
    )
    return row


HEADERS = [
    "McDiv_nuggets",
    "conTest",
    "decTest",
    "agreement_mean",
    "calib_rho",
    "calib_ratio",
    "ceiling",
]


def print_header(label_width: int = 30) -> None:
    print(f"  {'configuration':{label_width}s} " + " ".join(f"{h:>14s}" for h in HEADERS))
    print("  " + "-" * (label_width + 15 * len(HEADERS)))


def print_row(label: str, row: dict[str, Any], label_width: int = 30) -> None:
    cells = []
    for h in HEADERS:
        v = row[h]
        cells.append(f"{v:+14.4f}" if h != "ceiling" else f"{v:14.3f}")
    print(f"  {label:{label_width}s} " + " ".join(cells))
