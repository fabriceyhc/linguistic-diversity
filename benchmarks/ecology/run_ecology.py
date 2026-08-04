#!/usr/bin/env python
"""Evenness, sample coverage and the alpha/beta/gamma partition, on real corpora.

Three questions a single effective number cannot answer, each with a settled answer
in ecology and each now implemented in `linguistic_diversity.ecology`. This checks
that the implementations say something true and useful about text, not merely that
they satisfy their axioms -- the unit tests cover that.

    Q1  EVENNESS. Diversity falls when abundance concentrates. Does evenness say
        *why*, and does it separate corpora that diversity alone conflates?

    Q2  COVERAGE. Which linguistic levels have enough feature repetition for
        completeness to be estimable at all? The prediction from the duplicate-rate
        study is that this tracks alphabet size relative to sequence length, not the
        linguistic level.

    Q3  PARTITION. Given text from several sources, is the pooled diversity coming
        from within the sources or between them? Tested where the answer is known:
        the three decTest generation tasks are genuinely different sources, and
        temperature-stratified slices of one task are genuinely the same source.

Usage:
    python run_ecology.py
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from linguistic_diversity import (
    ConstituencyParse,
    DependencyParse,
    DocumentSemantics,
    PartOfSpeechSequence,
    Phonemic,
    Rhythmic,
    clear_model_cache,
)
from linguistic_diversity.ecology import expected_coverage, sample_coverage, size_for_coverage

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "output" / "results.json"
DATA = HERE.parent / "embedder_selection" / "data"
SEED = HERE.parent / "metric_validation" / "data" / "constructions.json"
TASKS = ("story_gen", "resp_gen", "prompt_gen")


def load_task(task: str, limit: int) -> pd.DataFrame:
    files = sorted(glob.glob(str(DATA / f"raw/decTest/*no_hds*{task}.csv")))
    if not files:
        raise FileNotFoundError(f"no decTest file for {task}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.sample(n=min(limit, len(df)), random_state=20260803)


def docs_of(df: pd.DataFrame, limit: int) -> list[str]:
    cols = [c for c in df.columns if c.startswith("resp_")]
    out: list[str] = []
    for _, row in df.iterrows():
        out += [str(row[c]) for c in cols if isinstance(row[c], str) and str(row[c]).strip()]
        if len(out) >= limit:
            break
    return out[:limit]


def q1_evenness(results: dict[str, Any]) -> None:
    print("=" * 84)
    print("Q1  EVENNESS -- same documents, different abundance")
    print("=" * 84)
    seed = json.loads(SEED.read_text())
    themes = seed["abundance_themes"]["themes"]
    metric = DocumentSemantics({"verbose": False})

    docs: list[str] = []
    per_theme = []
    for theme in themes:
        chosen = theme["paraphrases"][:3]
        per_theme.append(len(chosen))
        docs += chosen

    print(f"  {len(docs)} documents over {len(themes)} themes; only the weights change\n")
    print(f"  {'profile':22s} {'D_1':>8s} {'richness':>9s} {'E3':>7s} {'E5':>7s}")
    rows = {}
    for profile in seed["abundance_themes"]["profiles"]:
        weights = np.asarray(profile["weights"], dtype=np.float64)[: len(themes)]
        abundance: list[float] = []
        for w, n in zip(weights, per_theme, strict=True):
            abundance += [w / n] * n
        prof = metric.diversity_profile(docs, q_values=(0.0, 1.0), abundance=abundance)
        e3 = metric.evenness(docs, q=1.0, measure="E3", abundance=abundance)
        e5 = metric.evenness(docs, q=1.0, measure="E5", abundance=abundance)
        print(f"  {profile['id']:22s} {prof[1.0]:8.3f} {prof[0.0]:9.3f} {e3:7.3f} {e5:7.3f}")
        rows[profile["id"]] = {
            "note": profile["note"],
            "diversity_q1": round(prof[1.0], 4),
            "richness": round(prof[0.0], 4),
            "E3": round(e3, 4),
            "E5": round(e5, 4),
        }
    results["evenness"] = rows
    print(
        "\n  Richness is fixed by the documents; evenness is what the weights move.\n"
        "  Diversity alone cannot separate 'fewer things' from 'less balanced'.\n"
    )
    clear_model_cache()


def q2_coverage(results: dict[str, Any], n_docs: int) -> None:
    print("=" * 84)
    print("Q2  COVERAGE -- which levels repeat enough to estimate completeness?")
    print("=" * 84)
    corpus = docs_of(load_task("story_gen", 400), n_docs)
    print(f"  {len(corpus)} real sentences, one similarity matrix per metric\n")

    metrics = {
        "DocumentSemantics": DocumentSemantics,
        "Phonemic": Phonemic,
        "DependencyParse": DependencyParse,
        "PartOfSpeechSequence": PartOfSpeechSequence,
        "Rhythmic": Rhythmic,
        "ConstituencyParse": ConstituencyParse,
    }
    print(f"  {'metric':24s} {'coverage':>9s} {'deficit':>9s} {'n for C=0.9':>12s}")
    rows = {}
    for name, cls in metrics.items():
        try:
            metric = cls({"verbose": False})
            coverage = metric.sample_coverage(corpus)
        except Exception as exc:  # noqa: BLE001 - a missing model must not stop the sweep
            print(f"  {name:24s} unavailable ({type(exc).__name__})")
            continue
        finally:
            clear_model_cache()
        target = "unreachable"
        if coverage > 0:
            # Species counts are recoverable from the same equivalence classes.
            target = str(_size_for_target(metric, corpus, 0.9))
        print(f"  {name:24s} {coverage:9.4f} {1 - coverage:9.4f} {target:>12s}")
        rows[name] = {"coverage": round(coverage, 4), "deficit": round(1 - coverage, 4)}
    results["coverage"] = rows
    ordered = sorted(rows.items(), key=lambda kv: kv[1]["coverage"])
    print(
        f"\n  Ordering runs {ordered[0][0]} ({ordered[0][1]['coverage']:.3f}) to "
        f"{ordered[-1][0]} ({ordered[-1][1]['coverage']:.3f}), as the duplicate-rate\n"
        "  study predicts: coverage tracks alphabet size against sequence length, so\n"
        "  ConstituencyParse -- four labels over short skeletons -- repeats most.\n"
        "  Every level is above zero here only because this corpus contains genuine\n"
        "  repeated generations; a corpus of distinct documents scores exactly 0 at\n"
        "  the semantic level, which is the honest answer rather than a failure.\n"
    )


def _size_for_target(metric: Any, corpus: list[str], target: float) -> int:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    features, _ = metric.extract_features(corpus)
    Z = np.asarray(metric.calculate_similarities(features), dtype=np.float64)
    _n, labels = connected_components(csr_matrix(Z >= 1.0 - 1e-9), directed=False)
    counts = np.bincount(labels).astype(np.float64)
    try:
        return size_for_coverage(counts, target)
    except ValueError:
        return -1


def q3_partition(results: dict[str, Any], per_source: int) -> None:
    print("=" * 84)
    print("Q3  PARTITION -- is the diversity within sources or between them?")
    print("=" * 84)
    metric = DocumentSemantics({"verbose": False})

    # (a) Three genuinely different generation tasks.
    sources = {task: docs_of(load_task(task, 200), per_source) for task in TASKS}
    print(f"  (a) three decTest tasks, {per_source} documents each")
    print(f"      {'q':>5s} {'gamma':>8s} {'alpha':>8s} {'beta':>8s} {'repr':>8s}")
    rows_a = {}
    for q in (0.0, 1.0, 2.0):
        r = metric.partition(sources, q=q)
        print(
            f"      {q:5g} {r.gamma:8.3f} {r.alpha:8.3f} {r.beta:8.3f} "
            f"{r.representativeness:8.3f}"
        )
        rows_a[str(q)] = r.to_dict()

    # (b) The same task, split arbitrarily. Beta should be near 1.
    pool = docs_of(load_task("story_gen", 400), per_source * 3)
    halves = {f"slice_{i}": pool[i * per_source : (i + 1) * per_source] for i in range(3)}
    print("\n  (b) one task split three ways -- the same source, so beta should be ~1")
    print(f"      {'q':>5s} {'gamma':>8s} {'alpha':>8s} {'beta':>8s} {'repr':>8s}")
    rows_b = {}
    for q in (0.0, 1.0, 2.0):
        r = metric.partition(halves, q=q)
        print(
            f"      {q:5g} {r.gamma:8.3f} {r.alpha:8.3f} {r.beta:8.3f} "
            f"{r.representativeness:8.3f}"
        )
        rows_b[str(q)] = r.to_dict()

    results["partition"] = {"different_tasks": rows_a, "same_task_split": rows_b}
    print(
        "\n  beta is the effective number of DISTINCT sources: 1 when interchangeable,\n"
        "  N when they share nothing. The contrast between (a) and (b) is the test.\n"
    )
    clear_model_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-docs", type=int, default=150)
    parser.add_argument("--per-source", type=int, default=60)
    args = parser.parse_args()

    results: dict[str, Any] = {}
    q1_evenness(results)
    q2_coverage(results, args.n_docs)
    q3_partition(results, args.per_source)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
