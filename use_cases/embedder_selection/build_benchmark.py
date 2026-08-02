#!/usr/bin/env python
"""Build the embedder-screening benchmark from the concept seed data.

Each generated corpus carries a ground-truth effective diversity, so an embedder
can be scored on absolute calibration rather than ranking alone.

Usage:
    python build_benchmark.py [--out output/benchmark.json]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
DEFAULT_SEED_PATH = HERE / "data" / "concepts.json"
DEFAULT_OUT_PATH = HERE / "output" / "benchmark.json"

# Number of distinct concepts per corpus. 1 is the degenerate "all paraphrases of
# one idea" case; 8 approaches "everything is distinct".
K_VALUES = (1, 2, 3, 4, 6, 8)
# Paraphrases drawn per concept. m>1 is what makes the test non-trivial: corpus
# size is k*m but ground-truth diversity stays k.
M_VALUES = (2, 3)
# Independent draws per (k, m) cell, for variance.
DRAWS_PER_CELL = 3


def build_synonymy_corpora(
    clusters: list[dict[str, Any]], rng: random.Random
) -> list[dict[str, Any]]:
    """Build corpora of k concepts x m paraphrases, ground truth = k.

    Two sampling regimes:
      - "cross_domain": concepts drawn from different domains (easier to separate)
      - "within_domain": concepts drawn from one domain, so they are semantically
        adjacent (much harder, and closer to real corpora)

    Args:
        clusters: Concept clusters from the seed data.
        rng: Seeded RNG for reproducibility.

    Returns:
        List of corpus records.
    """
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        by_domain[cluster["domain"]].append(cluster)

    corpora = []
    for k in K_VALUES:
        for m in M_VALUES:
            for draw in range(DRAWS_PER_CELL):
                # Cross-domain: at most one concept per domain, so concepts differ broadly
                domains = rng.sample(sorted(by_domain), min(k, len(by_domain)))
                picked = [rng.choice(by_domain[d]) for d in domains]
                while len(picked) < k:  # small domain count; top up from anywhere
                    extra = rng.choice(clusters)
                    if extra["id"] not in {c["id"] for c in picked}:
                        picked.append(extra)
                corpora.append(
                    _make_record("synonymy", "cross_domain", k, m, draw, picked, rng)
                )

                # Within-domain: only from domains large enough to supply k concepts
                eligible = [d for d, cs in by_domain.items() if len(cs) >= k]
                if eligible:
                    domain = rng.choice(sorted(eligible))
                    picked = rng.sample(by_domain[domain], k)
                    corpora.append(
                        _make_record("synonymy", "within_domain", k, m, draw, picked, rng)
                    )
    return corpora


def _make_record(
    axis: str,
    regime: str,
    k: int,
    m: int,
    draw: int,
    picked: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:
    """Assemble one corpus record with its ground truth."""
    documents, labels = [], []
    for cluster in picked:
        for text in rng.sample(cluster["paraphrases"], m):
            documents.append(text)
            labels.append(cluster["id"])
    order = list(range(len(documents)))
    rng.shuffle(order)
    return {
        "id": f"{axis}-{regime}-k{k}-m{m}-{draw}",
        "axis": axis,
        "regime": regime,
        "documents": [documents[i] for i in order],
        "concept_labels": [labels[i] for i in order],
        "n_documents": len(documents),
        "true_diversity": float(k),
        "rationale": f"{k} distinct concepts, {m} paraphrases each",
    }


def build_polysemy_corpora(sets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build corpora where every sentence is a distinct sense, ground truth = n.

    These share a surface form, so a model driven by lexical overlap will
    under-report diversity here even if it scores well on the synonymy axis.

    Args:
        sets: Polysemy sets from the seed data.

    Returns:
        List of corpus records.
    """
    corpora = []
    for pset in sets:
        senses = pset["senses"]
        corpora.append({
            "id": f"polysemy-{pset['id']}",
            "axis": "polysemy",
            "regime": "shared_surface_form",
            "documents": list(senses),
            "concept_labels": [f"{pset['id']}_sense{i}" for i in range(len(senses))],
            "n_documents": len(senses),
            "true_diversity": float(len(senses)),
            "rationale": f"{len(senses)} unrelated senses of '{pset['surface_form']}'",
        })
    return corpora


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-data", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--random-seed", type=int, default=20260801)
    args = parser.parse_args()

    seed = json.loads(args.seed_data.read_text())
    rng = random.Random(args.random_seed)

    corpora = build_synonymy_corpora(seed["synonymy_clusters"], rng)
    corpora += build_polysemy_corpora(seed["polysemy_sets"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "_meta": {
            "random_seed": args.random_seed,
            "n_corpora": len(corpora),
            "source": str(args.seed_data.name),
            "scoring": "A calibrated metric reports true_diversity for every corpus.",
        },
        "corpora": corpora,
    }, indent=2))

    by_axis: dict[str, int] = defaultdict(int)
    for c in corpora:
        by_axis[f"{c['axis']}/{c['regime']}"] += 1
    print(f"Wrote {len(corpora)} corpora to {args.out}")
    for key in sorted(by_axis):
        print(f"  {key:34s} {by_axis[key]:3d}")
    sizes = [c["n_documents"] for c in corpora]
    print(f"  documents per corpus: min={min(sizes)} max={max(sizes)}")
    print(f"  total sentences: {sum(sizes)}")


if __name__ == "__main__":
    main()
