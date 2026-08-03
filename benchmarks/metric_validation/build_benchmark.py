#!/usr/bin/env python
"""Build the per-level metric validation benchmark from seed constructions.

The embedder-selection benchmark answers "which encoder should back document
semantics". This one answers a different question: does each metric respond to
its own linguistic level and stay flat on the others?

Every corpus therefore carries an expected value at *every* level, not only at
the level it targets. A corpus of syntactic alternations expects ~1 from a
semantic metric and ~n from a syntactic one; a corpus of one syntactic frame
lexicalised n ways expects the reverse. A metric that moves on both has failed
discriminant validity regardless of how well it tracks its own axis.

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
DEFAULT_SEED_PATH = HERE / "data" / "constructions.json"
DEFAULT_OUT_PATH = HERE / "output" / "benchmark.json"

LEVELS = ("semantic", "syntactic", "morphological", "rhythmic", "phonemic")

# Frames/meters per corpus, and realisations drawn from each. Ground-truth
# diversity at the target level is k regardless of m; that gap is the test.
K_VALUES = (1, 2, 3, 4)
M_VALUES = (2, 3)
DRAWS_PER_CELL = 3
# Cap on how many matched-size counterparts each corpus is paired against, so the
# contrast list stays interpretable rather than combinatorial.
MAX_PAIRS_PER_CORPUS = 3


def _record(
    corpus_id: str,
    family: str,
    target_level: str,
    documents: list[str],
    expected: dict[str, float | None],
    rationale: str,
    rng: random.Random,
) -> dict[str, Any]:
    """Assemble one corpus record, shuffled, with expectations at every level."""
    docs = list(documents)
    rng.shuffle(docs)
    return {
        "id": corpus_id,
        "family": family,
        "target_level": target_level,
        "documents": docs,
        "n_documents": len(docs),
        "expected": {level: expected.get(level) for level in LEVELS},
        "rationale": rationale,
    }


def build_syntactic_alternations(seed: dict, rng: random.Random) -> list[dict]:
    """One proposition, n structures. Semantics flat, syntax varies.

    Emitted at several sizes by subsampling, so that every corpus has a
    frame-family counterpart with the same document count. Document count is a
    confound for most diversity metrics, so the discriminant contrasts are only
    interpretable when it is matched.
    """
    corpora = []
    for s in seed["syntactic_alternations"]["sets"]:
        available = len(s["realisations"])
        for n in range(2, available + 1):
            docs = rng.sample(s["realisations"], n)
            corpora.append(_record(
                corpus_id=f"alt-{s['id']}-n{n}",
                family="syntactic_alternations",
                target_level="syntactic",
                documents=docs,
                # Meaning is held constant, so a semantic metric should collapse to ~1.
                # POS sequences differ across alternations, so morphology moves too --
                # this family separates syntax from semantics, not syntax from morphology.
                expected={"semantic": 1.0, "syntactic": float(n), "morphological": float(n)},
                rationale=f"{n} syntactic realisations of one proposition: {s['proposition']}",
                rng=rng,
            ))
    return corpora


def build_syntactic_frames(seed: dict, rng: random.Random) -> list[dict]:
    """k frames x m lexicalisations. Syntax flat within a frame, semantics varies."""
    frames = seed["syntactic_frames"]["frames"]
    corpora = []
    for k in K_VALUES:
        if k > len(frames):
            continue
        for m in M_VALUES:
            for draw in range(DRAWS_PER_CELL):
                picked = rng.sample(frames, k)
                docs = []
                for frame in picked:
                    docs.extend(rng.sample(frame["lexicalisations"], m))
                corpora.append(_record(
                    corpus_id=f"frame-k{k}-m{m}-{draw}",
                    family="syntactic_frames",
                    target_level="syntactic",
                    documents=docs,
                    # Structure and POS sequence are shared within a frame, so both
                    # syntax and morphology should report k. Every sentence means
                    # something different, so semantics should report k*m.
                    expected={
                        "semantic": float(k * m),
                        "syntactic": float(k),
                        "morphological": float(k),
                    },
                    rationale=f"{k} syntactic frames x {m} lexicalisations each",
                    rng=rng,
                ))
    return corpora


def build_morphological_templates(seed: dict, rng: random.Random) -> list[dict]:
    """One content, n POS realisations (nominalisation, passive, progressive)."""
    corpora = []
    for s in seed["morphological_templates"]["sets"]:
        n = len(s["realisations"])
        corpora.append(_record(
            corpus_id=f"morph-{s['id']}",
            family="morphological_templates",
            target_level="morphological",
            documents=s["realisations"],
            expected={"semantic": 1.0, "morphological": float(n), "syntactic": float(n)},
            rationale=f"{n} morphological realisations of: {s['gloss']}",
            rng=rng,
        ))
    return corpora


def build_pos_identical(seed: dict, rng: random.Random) -> list[dict]:
    """Same POS sequence, different dependency structure.

    The sharpest test that morphology and syntax are not the same measurement:
    a morphological metric must collapse these, a syntactic metric must not.
    """
    corpora = []
    for s in seed["pos_identical_structure_different"]["sets"]:
        n = len(s["realisations"])
        corpora.append(_record(
            corpus_id=f"posid-{s['id']}",
            family="pos_identical_structure_different",
            target_level="syntactic",
            documents=s["realisations"],
            expected={"semantic": float(n), "syntactic": float(n), "morphological": 1.0},
            rationale=f"identical POS sequence ({s['pattern']}), differing structure: {s['note']}",
            rng=rng,
        ))
    return corpora


def build_surface_parse_blind(seed: dict, rng: random.Random) -> list[dict]:
    """Distinctions a surface dependency parse provably cannot represent.

    spaCy assigns these pairs identical heads and dependency labels, so the
    expectation is collapse, not separation. Recorded as a known limitation so
    the boundary of the syntactic metric is documented rather than rediscovered.
    """
    corpora = []
    for s in seed["surface_parse_blind"]["sets"]:
        n = len(s["realisations"])
        corpora.append(_record(
            corpus_id=f"blind-{s['id']}",
            family="surface_parse_blind",
            target_level="known_limitation",
            documents=s["realisations"],
            expected={"semantic": float(n), "syntactic": 1.0, "morphological": 1.0},
            rationale=f"invisible to a surface parse: {s['note']}",
            rng=rng,
        ))
    return corpora


def build_rhythmic_meters(seed: dict, rng: random.Random) -> list[dict]:
    """k meters x m lines. Stress contour flat within a meter, content varies."""
    meters = seed["rhythmic_meters"]["meters"]
    corpora = []
    for k in K_VALUES:
        if k > len(meters):
            continue
        for m in M_VALUES:
            for draw in range(DRAWS_PER_CELL):
                picked = rng.sample(meters, k)
                docs = []
                for meter in picked:
                    docs.extend(rng.sample(meter["lines"], m))
                corpora.append(_record(
                    corpus_id=f"meter-k{k}-m{m}-{draw}",
                    family="rhythmic_meters",
                    target_level="rhythmic",
                    documents=docs,
                    expected={"rhythmic": float(k), "semantic": float(k * m)},
                    rationale=f"{k} metrical patterns x {m} lines each",
                    rng=rng,
                ))
    return corpora


def build_phonemic_oronyms(seed: dict, rng: random.Random) -> list[dict]:
    """Near-homophonic, semantically distinct. Phonemics flat, semantics varies."""
    corpora = []
    for s in seed["phonemic_oronyms"]["sets"]:
        n = len(s["realisations"])
        corpora.append(_record(
            corpus_id=f"oronym-{s['id']}",
            family="phonemic_oronyms",
            target_level="phonemic",
            documents=s["realisations"],
            expected={"phonemic": 1.0, "semantic": float(n)},
            rationale=f"{n} near-homophonic sentences with unrelated meanings",
            rng=rng,
        ))
    return corpora


def build_phonemic_pairs(seed: dict, rng: random.Random) -> list[dict]:
    """Minimal pairs vs phonemically distant controls, matched on n and frame.

    Absolute ground truth is n for both, so these are scored as a paired
    inequality rather than a calibration target.
    """
    corpora = []
    for s in seed["phonemic_minimal_pairs"]["sets"]:
        n = len(s["realisations"])
        corpora.append(_record(
            corpus_id=f"phon-{s['id']}",
            family="phonemic_minimal_pairs",
            target_level="phonemic",
            documents=s["realisations"],
            expected={"phonemic": float(n)},
            rationale=s.get("note", f"{n} sentences differing in one phoneme"),
            rng=rng,
        ))
    return corpora


def build_contrasts(corpora: list[dict]) -> list[dict]:
    """Paired inequalities that a valid metric must satisfy.

    Absolute calibration and rank agreement can disagree (the embedder-selection
    benchmark found exactly that), so the discriminant claims are stated as
    inequalities between matched corpora rather than as target numbers.
    """
    by_id = {c["id"]: c for c in corpora}
    contrasts = []

    def by_size(family: str) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = defaultdict(list)
        for c in corpora:
            if c["family"] == family:
                out[c["n_documents"]].append(c)
        return out

    alts_by_size = by_size("syntactic_alternations")
    frames_by_size = by_size("syntactic_frames")
    morphs_by_size = by_size("morphological_templates")

    # The inverse pair, and the core of the whole benchmark. At a matched document
    # count, semantics must rank frames above alternations (distinct propositions vs
    # one proposition restated) while syntax must rank them the other way (n
    # structures vs n lexicalisations of few structures). A metric that orders both
    # pairs the same way is not measuring the level it claims to.
    for size, alts in sorted(alts_by_size.items()):
        for alt in alts:
            for frame in frames_by_size.get(size, [])[:MAX_PAIRS_PER_CORPUS]:
                contrasts.append({
                    "level": "semantic",
                    "kind": "inverse_pair",
                    "greater": frame["id"],
                    "lesser": alt["id"],
                    "rationale": "distinct propositions vs one proposition restated",
                })
                contrasts.append({
                    "level": "syntactic",
                    "kind": "inverse_pair",
                    "greater": alt["id"],
                    "lesser": frame["id"],
                    "rationale": "n structures vs n lexicalisations of few structures",
                })

    # Morphology must rank n distinct POS realisations above n lexicalisations
    # sharing few POS templates, at a matched document count.
    for size, morphs in sorted(morphs_by_size.items()):
        for morph in morphs:
            for frame in frames_by_size.get(size, [])[:MAX_PAIRS_PER_CORPUS]:
                if (frame["expected"]["morphological"] or 0) < (morph["expected"]["morphological"] or 0):
                    contrasts.append({
                        "level": "morphological",
                        "kind": "inverse_pair",
                        "greater": morph["id"],
                        "lesser": frame["id"],
                        "rationale": "n distinct POS sequences vs n lexicalisations of few POS templates",
                    })

    # Phonemics must rank phonemically distant sentences above minimal pairs,
    # holding the frame and document count fixed.
    control = by_id.get("phon-distant_controls")
    if control:
        for cid in ("phon-at_rhymes", "phon-ight_rhymes"):
            if cid in by_id and by_id[cid]["n_documents"] == control["n_documents"]:
                contrasts.append({
                    "level": "phonemic",
                    "kind": "inverse_pair",
                    "greater": control["id"],
                    "lesser": cid,
                    "rationale": "phonemically distant heads vs single-phoneme minimal pairs",
                })

    # Within a POS-identical corpus, syntax must exceed morphology.
    for c in corpora:
        if c["family"] == "pos_identical_structure_different":
            contrasts.append({
                "level": "__within_corpus__",
                "corpus": c["id"],
                "greater_level": "syntactic",
                "lesser_level": "morphological",
                "rationale": "identical POS sequence, differing dependency structure",
            })

    # Every controlled family must fall below the random ceiling at matched size.
    # This is the check that a metric is measuring diversity rather than counting
    # documents: if an alternation set scores as high as n unrelated sentences,
    # the metric is responding to corpus size alone.
    randoms_by_size = by_size("random_controls")
    for family in ("syntactic_alternations", "syntactic_frames", "morphological_templates"):
        for size, members in sorted(by_size(family).items()):
            for member in members[:MAX_PAIRS_PER_CORPUS]:
                # Only meaningful where the family actually compresses at this level.
                # syntactic_frames is already semantically all-distinct, so there is no
                # headroom between it and a random draw of the same size -- asserting an
                # inequality there would penalise a correct metric.
                if (member["expected"]["semantic"] or 0) >= member["n_documents"]:
                    continue
                for ctrl in randoms_by_size.get(size, [])[:1]:
                    contrasts.append({
                        "level": "semantic",
                        "kind": "ceiling",
                        "greater": ctrl["id"],
                        "lesser": member["id"],
                        "rationale": f"random ceiling vs semantically compressed {family} at n={size}",
                    })

    return contrasts


def build_random_controls(seed: dict, rng: random.Random) -> list[dict]:
    """Unrelated sentences pooled from every family: the diversity ceiling.

    Zhang, Peng & Bollegala (ACL 2025) report that form-based metrics assign high
    diversity even to randomly assembled sentence sets. That finding is a matter
    of construction rather than annotation, so it can be replicated here without
    their data. These corpora carry no upper anchor of their own -- they are the
    reference every other family is read against. A metric that scores an
    alternation set near its random-control score at matched size is not
    measuring diversity, it is counting sentences.
    """
    pool: list[str] = []
    for family_key, subkey, textkey in (
        ("syntactic_frames", "frames", "lexicalisations"),
        ("rhythmic_meters", "meters", "lines"),
        ("morphological_templates", "sets", "realisations"),
    ):
        for group in seed[family_key][subkey]:
            pool.extend(group[textkey])

    corpora = []
    for n in sorted({k * m for k in K_VALUES for m in M_VALUES}):
        if n > len(pool):
            continue
        for draw in range(DRAWS_PER_CELL):
            docs = rng.sample(pool, n)
            corpora.append(_record(
                corpus_id=f"random-n{n}-{draw}",
                family="random_controls",
                target_level="ceiling",
                documents=docs,
                # Unrelated sentences: every level should be near its maximum, so
                # expectations are n across the board.
                expected={level: float(n) for level in LEVELS},
                rationale=f"{n} unrelated sentences drawn at random: diversity ceiling",
                rng=rng,
            ))
    return corpora


BUILDERS = (
    build_syntactic_alternations,
    build_syntactic_frames,
    build_morphological_templates,
    build_pos_identical,
    build_surface_parse_blind,
    build_rhythmic_meters,
    build_phonemic_oronyms,
    build_phonemic_pairs,
    build_random_controls,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-data", type=Path, default=DEFAULT_SEED_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--random-seed", type=int, default=20260802)
    args = parser.parse_args()

    seed = json.loads(args.seed_data.read_text())
    rng = random.Random(args.random_seed)

    corpora: list[dict] = []
    for builder in BUILDERS:
        corpora.extend(builder(seed, rng))

    contrasts = build_contrasts(corpora)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "_meta": {
            "random_seed": args.random_seed,
            "n_corpora": len(corpora),
            "n_contrasts": len(contrasts),
            "levels": list(LEVELS),
            "source": args.seed_data.name,
            "scoring": (
                "Two scores per metric. Calibration: does it report the expected "
                "value at its own level? Discriminance: does it stay flat at levels "
                "the corpus holds constant, and satisfy the paired contrasts?"
            ),
        },
        "corpora": corpora,
        "contrasts": contrasts,
    }, indent=2))

    by_family: dict[str, int] = defaultdict(int)
    for c in corpora:
        by_family[c["family"]] += 1
    print(f"Wrote {len(corpora)} corpora and {len(contrasts)} contrasts to {args.out}")
    for key in sorted(by_family):
        print(f"  {key:38s} {by_family[key]:3d}")
    sizes = [c["n_documents"] for c in corpora]
    print(f"  documents per corpus: min={min(sizes)} max={max(sizes)}")
    print(f"  total sentences: {sum(sizes)}")


if __name__ == "__main__":
    main()
