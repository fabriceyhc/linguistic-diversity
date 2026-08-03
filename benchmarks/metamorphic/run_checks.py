#!/usr/bin/env python
"""Metamorphic and consistency checks for every metric.

The other two benchmarks need ground truth: an authored expectation, or a
construction whose true diversity is known. Metamorphic relations need neither.
They state how a metric's output *must* change (or not change) when its input is
transformed, so they hold for any corpus and can be checked exhaustively.

That is what makes them worth running before a release. Every defect found in
this library so far -- constituency parses collapsing to a single node, tree
comparison ignoring part of speech, alignment similarity going negative -- would
have been caught by one of the relations below, without anyone authoring a
single expected value.

Relations are split by how strongly they bind:

  LAW        A mathematical property of similarity-sensitive Hill numbers. A
             violation is a bug, with no interpretation required.
  EXPECTED   Should hold for any sane diversity measure, but depends on the
             metric's definition of a species. Violations need a look.

Usage:
    python run_checks.py [--only METRIC ...] [--out output/results.json]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from linguistic_diversity import (
    ConstituencyParse,
    DependencyParse,
    DistinctN,
    DocumentSemantics,
    PartOfSpeechSequence,
    Phonemic,
    Rhythmic,
    SelfBLEU,
    TokenSemantics,
    TypeTokenRatio,
    clear_model_cache,
)

HERE = Path(__file__).parent
SEED_PATH = HERE.parent / "metric_validation" / "data" / "constructions.json"
DEFAULT_OUT = HERE / "output" / "results.json"

TOL = 1e-6

# Hill-number metrics: bounded below by 1 (one effective species) and above by
# the species count. The lexical baselines are ratios and are exempt from the
# bound and ordering laws, but not from invariance.
HILL_METRICS = {
    "DocumentSemantics", "TokenSemantics", "DependencyParse",
    "ConstituencyParse", "PartOfSpeechSequence", "Rhythmic", "Phonemic",
}
# Species are tokens, not documents, so document-count bounds do not apply.
TOKEN_UNIT = {"TokenSemantics"}
LOWER_IS_DIVERSE = {"SelfBLEU"}

BUILDERS: dict[str, Callable[..., Any]] = {
    "DocumentSemantics": DocumentSemantics,
    "TokenSemantics": TokenSemantics,
    "DependencyParse": DependencyParse,
    "ConstituencyParse": ConstituencyParse,
    "PartOfSpeechSequence": PartOfSpeechSequence,
    "Rhythmic": Rhythmic,
    "Phonemic": Phonemic,
    "TypeTokenRatio": TypeTokenRatio,
    "DistinctN": DistinctN,
    "SelfBLEU": SelfBLEU,
}


def load_corpora() -> dict[str, list[str]]:
    """A few structurally different corpora, so relations are not tested on one shape."""
    seed = json.loads(SEED_PATH.read_text())
    frames = seed["syntactic_frames"]["frames"]
    meters = seed["rhythmic_meters"]["meters"]
    alts = seed["syntactic_alternations"]["sets"]

    varied = [f["lexicalisations"][0] for f in frames] + [m["lines"][0] for m in meters]
    return {
        # Every document different in content and structure.
        "varied": varied[:8],
        # One syntactic frame, several lexicalisations: low structural diversity.
        "one_frame": list(frames[0]["lexicalisations"]),
        # One proposition, several structures: low semantic diversity.
        "one_meaning": list(alts[0]["realisations"]),
    }


# --------------------------------------------------------------------------
# Relations. Each returns (passed, detail).
# --------------------------------------------------------------------------

def r_determinism(metric: Any, corpus: list[str], **_: Any) -> tuple[bool, str]:
    """LAW. The same input must produce the same output."""
    a, b = float(metric(corpus)), float(metric(corpus))
    return abs(a - b) <= TOL, f"{a:.6f} vs {b:.6f}"


def r_permutation_invariance(metric: Any, corpus: list[str], rng: random.Random,
                             **_: Any) -> tuple[bool, str]:
    """LAW. A corpus is a multiset; document order carries no information."""
    base = float(metric(corpus))
    worst, detail = 0.0, ""
    for _ in range(3):
        shuffled = corpus[:]
        rng.shuffle(shuffled)
        got = float(metric(shuffled))
        if abs(got - base) > worst:
            worst, detail = abs(got - base), f"{base:.6f} vs {got:.6f}"
    return worst <= TOL, detail or f"{base:.6f} stable"


def r_replication_invariance(metric: Any, corpus: list[str], **_: Any) -> tuple[bool, str]:
    """LAW (Hill only). Duplicating a corpus leaves relative abundance untouched."""
    base = float(metric(corpus))
    got = float(metric(corpus * 3))
    return abs(got - base) <= 1e-4, f"{base:.6f} vs {got:.6f} at 3x"


def r_identical_corpus_is_one(metric: Any, _corpus: list[str], token_unit: bool = False,
                              **__: Any) -> tuple[bool, str]:
    """LAW (Hill, document-unit only). n copies of one document is one species.

    Skipped for token-unit metrics: repeating a document repeats its tokens, and
    the distinct token count is what they measure, so 1.0 is not the expectation.
    """
    if token_unit:
        return True, "not applicable: species are tokens"
    got = float(metric(["The committee reached a decision."] * 5))
    return abs(got - 1.0) <= 1e-3, f"{got:.6f}, expected 1.0"


def r_bounds(metric: Any, corpus: list[str], token_unit: bool = False,
             **_: Any) -> tuple[bool, str]:
    """LAW (Hill only). 1 <= D <= number of species."""
    got = float(metric(corpus))
    if got < 1.0 - 1e-6:
        return False, f"{got:.6f} below the floor of 1.0"
    if not token_unit and got > len(corpus) + 1e-6:
        return False, f"{got:.6f} above the species count {len(corpus)}"
    return True, f"{got:.6f} within [1, {len(corpus)}]"


def r_q_monotonicity(_metric: Any, corpus: list[str], builder: Callable[..., Any] | None = None,
                     **_kw: Any) -> tuple[bool, str]:
    """LAW (Hill only). Hill numbers are non-increasing in the order q."""
    if builder is None:
        return True, "skipped"
    values = []
    for q in (0.0, 1.0, 2.0):
        values.append(float(builder({"q": q, "verbose": False})(corpus)))
    ok = values[0] >= values[1] - 1e-4 >= values[2] - 2e-4
    return ok, f"q0={values[0]:.4f} q1={values[1]:.4f} q2={values[2]:.4f}"


# Removed: "appending a duplicate never increases diversity".
#
# It reads like a law and is not one. Adding a duplicate redistributes abundance:
# every other species receives a smaller share of the duplicated species'
# similarity mass, which lowers (Zp)_i and *raises* their contribution to D. That
# can outweigh the drop from the duplicated pair. Over 4000 random symmetric Z
# with unit diagonal at q=1, duplication increased D in 312 of them, by up to
# 0.098. Only q=0 (richness) is invariant to it.
#
# Kept as a comment rather than deleted because the intuition is compelling
# enough to be worth re-deriving: the corpus-wide version (replication_invariance)
# is the one that does hold.


def r_novel_never_decreases(metric: Any, corpus: list[str], **_: Any) -> tuple[bool, str]:
    """EXPECTED. Appending an unrelated document should not reduce diversity."""
    novel = "Quantum chromodynamics predicts asymptotic freedom at short distances."
    base = float(metric(corpus))
    got = float(metric(corpus + [novel]))
    if metric.__class__.__name__ in LOWER_IS_DIVERSE:
        base, got = -base, -got
    return got >= base - 1e-4, f"{base:.6f} -> {got:.6f} after adding an unrelated document"


def r_similarity_matrix_valid(metric: Any, corpus: list[str], **_: Any) -> tuple[bool, str]:
    """LAW. Z must be a similarity matrix: [0, 1], symmetric, unit diagonal."""
    if not hasattr(metric, "extract_features") or not hasattr(metric, "calculate_similarities"):
        return True, "no similarity matrix"
    features, _docs = metric.extract_features(corpus)
    Z = np.asarray(metric.calculate_similarities(features), dtype=float)
    problems = []
    if Z.min() < -1e-9:
        problems.append(f"min {Z.min():.4f} < 0")
    if Z.max() > 1 + 1e-9:
        problems.append(f"max {Z.max():.4f} > 1")
    if not np.allclose(Z, Z.T, atol=1e-6):
        problems.append("not symmetric")
    if not np.allclose(np.diag(Z), 1.0, atol=1e-6):
        problems.append(f"diagonal {np.diag(Z).min():.4f}..{np.diag(Z).max():.4f} != 1")
    return not problems, "; ".join(problems) or f"[{Z.min():.3f}, {Z.max():.3f}] symmetric"


RELATIONS: list[tuple[str, str, Callable[..., tuple[bool, str]], bool]] = [
    # (name, strength, fn, hill_only)
    ("determinism", "LAW", r_determinism, False),
    ("permutation_invariance", "LAW", r_permutation_invariance, False),
    ("similarity_matrix_valid", "LAW", r_similarity_matrix_valid, False),
    ("replication_invariance", "LAW", r_replication_invariance, True),
    ("identical_corpus_is_one", "LAW", r_identical_corpus_is_one, True),
    ("bounds", "LAW", r_bounds, True),
    ("q_monotonicity", "LAW", r_q_monotonicity, True),
    ("novel_never_decreases", "EXPECTED", r_novel_never_decreases, False),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="restrict to these metrics")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    corpora = load_corpora()
    names = args.only or list(BUILDERS)
    results: dict[str, Any] = {}

    for name in names:
        builder = BUILDERS[name]
        try:
            metric = builder({"verbose": False})
        except Exception as e:
            print(f"{name}: SKIPPED ({type(e).__name__}: {e})")
            continue
        print(f"{name} ...", end=" ", flush=True)
        rng = random.Random(20260803)
        per_relation: dict[str, Any] = {}

        for rel_name, strength, fn, hill_only in RELATIONS:
            if hill_only and name not in HILL_METRICS:
                continue
            outcomes = []
            for corpus_name, corpus in corpora.items():
                try:
                    ok, detail = fn(
                        metric, corpus, rng=rng, builder=builder,
                        token_unit=name in TOKEN_UNIT,
                    )
                except Exception as e:
                    ok, detail = False, f"raised {type(e).__name__}: {e}"
                outcomes.append({"corpus": corpus_name, "passed": bool(ok), "detail": detail})
            per_relation[rel_name] = {
                "strength": strength,
                "passed": all(o["passed"] for o in outcomes),
                "cases": outcomes,
            }
        results[name] = per_relation
        failed = [r for r, v in per_relation.items() if not v["passed"]]
        print("OK" if not failed else f"FAIL {failed}")
        clear_model_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    rel_names = [r[0] for r in RELATIONS]
    width = max(len(n) for n in results) if results else 10
    print(f"\n{'=' * 100}\nMETAMORPHIC RELATIONS  (. pass  X fail  - not applicable)\n{'-' * 100}")
    for i, rel in enumerate(rel_names):
        strength = next(r[1] for r in RELATIONS if r[0] == rel)
        print(f"  {i + 1}. {rel:26s} {strength}")
    print()
    print(f"  {'metric':{width}s}  " + " ".join(f"{i + 1:>2d}" for i in range(len(rel_names))))
    total_fail = 0
    for name, rels in results.items():
        cells = []
        for rel in rel_names:
            if rel not in rels:
                cells.append(" -")
            elif rels[rel]["passed"]:
                cells.append(" .")
            else:
                cells.append(" X")
                total_fail += 1
        print(f"  {name:{width}s}  " + " ".join(cells))

    if total_fail:
        print(f"\n{'-' * 100}\nFAILURES\n{'-' * 100}")
        for name, rels in results.items():
            for rel, v in rels.items():
                if v["passed"]:
                    continue
                for case in v["cases"]:
                    if not case["passed"]:
                        print(f"  {v['strength']:8s} {name}.{rel} [{case['corpus']}]: {case['detail']}")
    print(f"\n{total_fail} failing metric/relation pairs. Wrote {args.out}")


if __name__ == "__main__":
    main()
