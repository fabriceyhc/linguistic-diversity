#!/usr/bin/env python
"""Score every metric against the per-level validation benchmark.

Three readouts:

1. Calibration -- at a corpus's target level, does the metric report the
   expected effective diversity? Reported as Spearman rank agreement and as a
   median ratio (reported / expected).

2. Contrast satisfaction -- does the metric satisfy the paired inequalities the
   benchmark states for its level?

3. The inverse pair -- the single number that decides discriminant validity.
   Matched on document count, syntactic alternations hold meaning constant while
   varying structure; syntactic frames do the reverse. A semantic metric must
   rank frames above alternations (rate -> 1.0). A syntactic metric must rank
   them the other way (rate -> 0.0). A metric near 0.5 is not resolving either.

Usage:
    python evaluate_metrics.py [--out output/results.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import spearmanr

from linguistic_diversity import (
    DependencyParse,
    DistinctN,
    DocumentSemantics,
    PartOfSpeechSequence,
    Phonemic,
    Rhythmic,
    SelfBLEU,
    TypeTokenRatio,
    clear_model_cache,
)

HERE = Path(__file__).parent
DEFAULT_BENCHMARK = HERE / "output" / "benchmark.json"
DEFAULT_OUT = HERE / "output" / "results.json"

# metric key -> (constructor, the level it claims to measure)
# Lexical baselines claim no level; they are here because the point of the
# benchmark is that they respond to everything.
METRICS: dict[str, tuple[Any, str | None]] = {
    "DocumentSemantics": (lambda: DocumentSemantics({"verbose": False}), "semantic"),
    "DependencyParse": (lambda: DependencyParse({"verbose": False}), "syntactic"),
    "PartOfSpeechSequence": (lambda: PartOfSpeechSequence({"verbose": False}), "morphological"),
    "Rhythmic": (lambda: Rhythmic({"verbose": False}), "rhythmic"),
    "Phonemic": (lambda: Phonemic({"verbose": False}), "phonemic"),
    "TypeTokenRatio": (lambda: TypeTokenRatio(), None),
    "DistinctN": (lambda: DistinctN(), None),
    "SelfBLEU": (lambda: SelfBLEU(), None),
}

# SelfBLEU is a similarity, not a diversity: lower means more diverse.
LOWER_IS_DIVERSE = {"SelfBLEU"}


def score_corpora(benchmark: dict, only: list[str] | None) -> dict[str, dict[str, float]]:
    """Run every metric over every corpus. Returns metric -> corpus_id -> score."""
    corpora = benchmark["corpora"]
    scores: dict[str, dict[str, float]] = {}

    for name, (build, _level) in METRICS.items():
        if only and name not in only:
            continue
        print(f"  {name} ...", end=" ", flush=True)
        try:
            metric = build()
        except Exception as e:
            print(f"SKIPPED ({type(e).__name__}: {e})")
            continue

        per_corpus: dict[str, float] = {}
        failures = 0
        for corpus in corpora:
            try:
                value = float(metric(corpus["documents"]))
                if value != value:  # NaN
                    raise ValueError("NaN")
                per_corpus[corpus["id"]] = value
            except Exception:
                failures += 1
        scores[name] = per_corpus
        note = f" ({failures} failed)" if failures else ""
        print(f"{len(per_corpus)}/{len(corpora)} corpora{note}")
        clear_model_cache()

    return scores


def _oriented(name: str, value: float) -> float:
    """Return the value oriented so that larger always means more diverse."""
    return -value if name in LOWER_IS_DIVERSE else value


def calibration(benchmark: dict, scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Rank agreement and median ratio against expected values at the target level."""
    out: dict[str, Any] = {}
    for name, per_corpus in scores.items():
        level = METRICS[name][1]
        if level is None:
            continue
        observed, expected = [], []
        for corpus in benchmark["corpora"]:
            want = corpus["expected"].get(level)
            got = per_corpus.get(corpus["id"])
            if want is None or got is None:
                continue
            observed.append(_oriented(name, got))
            expected.append(want)
        if len(observed) < 3 or len(set(expected)) < 2:
            out[name] = {"level": level, "n": len(observed), "note": "insufficient variation"}
            continue
        rho, p = spearmanr(observed, expected)
        ratios = [o / e for o, e in zip(observed, expected) if e > 0 and name not in LOWER_IS_DIVERSE]
        out[name] = {
            "level": level,
            "n": len(observed),
            "spearman_vs_expected": round(float(rho), 4),
            "p_value": round(float(p), 6),
            "median_ratio": round(statistics.median(ratios), 4) if ratios else None,
        }
    return out


def contrast_accuracy(benchmark: dict, scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Fraction of paired inequalities each metric satisfies, by level."""
    out: dict[str, Any] = defaultdict(dict)
    for name, per_corpus in scores.items():
        tally: dict[str, list[int]] = defaultdict(list)
        for contrast in benchmark["contrasts"]:
            if contrast["level"] == "__within_corpus__":
                continue
            hi = per_corpus.get(contrast["greater"])
            lo = per_corpus.get(contrast["lesser"])
            if hi is None or lo is None:
                continue
            tally[f'{contrast["level"]}/{contrast.get("kind", "-")}'].append(
                int(_oriented(name, hi) > _oriented(name, lo))
            )
        for level, hits in tally.items():
            out[name][level] = {
                "accuracy": round(sum(hits) / len(hits), 4),
                "n": len(hits),
            }
    return dict(out)


def inverse_pair(benchmark: dict, scores: dict[str, dict[str, float]]) -> dict[str, Any]:
    """The discriminant headline: rate at which frames outrank alternations.

    Semantic metrics should approach 1.0, syntactic metrics 0.0. Anything near
    0.5 is not separating form from content on matched-size corpora.
    """
    pairs = [
        (c["greater"], c["lesser"])
        for c in benchmark["contrasts"]
        if c["level"] == "semantic" and c.get("kind") == "inverse_pair"
    ]
    out = {}
    for name, per_corpus in scores.items():
        hits, n = 0, 0
        for frame_id, alt_id in pairs:
            f, a = per_corpus.get(frame_id), per_corpus.get(alt_id)
            if f is None or a is None:
                continue
            n += 1
            hits += int(_oriented(name, f) > _oriented(name, a))
        if n:
            out[name] = {
                "frame_over_alternation_rate": round(hits / n, 4),
                "n_pairs": n,
                "claims_level": METRICS[name][1],
            }
    return out


def within_corpus_checks(benchmark: dict, scores: dict[str, dict[str, float]]) -> list[dict]:
    """Per-corpus level comparisons, e.g. syntax must exceed morphology."""
    results = []
    for contrast in benchmark["contrasts"]:
        if contrast["level"] != "__within_corpus__":
            continue
        hi_metric = next(
            (m for m, (_b, lv) in METRICS.items() if lv == contrast["greater_level"]), None
        )
        lo_metric = next(
            (m for m, (_b, lv) in METRICS.items() if lv == contrast["lesser_level"]), None
        )
        if hi_metric not in scores or lo_metric not in scores:
            continue
        hi = scores[hi_metric].get(contrast["corpus"])
        lo = scores[lo_metric].get(contrast["corpus"])
        if hi is None or lo is None:
            continue
        results.append({
            "corpus": contrast["corpus"],
            f"{contrast['greater_level']}": round(hi, 4),
            f"{contrast['lesser_level']}": round(lo, 4),
            "satisfied": bool(hi > lo),
            "rationale": contrast["rationale"],
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", help="Restrict to these metric names")
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text())
    print(f"Scoring {len(benchmark['corpora'])} corpora, "
          f"{len(benchmark['contrasts'])} contrasts\n")

    scores = score_corpora(benchmark, args.only)

    results = {
        "_meta": {
            "benchmark": args.benchmark.name,
            "n_corpora": len(benchmark["corpora"]),
            "n_contrasts": len(benchmark["contrasts"]),
        },
        "calibration": calibration(benchmark, scores),
        "contrast_accuracy": contrast_accuracy(benchmark, scores),
        "inverse_pair": inverse_pair(benchmark, scores),
        "within_corpus": within_corpus_checks(benchmark, scores),
        "raw_scores": scores,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print(f"\n{'=' * 74}")
    print("INVERSE PAIR -- frames ranked above alternations (matched size)")
    print("  semantic metrics -> 1.0 | syntactic metrics -> 0.0 | uninformative -> 0.5")
    print("-" * 74)
    print(f"  {'metric':24s} {'claims':14s} {'rate':>8s} {'n':>5s}")
    for name, r in sorted(
        results["inverse_pair"].items(), key=lambda kv: -kv[1]["frame_over_alternation_rate"]
    ):
        print(f"  {name:24s} {str(r['claims_level'] or '-'):14s} "
              f"{r['frame_over_alternation_rate']:8.3f} {r['n_pairs']:5d}")

    print(f"\n{'=' * 74}")
    print("CALIBRATION at each metric's own level")
    print("-" * 74)
    print(f"  {'metric':24s} {'level':14s} {'rho':>8s} {'ratio':>8s} {'n':>5s}")
    for name, r in results["calibration"].items():
        if "spearman_vs_expected" not in r:
            print(f"  {name:24s} {r['level']:14s} {r.get('note', '')}")
            continue
        ratio = r["median_ratio"]
        print(f"  {name:24s} {r['level']:14s} {r['spearman_vs_expected']:8.3f} "
              f"{ratio if ratio is not None else float('nan'):8.3f} {r['n']:5d}")

    print(f"\n{'=' * 74}")
    print("CONTRAST ACCURACY by level")
    print("-" * 74)
    levels = sorted({lv for m in results["contrast_accuracy"].values() for lv in m})
    print(f"  {'metric':24s} " + " ".join(f"{lv[:9]:>10s}" for lv in levels))
    for name in results["contrast_accuracy"]:
        cells = []
        for lv in levels:
            entry = results["contrast_accuracy"][name].get(lv)
            cells.append(f"{entry['accuracy']:10.3f}" if entry else f"{'-':>10s}")
        print(f"  {name:24s} " + " ".join(cells))

    if results["within_corpus"]:
        print(f"\n{'=' * 74}")
        print("WITHIN-CORPUS: syntax must exceed morphology on POS-identical sets")
        print("-" * 74)
        for r in results["within_corpus"]:
            mark = "PASS" if r["satisfied"] else "FAIL"
            print(f"  {mark}  {r['corpus']:34s} "
                  f"syn={r.get('syntactic', float('nan')):7.3f}  "
                  f"morph={r.get('morphological', float('nan')):7.3f}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
