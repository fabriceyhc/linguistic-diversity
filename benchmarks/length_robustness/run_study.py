#!/usr/bin/env python
"""Does a diversity score change when only length or corpus size changes?

Shaib et al. (Standardizing the Measurement of Text Diversity, 2024) name text
length as the field's unresolved confound: "Future research into a principled
solution for this problem is urgently needed." Ecology solved the same problem
for species counts with rarefaction and extrapolation (Chao et al. 2014), which
this library ships as ``estimate_diversity()``.

Two sweeps, both holding true diversity fixed by construction:

  replication   k distinct documents repeated j times. Exact duplicates are the
                same species, so true diversity stays k while the corpus grows
                to k*j documents. Tests sample-size sensitivity.

  padding       k distinct documents, each extended with the *same* boilerplate
                clause repeated t times. The number of distinct propositions
                stays k while mean document length grows. Note this manipulates
                length *and* shared content together: for a similarity-sensitive
                metric some decline is defensible, since the documents really do
                share more material. Read it against replication, where no such
                defence exists.

A length-robust metric returns the same number across every level of a sweep.
Reported per metric as drift (max-min over the sweep, relative to the baseline
score) and as Spearman correlation against the swept quantity. Both should be 0.

Usage:
    python run_study.py [--quick] [--out output/results.json]
"""
from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from statistics import median
from typing import Any, Callable

from scipy.stats import spearmanr

from linguistic_diversity import (
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

REPLICATION_LEVELS = (1, 2, 3, 4, 6)
PADDING_LEVELS = (0, 1, 2, 4, 8)
BASE_SIZE = 6  # distinct documents per base corpus
N_BASE_CORPORA = 5

# A neutral clause carrying no distinguishing content. Appended identically to
# every document, so it adds length without adding species at any level.
BOILERPLATE = "This was noted at the time."


def compression_ratio(corpus: list[str]) -> float:
    """Shaib et al.'s headline recommendation, as a baseline.

    Ratio of raw to gzip-compressed size. Repetitive text compresses well, so a
    *higher* ratio means *less* diverse -- it is similarity-oriented and is
    negated below alongside Self-BLEU. Included because it is the score they
    advise reporting, and because it is a pure surface statistic.
    """
    raw = "\n".join(corpus).encode("utf-8")
    if not raw:
        return 0.0
    return len(raw) / len(zlib.compress(raw, 9))


METRICS: dict[str, Callable[[], Any]] = {
    "DocumentSemantics": lambda: DocumentSemantics({"verbose": False}),
    "TokenSemantics": lambda: TokenSemantics({"verbose": False}),
    "DependencyParse": lambda: DependencyParse({"verbose": False}),
    "PartOfSpeechSequence": lambda: PartOfSpeechSequence({"verbose": False}),
    "Rhythmic": lambda: Rhythmic({"verbose": False}),
    "Phonemic": lambda: Phonemic({"verbose": False}),
    "TypeTokenRatio": lambda: TypeTokenRatio(),
    "DistinctN": lambda: DistinctN(),
    "SelfBLEU": lambda: SelfBLEU(),
    "CompressionRatio": lambda: compression_ratio,
}

LOWER_IS_DIVERSE = {"SelfBLEU", "CompressionRatio"}


def load_base_corpora(n: int, size: int) -> list[list[str]]:
    """Distinct sentences drawn from the validation seed pool, no repeats."""
    seed = json.loads(SEED_PATH.read_text())
    pool: list[str] = []
    for key, sub, text in (
        ("syntactic_frames", "frames", "lexicalisations"),
        ("rhythmic_meters", "meters", "lines"),
        ("morphological_templates", "sets", "realisations"),
    ):
        for group in seed[key][sub]:
            pool.extend(group[text])
    seen: set[str] = set()
    unique = [s for s in pool if not (s in seen or seen.add(s))]
    return [unique[i * size : (i + 1) * size] for i in range(n) if (i + 1) * size <= len(unique)]


def sweep_replication(base: list[str]) -> list[tuple[int, list[str]]]:
    """k documents repeated j times: corpus grows, species count does not."""
    return [(j, base * j) for j in REPLICATION_LEVELS]


def sweep_padding(base: list[str]) -> list[tuple[int, list[str]]]:
    """k documents each extended by shared boilerplate: length grows, species do not."""
    out = []
    for t in PADDING_LEVELS:
        suffix = (" " + BOILERPLATE) * t
        out.append((t, [doc + suffix for doc in base]))
    return out


def mean_tokens(corpus: list[str]) -> float:
    return sum(len(d.split()) for d in corpus) / len(corpus)


def run_sweep(
    name: str,
    sweep: Callable[[list[str]], list[tuple[int, list[str]]]],
    bases: list[list[str]],
    metric_names: list[str],
) -> dict[str, Any]:
    """Score every metric at every level of one sweep, over every base corpus."""
    print(f"\n{'=' * 74}\n{name.upper()} SWEEP\n{'-' * 74}")
    # metric -> list of (level, score) across all bases
    observations: dict[str, list[tuple[float, float]]] = {m: [] for m in metric_names}
    sizes: dict[int, dict[str, float]] = {}

    for mi, mname in enumerate(metric_names, 1):
        print(f"  [{mi}/{len(metric_names)}] {mname} ...", end=" ", flush=True)
        try:
            metric = METRICS[mname]()
        except Exception as e:
            print(f"SKIPPED ({type(e).__name__})")
            continue
        failures = 0
        for base in bases:
            for level, corpus in sweep(base):
                sizes.setdefault(level, {
                    "n_documents": len(corpus),
                    "mean_tokens": round(mean_tokens(corpus), 1),
                })
                try:
                    value = float(metric(corpus))
                    if value != value:
                        raise ValueError("NaN")
                    observations[mname].append((float(level), value))
                except Exception:
                    failures += 1
        print(f"{len(observations[mname])} points" + (f" ({failures} failed)" if failures else ""))
        clear_model_cache()

    return {"levels": sizes, "observations": observations}


def summarise(sweep_result: dict[str, Any]) -> dict[str, Any]:
    """Drift and rank correlation against the swept quantity, per metric."""
    out: dict[str, Any] = {}
    for mname, points in sweep_result["observations"].items():
        if len(points) < 4:
            continue
        levels = [p[0] for p in points]
        scores = [-p[1] if mname in LOWER_IS_DIVERSE else p[1] for p in points]

        # Drift is measured per base corpus, then pooled: within one base the
        # true diversity is constant, so any spread is the metric moving.
        by_level: dict[float, list[float]] = {}
        for lv, sc in zip(levels, scores):
            by_level.setdefault(lv, []).append(sc)
        baseline = median(by_level[min(by_level)])
        level_medians = {lv: median(v) for lv, v in by_level.items()}
        spread = max(level_medians.values()) - min(level_medians.values())
        drift = abs(spread / baseline) if baseline else float("inf")

        rho, p = spearmanr(levels, scores)
        out[mname] = {
            "drift": round(drift, 4),
            "spearman_vs_level": round(float(rho), 4),
            "p_value": round(float(p), 6),
            "baseline": round(baseline, 4),
            "by_level": {str(k): round(v, 4) for k, v in sorted(level_medians.items())},
            "n": len(points),
        }
    return out


def print_table(title: str, summary: dict[str, Any], levels: dict[int, dict[str, float]]) -> None:
    print(f"\n{'=' * 74}\n{title}\n  drift = (max-min)/baseline across the sweep; 0 is perfect")
    print("-" * 74)
    ordered = sorted(levels)
    header = " ".join(f"{lv:>8}" for lv in ordered)
    print(f"  {'metric':22s} {'drift':>7s} {'rho':>7s}  {header}")
    for mname, r in sorted(summary.items(), key=lambda kv: kv[1]["drift"]):
        cells = " ".join(f"{r['by_level'].get(str(float(lv)), float('nan')):8.3f}" for lv in ordered)
        print(f"  {mname:22s} {r['drift']:7.3f} {r['spearman_vs_level']:7.3f}  {cells}")


def run_extrapolation(metric_names: list[str]) -> dict[str, Any]:
    """Can a budgeted subsample recover the full corpus's diversity?

    The other two sweeps ask whether a score holds still when the corpus changes.
    This asks the question that matters at scale: exact diversity needs an O(n^2)
    similarity matrix, so given a measurement budget of m documents, does the
    fitted growth curve land closer to the true full-corpus value than simply
    scoring m documents would?

    That is what rarefaction and extrapolation are for. Scoring a subsample
    under-reports by construction, because the species it never saw are missing.

    Note estimate_diversity extrapolates to len(corpus) and measures directly
    when corpus_size <= max_sample_size, so the *full* corpus is passed with a
    budget below its size -- passing a pre-drawn sample would silently return a
    direct measurement of that sample.
    """
    print(f"\n{'=' * 74}\nEXTRAPOLATION FROM A MEASUREMENT BUDGET\n{'-' * 74}")
    seed = json.loads(SEED_PATH.read_text())
    pool: list[str] = []
    for key, sub, text in (
        ("syntactic_frames", "frames", "lexicalisations"),
        ("rhythmic_meters", "meters", "lines"),
        ("morphological_templates", "sets", "realisations"),
        ("syntactic_alternations", "sets", "realisations"),
    ):
        for group in seed[key][sub]:
            pool.extend(group[text])
    seen: set[str] = set()
    full = [s for s in pool if not (s in seen or seen.add(s))]

    import random
    rng = random.Random(20260803)
    budgets = [m for m in (10, 20, 30, 40) if m < len(full)]
    print(f"  full corpus: {len(full)} documents; budgets {budgets}\n")
    out: dict[str, Any] = {}

    for mname in metric_names:
        if mname not in METRICS or mname == "CompressionRatio":
            continue
        try:
            metric = METRICS[mname]()
        except Exception:
            continue
        if not hasattr(metric, "estimate_diversity"):
            continue
        try:
            truth = float(metric(full))
        except Exception:
            continue
        saturated = abs(truth - len(full)) < 0.5

        rows = []
        for m in budgets:
            try:
                raw = float(metric(rng.sample(full, m)))
            except Exception:
                continue
            try:
                est = metric.estimate_diversity(
                    full, base_sample_size=max(5, m // 4), max_sample_size=m,
                    num_trials=2, verbose=False,
                )
                extrap, model = float(est.diversity), est.model
            except Exception as e:
                extrap, model = float("nan"), type(e).__name__
            ok = extrap == extrap
            rows.append({
                "budget": m,
                "raw_subsample": round(raw, 3),
                "extrapolated": round(extrap, 3) if ok else None,
                "model": model,
                "raw_error": round(abs(raw - truth) / truth, 3),
                "extrap_error": round(abs(extrap - truth) / truth, 3) if ok else None,
            })
        out[mname] = {"truth": round(truth, 3), "n_full": len(full),
                      "saturated": saturated, "budgets": rows}
        clear_model_cache()

    print(f"  {'metric':22s} {'truth':>7s} {'bud':>4s} {'raw':>8s} {'extrap':>8s} "
          f"{'rawerr':>7s} {'exterr':>7s}  model")
    for mname, r in out.items():
        flag = "  [SATURATED: reports the species count]" if r["saturated"] else ""
        for row in r["budgets"]:
            ex = f"{row['extrapolated']:8.3f}" if row["extrapolated"] is not None else f"{'-':>8s}"
            ee = f"{row['extrap_error']:7.3f}" if row["extrap_error"] is not None else f"{'-':>7s}"
            print(f"  {mname:22s} {r['truth']:7.3f} {row['budget']:4d} "
                  f"{row['raw_subsample']:8.3f} {ex} {row['raw_error']:7.3f} {ee}  {row['model']}")
        if flag:
            print(f"  {'':22s}{flag}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--quick", action="store_true", help="two base corpora, fewer metrics")
    parser.add_argument("--only", nargs="*", help="restrict to these metrics")
    args = parser.parse_args()

    n_bases = 2 if args.quick else N_BASE_CORPORA
    bases = load_base_corpora(n_bases, BASE_SIZE)
    metric_names = args.only or list(METRICS)
    print(f"{len(bases)} base corpora of {BASE_SIZE} distinct documents, "
          f"{len(metric_names)} metrics")

    results: dict[str, Any] = {"_meta": {
        "base_corpora": len(bases),
        "base_size": BASE_SIZE,
        "replication_levels": list(REPLICATION_LEVELS),
        "padding_levels": list(PADDING_LEVELS),
        "boilerplate": BOILERPLATE,
    }}

    for name, sweep in (("replication", sweep_replication), ("padding", sweep_padding)):
        raw = run_sweep(name, sweep, bases, metric_names)
        summary = summarise(raw)
        results[name] = {"levels": {str(k): v for k, v in raw["levels"].items()},
                         "summary": summary}
        label = ("REPLICATION: corpus size grows, true diversity fixed"
                 if name == "replication"
                 else "PADDING: document length grows, true diversity fixed")
        print_table(label, summary, raw["levels"])

    results["extrapolation"] = run_extrapolation(metric_names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
