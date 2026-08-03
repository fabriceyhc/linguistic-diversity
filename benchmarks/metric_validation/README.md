# Per-Level Metric Validation

Does each metric respond to the linguistic level it claims to measure, and stay flat on
the others?

[`../embedder_selection/`](../embedder_selection/) answers a narrower question — which
sentence encoder should back `DocumentSemantics`. It validates one metric. This benchmark
validates all of them, and it is built around the property that a scalar diversity score
cannot express: **discriminant validity**.

```bash
python build_benchmark.py      # 116 corpora, 138 contrasts, deterministic
python evaluate_metrics.py     # scores all 10 metrics, writes output/results.json
```

## The design

Every corpus carries an expected value at **every** level, not only at the level it
targets. That is what turns a calibration check into a discriminant test.

| family | construction | semantic | syntactic | morphological | tests |
|---|---|---:|---:|---:|---|
| `syntactic_alternations` | one proposition, *n* structures (passive, cleft, topicalisation) | **1** | *n* | *n* | does syntax move when meaning does not? |
| `syntactic_frames` | *k* frames × *m* lexicalisations | *k·m* | **k** | *k* | does syntax stay flat when meaning moves? |
| `morphological_templates` | one content, *n* POS realisations (nominalisation, passive, progressive) | **1** | *n* | *n* | — |
| `pos_identical_structure_different` | same POS sequence, different grammatical relations | *n* | *n* | **1** | separates morphology from syntax |
| `rhythmic_meters` | *k* meters × *m* lines | *k·m* | — | — | does rhythm collapse metrically identical lines? |
| `phonemic_oronyms` | near-homophonic, semantically unrelated | *n* | — | — | does phonemics collapse homophones? |
| `phonemic_minimal_pairs` | single-phoneme differences vs distant controls | — | — | — | graded phonemic separation |
| `surface_parse_blind` | distinctions a surface parse cannot encode | *n* | **1** | **1** | records the metric's boundary |
| `random_controls` | unrelated sentences, matched size | *n* | *n* | *n* | the ceiling every family is read against |

The first two rows are the benchmark. They are **inverses of each other**: alternations
hold meaning constant and vary structure, frames do the reverse, and both are matched on
document count. A semantic metric must rank frames above alternations; a syntactic metric
must rank them the other way. A metric that orders both pairs the same way is not
measuring the level it claims to, however well it tracks its own axis.

`random_controls` follows Zhang, Peng & Bollegala ([ACL
2025](https://aclanthology.org/2025.acl-long.1181/)), who found that form-based metrics
assign high diversity even to randomly assembled sentence sets. That finding is a matter
of construction rather than annotation, so it is replicated here without their data.

## Results

### The inverse pair

Rate at which each metric ranks frames above alternations, matched on size. Semantic
metrics → 1.0, structural metrics → 0.0, uninformative → 0.5.

| Metric | claims | rate | n |
|---|---|---:|---:|
| `DocumentSemantics` | semantic | **1.000** | 54 |
| `TypeTokenRatio` | *(baseline)* | 1.000 | 54 |
| `DistinctN` | *(baseline)* | 1.000 | 54 |
| `SelfBLEU` | *(baseline)* | 0.944 | 54 |
| `TokenSemantics` | semantic | 0.833 | 54 |
| `Phonemic` | phonemic | 0.389 | 54 |
| `DependencyParse` | syntactic | **0.000** | 54 |
| `ConstituencyParse` | syntactic | **0.000** | 54 |
| `PartOfSpeechSequence` | morphological | **0.000** | 54 |
| `Rhythmic` | rhythmic | 0.315 | 54 |

The syntactic and morphological metrics sit at the floor and the semantic metric at the
ceiling — a total separation. The two phonological metrics land at 0.32–0.39, nearer the
uninformative midpoint: they lean structural but do not resolve form from content the way
the parse-based metrics do.

The alignment-similarity fix in v1.0.3 moved these numbers. Before it,
`PartOfSpeechSequence`, `Rhythmic` and `Phonemic` all produced *negative* similarities and
saturated on larger corpora, which made them look more sharply structural than they are.
The ratios below dropped for the same reason: the previous figures were flattered by
invalid negatives inflating diversity toward the truth.

**But the lexical baselines tie with `DocumentSemantics` at 1.000.** This benchmark does
not indict them, because its alternations vary vocabulary as well as structure (a passive
adds *was* and *by*), so vocabulary counting and meaning counting point the same way. The
axis that separates them is synonymy — paraphrases built from *disjoint* vocabulary —
which lives in [`../embedder_selection/`](../embedder_selection/). The two benchmarks are
complementary and neither is sufficient alone. Report both.

### Calibration at each metric's own level

| Metric | level | ρ vs expected | median ratio | n |
|---|---|---:|---:|---:|
| `DocumentSemantics` | semantic | **0.963** | 0.705 | 113 |
| `DependencyParse` | syntactic | 0.959 | **0.986** | 81 |
| `ConstituencyParse` | syntactic | 0.953 | 0.981 | 81 |
| `Phonemic` | phonemic | 0.889 | 0.363 | 32 |
| `PartOfSpeechSequence` | morphological | 0.877 | 0.419 | 81 |
| `TokenSemantics` | semantic | 0.839 | *n/a* | 113 |
| `Rhythmic` | rhythmic | 0.769 | 0.448 | 45 |

`TokenSemantics` has no ratio because its species are **tokens** while the benchmark's
ground truth counts documents — a corpus of *k* concepts contains far more than *k* token
species, so an absolute comparison is meaningless (it reads 3.3x if computed anyway). Its
rank agreement is meaningful and is the weakest of the two semantic metrics, as is its
inverse-pair rate (0.833 vs 1.000). If you want one semantic metric, `DocumentSemantics`
is the better instrument.

`ConstituencyParse` is validated here for the first time. It was entirely non-functional
before v1.0.2, and now tracks `DependencyParse` almost exactly — at roughly 20x the
runtime, since it needs benepar. Prefer `DependencyParse` unless you specifically need
phrase structure.

`DependencyParse` is the best-*scaled* metric in the library on this benchmark — it
recovers the true number of syntactic frames almost exactly (ratio 0.986).
`DocumentSemantics` ranks best but under-reports magnitude by ~30%, consistent with the
embedding-compression effect documented in the embedder-selection benchmark.

### Morphology vs syntax

The sharpest discriminant test in the suite. On sets with an identical POS sequence but
different grammatical relations, `DependencyParse` must separate them while
`PartOfSpeechSequence` collapses them:

| corpus | syntactic | morphological | |
|---|---:|---:|---|
| `posid-elect_vs_give` — object complement vs ditransitive | 1.762 | 1.000 | PASS |
| `posid-call_vs_give` — double object vs object complement | 1.762 | 1.000 | PASS |
| `posid-consider_vs_send` — object complement vs ditransitive | 1.905 | 1.000 | PASS |
| `posid-paint_vs_leave` — resultative vs temporal adjunct | 1.462 | 1.000 | PASS |

This passed only after a fix. `_tree_edit_distance` labelled ZSS nodes by **token index**,
discarding the `pos` and `dep` attributes the parse builder computes — so any two parses
sharing a tree *shape* compared as identical regardless of grammatical function. *She
sings beautifully* (intransitive + adverbial) and *Dogs eat bones* (transitive + direct
object) scored a distance of 0. Nodes are now labelled `POS:relation` and siblings ordered
by token index, so the comparison is over structure rather than shape. Regression tests
cover both directions in `tests/test_syntactic.py`.

### Recorded limitation: what a surface parse cannot see

The `surface_parse_blind` family holds distinctions that **no** metric built on a spaCy
dependency parse can represent — spaCy assigns these pairs byte-identical heads and
dependency labels:

| pair | distinction |
|---|---|
| *John is easy / eager to please* | tough-movement vs subject control |
| *The chicken is ready / likely to eat* | control vs raising |
| *watched the man with the telescope / beard* | instrument vs modifier PP |
| *promised / persuaded him to leave* | subject vs object control |

These expect collapse to 1.0, not separation. Recovering them needs semantic role
labelling, not a better tree distance. They are kept as corpora so the boundary is
documented rather than rediscovered — an earlier revision of this benchmark scored them
as failures of `DependencyParse`, which was a benchmark error, not a metric defect.

## Caveats

- **The seed constructions are hand-authored for this repository** and have not been
  validated by external annotators. Ground truth here is an authored construct. The
  human-anchored results in [`../embedder_selection/`](../embedder_selection/) (McDiv, 600
  human-scored sets) are the check on that, and where the two disagree, prefer the human
  judgments.
- Corpora are small (2–12 documents). That is deliberate — it keeps the ground truth
  exact — but it means these results say nothing about behaviour at corpus scale, where
  the O(n²) similarity matrix and the extrapolation path come into play.
- `Phonemic` has the least seed data (32 scored corpora): full-sentence homophony is rare
  in English, so the oronym family is inherently small and its ρ = 0.887 rests on a
  narrow base.
- The metric fix described above (`POS:relation` node labels) landed with this benchmark;
  results predating it are not comparable.
- `ConstituencyParse` is not scored here; it needs `benepar`, which is an optional extra.
  Add it to `METRICS` in `evaluate_metrics.py` if the extra is installed.
