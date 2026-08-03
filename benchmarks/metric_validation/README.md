# Per-Level Metric Validation

Does each metric respond to the linguistic level it claims to measure, and stay flat on
the others?

[`../embedder_selection/`](../embedder_selection/) answers a narrower question — which
sentence encoder should back `DocumentSemantics`. It validates one metric. This benchmark
validates all of them, and it is built around the property that a scalar diversity score
cannot express: **discriminant validity**.

```bash
python build_benchmark.py      # 154 corpora, 166 contrasts, deterministic
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
| `morphological_inflection` | *k* inflectional patterns × *m* lexicalisations, one frame | *k·m* | **1** | **k** | needs the fine tagset; invisible to UPOS |
| `constituency_contrasts` | phrase-structure contrasts, each verified against the parsers | *n* | *n* | — | which parser sees what |
| `rhythmic_stress_pairs` | noun/verb stress alternations (*a REcord* / *to reCORD*) | *n* | — | — | phonology where lexis barely moves |
| `phonemic_graded` | three tiers of phonological distance at one frame | — | — | — | scored as an **ordering**, not a target |
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
| `DocumentSemantics` | semantic | 1.000 | 54 |
| `TokenSemantics` | semantic | 1.000 | 54 |
| `TypeTokenRatio` | *(baseline)* | 1.000 | 54 |
| `DistinctN` | *(baseline)* | 1.000 | 54 |
| `SelfBLEU` | *(baseline)* | 0.944 | 54 |
| `Phonemic` | phonemic | 0.389 | 54 |
| `Rhythmic` | rhythmic | 0.315 | 54 |
| `PartOfSpeechSequence` | morphological | 0.056 | 54 |
| `DependencyParse` | syntactic | 0.000 | 54 |
| `ConstituencyParse` | syntactic | 0.000 | 54 |

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

### Calibration, and where the shortfall actually lives

| Metric | level | ρ vs expected | ratio | headroom | n |
|---|---|---:|---:|---:|---:|
| `DocumentSemantics` | semantic | **0.922** | 0.706 | 0.993 | 147 |
| `DependencyParse` | syntactic | 0.911 | 0.667 | 0.995 | 110 |
| `ConstituencyParse` | syntactic | 0.885 | 0.506 | 0.999 | 110 |
| `Phonemic` | phonemic | 0.874 | 0.372 | 0.995 | 32 |
| `TokenSemantics` | semantic | 0.840 | *n/a* | 0.844 | 147 |
| `PartOfSpeechSequence` | morphological | 0.772 | 0.437 | 0.995 | 105 |
| `Rhythmic` | rhythmic | 0.771 | 0.492 | 0.990 | 50 |

**ratio** is observed / authored ground truth. **headroom** is observed /
`max_diversity(Z)` — the largest value *any* abundance distribution could reach
given the similarity matrix the metric actually computed. By Leinster & Meckes
([2016](https://www.mdpi.com/1099-4300/18/3/88)) that ceiling is the same for
every order *q*, so it is a property of the similarity structure alone.

Read together they were supposed to localise the shortfall, and an earlier revision
of this file drew the wrong conclusion from them. It said: every metric reads 37–71%
of ground truth while sitting at 99% of its achievable ceiling, therefore the index
is extracting everything its similarity structure permits and the gap must lie in
the embedder.

**That inference does not hold.** Two things undercut it.

First, headroom is close to tautological at uniform abundance. For a similarity
matrix of the form `(1−z)I + zJ` — one baseline similarity between every pair — the
magnitude equals the Hill number at uniform **p** *exactly*, so headroom is 1.0000
by construction, not by merit. Every metric here uses uniform abundance. The number
mostly reports that the abundance distribution is optimal for this index, which it
is trivially, rather than that the index is extracting what the data holds.

Second, and decisively: given the *same* similarity matrix, the Vendi Score reads
0.986 of ground truth where the Hill number reads 0.706 (see
[`../vendi_comparison/`](../vendi_comparison/)). A different index does far better
on identical input, so the shortfall cannot be attributed to the representation
alone.

The mechanism is structural. A uniform baseline similarity contributes `zJ`, which
is **rank one**. Vendi's spectral decomposition confines it to a single eigenvalue
and leaves the other n−1 untouched; the Hill number spreads it through every
`(Zp)ᵢ = (1 + (n−1)z)/n`, where it accumulates linearly and drives the score toward
`1/z` regardless of n. At z = 0.3 and n = 50, the Hill number reads 3.18 against a
truth of 50; Vendi reads 26.90.

So the honest statement is that **both** components contribute: the encoder's floor
is real and worth correcting, and this index is unusually fragile to whatever floor
remains. The order parameter q cannot help — at uniform abundance every `(Zp)ᵢ` is
equal, so q cancels entirely and q = 0, 0.5, 1 and 2 all return the same value.

The practical consequence: **compare corpora with these scores, and read a single
score against `max_diversity(corpus)` rather than against the document count.**
`relative_diversity(corpus)` returns the pair as one number.

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

## What the newer families settled

Four families were added after the first round, because the non-semantic metrics were
being scored entirely on constructions built for semantics and syntax. `ConstituencyParse`
had none of its own at all.

**`phonemic_graded` is the one that mattered.** Three tiers of phonological distance at a
fixed syntactic frame — rhyme, shared onset, distant — scored as an ordering rather than a
calibration target. `Phonemic` gets it right **1.000** of the time; every other metric
scores 0.000–0.625, the three lexical baselines included. It is the only construction in
the suite where `Phonemic` is uniquely correct, and it moved the metric from "looks weak"
to "was untested".

**`constituency_contrasts` produced an unwelcome result and is kept for it.** Every pair
was checked against spaCy and benepar before authoring, and dependency parsing separates
most classic constituency contrasts *better* than constituency does: center-embedding
0.375 vs 0.571, complement-vs-relative 0.375 vs 0.714, PP stacking 0.400 vs 0.611. Only
coordination scope favours constituency, and NP-internal bracketing (*[French [history
teacher]]*) is invisible to both. Given `ConstituencyParse` costs roughly 20x the runtime
and an optional dependency, that is worth knowing before reaching for it.

**`morphological_inflection`** exists because `PartOfSpeechSequence` read UPOS until
v1.0.3, where *walks*, *walked* and *walking* are all `VERB`. It was not a morphological
metric. With the fine-grained PTB tagset a corpus varying only in tense scores above 1.0
where UPOS gives exactly 1.0, while one frame across different words still collapses.
