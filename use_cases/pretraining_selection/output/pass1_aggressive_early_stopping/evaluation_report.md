# Evaluation Results Summary

## Experimental Setup

- **Dataset**: HuggingFaceFW/fineweb-edu
- **Diversity-selected subsets**: 10% of data (9,000 documents)
- **Full dataset**: 100% of data (90,000 documents)
- **Seeds**: [42, 123, 456] (3 runs per evaluation)
- **Statistical tests**: Independent t-test vs random baseline

## GLUE Benchmark Results (Encoder Models)

| Regime | CoLA (Matthews) | SST-2 (Acc) | MRPC (F1) | RTE (Acc) | Average |
|--------|-----------------|-------------|-----------|-----------|---------|
| Semantic | 0.597 ± 0.013 | 0.948 ± 0.001 | 0.905 ± 0.003 | 0.578 ± 0.036 | **0.757** |
| Syntactic | 0.592 ± 0.016 | 0.950 ± 0.003 | 0.909 ± 0.005 | 0.580 ± 0.015 | **0.758** |
| Morphological | 0.567 ± 0.031 | 0.950 ± 0.002 | 0.906 ± 0.003 | 0.580 ± 0.023 | **0.751** |
| Phonological | 0.576 ± 0.056 | 0.945 ± 0.003 | 0.910 ± 0.006 | 0.575 ± 0.029 | **0.752** |
| Universal | 0.583 ± 0.004 | 0.950 ± 0.002 | 0.897 ± 0.003 | 0.574 ± 0.044 | **0.751** |
| Random | 0.586 ± 0.014 | 0.947 ± 0.000 | 0.899 ± 0.001 | 0.602 ± 0.015 | **0.758** |
| Full Dataset | 0.581 ± 0.020 | 0.946 ± 0.003 | 0.911 ± 0.005 | 0.570 ± 0.034 | **0.752** |

## Decoder Benchmark Results (Zero-Shot)

| Regime | HellaSwag | ARC-Easy | BoolQ | Average |
|--------|-----------|----------|-------|---------|
| Semantic | 0.621 ± 0.016 | 0.620 ± 0.007 | 0.625 ± 0.031 | **0.622** |
| Syntactic | 0.618 ± 0.014 | 0.618 ± 0.002 | 0.652 ± 0.036 | **0.629** |
| Morphological | 0.624 ± 0.007 | 0.594 ± 0.006 | 0.477 ± 0.017 | **0.565** |
| Phonological | 0.628 ± 0.012 | 0.627 ± 0.005 | 0.648 ± 0.040 | **0.634** |
| Universal | 0.625 ± 0.015 | 0.629 ± 0.006 | 0.640 ± 0.032 | **0.632** |
| Random | 0.613 ± 0.015 | 0.608 ± 0.007 | 0.590 ± 0.016 | **0.604** |
| Full Dataset | 0.614 ± 0.021 | 0.599 ± 0.004 | 0.642 ± 0.033 | **0.618** |

*Note: PIQA evaluation failed for all regimes.*

## Encoder-Decoder Task Results

| Regime | XSum (ROUGE-L) | SQuAD v2 (F1) |
|--------|----------------|---------------|
| Semantic | 0.256 ± 0.004 | 0.896 ± 0.006 |
| Syntactic | 0.246 ± 0.005 | 0.892 ± 0.009 |
| Morphological | 0.270 ± 0.005 | 0.893 ± 0.008 |
| Phonological | 0.261 ± 0.009 | 0.870 ± 0.014 |
| Universal | 0.267 ± 0.005 | 0.865 ± 0.014 |
| Random | 0.267 ± 0.003 | 0.879 ± 0.014 |
| Full Dataset | 0.266 ± 0.006 | 0.868 ± 0.016 |

*Note: SAMSum evaluation failed for all regimes.*

## Diversity Scores

| Regime | Universal Diversity | Std |
|--------|---------------------|-----|
| Universal | 462.38 | ± 4.84 |
| composite_diversity | 271.19 | ± 2.77 |
| Phonological | 155.64 | ± 0.28 |
| Morphological | 148.29 | ± 1.77 |
| Semantic | 111.94 | ± 1.32 |
| Random | 106.93 | ± 1.41 |
| Syntactic | 8.81 | ± 0.85 |

## Key Findings

1. **GLUE**: Random achieves highest average (0.758)
2. **Decoder benchmarks**: Phonological achieves highest average (0.634)
3. **Encoder-Decoder**: Morphological achieves highest average (0.582)

### Comparison with Baselines

- **Random baseline (10%)**: GLUE avg = 0.758
- **Full dataset (100%)**: GLUE avg = 0.752
- Full dataset vs random: -0.006 (-0.9%)

- **Random baseline (10%)**: Decoder avg = 0.604
- **Full dataset (100%)**: Decoder avg = 0.618
- Full dataset vs random: +0.014 (+2.4%)

### Statistical Significance

**GLUE** (vs random baseline, p < 0.05):
  - Semantic: significant on mrpc
  - Syntactic: significant on mrpc
  - Morphological: significant on mrpc
  - Full Dataset: significant on mrpc

**DECODER** (vs random baseline, p < 0.05):
  - Morphological: significant on boolq
  - Phonological: significant on arc_easy
  - Universal: significant on arc_easy

**ENCDEC** (vs random baseline, p < 0.05):
  - Semantic: significant on xsum
  - Syntactic: significant on xsum

## Conclusions

1. **Diversity-guided selection shows mixed results**: Performance varies by model type and evaluation task.
2. **10% diversity-selected data competitive with 100% data**: Diversity selection achieves similar or better performance with 10x less data.
3. **Different diversity types excel on different tasks**: No single diversity measure dominates across all evaluations.
4. **Decoder models benefit most from diversity selection**: Larger gaps between diversity-guided and random selection on decoder benchmarks.
