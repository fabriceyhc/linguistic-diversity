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
| Pretrained | 0.574 ± 0.008 | 0.951 ± 0.001 | 0.902 ± 0.001 | 0.613 ± 0.039 | **0.760** |
| Semantic | 0.573 ± 0.033 | 0.949 ± 0.005 | 0.915 ± 0.011 | 0.616 ± 0.030 | **0.763** |
| Syntactic | 0.570 ± 0.016 | 0.938 ± 0.001 | 0.905 ± 0.005 | 0.564 ± 0.011 | **0.744** |
| Morphological | 0.581 ± 0.007 | 0.948 ± 0.002 | 0.906 ± 0.009 | 0.560 ± 0.006 | **0.749** |
| Phonological | 0.542 ± 0.019 | 0.946 ± 0.004 | 0.892 ± 0.012 | 0.596 ± 0.008 | **0.744** |
| Universal | 0.544 ± 0.054 | 0.946 ± 0.003 | 0.901 ± 0.011 | 0.599 ± 0.028 | **0.748** |
| Random | 0.569 ± 0.005 | 0.945 ± 0.003 | 0.878 ± 0.028 | 0.604 ± 0.027 | **0.749** |
| Full Dataset | 0.553 ± 0.024 | 0.947 ± 0.002 | 0.904 ± 0.010 | 0.566 ± 0.012 | **0.742** |

## Decoder Benchmark Results (Zero-Shot)

| Regime | HellaSwag | ARC-Easy | BoolQ | Average |
|--------|-----------|----------|-------|---------|
| Pretrained | 0.625 ± 0.017 | 0.599 ± 0.007 | 0.648 ± 0.038 | **0.624** |
| Semantic | 0.490 ± 0.007 | 0.500 ± 0.011 | 0.629 ± 0.024 | **0.540** |
| Syntactic | 0.347 ± 0.024 | 0.374 ± 0.007 | 0.627 ± 0.034 | **0.449** |
| Morphological | 0.341 ± 0.001 | 0.397 ± 0.009 | 0.627 ± 0.035 | **0.455** |
| Phonological | 0.357 ± 0.032 | 0.448 ± 0.003 | 0.629 ± 0.036 | **0.478** |
| Universal | 0.522 ± 0.003 | 0.527 ± 0.004 | 0.591 ± 0.028 | **0.547** |
| Random | 0.505 ± 0.007 | 0.501 ± 0.010 | 0.622 ± 0.040 | **0.543** |
| Full Dataset | 0.395 ± 0.007 | 0.439 ± 0.007 | 0.595 ± 0.018 | **0.476** |

*Note: PIQA evaluation failed for all regimes.*

## Encoder-Decoder Task Results

| Regime | XSum (ROUGE-L) | SQuAD v2 (F1) |
|--------|----------------|---------------|
| Pretrained | 0.290 ± 0.009 | 0.886 ± 0.004 |
| Semantic | 0.119 ± 0.004 | 0.498 ± 0.008 |
| Syntactic | 0.123 ± 0.005 | 0.795 ± 0.016 |
| Morphological | 0.122 ± 0.004 | 0.692 ± 0.007 |
| Phonological | 0.115 ± 0.002 | 0.171 ± 0.012 |
| Universal | 0.177 ± 0.007 | 0.749 ± 0.021 |
| Random | 0.111 ± 0.001 | 0.162 ± 0.010 |
| Full Dataset | 0.114 ± 0.002 | 0.373 ± 0.006 |

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

1. **GLUE (Encoder)**: Semantic achieves highest average (0.763)
2. **Decoder benchmarks**: Pretrained achieves highest average (0.624)
3. **Encoder-Decoder**: Pretrained achieves highest average (0.588)

### Impact of Additional Pretraining

Comparing original pretrained models vs models with additional pretraining on fineweb-edu:

- **GLUE**: Pretrained=0.760, Best trained (Semantic)=0.763 → Additional pretraining **helps** (+0.003, +0.4%)
- **Decoder**: Pretrained=0.624, Best trained (Universal)=0.547 → Additional pretraining **hurts** (-0.077, -12.4%)
- **Encoder-Decoder**: Pretrained=0.588, Best trained (Universal)=0.463 → Additional pretraining **hurts** (-0.125, -21.2%)

### Data Efficiency: 10% Selection vs 100% Data

- **GLUE**: Random 10%=0.749, Full 100%=0.742 → **Random 10%** wins (-0.007)
- **Decoder**: Random 10%=0.543, Full 100%=0.476 → **Random 10%** wins (-0.066)
- **Encoder-Decoder**: Random 10%=0.137, Full 100%=0.244 → **Full dataset** wins (+0.107)

### Statistical Significance

**GLUE** (vs random baseline, p < 0.05):
  - Syntactic: significant on sst2

**DECODER** (vs random baseline, p < 0.05):
  - Pretrained: significant on hellaswag, arc_easy
  - Syntactic: significant on hellaswag, arc_easy
  - Morphological: significant on hellaswag, arc_easy
  - Phonological: significant on hellaswag, arc_easy
  - Universal: significant on hellaswag, arc_easy
  - Full Dataset: significant on hellaswag, arc_easy

**ENCDEC** (vs random baseline, p < 0.05):
  - Pretrained: significant on xsum, squad_v2
  - Semantic: significant on squad_v2
  - Syntactic: significant on xsum, squad_v2
  - Morphological: significant on xsum, squad_v2
  - Universal: significant on xsum, squad_v2
  - Full Dataset: significant on squad_v2

## Conclusions

1. **Catastrophic forgetting observed**: Additional pretraining on fineweb-edu **hurts** decoder and encoder-decoder model performance. The original pretrained models outperform all additionally-pretrained variants.
2. **More data is not always better**: Random 10% selection outperforms full dataset (100%) on GLUE, Decoder benchmarks, suggesting data quality/diversity matters more than quantity.
3. **Encoder models are most robust**: GLUE performance is relatively stable across different pretraining regimes, suggesting encoder architectures are less sensitive to pretraining data.
4. **Diversity-guided selection shows potential**: Some diversity selection methods (e.g., semantic, universal) achieve competitive or better performance than random selection with the same data budget.
