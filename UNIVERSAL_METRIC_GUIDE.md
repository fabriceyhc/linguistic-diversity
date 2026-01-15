# Universal Linguistic Diversity Metric

## Overview

The **Universal Linguistic Diversity** metric is a creative unified approach to measuring linguistic diversity that combines all 7 individual metrics across 4 branches of linguistics into a single, comprehensive diversity score.

## Metric Count

The repository contains **7 distinct metrics** across 4 linguistic branches:

### Semantic Branch (2 metrics)
1. **TokenSemantics** - Measures diversity of contextualized token embeddings
2. **DocumentSemantics** - Measures diversity of document-level semantic embeddings

### Syntactic Branch (2 metrics)
3. **DependencyParse** - Measures diversity of dependency parse tree structures
4. **ConstituencyParse** - Measures diversity of constituency (phrase structure) parse trees

### Morphological Branch (1 metric)
5. **PartOfSpeechSequence** - Measures diversity of POS tag sequences using biological sequence alignment

### Phonological Branch (2 metrics)
6. **Rhythmic** - Measures diversity of rhythmic patterns (stress and syllable weight)
7. **Phonemic** - Measures diversity of phoneme sequences (IPA representation)

## The Universal Metric

### Design Philosophy

The universal metric uses a **hierarchical aggregation strategy** that respects the structure of linguistic theory:

1. **Intra-branch aggregation**: Within each linguistic branch, metrics are combined using **geometric mean**, which:
   - Treats metrics equally within their domain
   - Prevents any single low score from dominating
   - Captures the multiplicative nature of linguistic diversity

2. **Inter-branch aggregation**: Across branches, we use a **weighted geometric mean** with configurable weights:
   - Semantic: 35% (default) - captures meaning diversity
   - Syntactic: 30% (default) - captures structural diversity
   - Morphological: 15% (default) - captures grammatical pattern diversity
   - Phonological: 20% (default) - captures sound diversity

### Why This Approach?

This hierarchical design has several advantages:

- **Linguistically Principled**: Respects the hierarchical organization of linguistic theory
- **Flexible**: Supports multiple aggregation strategies and weighting schemes
- **Robust**: Prevents any single metric from dominating the score
- **Interpretable**: Branch-level scores provide insight into which dimensions drive diversity
- **Extensible**: Easy to add new metrics or adjust weights for specific use cases

## Usage

### Basic Usage

```python
from linguistic_diversity import UniversalLinguisticDiversity

corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn canine leaps above an idle hound.",
    "Swift russet predator vaults past lethargic beast.",
]

metric = UniversalLinguisticDiversity()
diversity = metric(corpus)
print(f"Universal Diversity: {diversity:.2f}")
```

### Detailed Scores

```python
# Get breakdown by branch and metric
metric = UniversalLinguisticDiversity()
detailed = metric.get_detailed_scores(corpus)

print(f"Universal Score: {detailed['universal']:.3f}")
print("\nBranch Scores:")
for branch, score in detailed['branches'].items():
    print(f"  {branch}: {score:.3f}")

print("\nMetric Scores:")
for metric_name, score in detailed['metrics'].items():
    print(f"  {metric_name}: {score:.3f}")
```

### Preset Configurations

```python
from linguistic_diversity import UniversalLinguisticDiversity, get_preset_config

# Balanced (default)
config = get_preset_config("balanced")
metric = UniversalLinguisticDiversity(config)

# Semantic-focused (for content analysis)
config = get_preset_config("semantic_focus")
metric = UniversalLinguisticDiversity(config)

# Structure-focused (for grammatical analysis)
config = get_preset_config("structural_focus")
metric = UniversalLinguisticDiversity(config)

# Minimal (core metrics only, no optional dependencies)
config = get_preset_config("minimal")
metric = UniversalLinguisticDiversity(config)

# Conservative (harmonic mean - very sensitive to low scores)
config = get_preset_config("conservative")
metric = UniversalLinguisticDiversity(config)
```

### Custom Configuration

```python
custom_config = {
    # Aggregation strategy
    "strategy": "hierarchical",  # or: weighted_geometric, weighted_arithmetic, harmonic, minimum

    # Branch weights (should sum to 1.0)
    "semantic_weight": 0.40,
    "syntactic_weight": 0.35,
    "morphological_weight": 0.15,
    "phonological_weight": 0.10,

    # Enable/disable branches
    "use_semantic": True,
    "use_syntactic": True,
    "use_morphological": True,
    "use_phonological": True,

    # Enable/disable specific metrics
    "use_token_semantics": True,
    "use_document_semantics": True,
    "use_dependency_parse": True,
    "use_constituency_parse": False,  # Requires benepar
    "use_pos_sequence": True,
    "use_rhythmic": True,
    "use_phonemic": True,

    # Pass configs to individual metrics
    "semantic_config": {"use_cuda": False, "batch_size": 16},
    "syntactic_config": {"similarity_type": "ldp"},

    # Verbosity
    "verbose": False,
}

metric = UniversalLinguisticDiversity(custom_config)
```

## Aggregation Strategies

### 1. Hierarchical (Recommended Default)
```python
config = {"strategy": "hierarchical"}
```
- Geometric mean within branches
- Weighted geometric mean across branches
- Best balance of robustness and sensitivity
- Respects linguistic hierarchy

### 2. Weighted Geometric
```python
config = {"strategy": "weighted_geometric"}
```
- Direct weighted geometric mean of all 7 metrics
- Penalizes low scores more than arithmetic
- Good for identifying consistently diverse corpora

### 3. Weighted Arithmetic
```python
config = {"strategy": "weighted_arithmetic"}
```
- Direct weighted average of all 7 metrics
- Most lenient strategy
- Good for getting overall average

### 4. Harmonic
```python
config = {"strategy": "harmonic"}
```
- Harmonic mean (very conservative)
- Heavily penalizes low scores
- Good for identifying corpora that are diverse in ALL dimensions

### 5. Minimum
```python
config = {"strategy": "minimum"}
```
- Takes minimum score across all metrics
- Most conservative possible
- Good for identifying weakest dimension

## Preset Comparison

| Preset | Strategy | Semantic | Syntactic | Morphological | Phonological | Use Case |
|--------|----------|----------|-----------|---------------|--------------|----------|
| **balanced** | hierarchical | 35% | 30% | 15% | 20% | General-purpose analysis |
| **semantic_focus** | hierarchical | 60% | 20% | 10% | 10% | Content diversity, essays |
| **structural_focus** | hierarchical | 20% | 50% | 20% | 10% | Grammatical complexity |
| **minimal** | weighted_geometric | - | - | - | - | Fast, no optional deps |
| **conservative** | harmonic | - | - | - | - | Require diversity in ALL dims |

## Interpretation

### What Does the Score Mean?

The universal diversity score represents the **effective number of linguistically distinct units** in your corpus, considering all dimensions simultaneously.

- **Score ≈ N**: Your corpus has diversity equivalent to N completely distinct linguistic items
- **Higher scores**: More linguistically diverse corpus
- **Lower scores**: More linguistically uniform corpus

### Example Interpretations

```
Universal Diversity: 2.5
```
"The corpus has the linguistic diversity equivalent to 2.5 completely distinct items across all measured dimensions."

```
Branch Scores:
  Semantic: 1.8
  Syntactic: 3.2
  Morphological: 2.1
  Phonological: 2.9
```
"The corpus is most diverse syntactically (3.2) and least diverse semantically (1.8), suggesting similar meanings expressed with varied sentence structures."

## Use Cases

### 1. Content Analysis
Use `semantic_focus` preset to emphasize meaning diversity in:
- News article collections
- Essay datasets
- Creative writing corpora

### 2. Language Learning
Use `structural_focus` preset to analyze grammatical variety in:
- Textbook exercises
- Student writing samples
- Language proficiency assessments

### 3. Stylometry
Use `balanced` preset to characterize author style across:
- Multiple linguistic dimensions
- Historical text collections
- Author attribution tasks

### 4. Data Quality
Use `conservative` preset to ensure diversity in:
- Training data for NLP models
- Benchmark datasets
- Corpus construction

### 5. Comparative Analysis
Compare corpora using detailed scores to understand:
- Which dimensions differ most
- Whether diversity is balanced or specialized
- How to improve corpus diversity

## Technical Details

### Similarity-Sensitive Hill Numbers

All individual metrics use the similarity-sensitive Hill number framework:

```
D = (Σ p_i (Σ Z_ij p_j)^(q-1))^(1/(1-q))
```

Where:
- `p`: Abundance distribution over species
- `Z`: Similarity matrix between species
- `q`: Diversity order (default q=1 for Shannon diversity)

### Hierarchical Aggregation Formula

For the default hierarchical strategy:

1. **Within-branch geometric mean**:
   ```
   D_branch = (∏ D_i)^(1/n)
   ```

2. **Across-branch weighted geometric mean**:
   ```
   D_universal = exp(Σ w_k log(D_k))
   ```

Where `w_k` are normalized branch weights and `D_k` are branch-level diversities.

## Examples

See `examples/universal_metric.py` for comprehensive examples demonstrating:
- Default configuration
- Detailed score breakdown
- Preset configurations
- Custom configurations
- Aggregation strategy comparison
- Selective metrics

## Testing

Run tests with:
```bash
pytest tests/test_universal.py -v
```

## Citation

If you use the Universal Linguistic Diversity metric in your research, please cite:

```bibtex
@software{universal_linguistic_diversity_2026,
  title={Universal Linguistic Diversity: A Unified Metric Across Multiple Linguistic Dimensions},
  author={Harel-Canada, Fabrice},
  year={2026},
  url={https://github.com/fabriceyhc/linguistic-diversity}
}
```

## Future Directions

Potential extensions to the universal metric:

1. **Learned Weights**: Use supervised learning to optimize weights for specific tasks
2. **Adaptive Aggregation**: Dynamically adjust strategy based on corpus characteristics
3. **Pragmatic Dimension**: Add discourse and pragmatic diversity metrics
4. **Cross-lingual**: Extend to multilingual corpora
5. **Temporal**: Track how universal diversity evolves over time
6. **Hierarchical Decomposition**: Variance decomposition across dimensions

## Related Work

- **Ecological Diversity**: Hill numbers from ecology (Chao et al., 2014)
- **Linguistic Diversity**: Individual metrics from TextDiversity (Harel-Canada, 2022)
- **Composite Indices**: Similar to HDI (Human Development Index) methodology
- **Factor Analysis**: Related to dimensionality reduction in psychometrics

## License

MIT License - see LICENSE file for details.
