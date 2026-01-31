# Instruction Fine-tuning Data Selection Results

*Generated: 2026-01-26 18:11:20*

## Experimental Setup

- **Mode**: quick
- **Models**: Qwen/Qwen3-1.7B
- **Selection sizes**: ['size_1000']
- **Selection methods**: ['pretrained_baseline', 'random', 'semantic_diversity']
- **Evaluation tasks**: ['hellaswag']
- **Seeds**: [42]

### Dataset Statistics
- **Total samples**: 5000
- **Source datasets**: TokenBender/code_instructions_122k_alpaca_style, GAIR/lima, nvidia/OpenMathInstruct-1, PKU-Alignment/BeaverTails
- **Categories**: coding, instruction, reasoning, safety


## Results

### Qwen3-1.7B

*No evaluation results available*

## Key Findings

### Method Performance vs Random Baseline

- **Semantic Diversity**: 0/0 wins (0%)

### Statistical Significance (p < 0.05 vs Random)

*No statistically significant differences found (may need more seeds)*

## Conclusions

### Data Efficiency Analysis

Following the LIMA paper's insight that quality/diversity matters more than quantity:

1. No single selection method dominates across all configurations
2. Diversity-based selection methods aim to achieve comparable performance to larger randomly-selected datasets
3. The effectiveness of each method may vary based on the downstream task and model architecture

### Recommendations

- For resource-constrained fine-tuning, consider **combined diversity** selection
- Monitor both task performance and training efficiency
- Consider ensemble methods that combine multiple selection strategies

## Selection Method Overlap Analysis

Low overlap between methods indicates they capture different aspects of diversity.


### size_1000

- semantic_diversity_vs_combined_diversity: Jaccard=0.467 (637 shared samples)
- semantic_diversity_vs_instruction_diversity: Jaccard=0.300 (462 shared samples)
- combined_diversity_vs_instruction_diversity: Jaccard=0.299 (460 shared samples)
- semantic_diversity_vs_quality_filtered: Jaccard=0.208 (344 shared samples)
- combined_diversity_vs_quality_filtered: Jaccard=0.202 (336 shared samples)

### size_2500

- semantic_diversity_vs_combined_diversity: Jaccard=0.744 (2133 shared samples)
- semantic_diversity_vs_instruction_diversity: Jaccard=0.554 (1782 shared samples)
- combined_diversity_vs_instruction_diversity: Jaccard=0.547 (1767 shared samples)
- syntactic_diversity_vs_quality_filtered: Jaccard=0.424 (1489 shared samples)
- combined_diversity_vs_universal_embedding_diversity: Jaccard=0.350 (1295 shared samples)

### size_5000

- random_vs_semantic_diversity: Jaccard=1.000 (5000 shared samples)
- random_vs_syntactic_diversity: Jaccard=1.000 (5000 shared samples)
- random_vs_combined_diversity: Jaccard=1.000 (5000 shared samples)
- random_vs_length_diversity: Jaccard=1.000 (5000 shared samples)
- random_vs_quality_filtered: Jaccard=1.000 (5000 shared samples)

### size_10000

- random_vs_semantic_diversity: Jaccard=1.000 (5000 shared samples)
- random_vs_syntactic_diversity: Jaccard=1.000 (5000 shared samples)
- random_vs_combined_diversity: Jaccard=1.000 (5000 shared samples)
- random_vs_length_diversity: Jaccard=1.000 (5000 shared samples)
- random_vs_quality_filtered: Jaccard=1.000 (5000 shared samples)
