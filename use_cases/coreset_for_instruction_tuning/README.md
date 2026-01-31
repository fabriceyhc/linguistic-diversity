# Coreset Selection for Instruction Tuning

**Diversity-Guided Dataset Pruning for Efficient Instruction Fine-tuning**

## Overview

This use case demonstrates that instruction-tuning datasets can be pruned to **10% of their original size** without performance loss by selecting samples that maximize linguistic diversity.

### Key Hypothesis

> Quality and diversity matter more than quantity. By selecting the most diverse 10% of training data, we can:
> 1. Preserve (or improve) model performance
> 2. Reduce training compute by ~90%
> 3. Create efficient, high-quality training datasets for the community

### Success Criteria

- The "Diversity-Selected" (10%) model achieves AlpacaEval 2.0 win-rates **within 2%** of the "Full Dataset" model
- The "Diversity-Selected" model **outperforms** the "Random Selection" (10%) baseline

## Experiment Phases

### Phase 2: Pilot (Alpaca-GPT4)
- **Dataset**: `vicgalle/alpaca-gpt4` (~52k examples)
- **Target**: ~5,200 examples (10%)
- **Model**: Mistral-7B-v0.3 or Llama-3-8B
- **Evaluation**: lm-evaluation-harness (local, no API required) + optional local LLM-as-Judge

### Phase 3: Scale-up (Conditional on Phase 2 Success)
- **OpenOrca**: ~3.5M examples → 350k
- **UltraChat**: ~200k examples → 20k
- **Selection Method**: Clustering-based for scalability

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Install Unsloth for faster training
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 3. Run the pilot experiment
python run_pipeline.py --mode pilot

# 4. Run full experiment (after pilot succeeds)
python run_pipeline.py --mode full
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1 | `01_prepare_data.py` | Download datasets, generate embeddings |
| 2 | `02_select_subsets.py` | Create random and diversity-selected subsets |
| 3 | `03_finetune.py` | Fine-tune with LoRA (Unsloth-optimized) |
| 4 | `04_evaluate.py` | Run lm-eval-harness benchmarks + local LLM judge |
| 5 | `05_generate_report.py` | Generate visualizations and REPORT.md |

## Selection Algorithms

### Greedy Diversity Maximization (for datasets < 100k)

```python
# Algorithm:
# 1. Start with empty set S
# 2. Add random initial point
# 3. Iteratively add point that maximizes min_distance(point, S)
# 4. Repeat until |S| = target_size
```

This is a facility location / max-min diversity algorithm that maximizes coverage of the embedding space.

### Clustering-Based Selection (for datasets > 100k)

```python
# Algorithm:
# 1. Cluster embeddings into K clusters (K = target_size)
# 2. Select point closest to each cluster centroid
```

Uses MiniBatchKMeans for efficiency on large datasets.

## Configuration

Edit `config.yaml` to customize:

```yaml
# Experiment mode
mode: pilot  # or 'full' for Phase 3

# Selection parameters
selection:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2

# Fine-tuning parameters
finetuning:
  use_unsloth: true
  lora:
    r: 16
    lora_alpha: 32
  training:
    num_epochs: 1
    learning_rate: 2e-4

# Evaluation (all local, no paid APIs)
evaluation:
  local_benchmarks:
    enabled: true
    tasks: [hellaswag, arc_easy, winogrande, truthfulqa_mc1]
  llm_judge:
    enabled: true
    model: prometheus-eval/prometheus-7b-v2.0  # Local judge model
```

## Evaluation (100% Local)

All evaluation runs locally without paid APIs:

1. **lm-evaluation-harness**: Standard benchmarks (HellaSwag, ARC, WinoGrande, TruthfulQA, MMLU, GSM8K)
2. **Local LLM-as-Judge**: Uses Prometheus-7B or similar open models to score responses
3. **Linguistic Diversity Metrics**: Using the `linguistic_diversity` library:
   - **Semantic**: Document-level embedding diversity
   - **Syntactic**: Dependency parse tree diversity
   - **Morphological**: Part-of-speech sequence diversity
   - **Universal**: Combined diversity across all dimensions

No OpenAI/Anthropic API keys required.

## Length Bias Warning

> **Important**: Diversity metrics can favor long, verbose outputs. The pipeline monitors average token length of selected subsets vs. full dataset. A warning is issued if the selected subset is >3x longer than average.

## Output Structure

```
output/
├── adapters/           # Fine-tuned LoRA adapters
│   ├── Mistral-7B-v0.3_alpaca_gpt4_full_seed42/
│   ├── Mistral-7B-v0.3_alpaca_gpt4_random_seed42/
│   └── Mistral-7B-v0.3_alpaca_gpt4_diversity_seed42/
├── evaluation/         # Evaluation outputs
├── plots/              # Visualizations
│   ├── win_rate_comparison.png
│   ├── diversity_vs_performance.png
│   └── compute_savings.png
├── training_results.json
├── evaluation_results.json
└── REPORT.md           # Final report

selections/
├── alpaca_gpt4_full.jsonl
├── alpaca_gpt4_random.jsonl
├── alpaca_gpt4_diversity.jsonl
└── selection_summary.json
```

## Deliverables

Upon successful completion, this pipeline produces:

1. **Pruned Datasets**: Ready-to-use JSONL files for instruction tuning
2. **Fine-tuned Adapters**: LoRA adapters for Mistral/Llama models
3. **Report**: Comprehensive analysis with visualizations
4. **HuggingFace Upload** (optional): Share datasets with the community

## References

- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [AlpacaEval 2.0](https://github.com/tatsu-lab/alpaca_eval)
- [Unsloth](https://github.com/unslothai/unsloth)

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{diversity_coreset_instruction_tuning,
  title={Diversity-Guided Dataset Pruning for Instruction Tuning},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]}
}
```
