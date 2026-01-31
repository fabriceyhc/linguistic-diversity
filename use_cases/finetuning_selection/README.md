# Instruction Fine-tuning Data Selection

This experiment tests whether linguistic diversity metrics can automate the LIMA-style curation process for instruction fine-tuning data.

## Hypothesis

Based on the LIMA paper ("Less Is More for Alignment"), 1,000 carefully curated diverse examples can match or outperform 50,000+ randomly selected examples. This experiment automates the curation process using linguistic diversity metrics.

## Pipeline Overview

```
01_extract_features.py → 02_select_subsets.py → 03_finetune_evaluate.py → 04_generate_report.py
```

1. **Feature Extraction**: Extract semantic embeddings and syntactic features from instruction datasets
2. **Subset Selection**: Apply diversity-based selection methods to choose training subsets
3. **Fine-tuning**: Train models with LoRA on selected subsets
4. **Evaluation**: Evaluate on standard benchmarks and generate comparison report

## Selection Methods

- **Random**: Baseline random selection
- **Semantic Diversity**: Facility location on sentence embeddings
- **Syntactic Diversity**: Submodular selection on POS/dependency features
- **Combined Diversity**: Weighted combination of semantic + syntactic
- **Length Diversity**: Stratified sampling by response length
- **Quality-Filtered**: Filter by quality signals, then apply diversity selection
- **Instruction Diversity**: Diversity in the instruction space (task coverage)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run quick experiment (subset of data, one model)
python run_all.py --mode quick

# Run full experiment
python run_all.py --mode full

# Run specific step
python run_all.py --only 2  # Only run selection step
```

## Configuration

Edit `config.yaml` to customize:
- Datasets to use
- Target subset sizes (default: 1000, 2500, 5000, 10000)
- Selection method weights
- Fine-tuning hyperparameters
- Evaluation benchmarks

## Datasets

### General Instruction
- LIMA (gold standard reference)
- UltraChat (large synthetic)
- OpenOrca (diverse instructions)

### Reasoning
- OpenMathInstruct-1 (math with solutions)

### Coding
- Code instruction datasets

### Safety
- BeaverTails (safety-focused)

## Output

Results are saved to `output/`:
- `finetuning_results.json`: Raw evaluation results
- `finetuning_report.md`: Markdown summary report
- `plots/`: Comparison visualizations

## Key Metrics

- **HellaSwag**: Common sense reasoning
- **ARC-Easy**: Science questions
- **WinoGrande**: Coreference resolution
- **PIQA**: Physical intuition

## References

- Zhou et al. (2023). "LIMA: Less Is More for Alignment"
- Wang et al. (2023). "How Far Can Camels Go? Exploring the State of Instruction Tuning"
