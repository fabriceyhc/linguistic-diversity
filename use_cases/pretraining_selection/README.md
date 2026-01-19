# Pretraining Data Selection with Linguistic Diversity

This use case evaluates whether training data selected for high **linguistic diversity** (semantic or syntactic) yields better downstream performance in small language models compared to random selection.

## Objective

Experimentally determine if submodular optimization for linguistic diversity can improve pretraining data curation for language models.

**Research Question:** Does maximizing linguistic diversity in training data lead to better language model performance?

## Hypothesis

Training on linguistically diverse data will:
- ✅ Improve model generalization (lower perplexity on test data)
- ✅ Reduce repetitive generation patterns
- ✅ Outperform random data selection baseline

## Methodology

### Data Selection Regimes

We compare 4 data selection strategies:

1. **Semantic Diversity** - Maximize coverage of semantic space using Facility Location Selection on sentence embeddings
2. **Syntactic Diversity** - Maximize coverage of grammatical structures using Feature-based Selection on POS n-grams
3. **Composite Diversity** - Two-step selection: semantic first, then syntactic refinement
4. **Random Baseline** - Random sampling (control)

### Pipeline Stages

```
1. Data Download        → Download TinyStories & FineWeb-Edu datasets
2. Feature Extraction   → Extract semantic (embeddings) & syntactic (POS n-grams) features
3. Subset Selection     → Apply submodular optimization to select diverse 10% subsets
4. Model Training       → Train small GPT-2 models (10M params) on each subset
5. Evaluation          → Measure perplexity and generation quality
6. Report Generation   → Create comparative visualizations and analysis
```

## Quick Start

### Installation

```bash
# Navigate to the repository root
cd linguistic-diversity

# Install with pretraining dependencies
pip install -e ".[pretraining,viz]"

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Running the Pipeline

**Full pipeline (recommended for first run):**
```bash
cd use_cases/pretraining_selection
python run_pipeline.py
```

**Quick validation mode (100k samples, 10k training steps):**
- Default configuration in `config.yaml` is set to `mode: "quick"`
- Expected runtime: **2-4 hours** with multi-GPU setup

**Step-by-step execution:**
```bash
# Run individual steps
python processing/01_download_data.py
python processing/02_extract_features.py
python processing/03_select_subsets.py
python processing/04_train_models.py
python processing/05_evaluate_models.py
python processing/06_generate_report.py
```

**Advanced usage:**
```bash
# Skip completed steps automatically
python run_pipeline.py

# Force re-run all steps
python run_pipeline.py --force

# Run only specific step (e.g., training)
python run_pipeline.py --only-step 4

# Skip certain steps (e.g., skip download if data exists)
python run_pipeline.py --skip-steps 1,2
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Switch between quick validation and full experiment
mode: "quick"  # or "full"

# Adjust sample sizes
corpus:
  max_samples:
    quick: 100000    # Quick validation
    full: 1000000    # Full experiment

# Model configuration
training:
  model:
    n_layers: 6      # Transformer layers
    n_embed: 384     # Embedding dimension
    n_heads: 6       # Attention heads

  training:
    max_steps:
      quick: 10000   # Quick (~2-3 hours)
      full: 50000    # Full (~12-24 hours)
```

## Directory Structure

```
pretraining_selection/
├── input/                          # Raw corpus data
│   ├── tinystories_train.pkl
│   └── fineweb_edu_train.pkl
├── features/                       # Extracted features
│   ├── {dataset}_semantic_embeddings.npy
│   └── {dataset}_syntactic_features.npz
├── datasets/                       # Selected subsets
│   ├── semantic_diversity/
│   ├── syntactic_diversity/
│   ├── composite_diversity/
│   └── random_baseline/
├── models/                         # Trained models
│   ├── tinystories/
│   │   ├── semantic_diversity/
│   │   ├── syntactic_diversity/
│   │   ├── composite_diversity/
│   │   └── random_baseline/
│   └── fineweb-edu/
│       └── (same structure)
├── output/                         # Results and visualizations
│   ├── plots/
│   │   ├── training_curves_*.png
│   │   ├── perplexity_comparison.png
│   │   └── repetition_comparison.png
│   ├── report/
│   │   └── pretraining_selection_report.md
│   ├── evaluation_results.json
│   └── evaluation_summary.csv
├── processing/                     # Pipeline scripts
│   ├── 01_download_data.py
│   ├── 02_extract_features.py
│   ├── 03_select_subsets.py
│   ├── model_utils.py
│   ├── 04_train_models.py
│   ├── 05_evaluate_models.py
│   └── 06_generate_report.py
├── config.yaml                     # Configuration
├── run_pipeline.py                 # Master pipeline
└── README.md                       # This file
```

## Expected Runtime

**Quick Validation Mode (default):**
- Data download: 10-15 minutes
- Feature extraction: 15-20 minutes
- Subset selection: 20-30 minutes
- Model training: 60-90 minutes (with multi-GPU)
- Evaluation: 10-15 minutes
- **Total: 2-4 hours**

**Full Experiment Mode:**
- Model training: 8-12 hours (with multi-GPU)
- **Total: 12-24 hours**

## Dependencies

**Core:**
- `torch>=2.0.0` - PyTorch for model training
- `transformers>=4.35.0` - Hugging Face Transformers (GPT-2)
- `sentence-transformers>=2.3.0` - Sentence embeddings
- `spacy>=3.7.0` - Syntactic feature extraction
- `scikit-learn>=1.3.0` - Feature vectorization
- `apricot-select>=0.6.0` - Submodular optimization
- `datasets>=2.0.0` - Hugging Face datasets

**Visualization:**
- `matplotlib>=3.8.0`
- `seaborn>=0.13.0`
- `pandas>=2.0.0`

## Success Criteria

The experiment is considered **successful** if any diversity-based selection method shows:

✅ **Lower perplexity** than random baseline (≥2% improvement)
✅ **Lower repetition rate** in generated text
✅ **Consistent improvement** across both datasets

## Key Features

### 1. Scalable Selection Strategy

**Hierarchical Merge Approach:**
- Splits large corpus into manageable shards
- Performs selection on each shard
- Merges winners and performs final selection
- Enables submodular optimization on million-document corpora

### 2. Multi-Dataset Evaluation

- **TinyStories**: Simple, controlled narratives (~2M documents)
- **FineWeb-Edu**: Educational web content (more diverse and challenging)
- Cross-dataset validation of findings

### 3. Multi-GPU Support

- Automatic GPU detection
- DataParallel for efficient training
- Configurable batch sizes and gradient accumulation

### 4. Comprehensive Evaluation

- **Perplexity** on held-out test set
- **Generation quality** with multiple prompts
- **Repetition rate** analysis (n-gram overlap)
- Side-by-side comparison across all regimes

## Outputs

After running the pipeline, you'll find:

1. **Training curves** - Loss over time for each model
2. **Perplexity comparison** - Bar chart comparing test perplexity
3. **Repetition analysis** - Generation quality metrics
4. **Detailed report** - Markdown report with findings and sample generations
5. **Raw data** - JSON files with all evaluation metrics

## Interpreting Results

### Perplexity

- **Lower is better**
- Measures how well model predicts test data
- <50: Excellent for TinyStories
- <100: Good for FineWeb-Edu

### Repetition Rate

- **Lower is better**
- 0.0 = No repetition
- <0.1 = Acceptable
- >0.2 = Problematic (model collapse)

### Expected Outcomes

**If diversity helps:**
- Semantic/Syntactic/Composite models outperform Random baseline
- Lower perplexity and repetition rates
- Diverse training data → better generalization

**If diversity doesn't help:**
- Random baseline performs comparably
- Diversity-based selection adds complexity without benefit
- Other factors (data quality, quantity) may be more important

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size` in `config.yaml`
- Reduce `max_samples` for quick testing
- Use `gradient_accumulation_steps` to maintain effective batch size

### Slow Training

- Ensure GPU is being used (check `nvidia-smi`)
- Increase `batch_size` if you have memory headroom
- Use `mode: "quick"` for faster validation

### Missing Dependencies

```bash
# Install missing packages
pip install apricot-select datasets matplotlib seaborn

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Data Download Issues

- Check internet connection
- Verify Hugging Face datasets access
- Try manual download and place in `input/` directory

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{linguistic_diversity_pretraining_2026,
  title={Linguistic Diversity for Pretraining Data Selection},
  author={Harel-Canada, Fabrice},
  year={2026},
  howpublished={https://github.com/fabriceyhc/linguistic-diversity}
}
```

## Related Work

- **Submodular Optimization for NLP**: Mirzasoleiman et al. (2020)
- **Data Selection for LLMs**: Xie et al. (2023)
- **Linguistic Diversity**: Chao et al. (2014)

## Future Directions

- Test on larger models (100M+ parameters)
- Explore curriculum learning (easy → diverse data)
- Combine diversity with quality filtering
- Evaluate on downstream tasks (not just perplexity)
- Study optimal selection ratios (5%, 10%, 20%)

## Support

For questions or issues:
- 📧 Email: fabriceyhc@gmail.com
- 🐛 Issues: https://github.com/fabriceyhc/linguistic-diversity/issues
- 📖 Docs: See parent directory for framework documentation

---

**Last Updated:** 2026-01-14
