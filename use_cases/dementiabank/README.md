# DementiaBank Cognitive Impairment Detection

This use case evaluates whether linguistic diversity metrics can detect differences between cognitively impaired speech (Dementia) and healthy controls using the DementiaBank dataset.

## Objective

Determine if the linguistic-diversity framework provides statistically significant signal that distinguishes between:
- **Dementia group:** Transcripts from cognitively impaired individuals
- **Control group:** Transcripts from healthy age-matched controls

## Hypothesis

Cognitive impairment manifests in language as:
- **Reduced semantic diversity:** Repetitive loops, limited vocabulary
- **Reduced syntactic diversity:** Simpler grammar, less structural variation
- **Reduced morphological diversity:** Fewer grammatical patterns

## Dataset

**Source:** [MearaHe/dementiabank](https://huggingface.co/datasets/MearaHe/dementiabank) on Hugging Face

**Task:** Cookie Theft picture description
- Standard neuropsychological assessment task
- Elicits spontaneous speech describing a complex scene
- Allows direct comparison across subjects

## Directory Structure

```
dementiabank/
├── input/                         # Raw and preprocessed data
│   ├── dementiabank_raw.csv
│   ├── preprocessed_data.pkl
│   └── preprocessed_metadata.csv
├── processing/                    # Pipeline scripts
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   ├── 03_compute_metrics.py
│   ├── 04_analyze_results.py
│   ├── 05_create_plots.py
│   └── 06_generate_report.py
├── output/                        # Results
│   ├── scores/
│   │   ├── raw_scores.csv
│   │   ├── summary_stats.csv
│   │   ├── statistical_report.txt
│   │   └── error_log.csv
│   ├── plots/
│   │   ├── *_boxplot.png
│   │   └── all_metrics_comparison.png
│   └── report/
│       └── evaluation_report.md
├── config.yaml                    # Configuration
├── run_evaluation.py              # Master pipeline
└── README.md                      # This file
```

## Usage

### Quick Start

```bash
cd use_cases/dementiabank
python run_evaluation.py
```

This runs the complete evaluation pipeline:
1. Downloads DementiaBank dataset from Hugging Face
2. Filters Cookie Theft tasks and balances classes
3. Cleans transcripts (removes artifacts)
4. Computes all linguistic diversity metrics
5. Performs statistical tests (t-tests, effect sizes)
6. Generates visualizations (boxplots, distributions)
7. Creates final evaluation report

### Step-by-Step Execution

```bash
# 1. Download data
python processing/01_download_data.py

# 2. Preprocess
python processing/02_preprocess.py

# 3. Compute metrics (takes ~60-90 min with GPU)
python processing/03_compute_metrics.py

# 4. Statistical analysis
python processing/04_analyze_results.py

# 5. Create visualizations
python processing/05_create_plots.py

# 6. Generate report
python processing/06_generate_report.py
```

## Configuration

Edit `config.yaml` to customize:
- Sample size (use all data or subset)
- Metrics to compute
- GPU settings
- Statistical thresholds

## Metrics Evaluated

### Semantic Diversity
- **Document-level:** Sentence embeddings (all-MiniLM-L6-v2)
- **Token-level:** Contextualized word embeddings (BERT)
- Hypothesis: Lower in Dementia (repetitive content)

### Syntactic Diversity
- **Dependency parsing:** Grammatical structure
- **Constituency parsing:** Phrase structure
- Hypothesis: Lower in Dementia (simpler grammar)

### Morphological Diversity
- **POS sequences:** Part-of-speech patterns
- Hypothesis: Lower in Dementia (fewer grammatical variations)

### Phonological Diversity
- **Rhythmic:** Syllable stress and weight patterns
- **Phonemic:** Phoneme sequences
- Hypothesis: May differ in Dementia (prosody changes)

### Lexical Diversity
- **Type-Token Ratio:** Unique words / total words
- Baseline comparison metric

## Success Criteria

The framework is considered **useful** if:

✓ Semantic OR syntactic diversity shows:
  - Dementia group mean < Control group mean
  - p-value < 0.05 (statistically significant)
  - |Cohen's d| > 0.3 (meaningful effect size)

✓ At least 2 metrics show significant differences

✓ Visual separation in boxplots

## Expected Results

**If successful:**
- Semantic diversity: 15-30% lower in Dementia
- Syntactic diversity: 10-25% lower in Dementia
- Clear separation in visualizations
- **Conclusion:** Framework is useful for cognitive impairment detection

**If unsuccessful:**
- Small/no differences (<10%)
- p-values > 0.05
- Overlapping distributions
- **Conclusion:** Metrics don't capture cognitive decline signal

## Output Files

### Scores
- `raw_scores.csv`: Per-subject diversity scores for all metrics
- `summary_stats.csv`: Group means, std, p-values, effect sizes
- `statistical_report.txt`: Detailed statistical analysis
- `error_log.csv`: Any computation failures

### Plots
- `{metric}_boxplot.png`: Per-metric comparison (8 plots)
- `all_metrics_comparison.png`: Summary comparison

### Report
- `evaluation_report.md`: Complete evaluation report with conclusions

## Dependencies

```bash
# Install required packages
pip install datasets scipy matplotlib seaborn
pip install -e "../../.[viz]"  # Install main package with viz extras
```

## Timeline

- Download: 5-10 minutes
- Preprocessing: 10-15 minutes
- Metric computation: 60-90 minutes (with 8x A100 GPUs)
- Analysis + Viz: 10-15 minutes
- **Total: ~90-120 minutes**

## Notes

- All data stored locally in `input/` (not tracked in git)
- Results in `output/` can be committed for reference
- GPU acceleration dramatically speeds up semantic metrics
- First run will download Hugging Face dataset and model weights
- Subsequent runs reuse cached data

## Related Use Cases

See `use_cases/` directory for other applications:
- *(Add future use cases here)*
