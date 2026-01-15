# Linguistic Diversity Use Cases

This directory contains complete evaluation pipelines and applications of the linguistic-diversity framework for various tasks and domains.

## Available Use Cases

### 1. DementiaBank - Cognitive Impairment Detection

**Directory:** `dementiabank/`

**Objective:** Evaluate whether linguistic diversity metrics can distinguish between cognitively impaired speech (Dementia) and healthy controls.

**Dataset:** MearaHe/dementiabank from Hugging Face

**Status:** ✅ Complete implementation

**Key Features:**
- Automated data download and preprocessing
- Comprehensive metric computation (semantic, syntactic, morphological, phonological, lexical)
- Statistical analysis with effect sizes
- Publication-quality visualizations
- Automated report generation

**Quick Start:**
```bash
cd dementiabank
python run_evaluation.py
```

**Expected Runtime:** ~90-120 minutes (with GPU acceleration)

**Success Criteria:** Framework is considered useful if semantic OR syntactic diversity shows statistically significant differences (p < 0.05) with meaningful effect sizes (|d| > 0.3).

**Documentation:** See `dementiabank/README.md` for detailed documentation.

---

## Adding New Use Cases

To add a new use case evaluation:

1. **Create directory structure:**
   ```bash
   mkdir -p use_cases/your_use_case/{input,processing,output/{scores,plots,report}}
   ```

2. **Implement pipeline scripts:**
   - `processing/01_download_data.py` - Data acquisition
   - `processing/02_preprocess.py` - Data cleaning and preparation
   - `processing/03_compute_metrics.py` - Metric computation
   - `processing/04_analyze_results.py` - Statistical analysis
   - `processing/05_create_plots.py` - Visualizations
   - `processing/06_generate_report.py` - Report generation

3. **Create configuration:**
   - `config.yaml` - Pipeline configuration
   - `README.md` - Documentation

4. **Implement master pipeline:**
   - `run_evaluation.py` - Orchestrates all steps

5. **Follow the template:**
   - Use `dementiabank/` as a reference implementation
   - Adapt scripts to your specific dataset and objectives
   - Maintain consistent output formats for comparability

---

## Use Case Template

### Recommended Directory Structure

```
use_cases/
└── your_use_case/
    ├── input/                      # Raw and preprocessed data
    ├── processing/                 # Pipeline scripts (01-06)
    ├── output/
    │   ├── scores/                # CSV files with results
    │   ├── plots/                 # Visualizations
    │   └── report/                # Final evaluation report
    ├── config.yaml                # Configuration
    ├── run_evaluation.py          # Master pipeline
    └── README.md                  # Documentation
```

### Key Components

**1. Data Acquisition**
- Download or load your dataset
- Export to standardized format
- Document data source and structure

**2. Preprocessing**
- Clean text (remove artifacts, normalize)
- Segment into appropriate units (sentences, utterances, etc.)
- Apply quality filters
- Balance classes if needed

**3. Metric Computation**
- Initialize relevant diversity metrics
- Compute for each sample/subject
- Handle errors gracefully
- Save raw scores

**4. Statistical Analysis**
- Compute descriptive statistics
- Perform appropriate tests (t-test, ANOVA, etc.)
- Calculate effect sizes
- Evaluate success criteria

**5. Visualization**
- Create comparison plots (boxplots, violins)
- Show distributions and significance
- Generate summary figures
- Make plots publication-ready

**6. Reporting**
- Synthesize findings
- Provide actionable conclusions
- Document limitations
- Suggest next steps

---

## Best Practices

### Code Organization
- ✅ Keep scripts modular and independent
- ✅ Use consistent naming (01_, 02_, etc.)
- ✅ Save intermediate results
- ✅ Log errors and warnings

### Configuration
- ✅ Use YAML for configuration
- ✅ Document all parameters
- ✅ Provide sensible defaults
- ✅ Allow command-line overrides

### Documentation
- ✅ Write clear READMEs
- ✅ Include usage examples
- ✅ Document dependencies
- ✅ Provide expected outputs

### Testing
- ✅ Test on small data subset first
- ✅ Validate statistical assumptions
- ✅ Check for edge cases
- ✅ Verify output formats

### Reproducibility
- ✅ Set random seeds
- ✅ Document environment (Python version, package versions)
- ✅ Save configuration with results
- ✅ Include dataset citations

---

## Future Use Cases

Potential applications of the linguistic diversity framework:

### Healthcare & Clinical
- **Depression Detection:** Analyze speech patterns in depression
- **Autism Spectrum:** Evaluate linguistic diversity in ASD communication
- **Language Disorders:** Assess aphasia, dyslexia, and other disorders
- **Therapy Monitoring:** Track progress in speech therapy

### Education
- **Writing Assessment:** Evaluate essay diversity and sophistication
- **Language Learning:** Track L2 proficiency development
- **Reading Comprehension:** Analyze text complexity
- **Academic Writing:** Compare across disciplines

### Social Sciences
- **Authorship Attribution:** Distinguish authors by linguistic fingerprints
- **Social Media Analysis:** Measure diversity in online discourse
- **Political Speech:** Analyze campaign rhetoric and debates
- **Historical Linguistics:** Track language evolution over time

### Industry Applications
- **Content Generation:** Evaluate AI-generated text diversity
- **Customer Service:** Assess chatbot response variety
- **Product Reviews:** Analyze review authenticity
- **Content Moderation:** Detect spam and bot-generated content

### Research Methods
- **Survey Responses:** Measure open-ended response diversity
- **Interview Analysis:** Quantify qualitative data patterns
- **Corpus Linguistics:** Large-scale text collection analysis
- **Discourse Analysis:** Study conversation dynamics

---

## Contributing

To contribute a new use case:

1. Fork the repository
2. Create your use case following the template
3. Test thoroughly with multiple datasets
4. Document clearly (README, comments, docstrings)
5. Submit a pull request with:
   - Use case description
   - Example results
   - Performance benchmarks
   - Limitations and caveats

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{linguistic_diversity,
  title={Linguistic Diversity: Similarity-Sensitive Hill Numbers for Text Analysis},
  author={Harel-Canada, Fabrice},
  year={2024},
  url={https://github.com/fabriceyhc/linguistic-diversity}
}
```

---

## Support

For questions or issues:
- 📧 Email: fabriceyhc@gmail.com
- 🐛 Issues: https://github.com/fabriceyhc/linguistic-diversity/issues
- 📖 Docs: See individual use case READMEs

---

**Last Updated:** 2026-01-13
