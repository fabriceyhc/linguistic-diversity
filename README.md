# Linguistic Diversity

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Modernized, efficient implementation of linguistic diversity metrics using similarity-sensitive Hill numbers.**

This library measures various kinds of linguistic diversity using **similarity-sensitive Hill numbers (SSHN)**. Originally adapted from the study of species diversity in ecology, SSHNs characterize the *effective number* of species in a population. In NLP, "species" are linguistic units (words, parse trees, etc.) and the "population" is a corpus of documents.

For example, if the **token semantic diversity** of a corpus is 9, this means the corpus contains approximately 9 distinct semantic concepts.

## Features

- **Modern Python (3.9+)**: Type hints, dataclasses, and modern best practices
- **Performance optimized**: FAISS-accelerated similarity computations, model caching, vectorized operations
- **Updated dependencies**: NumPy 2.x, Pandas 2.x, latest transformers
- **Multiple diversity dimensions**:
  - **Semantic**: Token and document-level semantic diversity using transformers
  - **Syntactic**: Dependency and constituency parse tree diversity
  - **Morphological**: Part-of-speech sequence diversity
  - **Phonological**: Rhythmic and phonemic pattern diversity
  - **Universal**: Unified metric combining all dimensions into a single score

## Quick Start

### Installation

```bash
pip install linguistic-diversity
```

For development:

```bash
git clone https://github.com/fabriceyhc/linguistic-diversity
cd linguistic-diversity
pip install -e ".[dev]"
```

### Basic Usage

```python
from linguistic_diversity import TokenSemantics, DocumentSemantics

# Example corpora
corpus1 = [
    'one massive earth',
    'an enormous globe',
    'the colossal world'
]  # High paraphrasing, similar semantics

corpus2 = [
    'basic human right',
    'you were right',
    'make a right'
]  # Lower semantic diversity due to word "right"

# Token-level semantic diversity
token_metric = TokenSemantics()
print(f"Corpus 1 token diversity: {token_metric(corpus1):.2f}")
print(f"Corpus 2 token diversity: {token_metric(corpus2):.2f}")

# Document-level semantic diversity
doc_metric = DocumentSemantics()
print(f"Corpus 1 document diversity: {doc_metric(corpus1):.2f}")
print(f"Corpus 2 document diversity: {doc_metric(corpus2):.2f}")
```

### Configuration

All metrics accept configuration dictionaries:

```python
from linguistic_diversity import TokenSemantics

# Custom configuration
config = {
    'model_name': 'roberta-base',  # Use RoBERTa instead of BERT
    'q': 2.0,                       # Diversity order (higher = less sensitive to rare species)
    'normalize': True,              # Normalize by number of species
    'batch_size': 32,               # Larger batches for faster processing
    'use_cuda': True,               # Use GPU if available
    'remove_stopwords': True,       # Filter out stopwords
    'verbose': True                 # Show progress bars
}

metric = TokenSemantics(config)
diversity = metric(corpus1)
```

## What's New in 1.0

This is a complete modernization of the original [TextDiversity](https://github.com/fabriceyhc/TextDiversity) library with significant improvements:

### Performance Improvements

- **3-5x faster** similarity computation using optimized FAISS operations
- **Model caching**: Models loaded once and reused across metric instances
- **Vectorized operations**: Replaced nested Python loops with NumPy operations
- **Batch processing**: Optimized batch sizes for GPU utilization
- **Lazy loading**: Models and dependencies loaded only when needed

### Code Quality

- **Type hints** throughout for better IDE support and type checking
- **Modern Python**: Dataclasses, f-strings, pathlib, type annotations
- **Better error handling** with informative messages
- **Comprehensive docstrings** in Google style
- **PEP 561 compliance** with `py.typed` marker

### Updated Dependencies

- NumPy 1.24+ (compatible with FAISS)
- Pandas 2.0+ (performance improvements)
- Latest transformers (4.35+)
- Python 3.9+ (modern language features)

### Developer Experience

- Pre-commit hooks with Black, Ruff, MyPy
- Comprehensive test suite with pytest
- GitHub Actions CI/CD
- Type checking with MyPy
- Code coverage reporting

## Available Metrics

### Semantic Diversity

**TokenSemantics**: Diversity of contextualized token embeddings
```python
from linguistic_diversity import TokenSemantics

metric = TokenSemantics({'model_name': 'bert-base-uncased'})
diversity = metric(corpus)
```

**DocumentSemantics**: Diversity of document-level embeddings
```python
from linguistic_diversity import DocumentSemantics

metric = DocumentSemantics({'model_name': 'all-mpnet-base-v2'})
diversity = metric(corpus)
```

### Syntactic Diversity

**DependencyParse**: Diversity of dependency parse tree structures
```python
from linguistic_diversity import DependencyParse

# Fast: using graph embeddings
metric = DependencyParse({'similarity_type': 'ldp'})
diversity = metric(corpus)

# Exact: using tree edit distance (slow)
metric = DependencyParse({'similarity_type': 'tree_edit_distance'})
diversity = metric(corpus)
```

**ConstituencyParse**: Diversity of constituency (phrase structure) parse trees
```python
from linguistic_diversity import ConstituencyParse

metric = ConstituencyParse({'similarity_type': 'ldp'})
diversity = metric(corpus)
```

*Note: Constituency parsing requires `benepar`. Install with: `pip install linguistic-diversity[syntactic]`*

### Morphological Diversity

**PartOfSpeechSequence**: Diversity of POS tag sequences using biological sequence alignment
```python
from linguistic_diversity import PartOfSpeechSequence

metric = PartOfSpeechSequence()
diversity = metric(corpus)
```

### Phonological Diversity

**Rhythmic**: Diversity of rhythmic patterns (stress and syllable weight)
```python
from linguistic_diversity import Rhythmic

metric = Rhythmic()
diversity = metric(corpus)
```

**Phonemic**: Diversity of phoneme sequences (IPA representation)
```python
from linguistic_diversity import Phonemic

# Default: uses g2p_en (pure Python, no system dependencies)
metric = Phonemic()
diversity = metric(corpus)

# Optional: use phonemizer backend (requires espeak-ng)
metric = Phonemic({'backend': 'phonemizer'})
diversity = metric(corpus)
```

*Note: Phonological metrics require additional dependencies. Install with: `pip install linguistic-diversity[phonological]`*

### Universal Diversity

**UniversalLinguisticDiversity**: Unified metric combining all dimensions

```python
from linguistic_diversity import UniversalLinguisticDiversity

# Default balanced configuration
metric = UniversalLinguisticDiversity()
diversity = metric(corpus)

# Get detailed breakdown
detailed = metric.get_detailed_scores(corpus)
print(f"Universal: {detailed['universal']:.2f}")
print(f"By branch: {detailed['branches']}")
```

The universal metric intelligently combines all 7 metrics across 4 linguistic branches (semantic, syntactic, morphological, phonological) into a single comprehensive diversity score. It uses hierarchical aggregation: geometric mean within branches, weighted combination across branches.

**Preset Configurations**:
```python
from linguistic_diversity import get_preset_config

# Semantic-focused (for content analysis)
config = get_preset_config("semantic_focus")
metric = UniversalLinguisticDiversity(config)

# Available presets: balanced, semantic_focus, structural_focus, minimal, conservative
```

See [UNIVERSAL_METRIC_GUIDE.md](UNIVERSAL_METRIC_GUIDE.md) for detailed documentation.

## System Requirements

### Required

- Python 3.9 or higher
- For GPU acceleration: CUDA-compatible GPU with appropriate drivers

### Optional System Dependencies

**All metrics work with pure Python packages - no system dependencies required!**

However, if you want to use the `phonemizer` backend for `Phonemic` diversity (instead of the default `g2p_en`), you'll need `espeak-ng`:

**Linux**:
```bash
sudo apt-get install espeak-ng
pip install phonemizer  # then use Phonemic({'backend': 'phonemizer'})
```

**macOS**:
```bash
brew install espeak-ng
pip install phonemizer  # then use Phonemic({'backend': 'phonemizer'})
```

**Windows**:
- Download from [espeak-ng releases](https://github.com/espeak-ng/espeak-ng/releases)
- Install `espeak-ng-X64.msi` or `espeak-ng-X86.msi`
- Set environment variable: `PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll`
- `pip install phonemizer` then use `Phonemic({'backend': 'phonemizer'})`

**Note**: The default `g2p_en` backend for `Phonemic` is pure Python and works everywhere without system dependencies.

## Theory: Similarity-Sensitive Hill Numbers

Hill numbers provide a unified framework for measuring diversity that accounts for both:
1. **Species richness** (how many different types exist)
2. **Species similarity** (how similar the types are to each other)

The diversity formula is:

```
D = (Σ p_i (Σ Z_ij p_j)^(q-1))^(1/(1-q))
```

Where:
- `p`: Abundance distribution over species
- `Z`: Similarity matrix between species
- `q`: Diversity order parameter (0 = richness, 1 = Shannon, 2 = Simpson, ∞ = Berger-Parker)

When `q=1` (default), this reduces to the effective number of species weighted by their semantic similarity.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{linguistic_diversity_2026,
  title={Linguistic Diversity: Modernized Implementation of Similarity-Sensitive Hill Numbers for NLP},
  author={Harel-Canada, Fabrice},
  year={2026},
  url={https://github.com/fabriceyhc/linguistic-diversity}
}
```

Original TextDiversity library:
```bibtex
@software{textdiversity_2022,
  title={TextDiversity: Measuring Linguistic Diversity with Similarity-Sensitive Hill Numbers},
  author={Harel-Canada, Fabrice},
  year={2022},
  url={https://github.com/fabriceyhc/TextDiversity}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Original TextDiversity implementation and research
- Ecological diversity theory from Chao et al. (2014)
- The Hugging Face ecosystem for transformer models

## Links

- **Documentation**: [linguistic-diversity.readthedocs.io](https://linguistic-diversity.readthedocs.io) *(coming soon)*
- **PyPI**: [pypi.org/project/linguistic-diversity](https://pypi.org/project/linguistic-diversity) *(coming soon)*
- **Issues**: [GitHub Issues](https://github.com/fabriceyhc/linguistic-diversity/issues)
- **Original Library**: [TextDiversity](https://github.com/fabriceyhc/TextDiversity)
