# Quick Start Guide

## Installation

```bash
cd /data2/fabricehc/linguistic-diversity
pip install -e ".[dev]"
```

This installs the package in editable mode with all development dependencies.

## Verify Installation

```bash
# Run tests
pytest tests/ -v -m "not slow"

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/

# Run formatting check
black --check src/ tests/
```

## Run Examples

```bash
# Basic usage example
python examples/basic_usage.py
```

Expected output:
```
============================================================
Linguistic Diversity - Basic Examples
============================================================

1. Token-Level Semantic Diversity
------------------------------------------------------------
High paraphrase corpus: 7.42
Low diversity corpus:   7.99
...
```

## Usage in Your Code

```python
from linguistic_diversity import TokenSemantics, DocumentSemantics

# Simple usage
corpus = [
    "The quick brown fox jumps",
    "A fast auburn fox leaps",
    "The cat sits quietly"
]

# Token-level semantic diversity
token_metric = TokenSemantics()
token_div = token_metric(corpus)
print(f"Token diversity: {token_div:.2f}")

# Document-level semantic diversity
doc_metric = DocumentSemantics()
doc_div = doc_metric(corpus)
print(f"Document diversity: {doc_div:.2f}")

# With custom configuration
config = {
    'model_name': 'bert-base-uncased',
    'batch_size': 32,
    'use_cuda': True,
    'remove_stopwords': True,
}
metric = TokenSemantics(config)
diversity = metric(corpus)
```

## Next Steps

1. **Read the docs**: Check out README.md for detailed documentation
2. **Explore examples**: See examples/ directory for more use cases
3. **Run slow tests**: `pytest tests/ -v -m slow` (downloads models)
4. **Contribute**: See CONTRIBUTING.md for guidelines

## Common Issues

### CUDA not available

If you get CUDA-related warnings but don't have a GPU:
```python
metric = TokenSemantics({'use_cuda': False})
```

### Model downloads

First run will download models. Subsequent runs use cached models:
```python
# First run: downloads bert-base-uncased (~440MB)
metric = TokenSemantics()

# Subsequent runs: uses cached model (fast!)
metric2 = TokenSemantics()  # Reuses cached model
```

### Import errors

If you get import errors, make sure you installed with:
```bash
pip install -e .
```

Not just:
```bash
pip install .
```

The `-e` flag installs in editable mode, which is needed for development.

## Performance Tips

1. **Use GPU**: Set `use_cuda=True` for 5-10x speedup
2. **Batch size**: Increase `batch_size` for faster processing on GPU
3. **Model caching**: Models are cached automatically
4. **PCA**: Use `n_components='auto'` for faster similarity computation on large corpora

## Help

- **Issues**: https://github.com/fabriceyhc/linguistic-diversity/issues
- **Original library**: https://github.com/fabriceyhc/TextDiversity
- **Documentation**: See README.md and docstrings
