# Contributing to Linguistic Diversity

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/linguistic-diversity.git
cd linguistic-diversity
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

We use the following tools to maintain code quality:

- **Black** (code formatting): `black src/ tests/`
- **Ruff** (linting): `ruff check src/ tests/`
- **MyPy** (type checking): `mypy src/`

All of these run automatically via pre-commit hooks.

### Type Hints

All new code should include type hints. We aim for full type coverage:

```python
from __future__ import annotations

def calculate_diversity(corpus: list[str], q: float = 1.0) -> float:
    """Calculate diversity score.

    Args:
        corpus: List of documents.
        q: Diversity order parameter.

    Returns:
        Diversity score.
    """
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def function(arg1: str, arg2: int) -> bool:
    """Brief description.

    Longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is wrong.
    """
    ...
```

## Testing

### Running Tests

```bash
# All tests (excluding slow ones)
pytest

# Include slow tests (downloads models, may take time)
pytest -m slow

# With coverage
pytest --cov=linguistic_diversity --cov-report=html

# Specific test file
pytest tests/test_metric.py
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common test data
- Mark slow tests with `@pytest.mark.slow`

Example:

```python
import pytest
from linguistic_diversity import TokenSemantics

@pytest.fixture
def sample_corpus():
    return ["text 1", "text 2"]

def test_basic_functionality(sample_corpus):
    metric = TokenSemantics()
    result = metric(sample_corpus)
    assert result > 0

@pytest.mark.slow
def test_large_corpus():
    # Tests that download models or take time
    ...
```

## Pull Request Process

1. Create a new branch for your feature:
```bash
git checkout -b feature/my-new-feature
```

2. Make your changes and commit:
```bash
git add .
git commit -m "Add feature: description"
```

3. Push to your fork:
```bash
git push origin feature/my-new-feature
```

4. Open a Pull Request on GitHub

### PR Checklist

- [ ] Code follows the style guide (passes black, ruff, mypy)
- [ ] Tests added for new functionality
- [ ] All tests pass locally
- [ ] Documentation updated (if needed)
- [ ] Type hints added to all new functions
- [ ] CHANGELOG.md updated (for significant changes)

## Adding New Metrics

To add a new diversity metric:

1. Create a new module in `src/linguistic_diversity/diversities/`
2. Subclass `TextDiversity`
3. Implement required abstract methods:
   - `extract_features()`
   - `calculate_similarities()`
   - `calculate_abundance()`
4. Add comprehensive tests
5. Update documentation
6. Add examples

Example structure:

```python
from linguistic_diversity.metric import TextDiversity

class MyDiversity(TextDiversity):
    """My new diversity metric."""

    def extract_features(self, corpus):
        # Extract features
        ...

    def calculate_similarities(self, features):
        # Compute similarity matrix
        ...

    def calculate_abundance(self, species):
        # Compute abundance distribution
        ...
```

## Performance Guidelines

- Use NumPy operations instead of Python loops when possible
- Leverage FAISS for similarity computations on large feature matrices
- Cache expensive operations (models, embeddings) when appropriate
- Use `@lru_cache` for pure functions
- Profile code with `line_profiler` for bottlenecks

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Contact the maintainers for sensitive issues

## Code of Conduct

Be respectful and inclusive. We're all here to learn and improve the library together.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
