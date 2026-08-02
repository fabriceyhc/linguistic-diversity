# Makefile for linguistic-diversity project

.PHONY: help install install-dev test test-fast test-performance test-all coverage clean lint format format-check type-check check-all

help:
	@echo "Linguistic Diversity - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install package in development mode"
	@echo "  make install-dev      Install with all development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run unit tests (fast, excludes slow tests)"
	@echo "  make test-fast        Same as 'make test'"
	@echo "  make test-performance Run performance benchmarks (slow)"
	@echo "  make test-all         Run all tests including slow ones"
	@echo "  make coverage         Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linting checks (ruff)"
	@echo "  make format           Format code with black"
	@echo "  make format-check     Verify formatting without changing files"
	@echo "  make type-check       Run type checking with mypy (advisory)"
	@echo "  make check-all        Run everything CI runs"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove build artifacts and cache files"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	python scripts/run_tests.py

test-fast: test

test-performance:
	python scripts/run_tests.py --performance --verbose

test-all:
	python scripts/run_tests.py --all --verbose

coverage:
	python scripts/run_tests.py --coverage
	@echo ""
	@echo "Coverage report generated:"
	@echo "  - HTML: htmlcov/index.html"
	@echo "  - XML: coverage.xml"

# Code quality
lint:
	@echo "Running ruff..."
	ruff check src/ tests/

format:
	@echo "Formatting with black..."
	black src/ tests/ examples/
	@echo "Sorting imports with ruff..."
	ruff check --select I --fix src/ tests/

format-check:
	@echo "Checking formatting with black..."
	black --check src/ tests/

type-check:
	@echo "Type checking with mypy (advisory)..."
	-mypy src/

# Mirrors the CI pipeline: lint, formatting, types, then the test suite.
check-all: lint format-check type-check test
	@echo "✓ All checks passed!"

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/
	@echo "Cleaning Python cache..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	@echo "✓ Cleanup complete!"

# Development workflow
dev-setup: install-dev
	@echo "Installing pre-commit hooks..."
	pre-commit install
	@echo "✓ Development environment ready!"

# Quick check before commit
pre-commit: format lint type-check test-fast
	@echo "✓ Pre-commit checks passed!"
