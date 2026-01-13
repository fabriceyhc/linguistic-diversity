# Makefile for linguistic-diversity project

.PHONY: help install install-dev test test-fast test-performance test-all coverage clean lint format type-check

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
	@echo "  make type-check       Run type checking with mypy"
	@echo "  make check-all        Run all code quality checks"
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
	python run_tests.py

test-fast: test

test-performance:
	python run_tests.py --performance --verbose

test-all:
	python run_tests.py --all --verbose

coverage:
	python run_tests.py --coverage
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

type-check:
	@echo "Type checking with mypy..."
	mypy src/

check-all: lint type-check
	@echo "✓ All code quality checks passed!"

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
