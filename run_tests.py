#!/usr/bin/env python3
"""Unified test runner for the linguistic-diversity library.

This script provides a convenient interface to run different types of tests:
- Unit tests: Fast tests for basic functionality
- Performance tests: Benchmark and performance regression tests
- All tests: Complete test suite
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return the exit code."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run tests for linguistic-diversity library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run unit tests only (fast)
  %(prog)s --performance      # Run performance benchmarks
  %(prog)s --all              # Run all tests
  %(prog)s --coverage         # Run with coverage report
  %(prog)s --verbose          # Run with verbose output
        """
    )

    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance and benchmark tests (slow)"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests including slow ones"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        metavar="N",
        help="Run tests in parallel with N workers (requires pytest-xdist)"
    )

    parser.add_argument(
        "--markers",
        "-m",
        type=str,
        help="Run tests matching given mark expression (e.g., 'not slow')"
    )

    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional arguments to pass to pytest"
    )

    args = parser.parse_args()

    # Build pytest command
    cmd = ["pytest"]

    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
        cmd.append("-s")  # Don't capture output
    else:
        cmd.append("-v")

    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])

    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=linguistic_diversity",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-report=xml"
        ])

    # Determine which tests to run
    if args.performance:
        description = "Running Performance Benchmarks"
        cmd.append("tests/test_performance.py")
        cmd.extend(["-m", "slow"])
    elif args.all:
        description = "Running All Tests (including slow tests)"
        cmd.append("tests/")
    else:
        description = "Running Unit Tests (excluding slow tests)"
        cmd.append("tests/")
        if not args.markers:
            cmd.extend(["-m", "not slow"])

    # Add custom markers if specified
    if args.markers:
        cmd.extend(["-m", args.markers])

    # Add any additional pytest arguments
    if args.pytest_args:
        cmd.extend(args.pytest_args)

    # Run the tests
    exit_code = run_command(cmd, description)

    # Print summary
    print(f"\n{'='*60}")
    if exit_code == 0:
        print("✓ All tests passed!")
        if args.coverage:
            print("\nCoverage report generated:")
            print("  - Terminal: (shown above)")
            print("  - HTML: htmlcov/index.html")
            print("  - XML: coverage.xml")
    else:
        print("✗ Some tests failed!")

    print(f"{'='*60}\n")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
