#!/usr/bin/env python3
"""Master pipeline for pretraining data selection experiment.

This script orchestrates the complete evaluation pipeline:
1. Download and preprocess corpus data
2. Extract semantic and syntactic features
3. Select diverse subsets using submodular optimization
4. Train language models on each subset
5. Evaluate model performance
6. Generate comparative report with visualizations

Usage:
    python run_pipeline.py [--skip-steps STEPS] [--only-step STEP]

Examples:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --skip-steps 1,2   # Skip data download and feature extraction
    python run_pipeline.py --only-step 4      # Run only model training
"""

import sys
from pathlib import Path
import subprocess
import time
import argparse
from datetime import datetime, timedelta

# Add parent directories to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import os

# Set environment variables
os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)


class PipelineStep:
    """Represents a single pipeline step."""

    def __init__(self, number, name, script_name, output_indicators):
        """
        Args:
            number: Step number
            name: Step name
            script_name: Name of Python script to run
            output_indicators: List of file paths that indicate step completion
        """
        self.number = number
        self.name = name
        self.script_name = script_name
        self.output_indicators = output_indicators

    def is_completed(self, base_dir):
        """Check if step is already completed."""
        for indicator in self.output_indicators:
            path = base_dir / indicator
            if not path.exists():
                return False
        return True


def create_pipeline_steps():
    """Create pipeline step definitions."""
    return [
        PipelineStep(
            1,
            "Download and preprocess data",
            "processing/01_download_data.py",
            ["input/roneneldan_tinystories_train.pkl", "input/huggingfacefw_fineweb_edu_train.pkl"]
        ),
        PipelineStep(
            2,
            "Extract features",
            "processing/02_extract_features.py",
            ["features/roneneldan_tinystories_semantic_embeddings.npy",
             "features/huggingfacefw_fineweb_edu_semantic_embeddings.npy"]
        ),
        PipelineStep(
            3,
            "Select diverse subsets",
            "processing/03_select_subsets.py",
            ["datasets/semantic_diversity/roneneldan_tinystories/corpus.jsonl",
             "datasets/random_baseline/roneneldan_tinystories/corpus.jsonl"]
        ),
        PipelineStep(
            4,
            "Train models",
            "processing/04_train_models.py",
            ["models/training_results.json"]
        ),
        PipelineStep(
            5,
            "Evaluate models",
            "processing/05_evaluate_models.py",
            ["output/evaluation_results.json"]
        ),
        PipelineStep(
            6,
            "Generate report",
            "processing/06_generate_report.py",
            ["output/report/pretraining_selection_report.md"]
        ),
    ]


def run_step(step, base_dir, verbose=True):
    """Run a single pipeline step.

    Args:
        step: PipelineStep to run
        base_dir: Base directory path
        verbose: Whether to print verbose output

    Returns:
        tuple: (success, duration_seconds)
    """
    script_path = base_dir / step.script_name

    if not script_path.exists():
        print(f"   ✗ Script not found: {script_path}")
        return False, 0

    print(f"\n{'=' * 80}")
    print(f"STEP {step.number}: {step.name.upper()}")
    print(f"{'=' * 80}")

    start_time = time.time()

    try:
        # Run script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(base_dir),
            capture_output=not verbose,
            text=True,
            check=True
        )

        duration = time.time() - start_time

        print(f"\n✓ Step {step.number} completed in {duration:.1f} seconds")

        return True, duration

    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time

        print(f"\n✗ Step {step.number} failed after {duration:.1f} seconds")
        if not verbose and e.stdout:
            print(f"\nOutput:\n{e.stdout}")
        if e.stderr:
            print(f"\nError:\n{e.stderr}")

        return False, duration


def print_status(steps, base_dir):
    """Print pipeline status."""
    print("\nPipeline Status:")
    print("-" * 80)

    for step in steps:
        completed = step.is_completed(base_dir)
        status = "✓ Complete" if completed else "○ Pending"
        print(f"  Step {step.number}: {step.name:40s} {status}")

    print("-" * 80)


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run pretraining selection pipeline')
    parser.add_argument('--skip-steps', type=str, help='Comma-separated list of steps to skip (e.g., "1,2")')
    parser.add_argument('--only-step', type=int, help='Run only this step (1-6)')
    parser.add_argument('--force', action='store_true', help='Force re-run even if outputs exist')
    parser.add_argument('--quiet', action='store_true', help='Minimize output')
    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 80)
    print(" " * 20 + "PRETRAINING DATA SELECTION PIPELINE")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    base_dir = Path(__file__).parent
    steps = create_pipeline_steps()

    # Determine which steps to run
    if args.only_step:
        steps_to_run = [s for s in steps if s.number == args.only_step]
        if not steps_to_run:
            print(f"\n✗ Invalid step number: {args.only_step}")
            return 1
    elif args.skip_steps:
        skip_numbers = [int(n.strip()) for n in args.skip_steps.split(',')]
        steps_to_run = [s for s in steps if s.number not in skip_numbers]
    else:
        steps_to_run = steps

    # Print initial status
    if not args.only_step:
        print_status(steps, base_dir)

    # Run pipeline
    print(f"\n{len(steps_to_run)} steps to run")

    results = []
    total_start = time.time()

    for step in steps_to_run:
        # Check if already completed
        if not args.force and step.is_completed(base_dir):
            print(f"\n{'=' * 80}")
            print(f"STEP {step.number}: {step.name.upper()}")
            print(f"{'=' * 80}")
            print(f"⏭ Skipping - outputs already exist")
            print(f"   (use --force to re-run)")
            continue

        # Run step
        success, duration = run_step(step, base_dir, verbose=not args.quiet)
        results.append({
            'step': step.number,
            'name': step.name,
            'success': success,
            'duration': duration
        })

        # Stop if failed
        if not success:
            print(f"\n{'=' * 80}")
            print("✗ PIPELINE FAILED")
            print(f"{'=' * 80}")
            return 1

    # Print summary
    total_duration = time.time() - total_start

    print(f"\n{'=' * 80}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 80}")

    for result in results:
        status = "✓" if result['success'] else "✗"
        duration_str = str(timedelta(seconds=int(result['duration'])))
        print(f"  {status} Step {result['step']}: {result['name']:40s} ({duration_str})")

    print(f"\nTotal time: {timedelta(seconds=int(total_duration))}")

    # Final status
    if not args.only_step:
        print_status(steps, base_dir)

    # Success message
    if all(r['success'] for r in results):
        print(f"\n{'=' * 80}")
        print("✓ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'=' * 80}")

        # Point to report
        report_file = base_dir / "output" / "report" / "pretraining_selection_report.md"
        if report_file.exists():
            print(f"\n📊 Report: {report_file}")

        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
