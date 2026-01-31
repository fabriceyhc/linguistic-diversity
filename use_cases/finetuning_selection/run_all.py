#!/usr/bin/env python3
"""
Run the complete instruction fine-tuning data selection pipeline.

This script orchestrates:
1. Feature extraction from instruction datasets
2. Diversity-based subset selection
3. Fine-tuning with LoRA
4. Evaluation and report generation

Usage:
    python run_all.py              # Run in quick mode (default)
    python run_all.py --mode full  # Run full experiment
    python run_all.py --step 2     # Start from step 2 (selection)
    python run_all.py --force      # Force re-run even if outputs exist
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(step_num: int, script_name: str, description: str):
    """Run a processing step."""
    print(f"\n{'=' * 70}")
    print(f"STEP {step_num}: {description}")
    print(f"{'=' * 70}\n")

    script_path = Path(__file__).parent / "processing" / script_name

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(Path(__file__).parent),
    )

    if result.returncode != 0:
        print(f"\nERROR: Step {step_num} failed with return code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\nStep {step_num} completed successfully.")


def clear_outputs(base_dir: Path, step: int = None):
    """Clear output directories for fresh run."""
    dirs_to_clear = {
        1: ["datasets"],
        2: ["selections"],
        3: ["output"],
        4: [],  # Report doesn't need clearing
    }

    if step:
        dirs = dirs_to_clear.get(step, [])
    else:
        dirs = ["datasets", "selections", "output"]

    for dir_name in dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"Clearing {dir_path}...")
            shutil.rmtree(dir_path)
            dir_path.mkdir()


def main():
    parser = argparse.ArgumentParser(
        description="Run the instruction fine-tuning data selection pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="Run mode: quick (fast iteration) or full (complete experiment)"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Start from this step (1=extract, 2=select, 3=finetune, 4=report)"
    )
    parser.add_argument(
        "--only",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run only this step"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run by clearing existing outputs"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # Handle force flag
    if args.force:
        print("\n--force specified: Clearing existing outputs...")
        if args.only:
            clear_outputs(base_dir, args.only)
        elif args.step > 1:
            # Clear from starting step onwards
            for s in range(args.step, 5):
                clear_outputs(base_dir, s)
        else:
            clear_outputs(base_dir)

    # Update config mode if specified
    if args.mode:
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            config['mode'] = args.mode
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"Set mode to: {args.mode}")

    print("\n" + "=" * 70)
    print("INSTRUCTION FINE-TUNING DATA SELECTION PIPELINE")
    print("=" * 70)
    print(f"\nMode: {args.mode}")
    print(f"Starting from step: {args.step if not args.only else args.only}")
    if args.only:
        print(f"Running only step: {args.only}")

    steps = [
        (1, "01_extract_features.py", "EXTRACT FEATURES FROM INSTRUCTION DATASETS"),
        (2, "02_select_subsets.py", "SELECT DIVERSE SUBSETS"),
        (3, "03_finetune_evaluate.py", "FINE-TUNE AND EVALUATE"),
        (4, "04_generate_report.py", "GENERATE REPORT"),
    ]

    if args.only:
        # Run only the specified step
        for step_num, script_name, description in steps:
            if step_num == args.only:
                run_step(step_num, script_name, description)
                break
    else:
        # Run from starting step to end
        for step_num, script_name, description in steps:
            if step_num >= args.step:
                run_step(step_num, script_name, description)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {Path(__file__).parent / 'output'}")
    print("Check output/finetuning_report.md for the summary report.")


if __name__ == "__main__":
    main()
