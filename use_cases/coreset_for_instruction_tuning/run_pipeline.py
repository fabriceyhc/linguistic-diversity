#!/usr/bin/env python3
"""
Coreset Selection for Instruction Tuning: Main Pipeline

This script orchestrates the complete experiment pipeline:

Phase 2 (Pilot - Alpaca-GPT4):
1. Data Preparation: Download dataset, generate embeddings
2. Subset Selection: Random + Diversity-guided selection
3. Fine-tuning: Train LoRA adapters on Full/Random/Diversity
4. Evaluation: AlpacaEval 2.0 + local benchmarks
5. Report Generation: Visualizations and analysis

Phase 3 (Scale-up - OpenOrca/UltraChat):
Only runs if Phase 2 succeeds (diversity > random, within 2% of full)

Usage:
    # Run pilot phase
    python run_pipeline.py

    # Run full experiment (pilot + scale-up)
    python run_pipeline.py --mode full

    # Run specific step
    python run_pipeline.py --step 3

    # Run from specific step onwards
    python run_pipeline.py --from-step 2

    # Force re-run (clear existing outputs)
    python run_pipeline.py --force
"""

import argparse
import shutil
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_step(step_num: int, script_name: str, description: str, base_dir: Path) -> bool:
    """Run a processing step."""
    print(f"\n{'=' * 70}")
    print(f"STEP {step_num}: {description}")
    print(f"{'=' * 70}\n")

    script_path = base_dir / "processing" / script_name

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(base_dir),
    )

    if result.returncode != 0:
        print(f"\nERROR: Step {step_num} failed with return code {result.returncode}")
        return False

    print(f"\nStep {step_num} completed successfully.")
    return True


def check_phase2_success(base_dir: Path) -> bool:
    """
    Check if Phase 2 (pilot) succeeded.

    Success criteria:
    1. Diversity-selected model within 2% of full model
    2. Diversity-selected model outperforms random baseline
    """
    eval_file = base_dir / "output" / "evaluation_results.json"

    if not eval_file.exists():
        print("No evaluation results found - cannot verify Phase 2 success")
        return False

    with open(eval_file, 'r') as f:
        results = json.load(f)

    eval_results = results.get('results', [])

    # Find results for each method
    full_wr = None
    random_wr = None
    diversity_wr = None

    for r in eval_results:
        method = r.get('selection_method')
        wr = r.get('alpaca_eval_win_rate')

        if wr is None:
            continue

        if method == 'full':
            full_wr = wr
        elif method == 'random':
            random_wr = wr
        elif method == 'diversity':
            diversity_wr = wr

    if diversity_wr is None:
        print("No diversity selection results found")
        return False

    success = True

    # Check criterion 1: within 2% of full
    if full_wr is not None:
        diff = abs(full_wr - diversity_wr)
        within_2pct = diff <= 2.0
        print(f"Diversity vs Full: {diversity_wr:.1f}% vs {full_wr:.1f}% (diff: {diff:.1f}%)")
        print(f"  Within 2%: {'PASS' if within_2pct else 'FAIL'}")
        if not within_2pct:
            success = False

    # Check criterion 2: beats random
    if random_wr is not None:
        beats_random = diversity_wr > random_wr
        print(f"Diversity vs Random: {diversity_wr:.1f}% vs {random_wr:.1f}%")
        print(f"  Beats random: {'PASS' if beats_random else 'FAIL'}")
        if not beats_random:
            success = False

    return success


def clear_outputs(base_dir: Path, step: int = None) -> None:
    """Clear output directories for fresh run."""
    dirs_by_step = {
        1: ["data"],
        2: ["selections"],
        3: ["output/adapters", "output/runs"],
        4: ["output/evaluation"],
        5: ["output/plots", "output/REPORT.md"],
    }

    if step:
        dirs = dirs_by_step.get(step, [])
    else:
        # Clear all
        dirs = ["data", "selections", "output"]

    for dir_name in dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            if dir_path.is_file():
                print(f"Removing {dir_path}...")
                dir_path.unlink()
            else:
                print(f"Clearing {dir_path}...")
                shutil.rmtree(dir_path)
                dir_path.mkdir(parents=True, exist_ok=True)


def update_config_mode(base_dir: Path, mode: str) -> None:
    """Update the mode in config.yaml."""
    import yaml

    config_path = base_dir / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['mode'] = mode

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Set mode to: {mode}")


def print_banner():
    """Print pipeline banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║     CORESET SELECTION FOR INSTRUCTION TUNING                      ║
    ║     Diversity-Guided Dataset Pruning Experiment                   ║
    ║                                                                   ║
    ║     Objective: Demonstrate that 10% diverse selection             ║
    ║     matches full dataset performance                              ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    parser = argparse.ArgumentParser(
        description="Run the Coreset Selection for Instruction Tuning pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
        help="Run mode: pilot (Phase 2 only) or full (Phase 2 + Phase 3)"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run only this step"
    )
    parser.add_argument(
        "--from-step",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Start from this step (default: 1)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run by clearing existing outputs"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation step (useful for debugging)"
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent

    print_banner()

    print(f"Base directory: {base_dir}")
    print(f"Mode: {args.mode}")
    print(f"Starting from step: {args.from_step if not args.step else args.step}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Handle force flag
    if args.force:
        print("\n--force specified: Clearing existing outputs...")
        if args.step:
            clear_outputs(base_dir, args.step)
        else:
            for s in range(args.from_step, 6):
                clear_outputs(base_dir, s)

    # Update config mode
    update_config_mode(base_dir, args.mode)

    # Define pipeline steps
    steps = [
        (1, "01_prepare_data.py", "DATA PREPARATION"),
        (2, "02_select_subsets.py", "DIVERSITY-BASED SUBSET SELECTION"),
        (3, "03_finetune.py", "FINE-TUNING WITH LORA"),
        (4, "04_evaluate.py", "EVALUATION (AlpacaEval 2.0)"),
        (5, "05_generate_report.py", "REPORT GENERATION"),
    ]

    # Filter steps based on arguments
    if args.step:
        steps = [(n, s, d) for n, s, d in steps if n == args.step]
    else:
        steps = [(n, s, d) for n, s, d in steps if n >= args.from_step]

    if args.skip_eval:
        steps = [(n, s, d) for n, s, d in steps if n != 4]

    # Run Phase 2 (Pilot)
    print("\n" + "=" * 70)
    print("PHASE 2: PILOT (Alpaca-GPT4)")
    print("=" * 70)

    for step_num, script_name, description in steps:
        success = run_step(step_num, script_name, description, base_dir)
        if not success:
            print(f"\nPipeline stopped at step {step_num}")
            sys.exit(1)

    # Check Phase 2 success before proceeding to Phase 3
    if args.mode == 'full' and not args.step:
        print("\n" + "=" * 70)
        print("CHECKING PHASE 2 SUCCESS CRITERIA")
        print("=" * 70)

        phase2_success = check_phase2_success(base_dir)

        if phase2_success:
            print("\nPhase 2 SUCCESS - Proceeding to Phase 3 (Scale-up)")

            # Update config for scale-up datasets
            # Note: The actual scale-up would require additional configuration
            print("\nPhase 3 (OpenOrca + UltraChat) not yet implemented.")
            print("Current implementation focuses on the pilot phase.")

        else:
            print("\nPhase 2 did not meet success criteria.")
            print("Recommendation: Investigate selection algorithm or hyperparameters")
            print("before scaling up to larger datasets.")

    # Final summary
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    output_dir = base_dir / "output"
    print(f"\nOutputs saved to: {output_dir}")

    report_path = output_dir / "REPORT.md"
    if report_path.exists():
        print(f"Report: {report_path}")

    adapters_dir = output_dir / "adapters"
    if adapters_dir.exists():
        adapters = list(adapters_dir.iterdir())
        print(f"Adapters: {len(adapters)} saved to {adapters_dir}")

    selections_dir = base_dir / "selections"
    if selections_dir.exists():
        jsonl_files = list(selections_dir.glob("*.jsonl"))
        print(f"Selections: {len(jsonl_files)} JSONL files in {selections_dir}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
