#!/usr/bin/env python3
"""Master pipeline for DementiaBank evaluation.

This script runs the complete evaluation pipeline:
1. Download data
2. Preprocess
3. Compute metrics
4. Analyze results
5. Create visualizations
6. Generate report

Usage:
    python run_evaluation.py
"""

import sys
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime

# Set environment variables
os.environ['TMPDIR'] = '/data2/fabricehc/tmp'
os.environ['TEMP'] = '/data2/fabricehc/tmp'
os.environ['TMP'] = '/data2/fabricehc/tmp'
os.makedirs('/data2/fabricehc/tmp', exist_ok=True)


def check_step_completed(step_num):
    """Check if a pipeline step has already been completed.

    Args:
        step_num: Step number (1-6)

    Returns:
        bool: True if step output exists
    """
    project_dir = Path(__file__).parent

    output_files = {
        1: project_dir / "input" / "dementiabank_raw.csv",
        2: project_dir / "input" / "preprocessed_data.pkl",
        3: project_dir / "output" / "scores" / "raw_scores.csv",
        4: project_dir / "output" / "scores" / "summary_stats.csv",
        5: project_dir / "output" / "plots" / "all_metrics_comparison.png",
        6: project_dir / "output" / "report" / "evaluation_report.md",
    }

    output_file = output_files.get(step_num)
    if output_file and output_file.exists():
        return True
    return False


def run_step(script_name, description, step_num, skip_completed=True):
    """Run a pipeline step.

    Args:
        script_name: Name of the script to run
        description: Human-readable description
        step_num: Step number (1-6)
        skip_completed: If True, skip if output exists

    Returns:
        tuple: (success: bool, elapsed_time: float, skipped: bool)
    """
    print(f"\n{'=' * 80}")
    print(f"Step {step_num}: {description}")
    print(f"Script: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('=' * 80)

    # Check if already completed
    if skip_completed and check_step_completed(step_num):
        print(f"⏭️  Step already completed (output exists), skipping...")
        print(f"   To re-run, delete the output file or use --force")
        return True, 0.0, True

    script_path = Path(__file__).parent / "processing" / script_name

    if not script_path.exists():
        print(f"✗ Error: Script not found: {script_path}")
        return False, 0.0, False

    try:
        start_time = time.time()

        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            cwd=Path.cwd(),
            env=os.environ.copy()
        )

        elapsed_time = time.time() - start_time

        print(f"\n✓ {description} completed successfully")
        print(f"  Time elapsed: {elapsed_time / 60:.1f} minutes")

        return True, elapsed_time, False

    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {description} failed with error code {e.returncode}")
        print(f"  Time elapsed: {elapsed_time / 60:.1f} minutes")
        return False, elapsed_time, False

    except KeyboardInterrupt:
        print(f"\n⚠ Pipeline interrupted by user")
        raise

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Unexpected error: {e}")
        print(f"  Time elapsed: {elapsed_time / 60:.1f} minutes")
        return False, elapsed_time, False


def main():
    """Run the complete evaluation pipeline."""
    # Check for --force flag
    force_rerun = '--force' in sys.argv

    print("=" * 80)
    print("LINGUISTIC DIVERSITY EVALUATION PIPELINE")
    print("DementiaBank Cognitive Impairment Detection")
    print("=" * 80)

    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nConfiguration:")
    print(f"  - Dataset: All available DementiaBank subjects")
    print(f"  - Task: Cookie Theft picture description")
    print(f"  - Metrics: All (semantic, syntactic, morphological, phonological, lexical)")
    print(f"  - GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Auto-detect')}")
    print(f"  - Temp directory: {os.environ.get('TMPDIR', 'Not set')}")
    print(f"  - Resume mode: {'Disabled (--force)' if force_rerun else 'Enabled (skip completed steps)'}")

    print("\n" + "=" * 80)
    print("Pipeline Steps:")
    print("  1. Download data from Hugging Face")
    print("  2. Preprocess and clean transcripts")
    print("  3. Compute diversity metrics (~60-90 min)")
    print("  4. Statistical analysis")
    print("  5. Create visualizations")
    print("  6. Generate evaluation report")
    print("=" * 80)

    if not force_rerun:
        print("\n💡 Tip: Pipeline will skip steps that are already completed.")
        print("   Use --force to re-run all steps from scratch.")

    input("\nPress ENTER to start the pipeline (or Ctrl+C to cancel)...")

    # Define pipeline steps
    steps = [
        ('01_download_data.py', 'Download DementiaBank dataset'),
        ('02_preprocess.py', 'Preprocess and clean data'),
        ('03_compute_metrics.py', 'Compute diversity metrics'),
        ('04_analyze_results.py', 'Statistical analysis'),
        ('05_create_plots.py', 'Create visualizations'),
        ('06_generate_report.py', 'Generate evaluation report'),
    ]

    # Track progress
    completed_steps = []
    failed_step = None
    total_time = 0.0
    overall_start = time.time()

    # Run each step
    skipped_steps = []
    for i, (script_name, description) in enumerate(steps, 1):
        success, elapsed, skipped = run_step(script_name, description, step_num=i, skip_completed=(not force_rerun))
        total_time += elapsed

        if skipped:
            skipped_steps.append((script_name, description))
            completed_steps.append((script_name, description, 0.0))  # Add to completed with 0 time
        elif success:
            completed_steps.append((script_name, description, elapsed))
        else:
            failed_step = (script_name, description)
            break

    overall_time = time.time() - overall_start

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)

    if failed_step is None:
        print("\n✅ EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print(f"\n❌ PIPELINE FAILED AT: {failed_step[1]}")

    print(f"\nCompleted steps: {len(completed_steps)} / {len(steps)}")
    print(f"Total time: {overall_time / 60:.1f} minutes")

    if completed_steps:
        print(f"\n✓ Completed:")
        for script, desc, elapsed in completed_steps:
            if elapsed == 0.0 and any(s[0] == script for s in skipped_steps):
                print(f"  - {desc} (skipped - already done)")
            else:
                print(f"  - {desc} ({elapsed / 60:.1f} min)")

    if failed_step:
        print(f"\n✗ Failed:")
        print(f"  - {failed_step[1]}")
        print("\nTo resume:")
        print(f"  1. Fix the issue")
        print(f"  2. Run: python processing/{failed_step[0]}")
        print(f"  3. Continue with remaining steps")

    # Show output locations
    if len(completed_steps) >= 3:  # At least through metric computation
        print("\n" + "=" * 80)
        print("OUTPUT LOCATIONS")
        print("=" * 80)

        project_dir = Path(__file__).parent

        print(f"\n📂 Scores:")
        scores_dir = project_dir / "output" / "scores"
        if scores_dir.exists():
            for file in sorted(scores_dir.glob("*.csv")):
                size_kb = file.stat().st_size / 1024
                print(f"  - {file.name} ({size_kb:.1f} KB)")

        print(f"\n📊 Plots:")
        plots_dir = project_dir / "output" / "plots"
        if plots_dir.exists():
            n_plots = len(list(plots_dir.glob("*.png")))
            print(f"  - {n_plots} visualization(s) created")
            print(f"  - Location: {plots_dir}")

        print(f"\n📄 Report:")
        report_file = project_dir / "output" / "report" / "evaluation_report.md"
        if report_file.exists():
            print(f"  - {report_file.name}")
            print(f"  - Location: {report_file}")

    # Final instructions
    if failed_step is None:
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Review the evaluation report:")
        print(f"   open output/report/evaluation_report.md")
        print("\n2. Examine visualizations:")
        print(f"   open output/plots/all_metrics_comparison.png")
        print("\n3. Check statistical details:")
        print(f"   cat output/scores/statistical_report.txt")
        print("\n4. Review raw scores:")
        print(f"   less output/scores/raw_scores.csv")
        print("\n" + "=" * 80)
        print("SUCCESS! Framework evaluation complete.")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("PIPELINE INCOMPLETE")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠ Pipeline interrupted by user")
        print("Progress has been saved. You can resume by running individual scripts.")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
