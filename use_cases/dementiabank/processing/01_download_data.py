#!/usr/bin/env python3
"""Download DementiaBank dataset from Hugging Face.

This script:
1. Downloads the MearaHe/dementiabank dataset
2. Explores the structure to identify columns
3. Exports raw data to CSV for inspection
4. Caches the dataset locally
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from datasets import load_dataset


def main():
    print("=" * 80)
    print("STEP 1: DOWNLOAD DEMENTIABANK DATASET")
    print("=" * 80)

    # Ensure output directory exists
    output_dir = Path(__file__).parent.parent / "input"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Downloading dataset from Hugging Face...")
    print("   Source: MearaHe/dementiabank")

    try:
        dataset = load_dataset("MearaHe/dementiabank")
        print("   ✓ Dataset downloaded successfully")
    except Exception as e:
        print(f"   ✗ Error downloading dataset: {e}")
        print("\n   Troubleshooting:")
        print("   - Check internet connection")
        print("   - Verify dataset name is correct")
        print("   - Try: pip install --upgrade datasets")
        sys.exit(1)

    print("\n2. Exploring dataset structure...")
    print(f"   Dataset keys: {list(dataset.keys())}")

    # Use the appropriate split (usually 'train' or the only available split)
    split_name = list(dataset.keys())[0]
    data = dataset[split_name]

    print(f"   Using split: '{split_name}'")
    print(f"   Number of samples: {len(data)}")
    print(f"   Columns: {data.column_names}")

    print("\n3. Sample data inspection:")
    print("   " + "-" * 76)

    # Show first sample
    sample = data[0]
    for key, value in sample.items():
        # Truncate long values
        value_str = str(value)
        if len(value_str) > 100:
            value_str = value_str[:100] + "..."
        print(f"   {key}: {value_str}")

    print("   " + "-" * 76)

    print("\n4. Converting to pandas DataFrame...")
    df = data.to_pandas()
    print(f"   ✓ Converted to DataFrame ({len(df)} rows)")

    print("\n5. Data summary:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")

    # Check for label column
    label_columns = [col for col in df.columns if 'label' in col.lower() or 'diagnosis' in col.lower() or 'class' in col.lower()]
    if label_columns:
        print(f"\n   Label column(s): {label_columns}")
        for col in label_columns:
            print(f"   {col} distribution:")
            print(df[col].value_counts().to_string(indent='      '))

    # Check for task column
    task_columns = [col for col in df.columns if 'task' in col.lower() or 'test' in col.lower()]
    if task_columns:
        print(f"\n   Task column(s): {task_columns}")
        for col in task_columns:
            if df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df[col]):
                print(f"   {col} distribution:")
                print(df[col].value_counts().to_string(indent='      '))

    # Check for text/transcript column
    text_columns = [col for col in df.columns if any(word in col.lower() for word in ['text', 'transcript', 'speech', 'utterance'])]
    if text_columns:
        print(f"\n   Text column(s): {text_columns}")
        print(f"   Sample text (first 200 chars):")
        for col in text_columns[:1]:  # Just show first text column
            if col in df.columns:
                sample_text = str(df[col].iloc[0])[:200]
                print(f"      {sample_text}...")

    print("\n6. Saving raw data to CSV...")
    output_path = output_dir / "dementiabank_raw.csv"
    df.to_csv(output_path, index=False)
    print(f"   ✓ Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n7. Data validation...")
    print(f"   ✓ Total samples: {len(df)}")
    print(f"   ✓ Missing values per column:")
    missing = df.isnull().sum()
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"      {col}: {count} ({pct:.1f}%)")
    if missing.sum() == 0:
        print(f"      No missing values!")

    print("\n" + "=" * 80)
    print("✓ DATA DOWNLOAD COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review the raw data CSV to understand structure")
    print("  2. Identify the correct column names for:")
    print("     - Label/diagnosis (Dementia vs Control)")
    print("     - Task type (Cookie Theft)")
    print("     - Transcript/text")
    print("     - Subject ID")
    print("  3. Update config.yaml if needed")
    print("  4. Run: python processing/02_preprocess.py")
    print("=" * 80)


if __name__ == "__main__":
    main()
