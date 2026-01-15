#!/usr/bin/env python3
"""Preprocess DementiaBank data.

This script:
1. Loads raw data
2. Filters for Cookie Theft task
3. Balances classes (Dementia vs Control)
4. Cleans transcripts (removes artifacts)
5. Segments into sentences
6. Applies quality filters
7. Exports preprocessed data
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import re
import pandas as pd
import numpy as np
import yaml
from linguistic_diversity.utils import split_sentences


def load_config():
    """Load configuration file."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def clean_transcript(text, config):
    """Clean transcript by removing artifacts.

    Args:
        text: Raw transcript text
        config: Preprocessing configuration

    Returns:
        Cleaned text
    """
    if pd.isna(text) or text == '':
        return ''

    text = str(text)

    if config['remove_brackets']:
        # Remove bracketed content: [unintelligible], [laughter], etc.
        text = re.sub(r'\[.*?\]', '', text)

    if config['remove_timestamps']:
        # Remove timestamps: 00:00, 0:00:00, etc.
        text = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\b', '', text)

    if config['remove_speaker_labels']:
        # Remove speaker labels: INV:, PAR:, etc.
        text = re.sub(r'\b[A-Z]{2,}:', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def segment_transcript(text):
    """Segment transcript into sentences.

    Args:
        text: Cleaned transcript

    Returns:
        List of sentences
    """
    if not text or text == '':
        return []

    try:
        sentences = split_sentences(text)
        # Filter out very short sentences (< 3 words)
        sentences = [s for s in sentences if len(s.split()) >= 3]
        return sentences
    except Exception as e:
        print(f"      Warning: Segmentation failed: {e}")
        return []


def main():
    print("=" * 80)
    print("STEP 2: PREPROCESS DATA")
    print("=" * 80)

    # Load configuration
    config = load_config()
    prep_config = config['preprocessing']
    dataset_config = config['dataset']

    # Load raw data
    input_dir = Path(__file__).parent.parent / "input"
    raw_data_path = input_dir / "dementiabank_raw.csv"

    if not raw_data_path.exists():
        print(f"\n✗ Error: Raw data file not found: {raw_data_path}")
        print("   Please run: python processing/01_download_data.py first")
        sys.exit(1)

    print(f"\n1. Loading raw data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"   ✓ Loaded {len(df)} samples")

    # Identify columns (adjust based on actual dataset structure)
    print("\n2. Identifying columns...")
    print(f"   Available columns: {df.columns.tolist()}")

    # Auto-detect columns
    label_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['label', 'diagnosis', 'class', 'group', 'output']):
            label_col = col
            break

    task_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['task', 'test', 'type', 'instruction']):
            task_col = col
            break

    text_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['text', 'transcript', 'speech', 'utterance', 'input']):
            text_col = col
            break

    subject_col = None
    for col in df.columns:
        if any(word in col.lower() for word in ['subject', 'patient', 'id', 'participant']):
            subject_col = col
            break

    # Create subject_id from index if not found
    if subject_col is None:
        df['subject_id'] = df.index
        subject_col = 'subject_id'

    print(f"   Label column: {label_col}")
    print(f"   Task column: {task_col}")
    print(f"   Text column: {text_col}")
    print(f"   Subject column: {subject_col}")

    if not all([label_col, text_col]):
        print("\n✗ Error: Could not identify required columns")
        print("   Please manually specify column names in the script")
        sys.exit(1)

    # Filter for Cookie Theft task if task column exists
    print("\n3. Filtering data...")
    df_filtered = df.copy()

    if task_col and task_col in df.columns:
        task_filter = dataset_config['task_filter']
        print(f"   Task column found: {task_col}")
        print(f"   Sample task value: {df_filtered[task_col].iloc[0]}")

        # Check if task filtering is needed
        unique_tasks = df_filtered[task_col].unique()
        if len(unique_tasks) > 1:
            print(f"   Filtering for task: '{task_filter}'")
            # Flexible matching
            mask = df[task_col].astype(str).str.contains(task_filter, case=False, na=False)
            df_filtered = df_filtered[mask]
            print(f"   ✓ Retained {len(df_filtered)} samples with '{task_filter}' task")
        else:
            print(f"   ✓ All samples have same task, skipping filter")
    else:
        print(f"   No task column found, using all samples")

    # Filter for Dementia and Control labels
    print(f"\n   Filtering for labels...")
    print(f"   Available labels: {df_filtered[label_col].unique()}")

    # Flexible label matching
    dementia_mask = df_filtered[label_col].astype(str).str.contains('dementia|AD|alzheimer', case=False, na=False)
    control_mask = df_filtered[label_col].astype(str).str.contains('control|healthy|normal', case=False, na=False)

    df_dementia = df_filtered[dementia_mask].copy()
    df_control = df_filtered[control_mask].copy()

    print(f"   Dementia samples: {len(df_dementia)}")
    print(f"   Control samples: {len(df_control)}")

    if len(df_dementia) == 0 or len(df_control) == 0:
        print("\n✗ Error: Need both Dementia and Control samples")
        sys.exit(1)

    # Balance classes if specified
    n_samples = dataset_config.get('n_samples_per_class')
    if n_samples is not None:
        print(f"\n   Balancing to {n_samples} samples per class...")
        df_dementia = df_dementia.sample(n=min(n_samples, len(df_dementia)), random_state=42)
        df_control = df_control.sample(n=min(n_samples, len(df_control)), random_state=42)
        print(f"   ✓ Dementia: {len(df_dementia)}, Control: {len(df_control)}")

    # Standardize labels
    df_dementia['label_clean'] = 'Dementia'
    df_control['label_clean'] = 'Control'

    # Combine
    df_filtered = pd.concat([df_dementia, df_control], ignore_index=True)
    print(f"\n   ✓ Total filtered samples: {len(df_filtered)}")

    # Clean transcripts
    print("\n4. Cleaning transcripts...")
    df_filtered['cleaned_text'] = df_filtered[text_col].apply(
        lambda x: clean_transcript(x, prep_config)
    )

    # Count empty/very short transcripts
    empty_count = (df_filtered['cleaned_text'].str.len() < dataset_config.get('min_words', 10)).sum()
    if empty_count > 0:
        print(f"   ⚠ Warning: {empty_count} transcripts have < {dataset_config.get('min_words', 10)} characters")

    # Filter out empty transcripts
    df_filtered = df_filtered[df_filtered['cleaned_text'].str.len() >= dataset_config.get('min_words', 10)]
    print(f"   ✓ Retained {len(df_filtered)} transcripts after cleaning")

    # Segment into sentences
    print("\n5. Segmenting into sentences...")
    print("   (This may take a moment...)")

    df_filtered['sentences'] = df_filtered['cleaned_text'].apply(segment_transcript)
    df_filtered['n_sentences'] = df_filtered['sentences'].apply(len)

    # Filter: minimum sentences
    min_sents = dataset_config.get('min_sentences', 2)
    print(f"\n   Filtering for >= {min_sents} sentences...")
    df_filtered = df_filtered[df_filtered['n_sentences'] >= min_sents]
    print(f"   ✓ Retained {len(df_filtered)} transcripts")

    # Summary statistics
    print("\n6. Data summary:")
    print(f"   Total subjects: {len(df_filtered)}")
    print(f"   Dementia: {(df_filtered['label_clean'] == 'Dementia').sum()}")
    print(f"   Control: {(df_filtered['label_clean'] == 'Control').sum()}")
    print(f"\n   Sentence statistics:")
    print(f"   Min sentences: {df_filtered['n_sentences'].min()}")
    print(f"   Max sentences: {df_filtered['n_sentences'].max()}")
    print(f"   Mean sentences: {df_filtered['n_sentences'].mean():.1f}")
    print(f"   Median sentences: {df_filtered['n_sentences'].median():.1f}")

    # Sample examples
    print("\n7. Sample examples:")
    print("   " + "-" * 76)
    for idx in range(min(2, len(df_filtered))):
        row = df_filtered.iloc[idx]
        label = row['label_clean']
        n_sents = row['n_sentences']
        first_sent = row['sentences'][0] if row['sentences'] else ""
        print(f"   Sample {idx + 1} ({label}, {n_sents} sentences):")
        print(f"   First sentence: {first_sent}")
        print("   " + "-" * 76)

    # Save preprocessed data
    print("\n8. Saving preprocessed data...")

    # Prepare metadata
    metadata_df = df_filtered[[subject_col, 'label_clean', 'n_sentences']].copy()
    metadata_df.columns = ['subject_id', 'label', 'n_sentences']

    # Prepare full data (with sentences)
    output_df = df_filtered[[subject_col, 'label_clean', 'cleaned_text', 'sentences', 'n_sentences']].copy()
    output_df.columns = ['subject_id', 'label', 'text', 'sentences', 'n_sentences']

    # Save
    output_dir = Path(__file__).parent.parent / "input"
    metadata_path = output_dir / "preprocessed_metadata.csv"
    data_path = output_dir / "preprocessed_data.pkl"

    metadata_df.to_csv(metadata_path, index=False)
    output_df.to_pickle(data_path)

    print(f"   ✓ Saved metadata to: {metadata_path}")
    print(f"   ✓ Saved full data to: {data_path}")

    print("\n" + "=" * 80)
    print("✓ PREPROCESSING COMPLETE")
    print("=" * 80)
    print("\nData ready for metric computation:")
    print(f"  - {len(output_df)} subjects")
    print(f"  - {(output_df['label'] == 'Dementia').sum()} Dementia")
    print(f"  - {(output_df['label'] == 'Control').sum()} Control")
    print(f"  - Average {output_df['n_sentences'].mean():.1f} sentences per transcript")
    print("\nNext step:")
    print("  python processing/03_compute_metrics.py")
    print("  (This will take ~60-90 minutes with GPU)")
    print("=" * 80)


if __name__ == "__main__":
    main()
