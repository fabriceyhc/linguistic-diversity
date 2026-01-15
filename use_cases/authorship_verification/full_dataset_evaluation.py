"""Full Dataset Evaluation: Traditional vs Inter-Text vs Hybrid.

This script runs a comprehensive evaluation on the FULL dataset (30,772 pairs) to:
1. Test traditional stylometry baselines alone
2. Test inter-text similarity alone
3. Test hybrid combination
4. Validate that results hold at scale (no subset bias)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


# ============================================================================
# TRADITIONAL STYLOMETRY FEATURES (from hybrid_stylometry.py)
# ============================================================================

def extract_char_ngrams(text, n=3, top_k=100):
    """Extract character n-gram features."""
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i+n])

    freq = Counter(ngrams)
    top_ngrams = [ng for ng, _ in freq.most_common(top_k)]
    features = {f'char_{n}gram_{ng}': freq.get(ng, 0) for ng in top_ngrams}

    return features


def extract_function_words(text):
    """Extract function word frequencies."""
    function_words = [
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me'
    ]

    import re
    tokens = re.findall(r'\b\w+\b', text.lower())

    total = len(tokens)
    if total == 0:
        return {f'func_{word}': 0.0 for word in function_words}

    freq = Counter(tokens)
    return {f'func_{word}': freq.get(word, 0) / total for word in function_words}


def extract_punctuation_features(text):
    """Extract punctuation usage patterns."""
    import string

    punct_chars = string.punctuation
    total_chars = len(text)

    if total_chars == 0:
        return {f'punct_{p}': 0.0 for p in punct_chars}

    features = {}
    for p in punct_chars:
        features[f'punct_{p}'] = text.count(p) / total_chars

    return features


def extract_syntactic_patterns(text):
    """Extract simple syntactic patterns."""
    import re

    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) == 0:
        return {
            'avg_sentence_length': 0,
            'sentence_length_std': 0,
            'avg_words_per_sentence': 0,
        }

    sent_lengths = [len(s) for s in sentences]
    words_per_sent = [len(re.findall(r'\b\w+\b', s)) for s in sentences]

    return {
        'avg_sentence_length': np.mean(sent_lengths),
        'sentence_length_std': np.std(sent_lengths),
        'avg_words_per_sentence': np.mean(words_per_sent),
        'sentence_count': len(sentences),
    }


def compute_traditional_similarity(text1, text2):
    """Compute similarity using traditional stylometric features."""

    # Character 3-grams
    ngrams1 = extract_char_ngrams(text1, n=3, top_k=50)
    ngrams2 = extract_char_ngrams(text2, n=3, top_k=50)

    all_keys = set(list(ngrams1.keys()) + list(ngrams2.keys()))
    vec1 = np.array([ngrams1.get(k, 0) for k in all_keys])
    vec2 = np.array([ngrams2.get(k, 0) for k in all_keys])

    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        char_ngram_sim = 0.0
    else:
        char_ngram_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    # Function words
    func1 = extract_function_words(text1)
    func2 = extract_function_words(text2)

    func_keys = func1.keys()
    func_vec1 = np.array([func1[k] for k in func_keys])
    func_vec2 = np.array([func2.get(k, 0) for k in func_keys])

    if np.linalg.norm(func_vec1) == 0 or np.linalg.norm(func_vec2) == 0:
        func_sim = 0.0
    else:
        func_sim = np.dot(func_vec1, func_vec2) / (np.linalg.norm(func_vec1) * np.linalg.norm(func_vec2))

    # Punctuation
    punct1 = extract_punctuation_features(text1)
    punct2 = extract_punctuation_features(text2)

    punct_keys = punct1.keys()
    punct_vec1 = np.array([punct1[k] for k in punct_keys])
    punct_vec2 = np.array([punct2.get(k, 0) for k in punct_keys])

    if np.linalg.norm(punct_vec1) == 0 or np.linalg.norm(punct_vec2) == 0:
        punct_sim = 0.0
    else:
        punct_sim = np.dot(punct_vec1, punct_vec2) / (np.linalg.norm(punct_vec1) * np.linalg.norm(punct_vec2))

    # Syntactic patterns
    synt1 = extract_syntactic_patterns(text1)
    synt2 = extract_syntactic_patterns(text2)

    synt_features = ['avg_sentence_length', 'avg_words_per_sentence']
    synt_diffs = []
    for key in synt_features:
        val1 = synt1.get(key, 0)
        val2 = synt2.get(key, 0)
        if val1 + val2 > 0:
            synt_diffs.append(abs(val1 - val2) / (val1 + val2))
        else:
            synt_diffs.append(0)

    synt_sim = 1 - np.mean(synt_diffs)

    return {
        'char_ngram_sim': char_ngram_sim,
        'func_word_sim': func_sim,
        'punct_sim': punct_sim,
        'syntactic_sim': synt_sim,
    }


# ============================================================================
# FULL DATASET PROCESSING
# ============================================================================

def process_full_dataset_features(output_dir):
    """Process all 30,772 pairs with all features."""

    print(f"\n{'='*80}")
    print("FULL DATASET FEATURE EXTRACTION")
    print(f"{'='*80}")

    cache_file = output_dir / 'full_dataset_all_features.csv'

    if cache_file.exists():
        print(f"\nLoading cached features from {cache_file}")
        df = pd.read_csv(cache_file)
        print(f"Loaded {len(df)} pairs with all features")
        return df

    print("\nLoading inter-text similarity results...")
    df_similarity = pd.read_csv(output_dir / 'full_dataset_all-MiniLM-L12-v2.csv')
    print(f"Loaded {len(df_similarity)} pairs with inter-text features")

    print("\nLoading dataset to extract traditional features...")
    from datasets import load_dataset
    dataset = load_dataset("swan07/authorship-verification", split="validation")

    print(f"\nProcessing {len(df_similarity)} pairs with traditional stylometry...")
    print("This will take 2-4 hours. Progress will be saved every 1000 pairs.")

    # Process in batches and save intermediate results
    batch_size = 1000
    all_results = []

    start_time = time.time()

    for batch_start in range(0, len(df_similarity), batch_size):
        batch_end = min(batch_start + batch_size, len(df_similarity))
        batch_df = df_similarity.iloc[batch_start:batch_end]

        batch_results = []

        for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df),
                            desc=f"Batch {batch_start//batch_size + 1}/{(len(df_similarity)-1)//batch_size + 1}"):
            pair_id = row['id']

            # Get the texts
            example = dataset[int(pair_id)]
            text1 = example['text1']
            text2 = example['text2']

            # Compute traditional features
            trad_features = compute_traditional_similarity(text1, text2)

            # Combine with existing data
            result = row.to_dict()
            result.update(trad_features)
            batch_results.append(result)

        all_results.extend(batch_results)

        # Save intermediate results
        if batch_end % 5000 == 0 or batch_end == len(df_similarity):
            temp_df = pd.DataFrame(all_results)
            temp_df.to_csv(cache_file, index=False)
            elapsed = time.time() - start_time
            remaining = (elapsed / batch_end) * (len(df_similarity) - batch_end)
            print(f"  → Saved {batch_end} pairs. Elapsed: {elapsed/60:.1f}min, Remaining: {remaining/60:.1f}min")

    df_full = pd.DataFrame(all_results)
    df_full.to_csv(cache_file, index=False)

    total_time = time.time() - start_time
    print(f"\n✓ Complete! Total time: {total_time/60:.1f} minutes")
    print(f"✓ Saved all features to {cache_file}")

    return df_full


# ============================================================================
# COMPREHENSIVE EVALUATION
# ============================================================================

def evaluate_all_approaches(df_full):
    """Evaluate traditional, inter-text, and hybrid on full dataset."""

    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION - FULL DATASET")
    print(f"{'='*80}")

    # Prepare feature sets
    df_model = df_full.copy()

    # Convert distances to similarities
    df_model['semantic_sim'] = 1 - df_model['semantic_dist']
    df_model['lexical_sim'] = 1 - df_model['lexical_dist']

    # Define feature sets
    inter_text_features = ['semantic_sim', 'lexical_sim']
    traditional_features = ['char_ngram_sim', 'func_word_sim', 'punct_sim', 'syntactic_sim']
    hybrid_features = inter_text_features + traditional_features

    # Prepare data
    X_inter = df_model[inter_text_features].values
    X_trad = df_model[traditional_features].values
    X_hybrid = df_model[hybrid_features].values
    y = df_model['label'].values

    # Split data
    print(f"\nDataset size: {len(y)} pairs")
    print(f"  Same author: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"  Different author: {len(y)-sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")

    X_inter_train, X_inter_test, y_train, y_test = train_test_split(
        X_inter, y, test_size=0.3, random_state=42, stratify=y
    )
    X_trad_train, X_trad_test, _, _ = train_test_split(
        X_trad, y, test_size=0.3, random_state=42, stratify=y
    )
    X_hybrid_train, X_hybrid_test, _, _ = train_test_split(
        X_hybrid, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"Train size: {len(y_train)}")
    print(f"Test size: {len(y_test)}")

    # Test classifiers
    classifiers = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for clf_name, clf in classifiers.items():
        print(f"\n{'-'*80}")
        print(f"Classifier: {clf_name}")
        print(f"{'-'*80}")

        # 1. Traditional only (BASELINE)
        print(f"  Training traditional only...")
        clf.fit(X_trad_train, y_train)
        y_pred_trad = clf.predict_proba(X_trad_test)[:, 1]
        auc_trad = roc_auc_score(y_test, y_pred_trad)
        print(f"    Traditional only:    AUC = {auc_trad:.4f}")

        # 2. Inter-text only (OUR CONTRIBUTION)
        print(f"  Training inter-text only...")
        clf.fit(X_inter_train, y_train)
        y_pred_inter = clf.predict_proba(X_inter_test)[:, 1]
        auc_inter = roc_auc_score(y_test, y_pred_inter)
        print(f"    Inter-text only:     AUC = {auc_inter:.4f}")

        # 3. Hybrid (COMBINED)
        print(f"  Training hybrid...")
        clf.fit(X_hybrid_train, y_train)
        y_pred_hybrid = clf.predict_proba(X_hybrid_test)[:, 1]
        auc_hybrid = roc_auc_score(y_test, y_pred_hybrid)
        print(f"    Hybrid (combined):   AUC = {auc_hybrid:.4f}")

        # Calculate improvements
        improvement_vs_trad = ((auc_hybrid - auc_trad) / auc_trad * 100)
        improvement_vs_inter = ((auc_hybrid - auc_inter) / auc_inter * 100)

        print(f"\n    Improvement:")
        print(f"      vs Traditional: {improvement_vs_trad:+.1f}%")
        print(f"      vs Inter-text:  {improvement_vs_inter:+.1f}%")

        # Feature importance
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            feature_names = hybrid_features
            print(f"\n    Feature importances:")
            for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
                print(f"      {name:20s}: {imp:.4f}")

        results[clf_name] = {
            'traditional_auc': auc_trad,
            'inter_text_auc': auc_inter,
            'hybrid_auc': auc_hybrid,
            'improvement_vs_trad': improvement_vs_trad,
            'improvement_vs_inter': improvement_vs_inter,
            'y_pred_trad': y_pred_trad,
            'y_pred_inter': y_pred_inter,
            'y_pred_hybrid': y_pred_hybrid,
        }

    return results, y_test


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_full_evaluation_plot(results, y_test, output_dir):
    """Create comprehensive visualization of full dataset results."""

    print(f"\n{'='*80}")
    print("Generating Full Dataset Evaluation Plot")
    print(f"{'='*80}")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Main comparison (top-left, large)
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    classifiers = list(results.keys())
    trad_aucs = [results[c]['traditional_auc'] for c in classifiers]
    inter_aucs = [results[c]['inter_text_auc'] for c in classifiers]
    hybrid_aucs = [results[c]['hybrid_auc'] for c in classifiers]

    x = np.arange(len(classifiers))
    width = 0.25

    bars1 = ax1.bar(x - width, trad_aucs, width, label='Traditional Stylometry (Baseline)', color='coral')
    bars2 = ax1.bar(x, inter_aucs, width, label='Inter-text Similarity (Our Work)', color='skyblue')
    bars3 = ax1.bar(x + width, hybrid_aucs, width, label='Hybrid (Combined)', color='green')

    ax1.set_xticks(x)
    ax1.set_xticklabels(classifiers, fontsize=11)
    ax1.set_ylabel('ROC-AUC', fontsize=12)
    ax1.set_title('Full Dataset Results (30,772 pairs): Traditional vs Inter-text vs Hybrid',
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.axhline(y=0.65, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Viable (0.65)')
    ax1.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Strong (0.70)')
    ax1.set_ylim([0.55, max(hybrid_aucs) + 0.05])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (trad, inter, hyb) in enumerate(zip(trad_aucs, inter_aucs, hybrid_aucs)):
        ax1.text(i - width, trad, f'{trad:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i, inter, f'{inter:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax1.text(i + width, hyb, f'{hyb:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. Best model ROC curves (top-right)
    ax2 = fig.add_subplot(gs[0, 2])

    best_clf = max(results, key=lambda x: results[x]['hybrid_auc'])

    # Traditional
    fpr_trad, tpr_trad, _ = roc_curve(y_test, results[best_clf]['y_pred_trad'])
    ax2.plot(fpr_trad, tpr_trad, label=f'Traditional ({results[best_clf]["traditional_auc"]:.3f})',
            color='coral', linewidth=2)

    # Inter-text
    fpr_inter, tpr_inter, _ = roc_curve(y_test, results[best_clf]['y_pred_inter'])
    ax2.plot(fpr_inter, tpr_inter, label=f'Inter-text ({results[best_clf]["inter_text_auc"]:.3f})',
            color='skyblue', linewidth=2)

    # Hybrid
    fpr_hybrid, tpr_hybrid, _ = roc_curve(y_test, results[best_clf]['y_pred_hybrid'])
    ax2.plot(fpr_hybrid, tpr_hybrid, label=f'Hybrid ({results[best_clf]["hybrid_auc"]:.3f})',
            color='green', linewidth=2)

    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (0.50)')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'ROC Curves: {best_clf}', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # 3. Improvement percentages (middle-right)
    ax3 = fig.add_subplot(gs[1, 2])

    improvements = []
    for clf in classifiers:
        improvements.append({
            'Classifier': clf,
            'vs Traditional': results[clf]['improvement_vs_trad'],
            'vs Inter-text': results[clf]['improvement_vs_inter']
        })

    df_imp = pd.DataFrame(improvements)
    x_imp = np.arange(len(classifiers))
    width_imp = 0.35

    ax3.barh(x_imp, df_imp['vs Traditional'], width_imp, label='vs Traditional', color='coral', alpha=0.7)
    ax3.barh(x_imp + width_imp, df_imp['vs Inter-text'], width_imp, label='vs Inter-text', color='skyblue', alpha=0.7)

    ax3.set_yticks(x_imp + width_imp / 2)
    ax3.set_yticklabels(classifiers, fontsize=9)
    ax3.set_xlabel('Improvement (%)', fontsize=10)
    ax3.set_title('Hybrid Improvement Over Each Approach', fontweight='bold', fontsize=11)
    ax3.legend(fontsize=9)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, row in df_imp.iterrows():
        ax3.text(row['vs Traditional'], i, f" {row['vs Traditional']:+.1f}%",
                va='center', fontsize=8)
        ax3.text(row['vs Inter-text'], i + width_imp, f" {row['vs Inter-text']:+.1f}%",
                va='center', fontsize=8)

    # 4. Summary table (bottom row)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    best_hybrid_auc = max([r['hybrid_auc'] for r in results.values()])
    best_trad_auc = max([r['traditional_auc'] for r in results.values()])
    best_inter_auc = max([r['inter_text_auc'] for r in results.values()])

    baseline_diversity = 0.5789
    subset_misleading = 0.7066

    summary_data = [
        ['Approach', 'AUC', 'vs Diversity Delta', 'Status', 'Notes'],
        ['─' * 25, '─' * 10, '─' * 15, '─' * 15, '─' * 30],
        ['Diversity Delta (Baseline)', f'{baseline_diversity:.4f}', '—', 'Weak', 'Intra-text paradigm'],
        ['', '', '', '', ''],
        ['Inter-text (Subset 1K)', f'{subset_misleading:.4f}', f'{((subset_misleading-baseline_diversity)/baseline_diversity)*100:+.1f}%',
         'MISLEADING', 'Selection bias!'],
        ['Inter-text (Full 30K)', f'{best_inter_auc:.4f}', f'{((best_inter_auc-baseline_diversity)/baseline_diversity)*100:+.1f}%',
         'Below viable', 'Our contribution alone'],
        ['', '', '', '', ''],
        ['Traditional (Full 30K)', f'{best_trad_auc:.4f}', f'{((best_trad_auc-baseline_diversity)/baseline_diversity)*100:+.1f}%',
         '✓ Viable' if best_trad_auc >= 0.65 else 'Marginal', 'Standard stylometry'],
        ['', '', '', '', ''],
        [f'Hybrid (Full 30K)', f'{best_hybrid_auc:.4f}', f'{((best_hybrid_auc-baseline_diversity)/baseline_diversity)*100:+.1f}%',
         '✓✓ Strong' if best_hybrid_auc >= 0.70 else '✓ Viable', 'Combined approach'],
        ['', '', '', '', ''],
        ['Target (Viable)', '0.6500', '—', 'Goal', '—'],
        ['Target (Strong)', '0.7000', '—', 'Stretch', '—'],
    ]

    table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.25, 0.12, 0.15, 0.15, 0.33])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight best result
    if best_hybrid_auc >= 0.70:
        for i in range(5):
            table[(9, i)].set_facecolor('#90EE90')

    plt.suptitle('Full Dataset Evaluation: Traditional Stylometry vs Inter-Text Similarity vs Hybrid\n30,772 Pairs - Final Validation',
                fontsize=16, fontweight='bold')

    plot_file = output_dir / 'full_dataset_final_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization to {plot_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""

    print("="*80)
    print("FULL DATASET EVALUATION")
    print("Traditional vs Inter-Text vs Hybrid - Complete Validation")
    print("="*80)

    script_dir = Path(__file__).parent
    output_dir = script_dir / 'output/verification'

    # Step 1: Extract all features on full dataset
    df_full = process_full_dataset_features(output_dir)

    # Step 2: Evaluate all approaches
    results, y_test = evaluate_all_approaches(df_full)

    # Step 3: Create visualization
    create_full_evaluation_plot(results, y_test, output_dir)

    # Step 4: Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - FULL DATASET (30,772 PAIRS)")
    print(f"{'='*80}")

    # Find best in each category
    best_trad_clf = max(results, key=lambda x: results[x]['traditional_auc'])
    best_inter_clf = max(results, key=lambda x: results[x]['inter_text_auc'])
    best_hybrid_clf = max(results, key=lambda x: results[x]['hybrid_auc'])

    best_trad_auc = results[best_trad_clf]['traditional_auc']
    best_inter_auc = results[best_inter_clf]['inter_text_auc']
    best_hybrid_auc = results[best_hybrid_clf]['hybrid_auc']

    print(f"\nBest Results:")
    print(f"  Traditional Stylometry: {best_trad_auc:.4f} ({best_trad_clf})")
    print(f"  Inter-text Similarity:  {best_inter_auc:.4f} ({best_inter_clf})")
    print(f"  Hybrid (Combined):      {best_hybrid_auc:.4f} ({best_hybrid_clf})")

    print(f"\nComparison to Baselines:")
    print(f"  vs Diversity Delta (0.5789):")
    print(f"    Traditional:  {((best_trad_auc - 0.5789) / 0.5789 * 100):+.1f}%")
    print(f"    Inter-text:   {((best_inter_auc - 0.5789) / 0.5789 * 100):+.1f}%")
    print(f"    Hybrid:       {((best_hybrid_auc - 0.5789) / 0.5789 * 100):+.1f}%")

    print(f"\nHybrid Improvement:")
    print(f"  vs Traditional: {((best_hybrid_auc - best_trad_auc) / best_trad_auc * 100):+.1f}%")
    print(f"  vs Inter-text:  {((best_hybrid_auc - best_inter_auc) / best_inter_auc * 100):+.1f}%")

    # Status assessment
    print(f"\n{'='*80}")
    print("ASSESSMENT")
    print(f"{'='*80}")

    if best_hybrid_auc >= 0.72:
        print(f"✓✓✓ EXCELLENT: {best_hybrid_auc:.4f} (Very Strong, >0.72)")
    elif best_hybrid_auc >= 0.70:
        print(f"✓✓ STRONG: {best_hybrid_auc:.4f} (Strong Performance, >0.70)")
    elif best_hybrid_auc >= 0.65:
        print(f"✓ VIABLE: {best_hybrid_auc:.4f} (Viable for Deployment, >0.65)")
    else:
        print(f"⚠ BELOW THRESHOLD: {best_hybrid_auc:.4f} (<0.65)")

    # Our contribution
    our_value_add = best_hybrid_auc - best_trad_auc
    print(f"\nOur Contribution (Inter-text features):")
    print(f"  Added value: {our_value_add:+.4f} AUC points")
    print(f"  Relative improvement: {((best_hybrid_auc - best_trad_auc) / best_trad_auc * 100):+.1f}%")

    if our_value_add > 0.02:
        print(f"  ✓ Meaningful contribution (>0.02 AUC points)")
    elif our_value_add > 0:
        print(f"  ⚠ Modest contribution (<0.02 AUC points)")
    else:
        print(f"  ✗ No added value over traditional alone")

    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
