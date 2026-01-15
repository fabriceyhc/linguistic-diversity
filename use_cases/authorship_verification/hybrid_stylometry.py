"""Hybrid Stylometry: Combining Inter-Text Similarity with Traditional Features.

After discovering that inter-text similarity alone achieves only 0.6152 AUC (below
viable threshold), this script implements a hybrid approach combining:

1. Our inter-text similarity features (semantic + lexical)
2. Traditional stylometry features (character n-grams, function words, POS patterns)

Expected performance: 0.72-0.75 AUC (viable and strong)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")


# ============================================================================
# TRADITIONAL STYLOMETRY FEATURES
# ============================================================================

def extract_char_ngrams(text, n=3, top_k=100):
    """Extract character n-gram features."""
    # Get character n-grams
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i+n])

    # Count frequency
    freq = Counter(ngrams)

    # Return as feature vector (top-k most common)
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

    # Tokenize
    import re
    tokens = re.findall(r'\b\w+\b', text.lower())

    # Count frequencies
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

    # Tokenize into sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) == 0:
        return {
            'avg_sentence_length': 0,
            'sentence_length_std': 0,
            'avg_words_per_sentence': 0,
        }

    # Sentence length
    sent_lengths = [len(s) for s in sentences]

    # Words per sentence
    words_per_sent = [len(re.findall(r'\b\w+\b', s)) for s in sentences]

    return {
        'avg_sentence_length': np.mean(sent_lengths),
        'sentence_length_std': np.std(sent_lengths),
        'avg_words_per_sentence': np.mean(words_per_sent),
        'sentence_count': len(sentences),
    }


def compute_traditional_similarity(text1, text2):
    """Compute similarity using traditional stylometric features."""

    # Character 3-grams (most discriminative in literature)
    ngrams1 = extract_char_ngrams(text1, n=3, top_k=50)
    ngrams2 = extract_char_ngrams(text2, n=3, top_k=50)

    # Get all keys
    all_keys = set(list(ngrams1.keys()) + list(ngrams2.keys()))

    # Compute cosine similarity
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

    # Syntactic patterns (use absolute differences)
    synt1 = extract_syntactic_patterns(text1)
    synt2 = extract_syntactic_patterns(text2)

    synt_features = ['avg_sentence_length', 'avg_words_per_sentence']
    synt_diffs = []
    for key in synt_features:
        val1 = synt1.get(key, 0)
        val2 = synt2.get(key, 0)
        if val1 + val2 > 0:
            # Normalized absolute difference
            synt_diffs.append(abs(val1 - val2) / (val1 + val2))
        else:
            synt_diffs.append(0)

    synt_sim = 1 - np.mean(synt_diffs)  # Convert difference to similarity

    return {
        'char_ngram_sim': char_ngram_sim,
        'func_word_sim': func_sim,
        'punct_sim': punct_sim,
        'syntactic_sim': synt_sim,
    }


# ============================================================================
# HYBRID FEATURE EXTRACTION
# ============================================================================

def extract_hybrid_features(df_with_similarity):
    """Extract both inter-text similarity and traditional stylometry features.

    Args:
        df_with_similarity: DataFrame with existing semantic_dist, lexical_dist

    Returns:
        DataFrame with all hybrid features
    """
    print("\\nLoading dataset to extract traditional features...")
    from datasets import load_dataset
    dataset = load_dataset("swan07/authorship-verification", split="validation")

    print(f"Processing {len(df_with_similarity)} pairs with traditional stylometry...")

    # Initialize lists for new features
    char_ngram_sims = []
    func_word_sims = []
    punct_sims = []
    syntactic_sims = []

    for idx, row in tqdm(df_with_similarity.iterrows(), total=len(df_with_similarity), desc="Extracting features"):
        pair_id = row['id']

        # Get the texts
        example = dataset[int(pair_id)]
        text1 = example['text1']
        text2 = example['text2']

        # Compute traditional features
        trad_features = compute_traditional_similarity(text1, text2)

        char_ngram_sims.append(trad_features['char_ngram_sim'])
        func_word_sims.append(trad_features['func_word_sim'])
        punct_sims.append(trad_features['punct_sim'])
        syntactic_sims.append(trad_features['syntactic_sim'])

    # Add to dataframe
    df_hybrid = df_with_similarity.copy()
    df_hybrid['char_ngram_sim'] = char_ngram_sims
    df_hybrid['func_word_sim'] = func_word_sims
    df_hybrid['punct_sim'] = punct_sims
    df_hybrid['syntactic_sim'] = syntactic_sims

    print("✓ Traditional features extracted")

    return df_hybrid


# ============================================================================
# HYBRID MODEL TRAINING
# ============================================================================

def train_hybrid_models(df_hybrid):
    """Train and evaluate hybrid models."""
    print("\\n" + "="*80)
    print("Training Hybrid Models")
    print("="*80)

    # Prepare feature sets

    # 1. Inter-text similarity only (our previous best)
    inter_text_features = ['semantic_dist', 'lexical_dist']

    # 2. Traditional stylometry only (for comparison)
    traditional_features = ['char_ngram_sim', 'func_word_sim', 'punct_sim', 'syntactic_sim']

    # 3. Hybrid (combine both)
    hybrid_features = inter_text_features + traditional_features

    # Convert distances to similarities for inter-text features
    df_model = df_hybrid.copy()
    df_model['semantic_sim'] = 1 - df_model['semantic_dist']
    df_model['lexical_sim'] = 1 - df_model['lexical_dist']

    # Update feature lists
    inter_text_features_sim = ['semantic_sim', 'lexical_sim']
    hybrid_features_sim = inter_text_features_sim + traditional_features

    # Prepare data
    X_inter = df_model[inter_text_features_sim].values
    X_trad = df_model[traditional_features].values
    X_hybrid = df_model[hybrid_features_sim].values
    y = df_model['label'].values

    # Split data
    X_inter_train, X_inter_test, y_train, y_test = train_test_split(
        X_inter, y, test_size=0.3, random_state=42, stratify=y
    )
    X_trad_train, X_trad_test, _, _ = train_test_split(
        X_trad, y, test_size=0.3, random_state=42, stratify=y
    )
    X_hybrid_train, X_hybrid_test, _, _ = train_test_split(
        X_hybrid, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\\nTrain size: {len(y_train)}")
    print(f"Test size: {len(y_test)}")

    results = {}

    # Test different classifiers
    classifiers = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    for clf_name, clf in classifiers.items():
        print(f"\\n{'-'*80}")
        print(f"Classifier: {clf_name}")
        print(f"{'-'*80}")

        # 1. Inter-text similarity only
        clf.fit(X_inter_train, y_train)
        y_pred_inter = clf.predict_proba(X_inter_test)[:, 1]
        auc_inter = roc_auc_score(y_test, y_pred_inter)
        print(f"  Inter-text only:     AUC = {auc_inter:.4f}")

        # 2. Traditional stylometry only
        clf.fit(X_trad_train, y_train)
        y_pred_trad = clf.predict_proba(X_trad_test)[:, 1]
        auc_trad = roc_auc_score(y_test, y_pred_trad)
        print(f"  Traditional only:    AUC = {auc_trad:.4f}")

        # 3. Hybrid
        clf.fit(X_hybrid_train, y_train)
        y_pred_hybrid = clf.predict_proba(X_hybrid_test)[:, 1]
        auc_hybrid = roc_auc_score(y_test, y_pred_hybrid)
        print(f"  Hybrid (combined):   AUC = {auc_hybrid:.4f}")
        print(f"  Improvement:         {((auc_hybrid - auc_inter) / auc_inter * 100):+.1f}% vs inter-text")

        # Store results
        results[clf_name] = {
            'inter_text_auc': auc_inter,
            'traditional_auc': auc_trad,
            'hybrid_auc': auc_hybrid,
            'model': clf,
            'y_pred': y_pred_hybrid,
        }

        # Feature importance (if available)
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            feature_names = hybrid_features_sim
            print(f"\\n  Feature importances:")
            for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
                print(f"    {name:20s}: {imp:.4f}")

    # Find best model
    best_clf_name = max(results, key=lambda x: results[x]['hybrid_auc'])
    best_auc = results[best_clf_name]['hybrid_auc']

    print(f"\\n{'='*80}")
    print(f"BEST CLASSIFIER: {best_clf_name}")
    print(f"  Hybrid AUC: {best_auc:.4f}")
    print(f"{'='*80}")

    return results, y_test


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_hybrid_visualization(results, y_test, output_dir):
    """Create visualization comparing all approaches."""
    print("\\n" + "="*80)
    print("Generating Hybrid Model Visualization")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Model comparison (top-left)
    ax = axes[0, 0]
    classifiers = list(results.keys())
    inter_aucs = [results[c]['inter_text_auc'] for c in classifiers]
    trad_aucs = [results[c]['traditional_auc'] for c in classifiers]
    hybrid_aucs = [results[c]['hybrid_auc'] for c in classifiers]

    x = np.arange(len(classifiers))
    width = 0.25

    ax.bar(x - width, inter_aucs, width, label='Inter-text only', color='skyblue')
    ax.bar(x, trad_aucs, width, label='Traditional only', color='lightcoral')
    ax.bar(x + width, hybrid_aucs, width, label='Hybrid', color='green')

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=15, ha='right')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Classifier Comparison: Inter-text vs Traditional vs Hybrid', fontweight='bold')
    ax.legend()
    ax.axhline(y=0.65, color='orange', linestyle='--', alpha=0.5, label='Viable (0.65)')
    ax.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, label='Strong (0.70)')
    ax.set_ylim([0.55, max(hybrid_aucs) + 0.05])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (inter, trad, hyb) in enumerate(zip(inter_aucs, trad_aucs, hybrid_aucs)):
        ax.text(i - width, inter, f'{inter:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, trad, f'{trad:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width, hyb, f'{hyb:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 2. Improvement over baseline (top-right)
    ax = axes[0, 1]
    baseline_diversity_delta = 0.5789
    baseline_inter_text = 0.6152

    best_clf = max(results, key=lambda x: results[x]['hybrid_auc'])
    best_hybrid_auc = results[best_clf]['hybrid_auc']

    approaches = ['Diversity\\nDelta', 'Inter-text\\n(Full)', 'Hybrid\\n(Best)']
    aucs_prog = [baseline_diversity_delta, baseline_inter_text, best_hybrid_auc]
    colors_prog = ['gray', 'skyblue', 'green']

    bars = ax.bar(approaches, aucs_prog, color=colors_prog)
    ax.axhline(y=0.65, color='orange', linestyle='--', alpha=0.5, label='Viable')
    ax.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, label='Strong')
    ax.set_ylabel('ROC-AUC')
    ax.set_title('Progressive Improvement', fontweight='bold')
    ax.legend()
    ax.set_ylim([0.50, max(aucs_prog) + 0.05])

    # Add value labels with improvement
    improvements = [0, ((baseline_inter_text - baseline_diversity_delta) / baseline_diversity_delta) * 100,
                   ((best_hybrid_auc - baseline_diversity_delta) / baseline_diversity_delta) * 100]
    for bar, auc, imp in zip(bars, aucs_prog, improvements):
        label = f'{auc:.4f}\\n({imp:+.1f}%)' if imp != 0 else f'{auc:.4f}'
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
               label, ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Feature contribution (bottom-left)
    ax = axes[1, 0]
    # Get best model
    best_model = results[best_clf]['model']
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_names = ['Semantic', 'Lexical', 'Char N-grams', 'Function Words', 'Punctuation', 'Syntactic']

        # Sort by importance
        sorted_idx = np.argsort(importances)
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_importances = [importances[i] for i in sorted_idx]

        ax.barh(range(len(sorted_features)), sorted_importances, color='skyblue')
        ax.set_yticks(range(len(sorted_features)))
        ax.set_yticklabels(sorted_features)
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance: {best_clf}', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (feat, imp) in enumerate(zip(sorted_features, sorted_importances)):
            ax.text(imp, i, f' {imp:.3f}', va='center', fontsize=9)

    # 4. Summary table (bottom-right)
    ax = axes[1, 1]
    ax.axis('off')

    summary_data = [
        ['Approach', 'AUC', 'vs Baseline', 'Status'],
        ['─' * 20, '─' * 10, '─' * 12, '─' * 15],
        ['Diversity Delta', f'{baseline_diversity_delta:.4f}', '—', 'Weak'],
        ['Inter-text (Subset 1K)', '0.7066', '—', 'Misleading'],
        ['Inter-text (Full 30K)', f'{baseline_inter_text:.4f}',
         f'{((baseline_inter_text - baseline_diversity_delta)/baseline_diversity_delta)*100:+.1f}%',
         'Below viable'],
        ['', '', '', ''],
        [f'Hybrid ({best_clf})', f'{best_hybrid_auc:.4f}',
         f'{((best_hybrid_auc - baseline_diversity_delta)/baseline_diversity_delta)*100:+.1f}%',
         '✓ Viable' if best_hybrid_auc >= 0.65 else 'Below viable'],
        ['', '', '', ''],
        ['Target (Viable)', '0.6500', '—', 'Goal'],
        ['Target (Strong)', '0.7000', '—', 'Stretch goal'],
    ]

    table = ax.table(cellText=summary_data, cellLoc='left', loc='center',
                    colWidths=[0.35, 0.2, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight hybrid result
    if best_hybrid_auc >= 0.65:
        table[(6, 0)].set_facecolor('#90EE90')
        table[(6, 1)].set_facecolor('#90EE90')
        table[(6, 2)].set_facecolor('#90EE90')
        table[(6, 3)].set_facecolor('#90EE90')

    plt.suptitle('Hybrid Stylometry Results: Combining Inter-Text Similarity + Traditional Features',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_file = output_dir / 'hybrid_stylometry_results.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved visualization to {plot_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("="*80)
    print("HYBRID STYLOMETRY")
    print("Combining Inter-Text Similarity + Traditional Features")
    print("="*80)

    script_dir = Path(__file__).parent
    output_dir = script_dir / 'output/verification'

    # Load existing inter-text similarity results
    print("\\nLoading inter-text similarity results...")
    df_similarity = pd.read_csv(output_dir / 'full_dataset_all-MiniLM-L12-v2.csv')
    print(f"Loaded {len(df_similarity)} pairs")

    # Subsample for faster testing (optional - comment out for full dataset)
    SAMPLE_SIZE = 5000  # Use 5000 for faster testing, None for full dataset
    if SAMPLE_SIZE is not None and len(df_similarity) > SAMPLE_SIZE:
        print(f"\\n⚠ Subsampling to {SAMPLE_SIZE} pairs for faster testing")
        df_similarity = df_similarity.sample(n=SAMPLE_SIZE, random_state=42)

    # Extract traditional features
    cache_file = output_dir / f'hybrid_features_{len(df_similarity)}.csv'

    if cache_file.exists():
        print(f"\\nLoading cached hybrid features from {cache_file}")
        df_hybrid = pd.read_csv(cache_file)
    else:
        df_hybrid = extract_hybrid_features(df_similarity)
        df_hybrid.to_csv(cache_file, index=False)
        print(f"\\n✓ Saved hybrid features to {cache_file}")

    # Train hybrid models
    results, y_test = train_hybrid_models(df_hybrid)

    # Create visualization
    create_hybrid_visualization(results, y_test, output_dir)

    # Final summary
    print("\\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    best_clf = max(results, key=lambda x: results[x]['hybrid_auc'])
    best_result = results[best_clf]

    print(f"\\nBest Classifier: {best_clf}")
    print(f"\\nPerformance:")
    print(f"  Inter-text only:     {best_result['inter_text_auc']:.4f}")
    print(f"  Traditional only:    {best_result['traditional_auc']:.4f}")
    print(f"  Hybrid (combined):   {best_result['hybrid_auc']:.4f}")

    print(f"\\nComparison to baselines:")
    print(f"  vs Diversity Delta (0.5789):  {((best_result['hybrid_auc'] - 0.5789) / 0.5789 * 100):+.1f}%")
    print(f"  vs Inter-text (0.6152):       {((best_result['hybrid_auc'] - 0.6152) / 0.6152 * 100):+.1f}%")

    if best_result['hybrid_auc'] >= 0.72:
        print(f"\\n✓✓✓ EXCELLENT: Achieved {best_result['hybrid_auc']:.4f} (very strong, >0.72)")
    elif best_result['hybrid_auc'] >= 0.70:
        print(f"\\n✓✓ STRONG: Achieved {best_result['hybrid_auc']:.4f} (strong, >0.70)")
    elif best_result['hybrid_auc'] >= 0.65:
        print(f"\\n✓ VIABLE: Achieved {best_result['hybrid_auc']:.4f} (viable for deployment, >0.65)")
    else:
        print(f"\\n⚠ BELOW THRESHOLD: {best_result['hybrid_auc']:.4f} (<0.65)")

    print("\\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
