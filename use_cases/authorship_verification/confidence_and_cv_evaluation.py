"""Confidence Scoring and Cross-Validation: Final Validation with Calibration.

This script implements:
1. Calibrated probability confidence scoring (isotonic regression & Platt scaling)
2. K-fold cross-validation with multiple random seeds
3. Comprehensive evaluation with confidence intervals
4. Production-ready calibrated models

Improves on full_dataset_evaluation.py by adding:
- Calibration curves and Brier scores
- Cross-validation stability assessment
- Confidence intervals on performance metrics
- Reliability estimates for predictions
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")


# ============================================================================
# CALIBRATED CLASSIFIER EVALUATION
# ============================================================================

def evaluate_with_calibration(X, y, features_name, n_folds=5, random_seeds=[42, 123, 456]):
    """
    Evaluate classifiers with probability calibration.

    Args:
        X: Feature matrix
        y: Labels
        features_name: Name of feature set (for labeling)
        n_folds: Number of folds for cross-validation
        random_seeds: List of random seeds to test

    Returns:
        Dictionary with calibration results, CV scores, and trained models
    """

    print(f"\n{'='*80}")
    print(f"Evaluating: {features_name}")
    print(f"{'='*80}")

    # Classifiers to test
    classifiers = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}

    for clf_name, base_clf in classifiers.items():
        print(f"\n{'-'*80}")
        print(f"Classifier: {clf_name}")
        print(f"{'-'*80}")

        # Store results across seeds
        cv_aucs = []
        cv_brier_uncalibrated = []
        cv_brier_isotonic = []
        cv_brier_sigmoid = []

        all_predictions = {
            'uncalibrated': [],
            'isotonic': [],
            'sigmoid': [],
        }

        all_labels = []

        # Cross-validation with multiple seeds
        for seed in random_seeds:
            print(f"  Seed {seed}:")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )

            # 1. Uncalibrated baseline
            base_clf_copy = classifiers[clf_name]  # Fresh copy
            base_clf_copy.fit(X_train, y_train)
            y_pred_uncal = base_clf_copy.predict_proba(X_test)[:, 1]

            auc_uncal = roc_auc_score(y_test, y_pred_uncal)
            brier_uncal = brier_score_loss(y_test, y_pred_uncal)

            # 2. Isotonic calibration
            clf_isotonic = CalibratedClassifierCV(
                classifiers[clf_name],
                method='isotonic',
                cv=5
            )
            clf_isotonic.fit(X_train, y_train)
            y_pred_iso = clf_isotonic.predict_proba(X_test)[:, 1]

            auc_iso = roc_auc_score(y_test, y_pred_iso)
            brier_iso = brier_score_loss(y_test, y_pred_iso)

            # 3. Sigmoid (Platt scaling) calibration
            clf_sigmoid = CalibratedClassifierCV(
                classifiers[clf_name],
                method='sigmoid',
                cv=5
            )
            clf_sigmoid.fit(X_train, y_train)
            y_pred_sig = clf_sigmoid.predict_proba(X_test)[:, 1]

            auc_sig = roc_auc_score(y_test, y_pred_sig)
            brier_sig = brier_score_loss(y_test, y_pred_sig)

            print(f"    Uncalibrated:  AUC={auc_uncal:.4f}, Brier={brier_uncal:.4f}")
            print(f"    Isotonic:      AUC={auc_iso:.4f}, Brier={brier_iso:.4f}")
            print(f"    Sigmoid:       AUC={auc_sig:.4f}, Brier={brier_sig:.4f}")

            # Store results
            cv_aucs.append(auc_iso)  # Use isotonic as default
            cv_brier_uncalibrated.append(brier_uncal)
            cv_brier_isotonic.append(brier_iso)
            cv_brier_sigmoid.append(brier_sig)

            all_predictions['uncalibrated'].extend(y_pred_uncal)
            all_predictions['isotonic'].extend(y_pred_iso)
            all_predictions['sigmoid'].extend(y_pred_sig)
            all_labels.extend(y_test)

        # Compute statistics across seeds
        mean_auc = np.mean(cv_aucs)
        std_auc = np.std(cv_aucs)

        mean_brier_uncal = np.mean(cv_brier_uncalibrated)
        mean_brier_iso = np.mean(cv_brier_isotonic)
        mean_brier_sig = np.mean(cv_brier_sigmoid)

        print(f"\n  Cross-validation results (n={len(random_seeds)} seeds):")
        print(f"    AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"    Brier (uncalibrated): {mean_brier_uncal:.4f}")
        print(f"    Brier (isotonic):     {mean_brier_iso:.4f}")
        print(f"    Brier (sigmoid):      {mean_brier_sig:.4f}")

        # Determine best calibration method
        if mean_brier_iso < mean_brier_sig and mean_brier_iso < mean_brier_uncal:
            best_method = 'isotonic'
            best_brier = mean_brier_iso
        elif mean_brier_sig < mean_brier_uncal:
            best_method = 'sigmoid'
            best_brier = mean_brier_sig
        else:
            best_method = 'uncalibrated'
            best_brier = mean_brier_uncal

        print(f"    Best calibration: {best_method} (Brier={best_brier:.4f})")

        # Train final model on full dataset with best calibration
        print(f"\n  Training final production model with {best_method} calibration...")
        if best_method == 'uncalibrated':
            final_model = classifiers[clf_name]
            final_model.fit(X, y)
        else:
            final_model = CalibratedClassifierCV(
                classifiers[clf_name],
                method=best_method,
                cv=5
            )
            final_model.fit(X, y)

        print(f"    ✓ Final model trained on all {len(y)} samples")

        # Store results
        results[clf_name] = {
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'cv_aucs': cv_aucs,
            'mean_brier_uncalibrated': mean_brier_uncal,
            'mean_brier_isotonic': mean_brier_iso,
            'mean_brier_sigmoid': mean_brier_sig,
            'best_calibration': best_method,
            'best_brier': best_brier,
            'final_model': final_model,
            'all_predictions': all_predictions,
            'all_labels': np.array(all_labels),
        }

    return results


# ============================================================================
# K-FOLD CROSS-VALIDATION WITH CONFIDENCE INTERVALS
# ============================================================================

def kfold_cross_validation(X, y, features_name, classifier, k=5, n_repeats=3):
    """
    Perform k-fold cross-validation with multiple repeats.

    Args:
        X: Feature matrix
        y: Labels
        features_name: Name of feature set
        classifier: Classifier instance
        k: Number of folds
        n_repeats: Number of times to repeat with different seeds

    Returns:
        Dictionary with fold-wise results and statistics
    """

    print(f"\n{'='*80}")
    print(f"K-Fold Cross-Validation: {features_name}")
    print(f"{'='*80}")

    all_fold_aucs = []
    all_fold_briers = []

    for repeat in range(n_repeats):
        seed = 42 + repeat * 111
        print(f"\nRepeat {repeat + 1}/{n_repeats} (seed={seed}):")

        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

        fold_aucs = []
        fold_briers = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]

            # Train with calibration
            clf_cal = CalibratedClassifierCV(classifier, method='isotonic', cv=3)
            clf_cal.fit(X_train_fold, y_train_fold)

            y_pred = clf_cal.predict_proba(X_test_fold)[:, 1]

            auc = roc_auc_score(y_test_fold, y_pred)
            brier = brier_score_loss(y_test_fold, y_pred)

            fold_aucs.append(auc)
            fold_briers.append(brier)

            print(f"  Fold {fold_idx + 1}: AUC={auc:.4f}, Brier={brier:.4f}")

        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)

        print(f"  Mean: AUC={mean_auc:.4f} ± {std_auc:.4f}")

        all_fold_aucs.extend(fold_aucs)
        all_fold_briers.extend(fold_briers)

    # Overall statistics
    overall_mean_auc = np.mean(all_fold_aucs)
    overall_std_auc = np.std(all_fold_aucs)
    overall_mean_brier = np.mean(all_fold_briers)

    # 95% confidence interval
    ci_lower = overall_mean_auc - 1.96 * overall_std_auc / np.sqrt(len(all_fold_aucs))
    ci_upper = overall_mean_auc + 1.96 * overall_std_auc / np.sqrt(len(all_fold_aucs))

    print(f"\n{'='*40}")
    print(f"Overall Statistics ({k}-fold × {n_repeats} repeats = {len(all_fold_aucs)} folds):")
    print(f"  AUC: {overall_mean_auc:.4f} ± {overall_std_auc:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Brier: {overall_mean_brier:.4f}")
    print(f"{'='*40}")

    return {
        'mean_auc': overall_mean_auc,
        'std_auc': overall_std_auc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_brier': overall_mean_brier,
        'all_fold_aucs': all_fold_aucs,
        'all_fold_briers': all_fold_briers,
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_calibration_curves(results_dict, output_file):
    """Plot calibration curves for all approaches."""

    print(f"\nGenerating calibration curves...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    approach_names = list(results_dict.keys())

    for idx, approach_name in enumerate(approach_names):
        approach_results = results_dict[approach_name]

        # Find best classifier
        best_clf = max(approach_results, key=lambda x: approach_results[x]['mean_auc'])
        clf_results = approach_results[best_clf]

        ax = axes[idx]

        # Plot calibration curve for each method
        for method in ['uncalibrated', 'isotonic', 'sigmoid']:
            predictions = np.array(clf_results['all_predictions'][method])
            labels = clf_results['all_labels']

            # Bin predictions
            n_bins = 10
            bins = np.linspace(0, 1, n_bins + 1)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            bin_true_freq = []
            bin_pred_mean = []

            for i in range(n_bins):
                mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
                if mask.sum() > 0:
                    bin_true_freq.append(labels[mask].mean())
                    bin_pred_mean.append(predictions[mask].mean())
                else:
                    bin_true_freq.append(np.nan)
                    bin_pred_mean.append(bin_centers[i])

            # Plot
            label_name = method.capitalize()
            if method == clf_results['best_calibration']:
                label_name += ' (best)'
                linewidth = 2.5
                alpha = 1.0
            else:
                linewidth = 1.5
                alpha = 0.6

            ax.plot(bin_pred_mean, bin_true_freq, 'o-',
                   label=f"{label_name} (Brier={clf_results[f'mean_brier_{method}']:.3f})",
                   linewidth=linewidth, alpha=alpha, markersize=6)

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, linewidth=1.5, label='Perfect calibration')

        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('True Frequency', fontsize=11)
        ax.set_title(f'{approach_name}\n{best_clf} (AUC={clf_results["mean_auc"]:.4f})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # Hide unused subplots
    for idx in range(len(approach_names), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Calibration Curves: Predicted Probability vs True Frequency\nLower Brier Score = Better Calibration',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved calibration curves to {output_file}")


def plot_cv_distributions(cv_results_dict, output_file):
    """Plot cross-validation AUC distributions."""

    print(f"\nGenerating CV distributions plot...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    approach_names = list(cv_results_dict.keys())

    for idx, approach_name in enumerate(approach_names):
        ax = axes[idx]

        cv_results = cv_results_dict[approach_name]
        fold_aucs = cv_results['all_fold_aucs']

        # Histogram
        ax.hist(fold_aucs, bins=15, alpha=0.7, color='skyblue', edgecolor='black')

        # Statistics
        mean_auc = cv_results['mean_auc']
        std_auc = cv_results['std_auc']
        ci_lower = cv_results['ci_lower']
        ci_upper = cv_results['ci_upper']

        # Add vertical lines
        ax.axvline(mean_auc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_auc:.4f}')
        ax.axvline(ci_lower, color='orange', linestyle=':', linewidth=2, label=f'95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]')
        ax.axvline(ci_upper, color='orange', linestyle=':', linewidth=2)

        ax.set_xlabel('ROC-AUC', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{approach_name}\n{len(fold_aucs)} folds, σ={std_auc:.4f}',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle('Cross-Validation Stability: AUC Distribution Across Folds',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved CV distributions to {output_file}")


def plot_comprehensive_summary(results_dict, cv_results_dict, output_file):
    """Create comprehensive summary visualization."""

    print(f"\nGenerating comprehensive summary...")

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    approach_names = list(results_dict.keys())

    # 1. AUC comparison with confidence intervals (top-left, large)
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    x = np.arange(len(approach_names))
    width = 0.25

    # Get best classifier for each approach
    means = []
    stds = []
    for approach_name in approach_names:
        best_clf = max(results_dict[approach_name],
                      key=lambda x: results_dict[approach_name][x]['mean_auc'])
        means.append(results_dict[approach_name][best_clf]['mean_auc'])
        stds.append(results_dict[approach_name][best_clf]['std_auc'])

    bars = ax1.bar(x, means, width*2, yerr=stds, capsize=5,
                   alpha=0.7, color=['coral', 'skyblue', 'green'])

    ax1.set_xticks(x)
    ax1.set_xticklabels(approach_names, fontsize=12)
    ax1.set_ylabel('ROC-AUC', fontsize=13)
    ax1.set_title('Performance with Calibration: Mean ± Std Dev\n(Full Dataset: 30,772 pairs)',
                 fontsize=14, fontweight='bold')
    ax1.axhline(y=0.65, color='orange', linestyle='--', alpha=0.5, linewidth=2)
    ax1.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_ylim([0.55, max(means) + 0.05])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax1.text(i, mean + std + 0.01, f'{mean:.4f}\n±{std:.4f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Calibration quality (Brier scores) (top-right)
    ax2 = fig.add_subplot(gs[0, 2])

    brier_data = []
    for approach_name in approach_names:
        best_clf = max(results_dict[approach_name],
                      key=lambda x: results_dict[approach_name][x]['mean_auc'])
        brier_data.append({
            'Approach': approach_name,
            'Uncalibrated': results_dict[approach_name][best_clf]['mean_brier_uncalibrated'],
            'Isotonic': results_dict[approach_name][best_clf]['mean_brier_isotonic'],
            'Sigmoid': results_dict[approach_name][best_clf]['mean_brier_sigmoid'],
        })

    df_brier = pd.DataFrame(brier_data)
    x_pos = np.arange(len(approach_names))
    width_brier = 0.25

    ax2.bar(x_pos - width_brier, df_brier['Uncalibrated'], width_brier,
           label='Uncalibrated', alpha=0.7, color='gray')
    ax2.bar(x_pos, df_brier['Isotonic'], width_brier,
           label='Isotonic', alpha=0.7, color='blue')
    ax2.bar(x_pos + width_brier, df_brier['Sigmoid'], width_brier,
           label='Sigmoid', alpha=0.7, color='purple')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([a.split()[0] for a in approach_names], fontsize=9)
    ax2.set_ylabel('Brier Score (lower=better)', fontsize=10)
    ax2.set_title('Calibration Quality', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=0.15, color='orange', linestyle='--', alpha=0.3, linewidth=1)

    # 3. Cross-validation confidence intervals (middle-right)
    ax3 = fig.add_subplot(gs[1, 2])

    for i, approach_name in enumerate(approach_names):
        cv_res = cv_results_dict[approach_name]
        mean = cv_res['mean_auc']
        ci_lower = cv_res['ci_lower']
        ci_upper = cv_res['ci_upper']

        ax3.errorbar([i], [mean], yerr=[[mean - ci_lower], [ci_upper - mean]],
                    fmt='o', markersize=10, capsize=10, capthick=2, linewidth=2)

    ax3.set_xticks(range(len(approach_names)))
    ax3.set_xticklabels([a.split()[0] for a in approach_names], fontsize=9)
    ax3.set_ylabel('ROC-AUC', fontsize=10)
    ax3.set_title('95% Confidence Intervals\n(K-Fold CV)', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=0.65, color='orange', linestyle='--', alpha=0.3)
    ax3.axhline(y=0.70, color='green', linestyle='--', alpha=0.3)

    # 4. Summary table (bottom row)
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')

    summary_data = [['Approach', 'Best Classifier', 'AUC (Mean±Std)', '95% CI', 'Brier (Best)', 'Status']]
    summary_data.append(['─' * 20] * 6)

    for approach_name in approach_names:
        best_clf = max(results_dict[approach_name],
                      key=lambda x: results_dict[approach_name][x]['mean_auc'])

        mean_auc = results_dict[approach_name][best_clf]['mean_auc']
        std_auc = results_dict[approach_name][best_clf]['std_auc']
        best_brier = results_dict[approach_name][best_clf]['best_brier']
        ci_lower = cv_results_dict[approach_name]['ci_lower']
        ci_upper = cv_results_dict[approach_name]['ci_upper']

        if mean_auc >= 0.72:
            status = '✓✓✓ Excellent'
        elif mean_auc >= 0.70:
            status = '✓✓ Strong'
        elif mean_auc >= 0.65:
            status = '✓ Viable'
        else:
            status = '⚠ Below threshold'

        summary_data.append([
            approach_name,
            best_clf,
            f'{mean_auc:.4f} ± {std_auc:.4f}',
            f'[{ci_lower:.4f}, {ci_upper:.4f}]',
            f'{best_brier:.4f}',
            status
        ])

    table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                     colWidths=[0.20, 0.15, 0.18, 0.20, 0.12, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)

    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight best
    best_approach_idx = approach_names.index(max(approach_names,
        key=lambda x: results_dict[x][max(results_dict[x],
            key=lambda c: results_dict[x][c]['mean_auc'])]['mean_auc']))

    for i in range(6):
        table[(best_approach_idx + 2, i)].set_facecolor('#90EE90')

    plt.suptitle('Confidence Scoring & Cross-Validation: Final Validation Results\nCalibrated Probabilities with K-Fold CV (Full 30,772 pairs)',
                fontsize=16, fontweight='bold')

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved comprehensive summary to {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""

    print("="*80)
    print("CONFIDENCE SCORING & CROSS-VALIDATION EVALUATION")
    print("Calibrated Probabilities + K-Fold CV on Full Dataset")
    print("="*80)

    script_dir = Path(__file__).parent
    output_dir = script_dir / 'output/verification'

    # Load cached features
    print(f"\nLoading full dataset features...")
    cache_file = output_dir / 'full_dataset_all_features.csv'

    if not cache_file.exists():
        print(f"Error: {cache_file} not found!")
        print("Please run full_dataset_evaluation.py first to generate features.")
        return

    df_full = pd.read_csv(cache_file)
    print(f"✓ Loaded {len(df_full)} pairs with all features")

    # Prepare feature sets
    df_model = df_full.copy()
    df_model['semantic_sim'] = 1 - df_model['semantic_dist']
    df_model['lexical_sim'] = 1 - df_model['lexical_dist']

    inter_text_features = ['semantic_sim', 'lexical_sim']
    traditional_features = ['char_ngram_sim', 'func_word_sim', 'punct_sim', 'syntactic_sim']
    hybrid_features = inter_text_features + traditional_features

    X_inter = df_model[inter_text_features].values
    X_trad = df_model[traditional_features].values
    X_hybrid = df_model[hybrid_features].values
    y = df_model['label'].values

    print(f"\nDataset: {len(y)} pairs")
    print(f"  Traditional features: {len(traditional_features)}")
    print(f"  Inter-text features: {len(inter_text_features)}")
    print(f"  Hybrid features: {len(hybrid_features)}")

    # ========================================================================
    # PART 1: CALIBRATION EVALUATION
    # ========================================================================

    print(f"\n{'='*80}")
    print("PART 1: CALIBRATION EVALUATION")
    print(f"{'='*80}")

    results_trad = evaluate_with_calibration(X_trad, y, "Traditional Stylometry",
                                            n_folds=5, random_seeds=[42, 123, 456])

    results_inter = evaluate_with_calibration(X_inter, y, "Inter-text Similarity",
                                             n_folds=5, random_seeds=[42, 123, 456])

    results_hybrid = evaluate_with_calibration(X_hybrid, y, "Hybrid (Combined)",
                                              n_folds=5, random_seeds=[42, 123, 456])

    all_results = {
        'Traditional Stylometry': results_trad,
        'Inter-text Similarity': results_inter,
        'Hybrid (Combined)': results_hybrid,
    }

    # ========================================================================
    # PART 2: K-FOLD CROSS-VALIDATION
    # ========================================================================

    print(f"\n{'='*80}")
    print("PART 2: K-FOLD CROSS-VALIDATION WITH MULTIPLE REPEATS")
    print(f"{'='*80}")

    # Use best classifier from Part 1
    best_clf_type = max(results_hybrid, key=lambda x: results_hybrid[x]['mean_auc'])
    print(f"\nUsing best classifier: {best_clf_type}")

    if best_clf_type == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    elif best_clf_type == 'GradientBoosting':
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        clf = LogisticRegression(random_state=42, max_iter=1000)

    cv_trad = kfold_cross_validation(X_trad, y, "Traditional Stylometry", clf, k=5, n_repeats=3)
    cv_inter = kfold_cross_validation(X_inter, y, "Inter-text Similarity", clf, k=5, n_repeats=3)
    cv_hybrid = kfold_cross_validation(X_hybrid, y, "Hybrid (Combined)", clf, k=5, n_repeats=3)

    all_cv_results = {
        'Traditional Stylometry': cv_trad,
        'Inter-text Similarity': cv_inter,
        'Hybrid (Combined)': cv_hybrid,
    }

    # ========================================================================
    # PART 3: SAVE PRODUCTION MODELS
    # ========================================================================

    print(f"\n{'='*80}")
    print("PART 3: SAVING PRODUCTION-READY MODELS")
    print(f"{'='*80}")

    for approach_name in all_results.keys():
        best_clf = max(all_results[approach_name],
                      key=lambda x: all_results[approach_name][x]['mean_auc'])

        model = all_results[approach_name][best_clf]['final_model']

        model_filename = approach_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_path = output_dir / f'calibrated_model_{model_filename}_{best_clf.lower()}.pkl'

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"✓ Saved {approach_name} ({best_clf}) to {model_path}")

    # ========================================================================
    # PART 4: VISUALIZATIONS
    # ========================================================================

    print(f"\n{'='*80}")
    print("PART 4: GENERATING VISUALIZATIONS")
    print(f"{'='*80}")

    plot_calibration_curves(all_results, output_dir / 'calibration_curves.png')
    plot_cv_distributions(all_cv_results, output_dir / 'cv_distributions.png')
    plot_comprehensive_summary(all_results, all_cv_results, output_dir / 'confidence_cv_summary.png')

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print(f"\n{'='*80}")
    print("FINAL SUMMARY - CALIBRATED MODELS WITH CROSS-VALIDATION")
    print(f"{'='*80}")

    for approach_name in all_results.keys():
        print(f"\n{approach_name}:")

        best_clf = max(all_results[approach_name],
                      key=lambda x: all_results[approach_name][x]['mean_auc'])

        calib_results = all_results[approach_name][best_clf]
        cv_results = all_cv_results[approach_name]

        print(f"  Best Classifier: {best_clf}")
        print(f"  AUC (calibration): {calib_results['mean_auc']:.4f} ± {calib_results['std_auc']:.4f}")
        print(f"  AUC (k-fold CV):   {cv_results['mean_auc']:.4f} ± {cv_results['std_auc']:.4f}")
        print(f"  95% CI:            [{cv_results['ci_lower']:.4f}, {cv_results['ci_upper']:.4f}]")
        print(f"  Brier Score:       {calib_results['best_brier']:.4f} ({calib_results['best_calibration']})")

    # Best overall
    best_approach = max(all_results.keys(),
                       key=lambda x: all_results[x][max(all_results[x],
                           key=lambda c: all_results[x][c]['mean_auc'])]['mean_auc'])

    best_clf = max(all_results[best_approach],
                  key=lambda x: all_results[best_approach][x]['mean_auc'])

    best_auc = all_results[best_approach][best_clf]['mean_auc']
    best_ci_lower = all_cv_results[best_approach]['ci_lower']
    best_ci_upper = all_cv_results[best_approach]['ci_upper']

    print(f"\n{'='*80}")
    print(f"BEST OVERALL: {best_approach} ({best_clf})")
    print(f"  AUC: {best_auc:.4f}")
    print(f"  95% CI: [{best_ci_lower:.4f}, {best_ci_upper:.4f}]")

    if best_auc >= 0.72:
        print(f"  Status: ✓✓✓ EXCELLENT (>0.72)")
    elif best_auc >= 0.70:
        print(f"  Status: ✓✓ STRONG (>0.70)")
    elif best_auc >= 0.65:
        print(f"  Status: ✓ VIABLE (>0.65)")
    else:
        print(f"  Status: ⚠ BELOW THRESHOLD (<0.65)")

    print(f"{'='*80}")
    print("✓ COMPLETE")
    print(f"{'='*80}")

    print(f"\nOutput files:")
    print(f"  - calibration_curves.png")
    print(f"  - cv_distributions.png")
    print(f"  - confidence_cv_summary.png")
    print(f"  - calibrated_model_*.pkl (production models)")


if __name__ == "__main__":
    main()
