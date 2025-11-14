import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, accuracy_score
from scipy import stats
import pandas as pd
from statsmodels.stats.multitest import multipletests


def calculate_metrics(y_true, y_pred, baseline_scores=None, model_name="Model", alpha=0.05):


    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics_dict = {}

    # 1. Basic metrics calculation
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    metrics_dict['accuracy'] = accuracy

    # Kappa coefficient
    kappa = cohen_kappa_score(y_true, y_pred)
    metrics_dict['kappa'] = kappa

    # F1 score (macro average)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    metrics_dict['f1_macro'] = f1_macro

    # F1 scores per class
    f1_per_class = f1_score(y_true, y_pred, average=None)
    for i, f1 in enumerate(f1_per_class):
        metrics_dict[f'f1_class_{i}'] = f1

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics_dict['confusion_matrix'] = cm

    # 2. Statistical test - Wilcoxon signed-rank test (if baseline scores are provided)
    wilcoxon_results = {}

    if baseline_scores is not None:
        # Ensure current model also has multiple run results
        if hasattr(accuracy, '__len__') and len(accuracy) > 1:
            current_scores = accuracy
        else:
            # If only single result, assume multiple runs average
            current_scores = [accuracy]

        baseline_scores = np.array(baseline_scores)
        current_scores = np.array(current_scores)

        # Ensure both score arrays have same length
        min_len = min(len(baseline_scores), len(current_scores))
        baseline_scores = baseline_scores[:min_len]
        current_scores = current_scores[:min_len]

        if len(baseline_scores) > 1:
            # Perform Wilcoxon signed-rank test
            try:
                wilcoxon_stat, wilcoxon_p = stats.wilcoxon(current_scores, baseline_scores)
                wilcoxon_results['statistic'] = wilcoxon_stat
                wilcoxon_results['p_value'] = wilcoxon_p
                wilcoxon_results['significant'] = wilcoxon_p < alpha
            except Exception as e:
                print(f"Wilcoxon test execution failed: {e}")
                wilcoxon_results['error'] = str(e)
        else:
            wilcoxon_results['error'] = "At least 2 runs required for Wilcoxon test"

    metrics_dict['wilcoxon_test'] = wilcoxon_results

    # 3. Multiple model comparison with Bonferroni correction
    multiple_comparison_results = perform_multiple_comparisons(
        [current_scores] if 'current_scores' in locals() else [accuracy],
        baseline_scores if baseline_scores is not None else [accuracy],
        model_names=[model_name, "Baseline"],
        alpha=alpha
    )

    metrics_dict['multiple_comparisons'] = multiple_comparison_results

    return metrics_dict


def perform_multiple_comparisons(model_scores_list, baseline_scores, model_names, alpha=0.05):


    results = {}
    p_values = []
    comparisons = []

    # Compare each model with baseline
    for i, model_scores in enumerate(model_scores_list):
        if len(model_scores) > 1 and len(baseline_scores) > 1:
            try:
                # Wilcoxon test
                stat, p_val = stats.wilcoxon(model_scores, baseline_scores)
                p_values.append(p_val)
                comparisons.append(f"{model_names[i]} vs Baseline")

                results[f"{model_names[i]}_vs_baseline"] = {
                    'statistic': stat,
                    'p_value': p_val,
                    'significant_uncorrected': p_val < alpha
                }
            except Exception as e:
                print(f"{model_names[i]} vs Baseline test failed: {e}")

    # Apply Bonferroni correction
    if p_values:
        rejected, corrected_pvals, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')

        for i, comp in enumerate(comparisons):
            results[comp].update({
                'p_value_corrected': corrected_pvals[i],
                'significant_corrected': rejected[i]
            })

    return results


def calculate_cross_validation_metrics(all_fold_metrics):
    """
    Calculate cross-validation comprehensive metrics

    Parameters:
    ----------
    all_fold_metrics : list of dict
        List of metric dictionaries for all folds

    Returns:
    ----------
    cv_metrics : dict
        Cross-validation comprehensive metrics
    """

    cv_metrics = {}

    # Extract metrics from each fold
    accuracies = [fold['accuracy'] for fold in all_fold_metrics]
    kappas = [fold['kappa'] for fold in all_fold_metrics]
    f1_macros = [fold['f1_macro'] for fold in all_fold_metrics]

    # Calculate mean and standard deviation
    cv_metrics['mean_accuracy'] = np.mean(accuracies)
    cv_metrics['std_accuracy'] = np.std(accuracies)
    cv_metrics['mean_kappa'] = np.mean(kappas)
    cv_metrics['std_kappa'] = np.std(kappas)
    cv_metrics['mean_f1_macro'] = np.mean(f1_macros)
    cv_metrics['std_f1_macro'] = np.std(f1_macros)

    # Calculate confidence interval (95%)
    n_folds = len(accuracies)
    if n_folds > 1:
        t_value = stats.t.ppf(0.975, n_folds - 1)  # t-value for 95% confidence interval

        cv_metrics['accuracy_ci'] = {
            'lower': cv_metrics['mean_accuracy'] - t_value * (cv_metrics['std_accuracy'] / np.sqrt(n_folds)),
            'upper': cv_metrics['mean_accuracy'] + t_value * (cv_metrics['std_accuracy'] / np.sqrt(n_folds))
        }

    return cv_metrics


def print_metrics_summary(metrics_dict, model_name="Model"):
    """
    Print metrics summary

    Parameters:
    ----------
    metrics_dict : dict
        Metrics dictionary
    model_name : str
        Model name
    """

    print(f"\n{'=' * 50}")
    print(f"{model_name} Evaluation Results")
    print(f"{'=' * 50}")

    print(f"Accuracy (ACC): {metrics_dict['accuracy']:.4f}")
    print(f"Kappa coefficient: {metrics_dict['kappa']:.4f}")
    print(f"Macro F1: {metrics_dict['f1_macro']:.4f}")

    # F1 scores per class
    for i in range(len([k for k in metrics_dict.keys() if k.startswith('f1_class_')])):
        print(f"Class {i} F1: {metrics_dict[f'f1_class_{i}']:.4f}")

    # Wilcoxon test results
    if 'wilcoxon_test' in metrics_dict and 'p_value' in metrics_dict['wilcoxon_test']:
        wilcoxon = metrics_dict['wilcoxon_test']
        significance = "Significant" if wilcoxon['significant'] else "Not significant"
        print(f"Wilcoxon test: p-value = {wilcoxon['p_value']:.4f} ({significance})")

    # Multiple comparison results
    if 'multiple_comparisons' in metrics_dict:
        for comp, result in metrics_dict['multiple_comparisons'].items():
            if 'p_value_corrected' in result:
                sig_status = "Significant" if result['significant_corrected'] else "Not significant"
                print(f"{comp} (Bonferroni corrected): p-value = {result['p_value_corrected']:.4f} ({sig_status})")

