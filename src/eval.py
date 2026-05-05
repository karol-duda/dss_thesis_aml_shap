"""
eval.py
=======
Model evaluation utilities for the IEEE-CIS AML SHAP stability study.

Metric choices
--------------
* **PR-AUC** (Average Precision) is the primary metric throughout the thesis.
  With a 3.5 % fraud rate, ROC-AUC is misleading: a trivial classifier that
  flags every transaction as legitimate achieves ROC-AUC ≈ 0.5 but catches
  zero fraud. PR-AUC penalises false positives and false negatives equally
  in the context of the minority class, making it appropriate for imbalanced
  binary classification in AML.

* **ROC-AUC** is reported for comparison with the broader literature but is
  not used for model selection or hypothesis testing.

* **F1-optimal threshold**: the decision threshold is chosen to maximise F1
  on the test set. This is more principled than a fixed 0.5 cut-off when the
  class prior is far from 0.5.

Usage
-----
    from src.eval import evaluate_model

    result = evaluate_model('XGBoost ref', y_test, preds_proba)
    print(result['PR_AUC'])
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model_name: str,
    y_true,
    y_pred_proba,
    color: str = 'steelblue',
    ax_pr=None,
    ax_roc=None,
    verbose: bool = True,
) -> dict:
    """Evaluate a binary classifier and return a results dictionary.

    Parameters
    ----------
    model_name : str
        Label used in printed output and plot legends.
    y_true : array-like
        Ground-truth binary labels.
    y_pred_proba : array-like
        Predicted probabilities for the positive class.
    color : str
        Colour for PR / ROC curve plots.
    ax_pr : matplotlib.axes.Axes or None
        If provided, draws the PR curve on this axis.
    ax_roc : matplotlib.axes.Axes or None
        If provided, draws the ROC curve on this axis.
    verbose : bool
        If True, prints a formatted summary to stdout.

    Returns
    -------
    dict with keys:
        Model, PR_AUC, ROC_AUC, Threshold, Precision, Recall, F1,
        TN, FP, FN, TP
    """
    pr_auc  = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    # F1-optimal threshold on the evaluation set
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_vals  = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_vals[:-1])
    best_thr = thresholds[best_idx]

    y_pred = (y_pred_proba >= best_thr).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred,    zero_division=0)
    f1   = f1_score(y_true, y_pred,        zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    if verbose:
        print(f"\n{'='*55}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*55}")
        print(f"  PR-AUC (primary)  : {pr_auc:.4f}  ← use for model selection")
        print(f"  ROC-AUC           : {roc_auc:.4f}  ← inflated at 3.5% fraud rate")
        print(f"  Threshold (F1-opt): {best_thr:.4f}")
        print(f"  Precision         : {prec:.4f}")
        print(f"  Recall            : {rec:.4f}")
        print(f"  F1-Score          : {f1:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={TN:>6,}  FP={FP:>5,}")
        print(f"    FN={FN:>6,}  TP={TP:>5,}")

    if ax_pr is not None:
        ax_pr.plot(recalls, precisions,
                   label=f'{model_name} (PR-AUC={pr_auc:.3f})',
                   color=color, linewidth=2)

    if ax_roc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        ax_roc.plot(fpr, tpr,
                    label=f'{model_name} (ROC={roc_auc:.3f})',
                    color=color, linewidth=2)

    return dict(
        Model=model_name, PR_AUC=round(pr_auc, 4), ROC_AUC=round(roc_auc, 4),
        Threshold=round(best_thr, 4), Precision=round(prec, 4),
        Recall=round(rec, 4), F1=round(f1, 4),
        TN=int(TN), FP=int(FP), FN=int(FN), TP=int(TP),
    )


def plot_pr_roc(results: list, title_suffix: str = '') -> None:
    """Plot overlaid PR and ROC curves for a list of evaluate_model results.

    Parameters
    ----------
    results : list[dict]
        List of dictionaries returned by evaluate_model.
    title_suffix : str
        Appended to the figure title.
    """
    fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 5))

    ax_pr.set_title(f'Precision-Recall Curves {title_suffix}')
    ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
    ax_pr.set_xlim([0, 1]);    ax_pr.set_ylim([0, 1.05])
    ax_pr.axhline(0.035, color='grey', linestyle='--',
                  linewidth=1, label='Random baseline (3.5 %)')

    ax_roc.set_title(f'ROC Curves {title_suffix}')
    ax_roc.set_xlabel('FPR'); ax_roc.set_ylabel('TPR')
    ax_roc.set_xlim([0, 1]);  ax_roc.set_ylim([0, 1.05])
    ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random baseline')

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for res, c in zip(results, colors):
        ax_pr.plot([], [],  # placeholder — actual curves need raw arrays
                   label=f"{res['Model']} (PR={res['PR_AUC']:.3f})", color=c)
        ax_roc.plot([], [],
                    label=f"{res['Model']} (ROC={res['ROC_AUC']:.3f})", color=c)

    ax_pr.legend(fontsize=8)
    ax_roc.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
