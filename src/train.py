"""
train.py
========
XGBoost training with checkpoint / resume support for the IEEE-CIS AML
SHAP stability study.

Hyperparameter rationale
------------------------
* ``max_depth=12``: Deep trees are necessary to capture the complex
  interactions in 440 engineered features (UID aggregates × V-columns).
  Regularised via ``subsample`` and ``colsample_bytree``.

* ``learning_rate=0.02``: Conservative step size. Requires many estimators
  (5,000+) but produces more stable feature attributions across seeds.

* ``subsample=0.8``, ``colsample_bytree=0.4``: Row and column subsampling
  act as regularisation, reducing overfitting and providing stochasticity
  that drives the Rashomon Effect across seeds.

* ``eval_metric='aucpr'``: Early stopping monitors PR-AUC on the val set,
  consistent with the primary evaluation metric.

* ``missing=-1``: Missing values in the raw IEEE-CIS dataset are encoded as
  -1 by data_prep.py. XGBoost learns a default direction for missing values
  at each split, exploiting informative missingness patterns.

* ``device='cuda'``, ``tree_method='hist'``: GPU-accelerated histogram
  method. Reduces per-seed training time from ~4 h (CPU) to ~6 min (A100).

Checkpoint pattern
------------------
Training 30 seeds × 2 conditions (60 total) takes ~7 hours on an A100.
After each seed, the PR-AUC and SHAP ranking CSV are written to Drive, and
a JSON checkpoint file is updated. On kernel restart, the checkpoint is read
and completed seeds are skipped automatically.

Usage
-----
    from src.train import train_and_explain

    pr_auc, ranking = train_and_explain(
        seed=0,
        train=train, y=y,
        idxT=idxT, idxV=idxV, idxTE=idxTE,
        X_shap=X_shap, cols=cols,
        scale_pos_weight=None,   # Condition A (unweighted)
    )
"""

import gc
import json
import os

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import average_precision_score


# ---------------------------------------------------------------------------
# Default hyperparameters (Sprint 4b refined)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    n_estimators        = 5000,
    max_depth           = 12,
    learning_rate       = 0.02,
    subsample           = 0.8,
    colsample_bytree    = 0.4,
    missing             = -1,
    eval_metric         = 'aucpr',
    early_stopping_rounds = 200,
    tree_method         = 'hist',
    device              = 'cuda',
)


def train_and_explain(
    seed: int,
    train: pd.DataFrame,
    y: pd.Series,
    idxT,
    idxV,
    idxTE,
    X_shap: pd.DataFrame,
    cols: list,
    scale_pos_weight: float = None,
    n_estimators: int = None,
    early_stopping_rounds: int = None,
) -> tuple:
    """Train one XGBoost model and return its test PR-AUC and SHAP ranking.

    Parameters
    ----------
    seed : int
        ``random_state`` for XGBoost. The only difference between runs in
        the seed perturbation experiment.
    train : pd.DataFrame
        Full dataset (all rows).
    y : pd.Series
        Binary labels aligned with ``train``.
    idxT, idxV, idxTE : pd.Index
        Train / Val / Test partition indices.
    X_shap : pd.DataFrame
        Fixed SHAP subsample (10,000 rows, built by data_prep.build_shap_subsample).
    cols : list[str]
        Feature column names.
    scale_pos_weight : float or None
        Class weight for positive class (SPW).
        None = Condition A (unweighted).
        27.46 = Condition B (n_neg / n_pos = 427,342 / 15,563).
    n_estimators : int or None
        Overrides DEFAULT_PARAMS['n_estimators'] if provided.
    early_stopping_rounds : int or None
        Overrides DEFAULT_PARAMS['early_stopping_rounds'] if provided.

    Returns
    -------
    pr_auc_test : float
        PR-AUC on the held-out Test set.
    ranking : pd.Series
        Mean absolute SHAP values per feature, sorted descending.
    """
    params = {**DEFAULT_PARAMS}
    if n_estimators is not None:
        params['n_estimators'] = n_estimators
    if early_stopping_rounds is not None:
        params['early_stopping_rounds'] = early_stopping_rounds
    if scale_pos_weight is not None:
        params['scale_pos_weight'] = scale_pos_weight
    params['random_state'] = seed

    tag = f"SPW={scale_pos_weight:.1f}" if scale_pos_weight else "unweighted"
    print(f"  Seed {seed:2d} | {tag} | n_est={params['n_estimators']} ...", end=' ')

    clf = xgb.XGBClassifier(**params)
    clf.fit(
        train.loc[idxT, cols], y.loc[idxT],
        eval_set=[(train.loc[idxV, cols], y.loc[idxV])],
        verbose=False,
    )

    preds_test  = clf.predict_proba(train.loc[idxTE, cols])[:, 1]
    pr_auc_test = average_precision_score(y.loc[idxTE], preds_test)

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_shap)
    ranking   = pd.Series(
        np.abs(shap_vals).mean(axis=0), index=cols
    ).sort_values(ascending=False)

    print(f"PR-AUC(test)={pr_auc_test:.4f} | best_iter={clf.best_iteration}")

    del clf, explainer, shap_vals
    gc.collect()

    return pr_auc_test, ranking


def run_condition(
    seeds: list,
    train: pd.DataFrame,
    y: pd.Series,
    idxT, idxV, idxTE,
    X_shap: pd.DataFrame,
    cols: list,
    drive_path: str,
    prefix: str,
    condition: str,           # 'A' or 'B'
    scale_pos_weight: float = None,
) -> tuple:
    """Train all seeds for one condition with checkpoint / resume support.

    Writes per-seed SHAP ranking CSVs and a JSON checkpoint to ``drive_path``.
    Safe to interrupt and resume — completed seeds are skipped.

    Parameters
    ----------
    seeds : list[int]
    train, y, idxT, idxV, idxTE, X_shap, cols : see train_and_explain
    drive_path : str
        Root directory for artefact storage (e.g. '/content/drive/MyDrive/').
    prefix : str
        Artefact prefix (e.g. 's4r_').
    condition : str
        'A' (unweighted) or 'B' (SPW-weighted). Used for file naming.
    scale_pos_weight : float or None

    Returns
    -------
    rankings : dict[int, pd.Series]
    pr_aucs  : dict[int, float]
    """
    ckpt_path = os.path.join(drive_path, f'{prefix}ckpt_{condition}.json')
    rankings  = {}
    pr_aucs   = {}

    # Resume from checkpoint if it exists
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            pr_aucs = {int(k): v for k, v in json.load(f).items()}
        for seed in list(pr_aucs.keys()):
            csv_path = os.path.join(drive_path,
                                    f'{prefix}ranking_{condition}_seed{seed}.csv')
            df_r = pd.read_csv(csv_path, index_col=0, header=0)
            df_r.columns = ['mean_abs_shap']
            rankings[seed] = df_r['mean_abs_shap'].sort_values(ascending=False)
        print(f"  Resumed Condition {condition}: {len(pr_aucs)} / {len(seeds)} seeds done.")

    for seed in seeds:
        if seed in pr_aucs:
            continue
        pr, ranking = train_and_explain(
            seed=seed, train=train, y=y,
            idxT=idxT, idxV=idxV, idxTE=idxTE,
            X_shap=X_shap, cols=cols,
            scale_pos_weight=scale_pos_weight,
        )
        rankings[seed] = ranking
        pr_aucs[seed]  = pr

        # Save artefacts immediately after each seed
        csv_path = os.path.join(drive_path,
                                f'{prefix}ranking_{condition}_seed{seed}.csv')
        ranking.to_csv(csv_path, header=['mean_abs_shap'])
        with open(ckpt_path, 'w') as f:
            json.dump(pr_aucs, f)

    vals = list(pr_aucs.values())
    print(f"\n--- PR-AUC Summary (Condition {condition}, N={len(seeds)}) ---")
    print(f"  Range : {min(vals):.4f} – {max(vals):.4f}")
    print(f"  Mean  : {np.mean(vals):.4f}  ±  {np.std(vals):.4f}")

    return rankings, pr_aucs
