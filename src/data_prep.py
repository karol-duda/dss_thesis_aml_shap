"""
data_prep.py
============
Data loading and 4-way temporal split for the IEEE-CIS AML SHAP stability study.

Design decisions
----------------
* Strict temporal ordering — all splits are chronological (no random shuffle).
  This eliminates look-ahead bias and mimics a real deployment scenario where
  the model is trained on past data and evaluated on future data.

* 4-way split rationale (65 / 15 / 12 / 8 %):
    - Train  (65 %): main learning partition.
    - Val    (15 %): early-stopping signal only — never used for metric reporting.
    - Test   (12 %): held-out performance evaluation (PR-AUC, ROC-AUC, F1).
    - SHAP   ( 8 %): dedicated partition for SHAP stability experiments.
      Separating this from Test ensures that stability metrics are not
      contaminated by the same distribution used for performance reporting.

* Missing values encoded as -1 (XGBoost ``missing=-1`` parameter handles this
  natively via its split-finding algorithm).

* Categorical / object columns are label-encoded with pd.factorize, which
  assigns -1 to unseen categories automatically at inference time.

Usage
-----
    from src.data_prep import load_and_split

    splits = load_and_split('/content/drive/MyDrive/train_processed.parquet')
    train, y, idxT, idxV, idxTE, idxSH, cols = splits
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.65   # fraction used for model training
VAL_FRAC   = 0.15   # fraction used for early stopping (val)
TEST_FRAC  = 0.12   # fraction used for held-out performance evaluation
# SHAP fraction = remaining 8 %

EXCLUDE_COLS = ['isFraud', 'TransactionDT']


def load_and_split(parquet_path: str) -> tuple:
    """Load the frozen dataset artefact and apply the 4-way temporal split.

    Parameters
    ----------
    parquet_path : str
        Path to ``train_processed.parquet`` on Google Drive (or local disk).

    Returns
    -------
    train : pd.DataFrame
        Full dataset (all rows, sorted chronologically).
    y : pd.Series
        Binary fraud labels aligned with ``train``.
    idxT : pd.Index
        Row indices for the **Train** partition (65 %).
    idxV : pd.Index
        Row indices for the **Val** partition (15 %).
    idxTE : pd.Index
        Row indices for the **Test** partition (12 %).
    idxSH : pd.Index
        Row indices for the **SHAP** partition (8 %).
    cols : list[str]
        Feature column names (excludes ``isFraud`` and ``TransactionDT``).
    """
    print(f"Loading dataset from: {parquet_path}")
    train = pd.read_parquet(parquet_path)

    # Encode categoricals / objects as integers (XGBoost requirement)
    for col in train.columns:
        if train[col].dtype.name == 'category' or train[col].dtype == object:
            train[col] = pd.factorize(train[col])[0].astype('int16')

    # Sort chronologically — mandatory; TransactionDT is a relative timestamp
    train.sort_values('TransactionDT', inplace=True)
    train.reset_index(drop=True, inplace=True)

    y    = train['isFraud'].copy()
    cols = [c for c in train.columns if c not in EXCLUDE_COLS]
    n    = len(train)

    # Compute split boundaries
    i_train_end = int(n * TRAIN_FRAC)
    i_val_end   = int(n * (TRAIN_FRAC + VAL_FRAC))
    i_test_end  = int(n * (TRAIN_FRAC + VAL_FRAC + TEST_FRAC))

    idxT  = train.index[:i_train_end]            # Train  — 65 %
    idxV  = train.index[i_train_end:i_val_end]   # Val    — 15 %
    idxTE = train.index[i_val_end:i_test_end]    # Test   — 12 %
    idxSH = train.index[i_test_end:]             # SHAP   —  8 %

    # Summary
    print(f"\n{'='*50}")
    print(f"  4-WAY SPLIT SUMMARY")
    print(f"{'='*50}")
    for name, idx in [('TRAIN', idxT), ('VAL', idxV), ('TEST', idxTE), ('SHAP', idxSH)]:
        fraud_r = y.loc[idx].mean()
        print(f"  {name:<6}: {len(idx):>7,} rows | fraud rate = {fraud_r:.3%}")
    print(f"  TOTAL : {n:>7,} rows")
    print(f"\n  Features: {len(cols)}")
    print(f"{'='*50}\n")

    return train, y, idxT, idxV, idxTE, idxSH, cols


def build_shap_subsample(
    train: pd.DataFrame,
    y: pd.Series,
    idxSH: pd.Index,
    n_total: int = 10_000,
    random_state: int = 42,
) -> tuple:
    """Build a fixed stratified SHAP subsample from the SHAP partition.

    The subsample is stratified by fraud rate to maintain the class
    distribution of the SHAP partition. Using a fixed ``random_state``
    ensures the *same* 10,000 rows are used across all 30 seed experiments,
    isolating seed variation to the model itself rather than the SHAP input.

    Parameters
    ----------
    train : pd.DataFrame
    y : pd.Series
    idxSH : pd.Index
        Row indices of the SHAP partition.
    n_total : int
        Total subsample size (default 10,000).
    random_state : int
        Fixed seed for reproducibility (default 42).

    Returns
    -------
    X_shap : pd.DataFrame
    y_shap : pd.Series
    sample_idx : pd.Index
    """
    np.random.seed(random_state)
    fraud_rate = y.loc[idxSH].mean()
    shap_fraud = train.loc[idxSH][y.loc[idxSH] == 1]
    shap_legit = train.loc[idxSH][y.loc[idxSH] == 0]

    n_fraud = int(n_total * fraud_rate)
    n_legit = n_total - n_fraud

    sample_idx = pd.concat([
        shap_fraud.sample(n=n_fraud, random_state=random_state),
        shap_legit.sample(n=n_legit, random_state=random_state),
    ]).index

    X_shap = train.loc[sample_idx]
    y_shap = y.loc[sample_idx]

    print(f"SHAP subsample: {len(X_shap):,} obs | "
          f"fraud={y_shap.sum()} ({y_shap.mean():.2%})")

    return X_shap, y_shap, sample_idx
