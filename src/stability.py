"""
stability.py
============
SHAP explanation stability metrics for the IEEE-CIS AML study.

Metric design rationale
-----------------------
Two complementary metrics are used:

1. **Jaccard similarity @ k** — answers the question directly relevant to
   compliance officers: "Do two models agree on which k features to report?"
   Jaccard is defined as |A ∩ B| / |A ∪ B| for two sets A, B of top-k
   features. A value of 1.0 means perfect agreement; 0.0 means no overlap.

   We compute Jaccard at k = 5, 10, and 20 because regulatory reporting
   requirements vary: EU AI Act Article 13 may require top-5 explanations
   for individual decisions, while internal audit teams may review top-20.

2. **Spearman rank correlation (ρ)** — measures global rank ordering
   agreement across all 440 features. High Spearman ρ with low Jaccard k=5
   is the signature of the Rashomon Effect: global structure is stable, but
   the boundary between "important" and "not important" features is not.

Pairwise comparisons
--------------------
For N seeds, we compute C(N, 2) = N*(N-1)/2 pairwise comparisons:
  N=10 → 45 pairs   (Sprint 4 MVP)
  N=30 → 435 pairs  (Sprint 4b Refined — publication-grade power)

Usage
-----
    from src.stability import compute_stability, summarise_stability, jaccard

    stability_df = compute_stability(rankings_A)
    summarise_stability(stability_df, label='Condition A')
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel

K_VALS = [5, 10, 20]


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets.

    Parameters
    ----------
    set_a, set_b : set
        Sets of feature names (top-k from each model).

    Returns
    -------
    float in [0, 1]. Returns 0.0 if both sets are empty.
    """
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def spearman_rho(ranking_a: pd.Series, ranking_b: pd.Series) -> float:
    """Spearman rank correlation between two SHAP feature rankings.

    Uses only features present in both rankings (intersection).

    Parameters
    ----------
    ranking_a, ranking_b : pd.Series
        Mean absolute SHAP values, indexed by feature name, sorted descending.

    Returns
    -------
    float in [-1, 1].
    """
    common = ranking_a.index.intersection(ranking_b.index)
    rho, _ = spearmanr(
        ranking_a[common].rank(ascending=False, method='average'),
        ranking_b[common].rank(ascending=False, method='average'),
    )
    return float(rho)


# ---------------------------------------------------------------------------
# Pairwise stability computation
# ---------------------------------------------------------------------------

def compute_stability(
    rankings_dict: dict,
    k_values: list = None,
) -> pd.DataFrame:
    """Compute all C(N, 2) pairwise Jaccard + Spearman metrics.

    Parameters
    ----------
    rankings_dict : dict[int, pd.Series]
        Mapping from seed → SHAP ranking (mean |SHAP| per feature, sorted desc).
    k_values : list[int]
        Top-k values for Jaccard. Defaults to [5, 10, 20].

    Returns
    -------
    pd.DataFrame with columns:
        label_1, label_2, spearman_rho, jaccard_k5, jaccard_k10, jaccard_k20
        (jaccard columns depend on k_values).
    """
    if k_values is None:
        k_values = K_VALS

    labels = list(rankings_dict.keys())
    rows   = []

    for l1, l2 in combinations(labels, 2):
        r1, r2 = rankings_dict[l1], rankings_dict[l2]
        rho = spearman_rho(r1, r2)
        row = {'label_1': l1, 'label_2': l2, 'spearman_rho': round(rho, 4)}
        for k in k_values:
            row[f'jaccard_k{k}'] = round(
                jaccard(set(r1.head(k).index), set(r2.head(k).index)), 4
            )
        rows.append(row)

    return pd.DataFrame(rows)


def summarise_stability(df_stab: pd.DataFrame, label: str = '') -> pd.DataFrame:
    """Print and return mean / std / min / max for each stability metric.

    Parameters
    ----------
    df_stab : pd.DataFrame
        Output of compute_stability.
    label : str
        Descriptive label printed in the header.

    Returns
    -------
    pd.DataFrame (metrics × statistics).
    """
    metrics = ['spearman_rho'] + [f'jaccard_k{k}' for k in K_VALS
                                   if f'jaccard_k{k}' in df_stab.columns]
    summary = df_stab[metrics].agg(['mean', 'std', 'min', 'max']).round(4)
    n_pairs = len(df_stab)
    print(f"\n=== Stability Summary: {label} ({n_pairs} pairwise comparisons) ===")
    print(summary.to_string())
    return summary


# ---------------------------------------------------------------------------
# Condition comparison (RQ2)
# ---------------------------------------------------------------------------

def compare_conditions(
    stability_A: pd.DataFrame,
    stability_B: pd.DataFrame,
    k_values: list = None,
) -> pd.DataFrame:
    """Paired t-test comparison between Condition A and Condition B.

    The paired t-test is appropriate because A and B share the same 435 model
    pairs (same seed combinations, different hyperparameter setting), making
    the within-pair difference the correct unit of analysis.

    Parameters
    ----------
    stability_A, stability_B : pd.DataFrame
        Outputs of compute_stability for each condition.
        Must have the same length (same seed pairs).
    k_values : list[int]

    Returns
    -------
    pd.DataFrame with columns:
        Metric, A_mean, B_mean, Delta, p_value, Significance, Result
    """
    if k_values is None:
        k_values = K_VALS

    metrics = ['spearman_rho'] + [f'jaccard_k{k}' for k in k_values]
    rows    = []

    for metric in metrics:
        a_mean = stability_A[metric].mean()
        b_mean = stability_B[metric].mean()
        delta  = b_mean - a_mean
        _, pval = ttest_rel(stability_B[metric].values,
                            stability_A[metric].values)
        sig = ('*** (p<0.001)' if pval < 0.001
               else '* (p<0.05)' if pval < 0.05
               else '(n.s.)')
        direction = ('B more stable' if delta > 0.01
                     else 'A more stable' if delta < -0.01
                     else 'no difference')
        rows.append({
            'Metric': metric, 'A_mean': round(a_mean, 4),
            'B_mean': round(b_mean, 4), 'Delta': round(delta, 4),
            'p_value': round(pval, 6), 'Significance': sig, 'Result': direction,
        })
        print(f"  {metric:<20}: A={a_mean:.4f} | B={b_mean:.4f} | "
              f"Delta={delta:+.4f}  {sig}  -> {direction}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Ensemble construction (RQ3 / RQ3b)
# ---------------------------------------------------------------------------

def build_ensemble_ranking(rankings_dict: dict) -> pd.Series:
    """Average mean |SHAP| across all models to produce an ensemble ranking.

    Parameters
    ----------
    rankings_dict : dict[int, pd.Series]
        Mapping from seed → SHAP ranking.

    Returns
    -------
    pd.Series: ensemble mean |SHAP| per feature, sorted descending.
    """
    seeds        = list(rankings_dict.keys())
    all_features = rankings_dict[seeds[0]].index
    ensemble     = pd.Series(
        np.mean([rankings_dict[s].loc[all_features].values for s in seeds], axis=0),
        index=all_features,
    ).sort_values(ascending=False)
    return ensemble


def individual_vs_ensemble(
    rankings_dict: dict,
    ensemble_ranking: pd.Series,
    k_values: list = None,
) -> pd.DataFrame:
    """Compute each individual model's stability relative to the ensemble.

    Parameters
    ----------
    rankings_dict : dict[int, pd.Series]
    ensemble_ranking : pd.Series
        Output of build_ensemble_ranking.
    k_values : list[int]

    Returns
    -------
    pd.DataFrame with one row per seed.
    """
    if k_values is None:
        k_values = K_VALS

    rows = []
    for seed, ranking in rankings_dict.items():
        rho = spearman_rho(ranking, ensemble_ranking)
        row = {'seed': seed, 'spearman_vs_ensemble': round(rho, 4)}
        for k in k_values:
            j = jaccard(set(ranking.head(k).index),
                        set(ensemble_ranking.head(k).index))
            row[f'jaccard_k{k}_vs_ensemble'] = round(j, 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit tests (run with: python -m pytest src/stability.py or python src/stability.py)
# ---------------------------------------------------------------------------

def _run_unit_tests():
    """Lightweight unit tests — no pytest dependency required."""
    print("Running stability unit tests...")

    # ── jaccard ─────────────────────────────────────────────────────────────
    assert jaccard({'a', 'b', 'c'}, {'a', 'b', 'c'}) == 1.0,    "identical sets"
    assert jaccard({'a', 'b', 'c'}, {'d', 'e', 'f'}) == 0.0,    "disjoint sets"
    assert jaccard(set(), set()) == 0.0,                          "empty sets"
    assert abs(jaccard({'a', 'b'}, {'a', 'c'}) - 1/3) < 1e-9,   "partial overlap"

    # ── spearman_rho ────────────────────────────────────────────────────────
    r1 = pd.Series({'a': 3.0, 'b': 2.0, 'c': 1.0})
    r2 = pd.Series({'a': 3.0, 'b': 2.0, 'c': 1.0})
    assert abs(spearman_rho(r1, r2) - 1.0) < 1e-6,  "identical rankings -> rho=1"

    r3 = pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0})
    assert abs(spearman_rho(r1, r3) + 1.0) < 1e-6,  "reversed rankings -> rho=-1"

    # ── compute_stability ───────────────────────────────────────────────────
    r_a = pd.Series({'f1': 5.0, 'f2': 4.0, 'f3': 3.0, 'f4': 2.0, 'f5': 1.0})
    r_b = pd.Series({'f1': 5.0, 'f2': 4.0, 'f3': 3.0, 'f4': 2.0, 'f5': 1.0})
    df  = compute_stability({0: r_a, 1: r_b}, k_values=[5])
    assert df.shape[0] == 1,                          "C(2,2) = 1 pair"
    assert df['jaccard_k5'].iloc[0] == 1.0,           "identical -> Jaccard=1"
    assert df['spearman_rho'].iloc[0] == 1.0,         "identical -> rho=1"

    # ── build_ensemble_ranking ───────────────────────────────────────────────
    rankings = {
        0: pd.Series({'f1': 2.0, 'f2': 1.0}),
        1: pd.Series({'f1': 4.0, 'f2': 1.0}),
    }
    ens = build_ensemble_ranking(rankings)
    assert abs(ens['f1'] - 3.0) < 1e-9, "ensemble mean correct"
    assert abs(ens['f2'] - 1.0) < 1e-9, "ensemble mean correct"

    print("All unit tests passed. ✓")


if __name__ == '__main__':
    _run_unit_tests()
