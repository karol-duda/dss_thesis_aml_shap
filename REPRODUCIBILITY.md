# Reproducibility Guide

**Project:** Measuring and Controlling SHAP Explanation Instability in High-Stakes AML Systems  
**Author:** Karol Duda · MSc Data Science & Society · Tilburg University  
**Environment:** Google Colab (A100 GPU) · XGBoost 3.2.0 · SHAP 0.51.0

---

## 1. Prerequisites

### 1.1 Dataset

The raw dataset is the **IEEE-CIS Fraud Detection** dataset from Kaggle:

```
https://www.kaggle.com/c/ieee-fraud-detection
```

Download `train_transaction.csv` and `train_identity.csv` (approx. 500 MB combined).  
Place them in a folder on your Google Drive, e.g. `MyDrive/ieee_cis_raw/`.

The feature-engineered artefact `train_processed.parquet` is produced by **Sprint 1**
(`notebooks/01_sprint1_eda.ipynb`). It contains 590,540 rows and 442 columns (440 features
+ `isFraud` + `TransactionDT`).

> **Note:** The processed parquet is not tracked by git (see `.gitignore`). You must either
> run Sprint 1 to generate it, or obtain it from the project's shared Google Drive folder.

### 1.2 Python environment

```bash
pip install -r requirements.txt
```

All notebooks are designed for **Google Colab** with a GPU runtime (A100 recommended).
Runtime → Change runtime type → A100 GPU.

---

## 2. Repository Structure

```
dss_thesis_aml_shap/
├── notebooks/
│   ├── 01_sprint1_eda.ipynb               EDA & dataset profiling (Sprint 1)
│   ├── 02_sprint3_baseline_model.ipynb    Baseline XGBoost (Sprint 3)
│   ├── 03_sprint4_stability_mvp.ipynb     N=10 stability experiment (Sprint 4 MVP)
│   ├── 04_sprint4b_advanced_refined.ipynb N=30 full 2×2 matrix (Sprint 4b) ← main experiment
│   ├── 05_sprint5_ethics.ipynb            Error stratification & fairness (Sprint 5)
│   └── 06_sprint6_codebase.ipynb          Reproducibility audit (Sprint 6)
├── src/
│   ├── data_prep.py    Data loading & 4-way temporal split
│   ├── eval.py         Model evaluation (PR-AUC, ROC-AUC, F1, confusion matrix)
│   ├── train.py        XGBoost training with checkpoint/resume support
│   └── stability.py    SHAP stability metrics (Jaccard, Spearman) + unit tests
├── docs/
│   └── sprint2_methodology.md    Methodology design document
├── data/
│   ├── raw/            Original Kaggle files (not tracked by git)
│   └── processed/      Engineered parquet artefacts (not tracked by git)
├── requirements.txt
├── REPRODUCIBILITY.md  (this file)
└── .gitignore
```

---

## 3. Running the Experiments

### 3.1 Quick start (resume from saved artefacts)

All training artefacts (SHAP rankings, model predictions, stability CSVs) are saved to
Google Drive with the prefix `s4r_`. If artefacts already exist on Drive:

1. Open `notebooks/04_sprint4b_advanced_refined.ipynb` in Google Colab.
2. Run **Cell 1 (Setup)** — mounts Drive and imports libraries.
3. Run **Cell 2 (Master Reload)** — detects all `s4r_*` artefacts on Drive and
   populates all in-memory variables. No retraining required.
4. Skip directly to any analysis cell (RQ1, RQ2, RQ3, RQ3b, Section 10 summary).

### 3.2 Full replication (from scratch)

> **Warning:** Training 60 models (30 seeds × 2 conditions, N=5,000 estimators each)
> takes approximately **6–8 hours on an A100 GPU**. Checkpoint files are written after
> each seed, so training can be safely interrupted and resumed.

1. Run Sprint 1 to generate `train_processed.parquet`.
2. Open `04_sprint4b_advanced_refined.ipynb`.
3. Run Cell 1 (Setup).
4. Run Cell 2 (Master Reload) — will find no artefacts, initialises empty dicts.
5. Run cells in order: Section 1 (data) → Section 3 (k-fold) → Section 4 (reference
   model) → Section 7 (RQ1, Condition A training) → Section 8 (RQ2, Condition B) →
   Section 9 (RQ3 + RQ3b ensembles) → Section 10 (summary).

---

## 4. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Train/Val/Test/SHAP split | 65 / 15 / 12 / 8 % | Temporal; Val used only for early stopping |
| Primary metric | PR-AUC | ROC-AUC misleading at 3.5% fraud rate |
| SHAP subsample | 10,000 obs (fixed seed=42) | Stratified; fixed across all 60 models |
| N seeds | 30 → C(30,2) = 435 pairs | Publication-grade statistical power |
| n_estimators | 5,000 | Reference model hit 2,996/3,000 at n=3,000 |
| scale_pos_weight | 27.46 = 427,342 / 15,563 | Exact neg/pos ratio in Train partition |
| Stability metrics | Jaccard @ k=5/10/20 + Spearman ρ | Top-k relevant to compliance reporting |
| Artefact prefix | `s4r_` (Sprint 4 refined) | Preserves original `s4_` artefacts |

---

## 5. Random Seeds

All stochasticity is controlled:

| Source | Seed |
|---|---|
| SHAP subsample construction | `random_state=42` (fixed across all experiments) |
| Seed perturbation experiment | `random_state` ∈ {0, 1, …, 29} (the experimental variable) |
| TimeSeriesSplit k-fold | `random_state=42` (fixed; only data composition varies) |
| Reference model (seed=42) | `random_state=42` |

---

## 6. Artefact Index (Google Drive, prefix `s4r_`)

| File | Contents |
|---|---|
| `s4r_preds_ref_test.npy` | Reference model test set predictions |
| `s4r_shap_sample_idx.npy` | Fixed SHAP subsample indices |
| `s4r_shap_ranking_ref.csv` | SHAP ranking — reference model |
| `s4r_ranking_A_seed{0-29}.csv` | SHAP rankings — Condition A (30 files) |
| `s4r_ranking_B_seed{0-29}.csv` | SHAP rankings — Condition B (30 files) |
| `s4r_stability_A.csv` | 435 pairwise stability metrics, Condition A |
| `s4r_stability_B.csv` | 435 pairwise stability metrics, Condition B |
| `s4r_rq2_comparison.csv` | A vs B paired t-test results |
| `s4r_rq3_ensemble.csv` | Individual-vs-Ensemble A stability |
| `s4r_rq3b_ensemble_B.csv` | Individual-vs-Ensemble B stability |
| `s4r_shap_ranking_ensemble.csv` | Ensemble A ranking |
| `s4r_shap_ranking_ensemble_B.csv` | Ensemble B ranking |
| `s4r_ckpt_A.json` | Condition A checkpoint (resume on crash) |
| `s4r_ckpt_B.json` | Condition B checkpoint (resume on crash) |
| `s4r_kfold_results.csv` | 5-fold TimeSeriesSplit results |

---

## 7. Verifying Results

After running the full pipeline, the Section 10 summary cell in
`04_sprint4b_advanced_refined.ipynb` should produce the following key figures:

| Metric | Expected value |
|---|---|
| Reference PR-AUC (Test, seed=42, n=5000) | 0.5636 |
| Condition A Jaccard k=5 (435 pairs) | 0.6922 ± 0.1854 |
| Condition B Jaccard k=5 (435 pairs) | 0.8651 ± 0.1207 |
| SPW + Ensemble B Jaccard k=5 | 0.9111 |
| k=20 reversal (B − A) | −0.0305 (p < 0.001) |

Tolerance: results should match to ±0.001 (floating-point precision). Any larger
discrepancy indicates a dataset or environment mismatch.

---

## 8. Running Unit Tests

```bash
python src/stability.py
```

Tests cover: Jaccard (identical, disjoint, partial sets), Spearman ρ (identical and
reversed rankings), `compute_stability` (output shape and values), and
`build_ensemble_ranking` (mean computation). All tests pass without external data.

---

## 9. Version Control

Major milestones are tagged in git:

| Sprint | Description |
|---|---|
| Sprint 1 | EDA & dataset profiling |
| Sprint 3 | Baseline XGBoost model |
| Sprint 4 | N=10 stability MVP |
| Sprint 4b | N=30 refined + full 2×2 matrix |
| Sprint 5 | Ethics & fairness analysis |
| Sprint 6 | Codebase audit & reproducibility |

Repository: `https://github.com/karol-duda/dss_thesis_aml_shap`
