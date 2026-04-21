# Measuring and Controlling SHAP Explanation Instability in High-Stakes AML Systems

**Author:** Karol Duda  
**Programme:** MSc Data Science & Society — Tilburg University  
**Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) (590,540 transactions, ~3.5% fraud)

---

## Research Questions

| RQ | Question | Answer |
|---|---|---|
| **RQ1** | Does SHAP instability exist across random seeds in a production-grade AML model? | **Yes** — Jaccard k=5 = 0.692 across 435 pairs (N=30 seeds, indistinguishable PR-AUC) |
| **RQ2** | Does class weighting (SPW) affect SHAP stability? | **Yes, with a reversal** — SPW improves k=5/10 but reduces k=20 stability |
| **RQ3** | Does seed ensemble mitigate instability? | **Yes** — uniform improvement across all k, no reversal |
| **RQ3b** | Does SPW + Ensemble improve further? | **Yes** — best strategy for top-5/10 reporting scopes |

---

## Key Results (Sprint 4b — N=30, n_estimators=5,000)

### Full 2×2 Mitigation Matrix (Jaccard similarity — higher is better)

| Strategy | k=5 | k=10 | k=20 |
|---|---|---|---|
| Baseline A (individual) | 0.6922 | 0.8334 | **0.9452** |
| Ensemble A (no SPW) | 0.7921 | 0.8849 | **0.9701** |
| SPW B (individual) | 0.8651 | 0.9398 | 0.9147 |
| **SPW + Ensemble B** | **0.9111** | **0.9636** | 0.9374 |

**Optimal strategy is scope-dependent:**
- Top-5/10 explanation requirement → **SPW + Ensemble B**
- Top-20 / full attribution report → **Ensemble A** (SPW hurts here)

---

## Repository Structure

```
dss_thesis_aml_shap/
├── notebooks/
│   ├── 01_sprint1_eda.ipynb                   Sprint 1 — EDA & dataset profiling
│   ├── 02_sprint3_baseline_model.ipynb         Sprint 3 — Baseline XGBoost model
│   ├── 03_sprint4_stability_mvp.ipynb          Sprint 4 MVP — N=10 stability experiment
│   └── 04_sprint4b_advanced_refined.ipynb      Sprint 4b — N=30, full 2×2 matrix ← latest
├── docs/
│   └── sprint2_methodology.md                  Sprint 2 — Methodology design
├── src/
│   ├── data_prep.py                            Data loading & 4-way temporal split
│   ├── eval.py                                 Model evaluation (PR-AUC, ROC-AUC, F1)
│   └── train.py                                XGBoost training with checkpointing
├── data/
│   ├── raw/                                    Original Kaggle files (not tracked by git)
│   └── processed/                              Engineered parquet artefacts (not tracked)
├── requirements.txt
└── .gitignore
```

---

## Sprint Progress

| Sprint | Title | Status | Notebook |
|---|---|---|---|
| Sprint 1 | EDA & Dataset Profiling | Done | `01_sprint1_eda.ipynb` |
| Sprint 2 | Methodology Design | Done | `docs/sprint2_methodology.md` |
| Sprint 3 | Baseline XGBoost Model | Done | `02_sprint3_baseline_model.ipynb` |
| Sprint 4 MVP | N=10 Stability Experiment | Done | `03_sprint4_stability_mvp.ipynb` |
| Sprint 4b | N=30 Refined + Full 2x2 Matrix | Done | `04_sprint4b_advanced_refined.ipynb` |
| Sprint 5 | Ethics & Fairness Analysis | In progress | — |
| Sprint 6 | Codebase Audit & Reproducibility | Planned | — |
| Sprint 7 | Results Interpretation | Planned | — |

---

## Setup

```bash
pip install -r requirements.txt
```

> **Note:** All notebooks are designed to run on Google Colab (A100 GPU).
> Training artefacts (`*.npy`, `*.parquet`, checkpoint JSONs) are stored on Google Drive — not in this repo.
> After a kernel restart: run **Cell 1 (Setup)** then **Cell 2 (Master Reload)** to restore all variables from Drive without retraining.

---

## Model Architecture

- **Algorithm:** XGBoost with TreeSHAP
- **Split:** 4-way temporal (Train 65% / Val 15% / Test 12% / SHAP 8%)
- **Hyperparameters:** `max_depth=12`, `lr=0.02`, `n_estimators=5000`, `early_stopping=200`
- **Primary metric:** PR-AUC (ROC-AUC misleading for 3.5% fraud rate)
- **SHAP subsample:** 10,000 stratified observations (fixed seed=42)
- **Stability metrics:** Jaccard @ k=5/10/20, Spearman rho

---

## References

- Breiman, L. (2001). Statistical modeling: The two cultures. *Statistical Science*, 16(3), 199-231.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*.
