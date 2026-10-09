# `outputs/figures/` — figures arranged by notebook

Every generated figure, grouped into one folder per notebook. 38 figures, 7 folders.

> Notebooks currently still *write* to `outputs/dataset_validation/` (and NB04 to `outputs/` +
> `outputs/training_curves/`). After re-running any notebook, run `python outputs/figures/resort_figures.py`
> to move the fresh files back into the folders below. (Or update the notebook's output-path line —
> ask if you want that done.)

---

## nb01_data_preparation/  (Notebook 01 — Data Preparation)

| File | What it shows |
|---|---|
| `01_age_distribution_density.png` | Patient age distribution (density) |
| `02_ecg_counts_by_age_group.png` | ECG record counts per age band |
| `03_diagnostic_superclass_prevalence.png` | Superclass prevalence over the whole cohort |
| `04_diagnosis_prevalence_by_age_group.png` | Diagnosis prevalence by age band — the "aging heart" inversion |
| `05_multilabel_frequency_distribution.png` | How many superclasses co-occur per record |
| `06_split_sizes_patients_records.png` | Split sizes (patients vs records) |
| `07_age_distribution_by_split.png` | Age distribution per split (train / val / calibration / test) |
| `08_diagnosis_distribution_by_split.png` | Diagnosis distribution per split |
| `09_representative_raw_12lead_ecg.png` | A representative RAW 12-lead ECG |
| `10_representative_preprocessed_filtered_12lead_ecg.png` | The same record after mV calibration + 0.5–40 Hz band-pass |

*Note: the `NN_` prefixes here are figure sequence numbers inside NB01, not notebook numbers.*

## nb02_ensemble_baseline/  (Notebook 02 — 3-Model Heterogeneous Ensemble, equal-weight)

| File | What it shows |
|---|---|
| `03_dual_branch_confusion_matrices.png` | DualBranchECGNet (an ensemble member) — confusion matrices |
| `03_training_curves.png` | Training curves for the ensemble members |
| `04_ensemble_roc_curves.png` | 3-model **equal-weight** ensemble — per-class TEST ROC (NORM .947 / MI .920 / STTC .936 / CD .915 / HYP .903) |
| `04_ensemble_confusion_matrices.png` | 3-model equal-weight ensemble — confusion matrices |
| `04_ensemble_training_curves.png` | 3-model ensemble — training curves |

*Note: `03_`/`04_` prefixes date from when this notebook was numbered 04.*

## nb03_optimized_model/  (Notebook 03 — Optimized Best Model)

| File | What it shows |
|---|---|
| `05_optimized_inceptiontime_training_curves.png` | InceptionTime1D (optimized) — training curves |
| `05_optimized_best_roc_curves.png` | Optimized best model — per-class TEST ROC |
| `05_optimized_best_confusion_matrices.png` | Optimized best model — confusion matrices |
| `05_four_config_benchmark.png` | Four configurations head-to-head on TEST (chart title still reads "Notebook 05" — regenerate before publishing) |
| `05_age_stratified_metric_comparison.png` | Age-stratified audit of the winning config (title still reads "Notebook 05") |

*Note: `05_` prefix + internal "Notebook 05" titles date from when this notebook was numbered 05.*

## nb04_resnet_vs_transformer/  (Notebook 04 — ResNet-1D-18 vs ResNet-1D-Transformer)

| File | What it shows |
|---|---|
| `test_roc_clean.png` | Test ROC — best clean model, per-class AUC |
| `model_training_comparison.png` | ResNet-18 vs ResNet-Transformer — training comparison |
| `test_multilabel_roc_curves.png` | Multi-label TEST ROC curves |

## nb05_calibration_uncertainty/  (Notebook 05 — Calibration & Uncertainty)

| File | Module | What it shows |
|---|---|---|
| `05cal_reliability_uncalibrated.png` | 6 | Reliability diagram, uncalibrated (macro-ECE 0.082) |
| `05cal_reliability_calibrated.png` | 7 | Reliability after per-class temperature scaling (macro-ECE 0.075) |
| `05cal_uncertainty_vs_error.png` | 9 | Uncertainty on correct vs wrong predictions (error-detection AUC 0.74 / 0.78) |
| `05cal_risk_coverage.png` | 10 | Risk–coverage curve (defer the least-confident cases) |
| `05cal_age_stratified_ece.png` | 11 | Macro-ECE per age band, before vs after temp scaling |

## nb06_explainability/  (Notebook 06 — Explainability)

| File | Module | What it shows |
|---|---|---|
| `06xai_per_lead_importance.png` | 3 | Per-lead Integrated-Gradients importance heatmap |
| `06xai_map_NORM.png` … `06xai_map_HYP.png` | 4 | IG + Grad-CAM attribution maps, one per class (5 files) |
| `06xai_faithfulness_ablation.png` | 5 | Lead-ablation faithfulness curves (top-k IG vs random) |

## nb07_statistical_robustness/  (Notebook 07 — Statistical Robustness)

| File | What it shows |
|---|---|
| `07_bootstrap_headline_ci.png` | Bootstrap 95% CIs on macro-AUC / Fmax / macro-AUPRC (both headline models) |
| `07_per_class_auc_ci.png` | Per-class AUC ± 95% bootstrap CI (HYP interval 2.6× wider than NORM) |
| `07_ensemble_vs_single_delta.png` | Paired bootstrap: ensemble − single Δ macro-AUC (95% CI crosses 0) |

## nb08_improved_ensemble/  (Notebook 08 — Improved Ensemble)

| File | What it shows |
|---|---|
| `08_improved_ensemble_ci.png` | Blend-weights optimised on CALIBRATION → macro-AUC 0.9245 → **0.9269** (Δ +0.0025, 95% CI excludes 0). Left: vs NB05 and vs Strodthoff/Gitau. Right: paired bootstrap difference. |
