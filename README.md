# Age-Stratified Audit of Calibration, Conformal Uncertainty, and Explanation Stability in Multi-Label ECG Classification on PTB-XL

Research code for an age-stratified reliability audit of a literature-competitive 12-lead ECG classifier on the PhysioNet **PTB-XL v1.0.3** database.

**Authors:** Harshit Tandon (Chandigarh University), Bhekumuzi Mkhululi Mathunjwa (Yuan Ze University), Rachit Ranka (Chandigarh University)

> Most ECG deep-learning papers report a single pooled accuracy figure. That number says nothing about whether a model's confidence can be trusted, or whether it serves older patients as well as younger ones. This repository accompanies a manuscript that audits one such classifier along three axes a pooled number leaves untested — **calibration, distribution-free (conformal) uncertainty, and explanation stability** — each measured inside four age bands (`<40`, `40–65`, `65–80`, `80+`) rather than on the population average.

---

## Summary of findings

The ensemble matches published single-model discrimination but hides a steep, statistically unambiguous age gradient in its reliability.

| Metric | Youngest (`<40`) | Oldest (`80+`) | Trend |
|---|---|---|---|
| Macro-AUC (held-out test) | 0.929 | 0.877 | flat until 80+, then drops |
| Exact-match accuracy | 76.4% | 34.1% | monotone decline |
| Expected calibration error (after temperature scaling) | 0.045 | 0.133 | +0.016 / decade |
| Mean conformal prediction-set size (Mondrian, 90% coverage) | 1.42 | 2.98 | +0.079 labels / decade |

Headline discrimination: **macro-AUC 0.9245** (95% CI [0.9176, 0.9315]), macro-AUPRC 0.8150, F-max 0.7467 — on par with published single models and ~0.010 below the published six-model ensembles.

The age gradient:
- **survives confound control** — adjusting for four signal-quality covariates attenuates it by ~3%, and in-band SNR is not associated with age (Spearman ρ = −0.007, p = 0.74);
- **reproduces on a classical baseline** — a Random Forest on 173 handcrafted features (macro-AUC 0.8837) traces the same curves, so the effect lives in the data, not one architecture;
- **holds across 300 calibration resamples** (positive set-size slope in 100% of resamples) and **across four training seeds** with all three ensemble members reseeded.

Two results are reported as negative results: per-band ("Mondrian") recalibration **fails** — the oldest band's calibration cell is too small to estimate its own quantile (calibration-cell fragmentation) — and **explanation stability does not degrade** with age (split-half Integrated-Gradients rank correlation 0.980 vs. 0.955 for confident vs. unconfident elderly cases; Wilcoxon p = 0.13).

---

## Methodology at a glance

- **Dataset:** PTB-XL v1.0.3 (21,799 records, 18,869 patients, 100 Hz, 10-second 12-lead), five diagnostic superclasses (NORM, MI, STTC, CD, HYP), multi-label.
- **Splits:** patient-disjoint TRAIN / VALIDATION / CONFORMAL_CALIBRATION / TEST from PTB-XL's native 10-fold structure; the calibration split is never used for model selection.
- **Model:** AUC-weighted soft-voting ensemble of a demographic-fused dual-branch CNN, InceptionTime1D, and ResNet1D-101.
- **Uncertainty:** per-class temperature scaling + split-conformal and age-conditional (Mondrian) conformal prediction at α = 0.10.
- **Statistics:** 1,000-resample paired bootstrap on every headline metric; Kruskal–Wallis, Spearman, and HC0-robust OLS slopes for every age-stratified claim.
- **Explainability:** Integrated Gradients and Grad-CAM-1D with a lead-ablation faithfulness test and a model-randomization sanity check.

---

## Repository structure

```text
.
├── notebooks/                 # Numbered, ordered research pipeline (01–14) + run-order README
│   ├── 01_Data_Preparation.ipynb
│   ├── 02_3Model_Heterogeneous_Ensemble.ipynb
│   ├── 03_Optimized_Best_Model.ipynb            # AUC-weighted ensemble + per-class thresholds
│   ├── 03b_hyp_augmentation_ablation.ipynb
│   ├── 04_ResNet18_vs_Transformer_Baseline.ipynb
│   ├── 05_Calibration_and_Uncertainty.ipynb     # Reliability/ECE, temperature scaling, conformal
│   ├── 06_Explainability.ipynb
│   ├── 07_Statistical_Robustness.ipynb
│   ├── 08_Improved_Ensemble.ipynb
│   ├── 09_RandomForest_Baseline.ipynb
│   ├── 10_Final_Hypothesis_and_Rigor.ipynb
│   ├── 11_Confound_Control_and_Trend_Tests.ipynb
│   ├── 12_XAI_Elderly_Confidence_Pairs.ipynb
│   ├── 13_MultiSeed_Replication.ipynb           # Anchor reseeding
│   └── 14_Full_Ensemble_MultiSeed.ipynb         # All three members reseeded
├── analysis/                  # Standalone scripts mirroring notebooks 11–14
├── outputs/
│   ├── dataset_validation/    # Archived metric JSONs — the authoritative numbers (see below)
│   ├── figures/               # Research figures (PNG)
│   └── rf_baseline/           # Random Forest baseline artifacts
├── data/processed/            # Derived metadata (MASTER_RESEARCH_METADATA.csv, manifest, label CSVs)
├── docs/                      # Pipeline reference and engineering log
└── README.md
```

### Authoritative numbers

Every figure in the manuscript is read from one of the archived metric files under `outputs/dataset_validation/`:

| File | Contents |
|---|---|
| `review_closeout_metrics.json` | Headline discrimination, conformal, per-band temperature |
| `11_confound_and_trend_tests.json` | Signal-quality covariates, confound-adjusted trend tests, calibration-resampling stability |
| `12_xai_pair_cases.json` | Elderly confident/unconfident attribution case pairs |
| `13_multiseed_replication.json` | Anchor reseeding |
| `14_full_ensemble_multiseed.json` | Full three-member reseeding (supersedes the ensemble column of `13`) |

---

## Reproducing the results

1. Obtain **PTB-XL v1.0.3** from PhysioNet (see *Data availability*) and place it at the repository root.
2. Create the environment: Python 3.12, PyTorch 2.x, NumPy, SciPy, scikit-learn.
3. Run `notebooks/01_Data_Preparation.ipynb` to build the patient-disjoint splits and cached arrays, then the remaining notebooks in order (see `notebooks/README.md`).
4. The four manuscript analyses (11–14) are also runnable as standalone scripts in `analysis/`. Each re-derives the full calibration and conformal pipeline from the saved checkpoints and asserts agreement with the archived JSON before computing anything new.

Global seed `42` is used throughout; all reported runs are CPU-deterministic given the released checkpoints.

> **Note on large files.** Raw PTB-XL waveforms, cached `.npy` arrays, and model checkpoints (`.pth`) are **not** tracked in this repository (they are large and regenerable). A fresh clone will not contain them; run notebook 01 after downloading PTB-XL to reconstruct the derived data, and the training notebooks to reproduce the checkpoints.

---

## Data availability and ethics

This study uses **PTB-XL v1.0.3**, publicly available from PhysioNet under a Creative Commons Attribution 4.0 licence (Wagner et al., 2020; Goldberger et al., 2000). No data were collected for this study and no raw PTB-XL data are redistributed here. The record-count difference against the original PTB-XL publication (21,799 vs. 21,837) reflects an upstream de-duplication fix documented in the v1.0.3 changelog.

PTB-XL is fully de-identified and publicly released; no IRB approval or individual consent was required for this secondary analysis, and no attempt was made to re-identify any subject. This work reports subgroup performance disparities for research purposes; **it does not propose a deployable clinical device, and none of the results should be used to guide patient care.**

---

## Citation

If you use this code or its findings, please cite the accompanying manuscript:

```bibtex
@misc{tandon2026agestratified,
  title  = {Age-Stratified Audit of Calibration, Conformal Uncertainty,
            and Explanation Stability in Multi-Label ECG Classification on PTB-XL},
  author = {Tandon, Harshit and Mathunjwa, Bhekumuzi Mkhululi and Ranka, Rachit},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

Please also cite the PTB-XL dataset: Wagner, P. et al. (2020), *PTB-XL, a Large Publicly Available Electrocardiography Dataset*, Scientific Data 7, 154.
