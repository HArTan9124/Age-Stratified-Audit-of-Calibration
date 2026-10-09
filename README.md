# Age-Stratified Explainable & Uncertainty-Calibrated ECG Diagnosis (PTB-XL)

A rigorous research framework investigating age-stratified electrocardiogram (ECG) diagnosis, uncertainty calibration, and interpretability using the PhysioNet **PTB-XL v1.0.3** clinical database.

---

## 1. Project Directory Structure

```text
age with ecg/
├── notebooks/
│   ├── 01_Data_Preparation.ipynb              # Data pipeline (bandpass filter, patient-safe split, .npy arrays)
│   ├── 02_3Model_Heterogeneous_Ensemble.ipynb # Equal-weight 3-model soft-voting ensemble
│   ├── 03_Optimized_Best_Model.ipynb          # AUC-weighted ensemble + per-class thresholds (RECOMMENDED)
│   ├── 04_ResNet18_vs_Transformer_Baseline.ipynb  # Standalone ResNet1D-18 vs ResNet1D-Transformer baseline comparison
│   ├── 05_Calibration_and_Uncertainty.ipynb  # Reliability/ECE, temperature scaling, conformal, ensemble uncertainty (reloads NB03)
│   └── README.md                             # Notebook run-order guide
├── data/
│   ├── ptb-xl/                         # Official PhysioNet PTB-XL v1.0.3 dataset root (read-only)
│   └── processed/                      # Final research artifacts (MASTER_RESEARCH_METADATA.csv, dataset_manifest.json, .npy arrays)
├── outputs/                            # Model checkpoints (.pth), training curves, validation figures
│   ├── dataset_validation/             # High-resolution research validation figures (PNG)
│   ├── training_curves/
│   └── download_diff_report.txt
├── papers/                             # Reference papers (PDF) + Research_and_Analysis.md + Whats_Needed_Next.md
├── presentations/                      # Project decks (.pptx)
├── docs/
│   ├── DAILY_DEVELOPMENT_LOG.md               # Contributor engineering and research diary
│   └── RESEARCH_DATASET_AND_PIPELINE_REFERENCE.md  # Slide-by-slide presentation deck with embedded figures
├── records100/ · records500/          # Official PTB-XL waveform data (read-only)
├── ptbxl_database.csv · scp_statements.csv · RECORDS · SHA256SUMS.txt  # Official PTB-XL files (read-only)
└── README.md                           # Project documentation
```

---

## 2. Research Workflow

Run the numbered notebooks in order (see `notebooks/README.md`). The data lifecycle for both
**100 Hz (low-resolution)** and **500 Hz (high-resolution)** signals lives in:

### **[notebooks/01_Data_Preparation.ipynb](file:///home/tandon/Teep/age%20with%20ecg/notebooks/01_Data_Preparation.ipynb)**
* **Module 1:** Environment Setup & Dual-Resolution Path Resolver
* **Module 2:** Dataset Structure & Download Completion Diff Inspection
* **Module 3:** Metadata Ingestion, Diagnostic Mapping (5 Superclasses) & Feature Engineering
* **Module 4:** Demographic Profiling & Epidemiological Analysis (EDA)
* **Module 5:** Dual-Resolution 12-Lead ECG Waveform Visualization (100 Hz & 500 Hz) & 0.5–40 Hz Filtering
* **Module 6:** Leakage-Free 4-Way Patient-Safe Splitting (`TRAIN`, `VAL`, `CAL`, `TEST`)
* **Module 7:** Unified Batch Signal Extraction & Fast Disk Caching (`.npy` arrays)
* **Module 8:** Production PyTorch Dataset & Multi-Task DataLoader Pipeline
* **Module 9:** Automated 17-Point Research Dataset Readiness Audit
* **Module 10:** Structured Research Findings & Model Training Strategy

---

## 3. Key Rules & Reproducibility Standards
- Original PTB-XL files (`ptbxl_database.csv`, `scp_statements.csv`, `records100/`, `records500/`) must remain untouched and read-only.
- All derived datasets are stored in `data/processed/` and figures in `outputs/`.
- Zero data leakage: Normalization parameters (mean, std) are fitted strictly on the `TRAIN` split.


