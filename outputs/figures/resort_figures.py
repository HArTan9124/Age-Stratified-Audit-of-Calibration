#!/usr/bin/env python3
"""Re-sort freshly generated figures into outputs/figures/nbXX_*/ by filename.

Notebooks still write PNGs into outputs/dataset_validation/ (and NB04 into outputs/
+ outputs/training_curves/). Run this after re-executing any notebook:

    python outputs/figures/resort_figures.py
"""
import shutil
from pathlib import Path

OUT  = Path(__file__).resolve().parent.parent          # .../outputs
DEST = OUT / "figures"

# filename  ->  notebook folder
ROUTE = {
    # NB01
    **{f: "nb01_data_preparation" for f in [
        "01_age_distribution_density.png", "02_ecg_counts_by_age_group.png",
        "03_diagnostic_superclass_prevalence.png", "04_diagnosis_prevalence_by_age_group.png",
        "05_multilabel_frequency_distribution.png", "06_split_sizes_patients_records.png",
        "07_age_distribution_by_split.png", "08_diagnosis_distribution_by_split.png",
        "09_representative_raw_12lead_ecg.png",
        "10_representative_preprocessed_filtered_12lead_ecg.png"]},
    # NB02
    **{f: "nb02_ensemble_baseline" for f in [
        "03_dual_branch_confusion_matrices.png", "03_training_curves.png",
        "04_ensemble_roc_curves.png", "04_ensemble_confusion_matrices.png",
        "04_ensemble_training_curves.png"]},
    # NB03
    **{f: "nb03_optimized_model" for f in [
        "05_optimized_inceptiontime_training_curves.png", "05_optimized_best_roc_curves.png",
        "05_optimized_best_confusion_matrices.png", "05_four_config_benchmark.png",
        "05_age_stratified_metric_comparison.png"]},
    # NB04
    **{f: "nb04_resnet_vs_transformer" for f in [
        "test_roc_clean.png", "model_training_comparison.png", "test_multilabel_roc_curves.png"]},
    # NB05
    **{f: "nb05_calibration_uncertainty" for f in [
        "05cal_reliability_uncalibrated.png", "05cal_reliability_calibrated.png",
        "05cal_uncertainty_vs_error.png", "05cal_risk_coverage.png",
        "05cal_age_stratified_ece.png"]},
    # NB06
    **{f: "nb06_explainability" for f in [
        "06xai_per_lead_importance.png", "06xai_map_NORM.png", "06xai_map_MI.png",
        "06xai_map_STTC.png", "06xai_map_CD.png", "06xai_map_HYP.png",
        "06xai_faithfulness_ablation.png"]},
    # NB07
    **{f: "nb07_statistical_robustness" for f in [
        "07_bootstrap_headline_ci.png", "07_per_class_auc_ci.png",
        "07_ensemble_vs_single_delta.png"]},
}

SEARCH = [OUT / "dataset_validation", OUT / "training_curves", OUT]
moved = 0
for base in SEARCH:
    if not base.exists():
        continue
    for png in list(base.glob("*.png")):
        folder = ROUTE.get(png.name)
        if not folder:
            continue
        d = DEST / folder
        d.mkdir(parents=True, exist_ok=True)
        shutil.move(str(png), str(d / png.name))
        print(f"  {png.name:52s} -> figures/{folder}/")
        moved += 1

for sub in (OUT / "dataset_validation", OUT / "training_curves"):
    if sub.exists() and not any(sub.iterdir()):
        sub.rmdir()

print(f"\n{moved} figure(s) re-sorted." if moved else "Nothing to move — already sorted.")
