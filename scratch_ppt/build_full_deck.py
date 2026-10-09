"""ECG_PTBXL_NB01-06 — full professional deck (English, formal).
Rebuilds the merged deck with a single consistent notebook-numbering scheme,
all fixes from the review, a 6-paper literature comparison (2 slides + refs),
and a prioritized next-steps pipeline.
"""
from pathlib import Path
from pptx.util import Inches, Pt
from pptbuild import (new_deck, title_slide, content_slide, bullets, panel,
                      panel_title, image_fit, caption, simple_table, flow_boxes,
                      divider_slide, _tb, _set, add_run,
                      INK, ACCENT, ACCENT2, MUTE, WHITE)

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "outputs" / "dataset_validation"
A = ROOT / "presentations" / "assets_v2"
OUT = ROOT / "presentations" / "ECG_PTBXL_NB01-06_Professional.pptx"

prs = new_deck()
TOTAL = 24
_pg = [0]
def N():
    _pg[0] += 1; return _pg[0]

# ============================================================ 1 — TITLE
N()
title_slide(
    prs,
    "PTB-XL 12-Lead ECG  ·  Notebooks 01–06",
    "From Raw Signals to a Calibrated, Explainable Diagnostic Model",
    "A reproducible pipeline for age-stratified, explainable and uncertainty-calibrated "
    "diagnosis of 5 ECG superclasses (NORM, MI, STTC, CD, HYP).",
    [
        "Dataset: PhysioNet PTB-XL v1.0.3 — 21,799 records · 18,869 patients · 100 Hz (1000 samples × 12 leads).",
        "Task: multi-label classification of 5 diagnostic superclasses; continuous age kept as an auxiliary target.",
        "NB01 data preparation → NB02 heterogeneous ensemble baseline → NB04 architecture check → "
        "NB03 optimized best model → NB05 calibration & uncertainty → NB06 explainability.",
        "Headline: macro-AUC ≈ 0.923 (best single) / 0.924 (AUC-weighted ensemble) on the held-out fold-10 TEST set (N = 2,198).",
        "Contribution beyond the cited literature: measured calibration, conformal guarantees, uncertainty-based "
        "abstention, and faithfulness-tested explanations.",
    ],
)

# ============================================================ 2 — NB01 process
s = content_slide(prs, N(), TOTAL, "Notebook 01 — Data preparation pipeline",
                  "Raw WFDB signals → analysis-ready, leakage-free float32 tensors")
steps = [
    ("1 · Ingest & map", "Parse ptbxl_database.csv + scp_statements.csv; map SCP codes to 5 superclasses; keep sub-classes + multi-label counts."),
    ("2 · Feature engineering", "Age bands <40 / 40–65 / 65–80 / 80+; sex → F/M; BMI from height & weight; continuous age kept as a regression target."),
    ("3 · Signal conditioning", "Read WFDB 12-lead signals; mV calibration via per-lead gain & baseline; 0.5–40 Hz zero-phase Butterworth band-pass (filtfilt)."),
    ("4 · Patient-safe split", "Folds 1–8 → TRAIN (17,418); Fold 10 → TEST (2,198); Fold 9 split by patient → VAL (1,080) + CALIBRATION (1,103). All 6 pairwise patient intersections = ∅."),
    ("5 · Array caching", "Batch-extract to float32 .npy tensors (N, 1000, 12); export row-aligned MASTER_RESEARCH_METADATA.csv; sub-100 ms tensor loading."),
    ("6 · Readiness audit", "17-point QA checklist (schema, demographics, mapping); zero-leakage asserts on every split; PyTorch Dataset → (B, 12, 1000) batches."),
]
y0 = 1.28
for i, (h, sub) in enumerate(steps):
    col = i % 2; row = i // 2
    x = 0.5 + col * 6.45
    yy = y0 + row * 1.94
    panel(s, Inches(x), Inches(yy), Inches(6.2), Inches(1.78))
    _, tf = _tb(s, Inches(x + 0.18), Inches(yy + 0.12), Inches(5.85), Inches(1.55))
    _set(tf.paragraphs[0], h, 12.5, ACCENT2, bold=True, space_after=3)
    p = tf.add_paragraph(); _set(p, sub, 9.7, INK)

# ============================================================ 3 — NB01 signal conditioning viz
s = content_slide(prs, N(), TOTAL, "Notebook 01 — Signal conditioning, before vs after",
                  "Per-lead mV calibration + 0.5–40 Hz zero-phase band-pass, shown on a representative record")
image_fit(s, str(A / "nb01_cohort2.png"), Inches(0.45), Inches(1.25), Inches(6.25), Inches(4.4), valign="top")
caption(s, Inches(0.45), Inches(5.75), Inches(6.25), "Raw 100 Hz 12-lead acquisition (ID 1, age 56, NORM).")
image_fit(s, str(VAL / "10_representative_preprocessed_filtered_12lead_ecg.png"),
          Inches(6.95), Inches(1.25), Inches(5.95), Inches(4.4), valign="top")
caption(s, Inches(6.95), Inches(5.75), Inches(5.95), "After calibration + band-pass filtering — the signal fed to every model.")
bullets(s, Inches(0.45), Inches(6.25), Inches(12.5), Inches(1.0), [
    ("Filtering removes baseline wander and high-frequency noise while preserving P/QRS/T morphology; "
     "100 Hz is retained (Strodthoff 2020: 500 Hz gives no measurable benefit).", 1, MUTE, False),
], size=9.5, gap=0)

# ============================================================ 4 — NB01 cohort profile
s = content_slide(prs, N(), TOTAL, "Notebook 01 — Cohort profile & the 'aging heart'",
                  "Why every downstream notebook evaluates performance stratified by age")
image_fit(s, str(A / "nb01_viz3.png"), Inches(0.45), Inches(1.25), Inches(8.1), Inches(5.4), valign="top")
caption(s, Inches(0.45), Inches(6.75), Inches(8.1),
        "Age × sex distribution · superclass prevalence · prevalence by age band · multi-label count per record.")
panel(s, Inches(8.75), Inches(1.25), Inches(4.15), Inches(5.5))
bullets(s, Inches(8.95), Inches(1.4), Inches(3.8), Inches(5.2), [
    ("Cohort scale", 0, ACCENT2, True),
    ("21,799 ECGs · 18,869 patients · ~52% male · mean age 59.8 ± 17 y.", 1, INK, False),
    ("Superclass prevalence", 0, ACCENT2, True),
    ("NORM 43.6% · MI 25.1% · STTC 24.0% · CD 22.5% · HYP 12.2%. HYP is rarest and weakest → targeted in NB03.", 1, INK, False),
    ("The aging-heart inversion", 0, ACCENT2, True),
    ("Age <40: ~78% normal, CD rare (~6%). Age 80+: normal collapses to ~17%; CD ~43%, MI ~38%.", 1, INK, False),
    ("→ motivates the age-stratified audit repeated in NB02–NB06.", 1, ACCENT, True),
], size=10.5, gap=5)

# ============================================================ 5 — end-to-end pipeline
s = content_slide(prs, N(), TOTAL, "End-to-end pipeline & notebook ownership",
                  "NB01 owns RAW → LOAD (run once, shared). All model notebooks consume the same cached tensors and TEST split.")
flow_boxes(s, Inches(0.5), Inches(1.5), Inches(12.4), Inches(1.5), [
    ("RAW", "WFDB 12-lead\n100 Hz"),
    ("CLEAN", "mV calibrate\n0.5–40 Hz"),
    ("CACHE", "float32 .npy\n(N,1000,12)+meta"),
    ("SPLIT", "patient-safe\nTRAIN/VAL/CALIB/TEST"),
    ("LOAD", "DataLoader\n(B,12,1000)"),
    ("TRAIN", "ResNet / Inception\n/ Transformer"),
    ("TUNE", "early stop · LR\nper-class thr"),
    ("EVAL", "macro-AUC\nage audit"),
], fill=ACCENT2, size=8, head_size=11)
panel(s, Inches(0.5), Inches(3.5), Inches(12.4), Inches(3.4))
bullets(s, Inches(0.75), Inches(3.7), Inches(11.9), Inches(3.0), [
    ("NB01 — builds RAW → LOAD and the 4-way patient-disjoint split; every later notebook reloads these tensors, never re-derives them.", 1, INK, False),
    ("NB02 — trains 3 architecturally diverse models (DualBranchECGNet, InceptionTime1D, ResNet1D-101) and combines them by equal-weight soft voting (baseline).", 1, INK, False),
    ("NB04 — standalone architecture check: ResNet1D-18 vs ResNet1D-Transformer, plus an age-only logistic baseline and an ECE calibration probe.", 1, INK, False),
    ("NB03 — the optimized configuration: literature-informed InceptionTime1D + AUC-weighted ensemble (six fixes, next slide).", 1, INK, False),
    ("NB05 / NB06 — post-hoc audits on the frozen NB03 model: calibration & uncertainty, then explainability.", 1, INK, False),
    ("Common protocol — identical held-out fold-10 TEST set, macro-AUC as the headline metric, age-stratified fairness audit everywhere.", 1, ACCENT, True),
], size=10.5, gap=6)

# ============================================================ 6 — NB02 baseline ensemble
s = content_slide(prs, N(), TOTAL, "Notebook 02 — 3-Model heterogeneous ensemble (baseline)",
                  "Equal-weight soft voting over three architecturally diverse models — the honest starting point NB03 improves on")
image_fit(s, str(VAL / "04_ensemble_roc_curves.png"), Inches(0.45), Inches(1.25), Inches(12.45), Inches(2.35), valign="top")
caption(s, Inches(0.45), Inches(3.7), Inches(12.45),
        "Per-class TEST ROC for the equal-weight 3-model ensemble (N = 2,198).")
panel(s, Inches(0.45), Inches(4.1), Inches(7.4), Inches(3.0))
panel_title(s, Inches(0.65), Inches(4.22), Inches(7.0), "Measured on the real TEST set")
simple_table(s, Inches(0.65), Inches(4.62), Inches(7.0), [
    ["model", "macro-AUC", "per-label acc", "exact-match"],
    ["DualBranchECGNet", "0.899", "85.8%", "54.4%"],
    ["InceptionTime1D", "0.928", "87.1%", "55.6%"],
    ["ResNet1D-101", "0.918", "85.9%", "52.7%"],
    ["Equal-weight ensemble", "0.924", "87.3%", "56.7%"],
], col_ratio=[0.40, 0.20, 0.22, 0.18], size=9.5, row_h=Inches(0.34))
bullets(s, Inches(8.1), Inches(4.15), Inches(4.8), Inches(3.0), [
    ("Key finding", 0, ACCENT2, True),
    ("The equal-weight ensemble (0.924) does NOT beat its best single member, InceptionTime1D (0.928) — "
     "the weak DualBranch model drags the average down.", 1, INK, False),
    ("Matches Strodthoff 2020: “the best single resnet/inception model remains compatible with the ensemble within error bars.”", 1, INK, False),
    ("→ NB03 replaces equal weights with AUC-weighting.", 1, ACCENT, True),
], size=10.5, gap=6)

# ============================================================ 7 — NB04 architecture check
s = content_slide(prs, N(), TOTAL, "Notebook 04 — ResNet-1D-18 vs ResNet-1D-Transformer",
                  "Does adding a Transformer encoder over convolutional tokens help? — a controlled architecture check")
image_fit(s, str(A / "nb04_test_roc.png"), Inches(0.5), Inches(1.3), Inches(6.3), Inches(4.6), valign="top", align="left")
caption(s, Inches(0.5), Inches(6.05), Inches(6.6), "Test ROC — best clean model, per-class AUC.")
bullets(s, Inches(7.15), Inches(1.5), Inches(5.75), Inches(5.5), [
    ("Setup", 0, ACCENT2, True),
    ("ResNet1D-18 (≈4 stages, global average pool) vs the same backbone + a 2-layer Transformer encoder over conv tokens.", 1, INK, False),
    ("BCEWithLogits + per-class pos_weight; AdamW, cosine LR, early stopping on validation macro-AUC.", 1, INK, False),
    ("Bug fixes: single-pass val-AUC collection, shuffle=False on val/test, age=300 sentinel → NaN.", 1, INK, False),
    ("Verdict", 0, ACCENT, True),
    ("The Transformer gives no meaningful gain over the plain ResNet — keep the simpler model.", 1, INK, True),
    ("Independently corroborated by Jafari & Jafari 2026: stacked / bidirectional recurrent layers also fail to beat a "
     "well-tuned CNN on PTB-XL (same split).", 1, MUTE, False),
], size=11, gap=7)

# ============================================================ 8 — NB03 six fixes
s = content_slide(prs, N(), TOTAL, "Notebook 03 — Six literature-grounded fixes",
                  "Turning the NB02 baseline into the optimized configuration audited by NB05 / NB06")
simple_table(s, Inches(0.5), Inches(1.35), Inches(12.35), [
    ["#  Fix", "What it does — and why"],
    ["1 · Strongest single model", "Train InceptionTime1D properly (Strodthoff 2020; Gitau 2025) instead of leaning on an ensemble to hide a weak member."],
    ["2 · Minority-class augmentation", "Gaussian noise (σ = 0.01) on every HYP-positive record — the rarest & weakest superclass (Atwa 2025, Table 4).  ⚠ ablate: Atwa's own results show this recipe can hurt."],
    ["3 · Adaptive LR scheduling", "ReduceLROnPlateau on validation macro-AUC + EarlyStopping, replacing a blind fixed schedule."],
    ["4 · Per-class thresholds", "Fmax-style: sweep thresholds on VAL, apply frozen to TEST — no 0.5 default, no test leakage."],
    ["5 · AUC-weighted ensemble", "Weight each member by its validation macro-AUC instead of equal 1/3 voting (reuses NB02 checkpoints)."],
    ["6 · Honest metrics", "Headline = macro-AUC + per-label accuracy; exact-match accuracy shown only as a non-comparable reference."],
], col_ratio=[0.26, 0.74], size=10, row_h=Inches(0.6))

# ============================================================ 9 — NB03 four configs
s = content_slide(prs, N(), TOTAL, "Notebook 03 — Four configurations on the TEST set",
                  "Real held-out test set, N = 2,198 — exact-match vs per-label accuracy vs macro-AUC")
image_fit(s, str(A / "nb03_four_config.png"), Inches(0.45), Inches(1.3), Inches(12.45), Inches(3.7), valign="top")
caption(s, Inches(0.45), Inches(5.15), Inches(12.45),
        "Blue = exact-match accuracy · navy = per-label (Hamming) accuracy · green line = macro-AUC.")
panel(s, Inches(0.45), Inches(5.5), Inches(12.45), Inches(1.7))
bullets(s, Inches(0.7), Inches(5.62), Inches(12.0), Inches(1.5), [
    ("Thresholds recover accuracy at zero retrain cost: InceptionTime1D per-label 84.3% → 86.8% just by moving off the 0.5 default.", 1, INK, False),
    ("The equal-weight ensemble underperforms its own best member; the AUC-weighted ensemble + optimized thresholds is the best config "
     "(per-label 87.9%, macro-AUC ≈ 0.92–0.925 depending on run).", 1, ACCENT, True),
], size=10, gap=6)

# ============================================================ 10 — divider NB05
N(); divider_slide(prs, "Notebook 05", "Uncertainty Calibration & Predictive Uncertainty",
              "Are the model's probabilities trustworthy — and does it know when it is wrong? "
              "Calibrated on the patient-disjoint CONFORMAL_CALIBRATION split (1,103 records), never on VALIDATION.")

# ============================================================ 11 — NB05 M6
s = content_slide(prs, N(), TOTAL, "NB05 · Module 6 — Reliability & calibration error",
                  "Uncalibrated. 15 equal-width bins per superclass;  ECE = Σ (n_b/N)·|acc_b − conf_b|,  MCE = max bin gap")
image_fit(s, str(VAL / "05cal_reliability_uncalibrated.png"), Inches(0.45), Inches(1.2), Inches(12.45), Inches(2.45), valign="top")
caption(s, Inches(0.45), Inches(3.6), Inches(12.45),
        "Observed positive-rate vs predicted probability. Bars below the diagonal = the ensemble is over-confident.")
panel(s, Inches(0.45), Inches(4.05), Inches(7.35), Inches(3.05))
panel_title(s, Inches(0.65), Inches(4.18), Inches(6.9), "Per-class TEST calibration (uncalibrated)")
simple_table(s, Inches(0.65), Inches(4.62), Inches(6.95), [
    ["", "NORM", "MI", "STTC", "CD", "HYP", "MACRO"],
    ["ECE", "0.082", "0.076", "0.063", "0.090", "0.097", "0.082"],
    ["MCE", "0.270", "0.272", "0.260", "0.346", "0.376", "0.305"],
    ["Brier", "0.100", "0.104", "0.090", "0.096", "0.083", "0.094"],
], col_ratio=[0.16] + [0.14]*6, size=10.5, row_h=Inches(0.4))
bullets(s, Inches(8.05), Inches(4.15), Inches(4.85), Inches(3.0), [
    ("What it establishes", 0, ACCENT2, True),
    ("Probabilities are NOT calibrated as-is: macro-ECE 0.082; worst bins off by ~30–38 points.", 1, INK, False),
    ("Miscalibration is worst on CD and HYP (the rarest classes).", 1, INK, False),
    ("Bias is consistent over-confidence → one softening temperature per class should help.", 1, ACCENT, True),
], size=12, gap=7)

# ============================================================ 12 — NB05 M7
s = content_slide(prs, N(), TOTAL, "NB05 · Module 7 — Per-class temperature scaling",
                  "Fit one scalar T per class by NLL on CALIBRATION, freeze it, apply to TEST")
image_fit(s, str(VAL / "05cal_reliability_calibrated.png"), Inches(0.45), Inches(1.2), Inches(12.45), Inches(2.45), valign="top")
caption(s, Inches(0.45), Inches(3.6), Inches(12.45),
        "Reliability after per-class temperature scaling (TEST, N = 2,198). Curves move toward the diagonal, most visibly for CD and HYP.")
panel(s, Inches(0.45), Inches(4.05), Inches(7.35), Inches(3.05))
panel_title(s, Inches(0.65), Inches(4.18), Inches(6.9), "Fitted T and macro effect (TEST)")
simple_table(s, Inches(0.65), Inches(4.62), Inches(6.95), [
    ["", "NORM", "MI", "STTC", "CD", "HYP"],
    ["fitted T", "1.29", "0.94", "1.10", "0.85", "0.79"],
], col_ratio=[0.18] + [0.164]*5, size=10.5, row_h=Inches(0.4))
simple_table(s, Inches(0.65), Inches(5.55), Inches(6.95), [
    ["macro ECE", "0.082 → 0.075", "macro MCE", "0.305 → 0.298"],
    ["macro Brier", "0.0945 → 0.0939", "macro-AUC", "0.9245 (unchanged)"],
], col_ratio=[0.25, 0.28, 0.22, 0.25], size=9.5, header=False, row_h=Inches(0.4))
bullets(s, Inches(8.05), Inches(4.15), Inches(4.85), Inches(3.0), [
    ("Reading", 0, ACCENT2, True),
    ("T > 1 (NORM, STTC) softens over-confidence; T < 1 (CD, HYP) sharpens under-confident heads.", 1, INK, False),
    ("Macro-ECE 0.082 → 0.075 (−15%); σ(z/T) is strictly monotone, so macro-AUC is provably unchanged (asserted in code).", 1, INK, False),
    ("Limit: one scalar per class can't fix bin-shape miscalibration — MCE barely moves.", 1, ACCENT, True),
], size=12, gap=7)

# ============================================================ 13 — NB05 M8-9
s = content_slide(prs, N(), TOTAL, "NB05 · Modules 8 & 9 — Conformal thresholds + predictive uncertainty",
                  "Per-class conformal sensitivity guarantee (α = 0.10)  |  disagreement & entropy as error detectors")
panel(s, Inches(0.45), Inches(1.2), Inches(6.05), Inches(3.15))
panel_title(s, Inches(0.65), Inches(1.32), Inches(5.6), "Module 8 — split-conformal @ 90% target sensitivity")
simple_table(s, Inches(0.65), Inches(1.72), Inches(5.65), [
    ["class", "thr p≥", "realised sens.", "alert rate"],
    ["NORM", "0.61", "0.913", "48.5%"],
    ["MI", "0.25", "0.907", "42.3%"],
    ["STTC", "0.37", "0.889", "33.9%"],
    ["CD", "0.30", "0.853", "34.7%"],
    ["HYP", "0.13", "0.889", "36.6%"],
], col_ratio=[0.22, 0.20, 0.34, 0.24], size=10, row_h=Inches(0.29))
caption(s, Inches(0.65), Inches(3.7), Inches(5.7),
        "Mean predicted-set size ≈ 1.9 / 5; guarantee needs only CAL–TEST exchangeability (holds: same prep, disjoint folds).")
image_fit(s, str(VAL / "05cal_uncertainty_vs_error.png"), Inches(6.75), Inches(1.3), Inches(6.15), Inches(2.7), valign="top")
caption(s, Inches(6.75), Inches(4.02), Inches(6.15),
        "Uncertainty on correctly vs incorrectly classified TEST records (5-way exact match).")
panel(s, Inches(0.45), Inches(4.5), Inches(12.45), Inches(2.6))
panel_title(s, Inches(0.65), Inches(4.62), Inches(11.8), "Module 9 — does uncertainty flag the model's own errors?")
bullets(s, Inches(0.65), Inches(5.0), Inches(12.0), Inches(2.0), [
    ("Exact-match correct on TEST = 55.6% (1221 / 2198) at the calibrated 0.5 operating point.", 1, INK, False),
    ("Disagreement (mean class-wise std of the 3 members): 0.046 correct vs 0.076 wrong → error-detection ROC-AUC 0.737.", 1, INK, False),
    ("Predictive entropy of the calibrated ensemble prob.: 0.237 correct vs 0.379 wrong → error-detection ROC-AUC 0.775.", 1, INK, False),
    ("Both scores are clearly higher on wrong predictions — usable as an abstention signal (next slide).", 1, ACCENT, True),
], size=11, gap=5)

# ============================================================ 14 — NB05 M10-12
s = content_slide(prs, N(), TOTAL, "NB05 · Modules 10–12 — Selective prediction, age-band calibration, verdict",
                  "Abstain on the least-confident cases  |  is 80+ also the worst-calibrated group?")
image_fit(s, str(VAL / "05cal_risk_coverage.png"), Inches(0.6), Inches(1.2), Inches(5.55), Inches(3.05), valign="top")
caption(s, Inches(0.45), Inches(4.3), Inches(6.15), "Retain most-confident c%; performance on the kept set vs coverage.")
image_fit(s, str(VAL / "05cal_age_stratified_ece.png"), Inches(7.0), Inches(1.2), Inches(5.75), Inches(3.05), valign="top")
caption(s, Inches(6.95), Inches(4.3), Inches(6.0), "Macro-ECE per age band, before vs after temperature scaling.")
panel(s, Inches(0.45), Inches(4.7), Inches(6.15), Inches(1.95))
panel_title(s, Inches(0.65), Inches(4.8), Inches(5.7), "Risk–coverage (defer by disagreement)")
simple_table(s, Inches(0.65), Inches(5.15), Inches(5.75), [
    ["coverage", "100%", "90%", "80%", "70%"],
    ["macro-AUC", "0.925", "0.931", "0.937", "0.942"],
    ["exact-match", "55.6%", "58.1%", "60.9%", "64.4%"],
], col_ratio=[0.30, 0.175, 0.175, 0.175, 0.175], size=9.5, row_h=Inches(0.36))
panel(s, Inches(6.95), Inches(4.7), Inches(5.85), Inches(1.95))
panel_title(s, Inches(7.15), Inches(4.8), Inches(5.5), "Age-band calibration (Module 11)")
bullets(s, Inches(7.15), Inches(5.12), Inches(5.45), Inches(1.5), [
    ("Macro-ECE climbs <40: 0.055 → 80+: 0.135; temp-scaling barely helps 80+ (0.135 → 0.133).", 1, INK, False),
    ("Mean uncertainty also rises with age (0.038 → 0.078) — the model does signal it struggles on 80+.", 1, INK, False),
], size=9.5, gap=5)
bullets(s, Inches(0.45), Inches(6.85), Inches(12.45), Inches(0.55), [
    ("Verdict: the “uncertainty-calibrated” claim now has measured results. Open: single fold-10 TEST set, no bootstrap CI; "
     "heterogeneous (not N-reinit) uncertainty ensemble; per-class marginal (not joint) conformal.", 1, MUTE, False),
], size=9, gap=0)

# ============================================================ 15 — divider NB06
N(); divider_slide(prs, "Notebook 06", "Explainability of the InceptionTime1D Model",
              "Which leads and which parts of the beat drive each decision — and does the explanation survive "
              "faithfulness and sanity checks the cited XAI work skipped?")

# ============================================================ 16 — NB06 M3
s = content_slide(prs, N(), TOTAL, "NB06 · Module 3 — Per-lead Integrated-Gradients importance",
                  "I_l = mean_i mean_t |IG(i,l,t)| over 160 true-positive TEST records / class, row-normalised (Atwa 2025 §3.6 formula, SHAP → IG)")
image_fit(s, str(VAL / "06xai_per_lead_importance.png"), Inches(0.45), Inches(1.2), Inches(12.45), Inches(3.0), valign="top")
caption(s, Inches(0.45), Inches(4.33), Inches(12.45),
        "Share of |IG| mass per lead, per superclass (brighter = more). Lead III is consistently the least used.")
panel(s, Inches(0.45), Inches(4.72), Inches(8.15), Inches(2.5))
panel_title(s, Inches(0.65), Inches(4.82), Inches(7.7), "Top-3 IG leads  vs  clinical expectation (Atwa 2025 §4.6)")
simple_table(s, Inches(0.65), Inches(5.18), Inches(7.75), [
    ["class", "top-3 leads (IG)", "clinical expectation"],
    ["NORM", "V2, aVF, V6", "diffuse / no single lead"],
    ["MI", "V2, V1, II", "V2, II, V3, V4"],
    ["STTC", "V6, V1, V2", "V4, V5, aVR"],
    ["CD", "V1, V2, aVF", "V1, V2, V3"],
    ["HYP", "aVL, aVF, V6", "V5, V6, V1"],
], col_ratio=[0.16, 0.40, 0.44], size=9.5, row_h=Inches(0.3))
bullets(s, Inches(8.8), Inches(4.8), Inches(4.1), Inches(2.4), [
    ("Reading", 0, ACCENT2, True),
    ("Precordial V1/V2 dominate MI, STTC and CD — the anteroseptal leads clinicians read for those.", 1, INK, False),
    ("HYP leans on limb leads aVL / aVF (the LVH-axis leads) + V6.", 1, INK, False),
    ("Module 3b: ranking near-identical under a per-lead-mean IG baseline — Spearman ρ 0.98–1.00.", 1, ACCENT, True),
], size=10, gap=5)

# ============================================================ 17 — NB06 M4
s = content_slide(prs, N(), TOTAL, "NB06 · Module 4 — Time-domain attribution maps",
                  "IG overlay on every lead + Grad-CAM-1D temporal strip, for the most-confident true positive per class (Strodthoff 2020, Fig 8 style)")
image_fit(s, str(VAL / "06xai_map_MI.png"), Inches(0.4), Inches(1.12), Inches(6.2), Inches(5.72), valign="top", align="left")
caption(s, Inches(0.4), Inches(6.95), Inches(6.5), "MI, most-confident true positive (TEST idx 2035, p = 1.00, age 87).")
bullets(s, Inches(6.7), Inches(1.4), Inches(6.2), Inches(5.7), [
    ("What to look for", 0, ACCENT2, True),
    ("Red = per-sample |IG|; brighter dots mark the timepoints that move the MI logit most.", 1, MUTE, False),
    ("IG mass concentrates on the QRS deflection / R-peak in the precordial leads (V1–V3, V5) — "
     "the model reads beat morphology, not the flat baseline segment.", 1, INK, False),
    ("The Grad-CAM-1D strip lights up in periodic bands locked to every QRS complex — temporal localisation is beat-synchronous.", 1, INK, False),
    ("Limb leads carry visibly less relevance here, matching the precordial emphasis in the Module 3 heatmap.", 1, INK, False),
    ("All five class maps are saved to outputs/dataset_validation/06xai_map_*.png.", 1, MUTE, False),
], size=11, gap=8)

# ============================================================ 18 — NB06 M5
s = content_slide(prs, N(), TOTAL, "NB06 · Module 5 — Faithfulness: lead-ablation test",
                  "Replace top-k IG leads (vs k random leads) with their TEST-set mean trace; measure per-class AUC drop on 800 records")
image_fit(s, str(VAL / "06xai_faithfulness_ablation.png"), Inches(0.45), Inches(1.2), Inches(12.45), Inches(2.5), valign="top")
caption(s, Inches(0.45), Inches(3.62), Inches(12.45),
        "Deletion-metric curves: solid red = remove top-k IG leads, dashed grey = remove k random leads (mean of 3 draws).")
panel(s, Inches(0.45), Inches(4.05), Inches(8.35), Inches(3.1))
panel_title(s, Inches(0.65), Inches(4.15), Inches(7.9), "AUC drop:  top-k IG leads  /  random k leads")
simple_table(s, Inches(0.65), Inches(4.52), Inches(7.9), [
    ["class", "k=1", "k=2", "k=3", "k=5"],
    ["NORM", "0.003 / 0.005", "0.022 / 0.023", "0.044 / 0.026", "0.074 / 0.063"],
    ["MI", "0.035 / 0.010", "0.065 / 0.020", "0.106 / 0.048", "0.210 / 0.116"],
    ["STTC", "0.015 / 0.005", "0.029 / 0.011", "0.051 / 0.023", "0.081 / 0.079"],
    ["CD", "0.070 / 0.007", "0.109 / 0.036", "0.118 / 0.048", "0.164 / 0.042"],
    ["HYP", "0.042 / 0.041", "0.072 / 0.025", "0.174 / 0.056", "0.163 / 0.147"],
], col_ratio=[0.13, 0.2175, 0.2175, 0.2175, 0.2175], size=8.5, row_h=Inches(0.33))
bullets(s, Inches(9.0), Inches(4.1), Inches(3.9), Inches(3.0), [
    ("Reading", 0, ACCENT2, True),
    ("MI, STTC, CD: solid stays clearly above dashed at every k → attribution is faithful, not just plausible "
     "(CD strongest: 10× at k=1).", 1, INK, False),
    ("NORM: near-parity — expected, “normal” = absence of focal pathology.", 1, INK, False),
    ("Top-3 |IG| mass share: NORM .31 · MI .33 · STTC .34 · CD .36 · HYP .35 (0.25 = uniform).", 1, MUTE, False),
], size=9.5, gap=5)

# ============================================================ 19 — NB06 M6
s = content_slide(prs, N(), TOTAL, "NB06 · Module 6 — Model-randomisation sanity check + verdict",
                  "Adebayo et al. 2018: attribution from a randomly re-initialised model must NOT match the trained model's")
panel(s, Inches(0.45), Inches(1.2), Inches(5.75), Inches(3.45))
panel_title(s, Inches(0.65), Inches(1.32), Inches(5.3), "Spearman ρ — trained vs random-init lead ranking")
simple_table(s, Inches(0.65), Inches(1.72), Inches(5.35), [
    ["class", "ρ", "verdict"],
    ["NORM", "0.538", "CHECK (high)"],
    ["MI", "0.448", "ok (low)"],
    ["STTC", "0.350", "ok (low)"],
    ["CD", "−0.021", "ok (low)"],
    ["HYP", "−0.133", "ok (low)"],
], col_ratio=[0.3, 0.28, 0.42], size=10, row_h=Inches(0.34))
caption(s, Inches(0.65), Inches(4.08), Inches(5.4),
        "Low ρ → explanation depends on learned weights. NORM flagged: its lead ranking is partly recoverable from an untrained net.")
panel(s, Inches(6.45), Inches(1.2), Inches(6.45), Inches(3.45))
panel_title(s, Inches(6.65), Inches(1.32), Inches(6.0), "What NB06 establishes for the 'explainable' claim")
bullets(s, Inches(6.65), Inches(1.72), Inches(6.05), Inches(2.85), [
    ("Which leads drive each class — Module 3 heatmap + top-3 vs Atwa §4.6.", 1, INK, False),
    ("Ranking robust to the IG baseline — Module 3b, ρ 0.98–1.00.", 1, INK, False),
    ("Relevance lands on real ECG morphology — Module 4 overlays + Grad-CAM.", 1, INK, False),
    ("Attribution is faithful, not just plausible — Module 5 ablation > random (MI/STTC/CD).", 1, INK, False),
    ("Explanation is model-dependent — Module 6, low ρ for 4 / 5 classes.", 1, INK, False),
], size=10, gap=5)
panel(s, Inches(0.45), Inches(4.85), Inches(12.45), Inches(2.3))
panel_title(s, Inches(0.65), Inches(4.95), Inches(11.8), "Honest limitations")
bullets(s, Inches(0.65), Inches(5.3), Inches(12.0), Inches(1.7), [
    ("IG ≠ SHAP (different axioms / baseline); no LRP-ε (no zennit / captum); single fold-10 TEST set, subsampled I_l, no bootstrap CI.", 1, MUTE, False),
    ("Explains the single InceptionTime1D — not the full ensemble or the DualBranch demographic pathway.", 1, MUTE, False),
], size=10, gap=6)

# ============================================================ 20 — divider LIT
N(); divider_slide(prs, "Positioning", "Comparison with the Literature (6 papers)",
              "Three original reference papers + three added 2026 papers, cross-checked against this project's "
              "notebooks and results.")

# ============================================================ 21 — LIT part 1
s = content_slide(prs, N(), TOTAL, "Literature comparison (1/2) — task framing, rigor, discrimination",
                  "All on PTB-XL v1.0.3. Only multi-label super-diagnostic macro-AUC rows are directly comparable.  *single-label = easier framing")
simple_table(s, Inches(0.4), Inches(1.25), Inches(12.55), [
    ["Study", "Framing", "Split", "Headline macro-AUC (5-super)", "CIs", "Calibration / uncert. / XAI"],
    ["Strodthoff 2020 [1]", "multi-label", "patient 1–8/9/10", "xresnet 0.920 · ensemble 0.928", "1000× bootstrap", "uncert. (corr. only); LRP demo"],
    ["Atwa 2025 [2]", "single-label*", "patient folds", "no AUC (accuracy 0.795)", "none", "SHAP leads; no faithfulness"],
    ["Gitau 2025 [3]", "multi-label", "patient 1–8/9/10", "0.918 repro · 0.922 tuned · 0.934 ens", "3× bootstrap", "noise-robustness only"],
    ["Jafari 2026 [4]", "multi-label", "patient 1–8/9/10", "0.920–0.923 (on 23 classes)", "5-fold CV", "none"],
    ["Naqcho 2026 [5]", "multi-label", "stratified", "0.896 (CNN-VAE, 197k params)", "none", "latent space only"],
    ["Garg 2026 [6]", "single-label*", "13k / 1.6k / 1.6k", "0.90 ROC-AUC (ECG-Lens CNN)", "none", "confusion matrix only"],
    ["This work (NB01–06)", "multi-label", "patient 4-way", "0.923 single · 0.924 ensemble", "none (gap)", "ECE+temp+conformal; IG/GradCAM +faithfulness"],
], col_ratio=[0.155, 0.10, 0.135, 0.27, 0.11, 0.23], size=8, row_h=Inches(0.62))
bullets(s, Inches(0.4), Inches(6.55), Inches(12.55), Inches(0.8), [
    ("Validated by the new papers: 'Transformer/RNN depth doesn't help' (Jafari) · small model is enough & HYP is hardest (Naqcho) · "
     "DL ≫ classic ML on raw signal (Garg) · lead-wise z-score on train stats (Naqcho).", 1, MUTE, False),
], size=8.5, gap=0)

# ============================================================ 22 — LIT part 2 + refs
s = content_slide(prs, N(), TOTAL, "Literature comparison (2/2) — where we lead, where we lag, references",
                  "")
panel(s, Inches(0.4), Inches(1.2), Inches(6.15), Inches(2.9))
panel_title(s, Inches(0.6), Inches(1.3), Inches(5.7), "Ahead of all six papers")
bullets(s, Inches(0.6), Inches(1.68), Inches(5.75), Inches(2.3), [
    ("Measured calibration (ECE/MCE/Brier) + per-class temperature scaling.", 1, INK, False),
    ("Distribution-free conformal per-class sensitivity guarantee.", 1, INK, False),
    ("Uncertainty used for abstention (risk–coverage), not just correlation.", 1, INK, False),
    ("Faithfulness + model-randomisation tests of the explanation.", 1, INK, False),
    ("Age-stratified ECE, not only age-stratified accuracy.", 1, INK, False),
], size=9.5, gap=4)
panel(s, Inches(6.75), Inches(1.2), Inches(6.15), Inches(2.9))
panel_title(s, Inches(6.95), Inches(1.3), Inches(5.7), "Behind the literature (to close)")
bullets(s, Inches(6.95), Inches(1.68), Inches(5.75), Inches(2.3), [
    ("No bootstrap 95% CIs — 4 of 6 papers report them.", 1, ACCENT, True),
    ("No Fmax / macro-AUPRC (Jafari's AUPRC 0.47 shows how hard rare classes are).", 1, INK, False),
    ("macro-AUC ~0.005 below SOTA; Gitau gives an unused known-good InceptionTime config.", 1, INK, False),
    ("HYP augmentation never ablated (Atwa's own result: it can hurt).", 1, INK, False),
    ("80+ subgroup surfaced but not improved.", 1, INK, False),
], size=9.5, gap=4)
panel(s, Inches(0.4), Inches(4.25), Inches(12.5), Inches(2.95))
panel_title(s, Inches(0.6), Inches(4.35), Inches(11.8), "References")
bullets(s, Inches(0.6), Inches(4.72), Inches(12.0), Inches(2.4), [
    ("[1] Strodthoff N, Wagner P, Schaeffter T, Samek W. Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. arXiv:2004.13701 (2020); IEEE JBHI 25(5), 2021.", 1, INK, False),
    ("[2] Atwa AEM et al. Interpretable Deep Learning Models for Arrhythmia Classification … PTB-XL. Diagnostics 2025;15(15):1950.", 1, INK, False),
    ("[3] Gitau AM, Adeyemi A, Tavashi B, Singstad B-J. [Re] Deep Learning for ECG Analysis … PTB-XL. medRxiv 2025.01.27.25321112 (2025).", 1, INK, False),
    ("[4] Jafari A, Jafari F. How Much Temporal Modeling is Enough? Hybrid CNN–RNN Architectures for Multi-Label ECG Classification. arXiv:2601.18830 (2026).", 1, INK, False),
    ("[5] Naqcho Ali Mehdi, Aamir Ali Drigh. ECG Classification on PTB-XL: A Data-Centric Approach with Simplified CNN-VAE. arXiv:2603.07558 (2026).", 1, INK, False),
    ("[6] Garg S, Jadia U, Sagtani A, Hiran KK. ECG-Lens: Benchmarking ML & DL Models on PTB-XL Dataset. arXiv:2604.15822 (2026).", 1, INK, False),
    ("Methods: Guo 2017 (temp. scaling) · Angelopoulos & Bates 2021 (conformal) · Sundararajan 2017 (IG) · "
     "Selvaraju 2017 (Grad-CAM) · Petsiuk 2018 (deletion metric) · Adebayo 2018 (sanity checks) · Wagner 2020 (PTB-XL).", 1, MUTE, False),
], size=8.3, gap=3)

# ============================================================ 23 — NEXT STEPS pipeline
s = content_slide(prs, N(), TOTAL, "Next steps — prioritized pipeline",
                  "Three tiers; Tier 1 is what makes the current results paper-defensible")
flow_boxes(s, Inches(0.5), Inches(1.4), Inches(12.4), Inches(1.35), [
    ("TIER 1\nStatistical rigor", "bootstrap 95% CIs\n+ Fmax + macro-AUPRC\non every headline number"),
    ("TIER 2\nClose the AUC gap", "port Gitau config\n(depth 12, k=(40,20,10),\nweighted-BCE, ≥25 ep)"),
    ("TIER 3\nFairness + XAI depth", "fix 80+ subgroup;\nablate HYP augmentation;\nadd LRP-ε, bootstrap I_l"),
], fill=ACCENT2, size=8.5, head_size=11)
panel(s, Inches(0.5), Inches(3.0), Inches(12.4), Inches(4.15))
simple_table(s, Inches(0.7), Inches(3.2), Inches(12.0), [
    ["#", "Action", "NB", "Why / source", "Eff."],
    ["1.1", "1000× test-set bootstrap → 95% CIs for macro-AUC, ECE, error-detection AUC, ablation drops", "NB03/05/06", "Strodthoff, Gitau, Jafari all do this", "S"],
    ["1.2", "Report Fmax and macro-AUPRC alongside macro-AUC", "NB03", "Strodthoff (Fmax), Jafari (AUPRC 0.47)", "S"],
    ["2.1", "Retrain InceptionTime1D with Gitau's super-diag config; re-run NB05/NB06 on the new checkpoint", "NB03", "Gitau 2025 hands the exact recipe", "M"],
    ["2.2", "Ablation: HYP Gaussian-noise aug ON vs OFF vs SWT / lead-dropout", "NB03", "Atwa's own result: plain noise can hurt", "M"],
    ["3.1", "80+ subgroup: per-band temperature, per-band threshold, or targeted resampling", "NB05", "ECE 0.133 & 34% exact-match unresolved", "M"],
    ["3.2", "Add LRP-ε (zennit) as a second attribution; bootstrap the per-lead I_l ranking; investigate NORM ρ = 0.54", "NB06", "Strodthoff Fig 8; Adebayo flag", "M"],
    ["3.3", "Joint (not per-class marginal) multi-label conformal prediction sets", "NB05", "current guarantee is marginal only", "L"],
], col_ratio=[0.05, 0.47, 0.08, 0.31, 0.06], size=8.2, row_h=Inches(0.46))

# ============================================================ 24 — roadmap / close
s = content_slide(prs, N(), TOTAL, "Roadmap & definition of done",
                  "What 'finished' looks like for each clause of the project title")
simple_table(s, Inches(0.5), Inches(1.35), Inches(12.35), [
    ["Title clause", "Status now", "Done when"],
    ["Age-stratified", "Audited in every notebook; 80+ gap surfaced (AUC ↓, ECE 0.133)", "80+ ECE < 0.08 after a per-band remedy, with CIs"],
    ["Explainable", "IG + Grad-CAM + faithfulness + randomisation done; NORM ρ = 0.54 flagged", "LRP-ε cross-check + bootstrapped I_l ranking; NORM caveat explained"],
    ["Uncertainty-calibrated", "Temp-scaling + conformal + abstention done on 1 fold", "Bootstrap CIs on ECE / risk–coverage; joint conformal sets"],
    ["Discrimination", "macro-AUC 0.923 / 0.924 — ~0.005 below SOTA", "≥ 0.928 with Gitau config, reported with Fmax + AUPRC + CIs"],
], col_ratio=[0.20, 0.44, 0.36], size=9.5, row_h=Inches(0.82))
panel(s, Inches(0.5), Inches(5.55), Inches(12.35), Inches(1.6))
bullets(s, Inches(0.75), Inches(5.7), Inches(11.9), Inches(1.3), [
    ("Bottom line: the pipeline is methodologically sound and, on calibration + explainability, ahead of all six reference papers. "
     "The remaining work is statistical rigor (Tier 1) and ~0.005 of AUC (Tier 2) — both low-risk and well-specified.", 1, ACCENT, True),
], size=10, gap=0)

OUT.parent.mkdir(exist_ok=True)
prs.save(str(OUT))
print("saved", OUT, "with", len(prs.slides._sldIdLst), "slides")
