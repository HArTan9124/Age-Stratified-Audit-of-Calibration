"""NB05 - Uncertainty Calibration & Predictive Uncertainty : 5-slide result deck."""
from pathlib import Path
from pptx.util import Inches, Pt
from pptbuild import (new_deck, title_slide, content_slide, bullets, panel,
                      panel_title, image_fit, caption, simple_table,
                      INK, ACCENT, ACCENT2, MUTE)

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "outputs" / "dataset_validation"
OUT = ROOT / "presentations" / "NB05_Calibration_Uncertainty_Deck.pptx"
N = 5
prs = new_deck()

# ============================================================ Slide 1 - title
title_slide(
    prs,
    "PTB-XL  ·  Notebook 05",
    "Uncertainty Calibration & Predictive Uncertainty",
    "Auditing whether the NB03 winning ensemble's probabilities can be trusted "
    "- and whether it knows when it is wrong.",
    [
        "Motivation: NB01-04 reach macro-AUC ~ 0.92 but never measure calibration or uncertainty "
        "- the “uncertainty-calibrated” half of the project title had no result.",
        "Model under audit: AUC-weighted soft-voting ensemble of 3 members "
        "(DualBranchECGNet, InceptionTime1D, ResNet1D101) - TEST macro-AUC 0.9245.",
        "Fit split: CONFORMAL_CALIBRATION (1103 records, patient-disjoint, untouched by NB03) "
        "- never VALIDATION, which NB03 already used for early-stopping.",
        "Pipeline: 15-bin reliability / ECE / MCE / Brier  ->  per-class temperature scaling  ->  "
        "split-conformal thresholds  ->  ensemble-variance uncertainty  ->  risk-coverage  ->  age bands.",
        "Anchors: Guo et al. 2017 (temperature scaling); Angelopoulos & Bates 2021 (conformal); "
        "Strodthoff et al. 2020 §IV-C (ensemble-variance uncertainty).",
    ],
)

# ============================================================ Slide 2 - uncalibrated reliability
s = content_slide(prs, 2, N, "Module 6 - Reliability & calibration error (uncalibrated)",
                  "15 equal-width bins per superclass;  ECE = Σ (n_b/N)·|acc_b - conf_b|,  MCE = max bin gap")
image_fit(s, str(IMG / "05cal_reliability_uncalibrated.png"),
          Inches(0.45), Inches(1.2), Inches(12.45), Inches(2.45), valign="top")
caption(s, Inches(0.45), Inches(3.6), Inches(12.45),
        "Observed positive-rate vs predicted probability, 5 superclasses. Bars below the diagonal = the ensemble is over-confident.")
panel(s, Inches(0.45), Inches(4.05), Inches(7.35), Inches(3.05))
panel_title(s, Inches(0.65), Inches(4.18), Inches(6.9), "Per-class TEST calibration (uncalibrated)")
simple_table(s, Inches(0.65), Inches(4.62), Inches(6.95), [
    ["", "NORM", "MI", "STTC", "CD", "HYP", "MACRO"],
    ["ECE",   "0.082", "0.076", "0.063", "0.090", "0.097", "0.082"],
    ["MCE",   "0.270", "0.272", "0.260", "0.346", "0.376", "0.305"],
    ["Brier", "0.100", "0.104", "0.090", "0.096", "0.083", "0.094"],
], col_ratio=[0.16] + [0.14]*6, size=10.5, row_h=Inches(0.4))
bullets(s, Inches(8.05), Inches(4.15), Inches(4.85), Inches(3.0), [
    ("What it establishes", 0, ACCENT2, True),
    ("Probabilities are NOT calibrated as-is: macro-ECE 0.082; worst bins off by ~30-38 points.", 1, INK, False),
    ("Miscalibration is worst on CD and HYP (the rarest classes).", 1, INK, False),
    ("Bias is consistent over-confidence -> one softening temperature per class should help.", 1, ACCENT, True),
], size=12, gap=7)

# ============================================================ Slide 3 - temperature scaling
s = content_slide(prs, 3, N, "Module 7 - Per-class temperature scaling on CALIBRATION",
                  "Fit one scalar T per class by NLL on CALIBRATION, freeze it, apply to TEST")
image_fit(s, str(IMG / "05cal_reliability_calibrated.png"),
          Inches(0.45), Inches(1.2), Inches(12.45), Inches(2.45), valign="top")
caption(s, Inches(0.45), Inches(3.6), Inches(12.45),
        "Reliability after per-class temperature scaling (TEST, N = 2198). Curves move toward the diagonal, most visibly for CD and HYP.")
panel(s, Inches(0.45), Inches(4.05), Inches(7.35), Inches(3.05))
panel_title(s, Inches(0.65), Inches(4.18), Inches(6.9), "Fitted T and macro effect (TEST)")
simple_table(s, Inches(0.65), Inches(4.62), Inches(6.95), [
    ["", "NORM", "MI", "STTC", "CD", "HYP"],
    ["fitted T", "1.29", "0.94", "1.10", "0.85", "0.79"],
], col_ratio=[0.18] + [0.164]*5, size=10.5, row_h=Inches(0.4))
simple_table(s, Inches(0.65), Inches(5.55), Inches(6.95), [
    ["macro ECE", "0.082 -> 0.075", "macro MCE", "0.305 -> 0.298"],
    ["macro Brier", "0.0945 -> 0.0939", "macro-AUC", "0.9245 (unchanged)"],
], col_ratio=[0.25, 0.28, 0.22, 0.25], size=9.5, header=False, row_h=Inches(0.4))
bullets(s, Inches(8.05), Inches(4.15), Inches(4.85), Inches(3.0), [
    ("Reading", 0, ACCENT2, True),
    ("T > 1 (NORM, STTC) softens over-confidence; T < 1 (CD, HYP) sharpens under-confident heads.", 1, INK, False),
    ("Macro-ECE 0.082 -> 0.075 (-15%); σ(z/T) is strictly monotone, so macro-AUC is provably unchanged (asserted in code).", 1, INK, False),
    ("Limit: one scalar per class can't fix bin-shape miscalibration - MCE barely moves.", 1, ACCENT, True),
], size=12, gap=7)

# ============================================================ Slide 4 - conformal + uncertainty
s = content_slide(prs, 4, N, "Modules 8 & 9 - Conformal thresholds + predictive uncertainty",
                  "Per-class conformal sensitivity guarantee (α = 0.10)  |  disagreement & entropy as error detectors")
panel(s, Inches(0.45), Inches(1.2), Inches(6.05), Inches(3.15))
panel_title(s, Inches(0.65), Inches(1.32), Inches(5.6), "Module 8 - split-conformal @ 90% target sensitivity")
simple_table(s, Inches(0.65), Inches(1.72), Inches(5.65), [
    ["class", "thr p≥", "realised sens.", "alert rate"],
    ["NORM", "0.61", "0.913", "48.5%"],
    ["MI",   "0.25", "0.907", "42.3%"],
    ["STTC", "0.37", "0.889", "33.9%"],
    ["CD",   "0.30", "0.853", "34.7%"],
    ["HYP",  "0.13", "0.889", "36.6%"],
], col_ratio=[0.22, 0.20, 0.34, 0.24], size=10, row_h=Inches(0.29))
caption(s, Inches(0.65), Inches(3.7), Inches(5.7),
        "Mean predicted-set size ~ 1.9 / 5;  guarantee needs only CAL-TEST exchangeability (holds: same prep, disjoint folds).")
image_fit(s, str(IMG / "05cal_uncertainty_vs_error.png"),
          Inches(6.75), Inches(1.3), Inches(6.15), Inches(2.7), valign="top")
caption(s, Inches(6.75), Inches(4.02), Inches(6.15),
        "Uncertainty on correctly vs incorrectly classified TEST records (5-way exact match).")
panel(s, Inches(0.45), Inches(4.5), Inches(12.45), Inches(2.55))
panel_title(s, Inches(0.65), Inches(4.62), Inches(11.8), "Module 9 - does uncertainty flag the model's own errors?")
bullets(s, Inches(0.65), Inches(5.0), Inches(12.0), Inches(2.0), [
    ("Exact-match correct on TEST = 55.6%  (1221 / 2198) at the calibrated 0.5 operating point.", 1, INK, False),
    ("Disagreement (mean class-wise std of the 3 members):  0.046 correct  vs  0.076 wrong   ->   error-detection ROC-AUC 0.737.", 1, INK, False),
    ("Predictive entropy of the calibrated ensemble prob.:  0.237 correct  vs  0.379 wrong   ->   error-detection ROC-AUC 0.775.", 1, INK, False),
    ("Both scores are clearly higher on wrong predictions - usable as an abstention signal (next slide).", 1, ACCENT, True),
], size=11, gap=5)

# ============================================================ Slide 5 - selective + age + verdict
s = content_slide(prs, 5, N, "Modules 10-12 - Selective prediction, age-band calibration, verdict",
                  "Abstain on the least-confident cases  |  is 80+ also the worst-calibrated group?")
image_fit(s, str(IMG / "05cal_risk_coverage.png"),
          Inches(0.6), Inches(1.2), Inches(5.55), Inches(3.05), valign="top")
caption(s, Inches(0.45), Inches(4.3), Inches(6.15),
        "Retain most-confident c%; performance on the kept set vs coverage.")
image_fit(s, str(IMG / "05cal_age_stratified_ece.png"),
          Inches(7.05), Inches(1.2), Inches(5.75), Inches(3.05), valign="top")
caption(s, Inches(6.95), Inches(4.3), Inches(6.0),
        "Macro-ECE per age band, before vs after temperature scaling.")
panel(s, Inches(0.45), Inches(4.7), Inches(6.15), Inches(1.95))
panel_title(s, Inches(0.65), Inches(4.8), Inches(5.7), "Risk-coverage (defer by disagreement)")
simple_table(s, Inches(0.65), Inches(5.15), Inches(5.75), [
    ["coverage", "100%", "90%", "80%", "70%"],
    ["macro-AUC", "0.925", "0.931", "0.937", "0.942"],
    ["exact-match", "55.6%", "58.1%", "60.9%", "64.4%"],
], col_ratio=[0.30, 0.175, 0.175, 0.175, 0.175], size=9.5, row_h=Inches(0.36))
panel(s, Inches(6.95), Inches(4.7), Inches(5.85), Inches(1.95))
panel_title(s, Inches(7.15), Inches(4.8), Inches(5.5), "Age-band calibration (Module 11)")
bullets(s, Inches(7.15), Inches(5.12), Inches(5.45), Inches(1.5), [
    ("Macro-ECE climbs  <40: 0.055  ->  80+: 0.135; temp-scaling barely helps 80+ (0.135 -> 0.133).", 1, INK, False),
    ("Mean uncertainty also rises with age (0.038 -> 0.078) - the model does signal it struggles on 80+.", 1, INK, False),
], size=9.5, gap=5)
bullets(s, Inches(0.45), Inches(6.85), Inches(12.45), Inches(0.55), [
    ("Verdict: all three title clauses - age-stratified, explainable (NB06), uncertainty-calibrated - now have measured results.  "
     "Limits: single fold-10 TEST set, no bootstrap CI; heterogeneous (not N-reinit) uncertainty ensemble; per-class marginal conformal.",
     1, MUTE, False),
], size=9, gap=0)

OUT.parent.mkdir(exist_ok=True)
prs.save(str(OUT))
print("saved", OUT)
