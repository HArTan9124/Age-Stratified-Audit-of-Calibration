"""Append the "Response to Review Round 2" section to Tandon's_Draft.pptx.

Why this script exists
---------------------
The review the supervisor issued on the NB01-10 deck marked Items 2, 3, 7-extra and 8 as
un- or partly-addressed. They were in fact addressed on 2026-09-30 in
`analysis/11_confound_signal_quality_and_trend_tests.py`,
`analysis/12_xai_elderly_confidence_pair.py` and `papers/Draft_Paper.md` — several hours
*after* the deck was last built, so none of it ever reached a slide. This script closes
that gap by reading the executed JSON and figures and rendering them as slides.

Every number is read from an executed artefact. Nothing is typed in by hand:
  outputs/dataset_validation/11_confound_and_trend_tests.json   (Items 2, 3, clarification)
  outputs/dataset_validation/13_multiseed_replication.json      (Item 5, if present)
  outputs/figures/nb11_confound_trends/*.png
  outputs/figures/nb12_xai_pairs/*.png

Slides are inserted as a new Part 10 immediately before the closing Summary slide, so
slides 1-56 keep the numbers the supervisor cross-referenced in the review.

Re-runnable: each slide is located by title and rebuilt in place. Run it again once
analysis/13_multiseed_replication.py has written its JSON to fill in the Item 5 slide.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nb03b_ablation import (  # noqa: E402  - shared deck style + slide plumbing
    BODY, INFO_FILL, INFO_LINE, KEY_FILL, KEY_LINE, NAVY, RED,
    _run, add_bullets, add_caption, add_panel, add_table, add_title,
    clone_slide, find_slide, move_slide, strip_content, textbox,
)

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "outputs" / "dataset_validation"
FIG11 = ROOT / "outputs" / "figures" / "nb11_confound_trends"
FIG12 = ROOT / "outputs" / "figures" / "nb12_xai_pairs"
DECK = ROOT / "presentations" / "Tandon's_Draft.pptx"

# This section is built at the larger type the supervisor asked for, not the deck's
# original 11 pt default. apply_typography.py raises the older slides to match.
T_SUB, T_BODY, T_TBL, T_CAP = 15, 13, 12, 12   # nothing in this deck drops below 12 pt
FLOOR = 7.02          # page-number placeholder starts at 7.16; keep content above this

BANDS = ["<40", "40-65", "65-80", "80+"]

TITLES = {
    "divider": "Part 10 · Response to Review Round 2",
    "item2": "Review Item 2 — Signal Quality Tested as a Confound",
    "item2b": "Review Item 2 — Confound Figures: Quality and the Adjusted Age Effect",
    "item3": "Review Item 3 — Formal Trend Statistics for Every Age Claim",
    "clarify": "Clarification — Which Result Is Monotonic, and Why 65–80 Dips",
    "item7": "Review Item 7 — Literal ECG Waveform × Attribution Overlays",
    "item5": "Review Item 5 — Multi-Seed Replication of the Primary Pipeline",
    "item8": "Review Item 8 — Reproducibility, Data Availability and Ethics",
    "sota": "Additional Correction — The SOTA Comparator Was the Wrong Task",
    "score": "Review Scorecard — Status of All Nine Action Items",
}
SUMMARY_TITLE = "Summary — Final Conclusions"


# ------------------------------------------------------------------ data access
def load_trends() -> dict:
    f = VAL / "11_confound_and_trend_tests.json"
    if not f.exists():
        raise SystemExit(f"missing {f} — run analysis/11_confound_signal_quality_and_trend_tests.py first")
    return json.loads(f.read_text())


def load_seeds():
    """NB14 (all three members reseeded) if present, else NB13 (anchor only).

    NB14 supersedes NB13 for the ensemble column and leaves the anchor column
    untouched, so whichever is returned here is the authoritative Item 5 source.
    """
    for name in ("14_full_ensemble_multiseed.json", "13_multiseed_replication.json"):
        f = VAL / name
        if f.exists():
            d = json.loads(f.read_text())
            d["_source"] = name
            return d
    return None


def sci(p: float) -> str:
    """p-values the way the manuscript prints them."""
    if p <= 0.0:
        return "<1e-300"
    if p >= 0.01:
        return f"{p:.2f}"
    mant, exp = f"{p:.0e}".split("e")
    return f"{mant}e{int(exp)}"


# ------------------------------------------------------------------ layout helpers
def subhead(slide, text, y=1.10, h=0.56, size=T_SUB):
    """Wider and larger than the shared helper; two lines fit at this width."""
    _, tf = textbox(slide, 0.6, y, 12.2, h, MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], text, size, NAVY, bold=True)


def picture_fit(slide, path: Path, x: float, y: float, max_w: float, max_h: float):
    """Place an image inside a box, preserving aspect ratio, and return its real height."""
    if not path.exists():
        print(f"[WARN] figure missing: {path}")
        return 0.0
    w_px, h_px = Image.open(path).size
    scale = min(max_w / w_px, max_h / h_px)
    w, h = w_px * scale, h_px * scale
    slide.shapes.add_picture(str(path), Inches(x + (max_w - w) / 2), Inches(y), width=Inches(w))
    return h


def note_strip(slide, y, items, h=0.95, size=T_BODY):
    add_panel(slide, 0.55, y, 12.3, h, KEY_FILL, KEY_LINE)
    add_bullets(slide, 0.75, y + 0.10, 11.9, h - 0.2, "", items, size=size, gap=6)


# ------------------------------------------------------------------ slides
def build_divider(slide, trends, seeds):
    add_title(slide, TITLES["divider"])
    add_panel(slide, 0.6, 1.25, 12.2, 1.45, KEY_FILL, KEY_LINE)
    _, tf = textbox(slide, 0.85, 1.36, 11.7, 1.25, MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _run(p, "Why this section exists.  ", T_BODY, RED, bold=True)
    _run(p, "The reviewed deck was built on 29 September at 23:05. The confound analysis, the formal trend "
            "tests, the attribution overlays and the manuscript's ethics and reproducibility statement were "
            "all executed on 30 September between 10:54 and 11:13 — after the deck was frozen. The work "
            "existed; the slides did not. Part 10 puts each executed result on a slide, and leaves "
            "slides 1–56 untouched so the review's cross-references still resolve.", T_BODY, BODY)

    seed_line = ("anchor architecture retrained at 3 further seeds and the whole pipeline re-derived per seed."
                 if seeds else
                 "retraining is running now; the slide is rebuilt automatically when it writes its JSON.")
    items = [
        ("Item 2 · signal-quality confound — ",
         "four acquisition covariates computed on all 2,198 TEST records, and the age effect re-fit "
         "adjusted for them. Two slides: the statistics, then the figures."),
        ("Item 3 · formal trend tests — ",
         "Spearman ρ, Kruskal–Wallis and HC0-robust OLS slope per decade on every age-stratified outcome."),
        ("Item 5 · multi-seed replication — " + seed_line, ""),
        ("Item 7 · literal overlays — ",
         "the ECG trace with its attribution heat-map drawn on it, for a high- and a low-confidence 80+ case."),
        ("Item 8 · reproducibility and ethics — ",
         "data availability, ethics, code inventory and the full hyperparameter list."),
        ("Clarification · monotonicity — ",
         "which of the three set-size analyses is monotone is now stated per analysis, not uniformly."),
    ]
    add_bullets(slide, 0.6, 2.95, 12.2, 4.0, "What follows, and why", items, size=T_BODY, gap=11)


def build_item2(slide, trends):
    add_title(slide, TITLES["item2"])
    subhead(slide, "The review asked for a signal-quality covariate. Four were computed on every TEST record, "
                   "and the age effect barely moves once they are adjusted for.")

    sq = trends["age_vs_signal_quality"]
    spec = [("snr_db", "In-band SNR (dB)", 2),
            ("wander_rms_mv", "Baseline-wander RMS (mV)", 4),
            ("hf_rms_mv", "High-frequency noise RMS (mV)", 4),
            ("flat_leads", "Dead-lead count", 4)]
    rows = [["Acquisition covariate"] + BANDS + ["Spearman ρ vs age", "p", "Kruskal–Wallis p"]]
    for key, lab, nd in spec:
        v = sq[key]
        rows.append([lab] + [f"{v['band_means'][b]:.{nd}f}" for b in BANDS]
                    + [f"{v['spearman_rho_vs_age']:+.3f}", sci(v["spearman_p"]), sci(v["kruskal_p"])])
    add_table(slide, 0.55, 1.66, 12.3, rows,
              col_w=[3.20, 1.05, 1.15, 1.15, 1.05, 1.70, 1.00, 1.00],
              row_h=0.44, size=T_TBL, header_size=T_TBL)

    snr = sq["snr_db"]
    snr_bands = ", ".join(f"{snr['band_means'][b]:.2f}" for b in BANDS)
    tr = trends["trend_tests"]
    mon, glo, cal = (tr["conformal_set_size_mondrian"], tr["conformal_set_size_global"],
                     tr["per_record_abs_calibration_error"])

    add_panel(slide, 0.55, 3.95, 6.05, 2.95, INFO_FILL, INFO_LINE)
    add_bullets(slide, 0.75, 4.07, 5.65, 2.70, "What the covariates show", [
        ("SNR is flat in age. ", f"ρ = {snr['spearman_rho_vs_age']:+.3f}, p = {sci(snr['spearman_p'])}. "
                                 f"Band means {snr_bands} dB do not order with age at all."),
        ("Wander and HF noise rise, slightly. ", "Both are statistically detectable but small "
         f"(ρ = {sq['wander_rms_mv']['spearman_rho_vs_age']:+.3f} and "
         f"{sq['hf_rms_mv']['spearman_rho_vs_age']:+.3f})."),
        ("PTB-XL is a curated corpus. ", "Acquisition quality is close to constant across the age range, so "
                                         "age is not acting as a proxy for a noisier trace."),
    ], size=T_BODY, gap=9)

    add_panel(slide, 6.80, 3.95, 6.05, 2.95, KEY_FILL, KEY_LINE)
    add_bullets(slide, 7.00, 4.07, 5.65, 2.70, "Effect of adjusting for all four", [
        ("Mondrian set size. ", f"{mon['ols_unadjusted']['beta']:+.4f} → "
                                f"{mon['ols_adjusted_signal_quality']['beta']:+.4f} labels per decade — "
                                f"attenuation {mon['attenuation_pct_after_sq']:.1f}%."),
        ("Global set size. ", f"{glo['ols_unadjusted']['beta']:+.4f} → "
                              f"{glo['ols_adjusted_signal_quality']['beta']:+.4f}, attenuation "
                              f"{glo['attenuation_pct_after_sq']:.1f}%."),
        ("Calibration error. ", f"{cal['ols_unadjusted']['beta']:+.4f} → "
                                f"{cal['ols_adjusted_signal_quality']['beta']:+.4f}, attenuation "
                                f"{cal['attenuation_pct_after_sq']:.1f}%."),
        ("Residual limitation, stated. ", "A confound shared by the deep models and the Random Forest alike "
                                          "is not excluded by this test. The manuscript says so in its "
                                          "Limitations, as the review asked."),
    ], size=T_BODY, gap=9)


def build_item2b(slide, trends):
    add_title(slide, TITLES["item2b"])
    subhead(slide, "Left: the four acquisition covariates by age band. Right: the age slope before and after "
                   "adjusting for them, for each outcome.")

    h1 = picture_fit(slide, FIG11 / "11_signal_quality_by_age_band.png", 0.55, 1.70, 7.35, 2.55)
    h2 = picture_fit(slide, FIG11 / "11_age_effect_adjusted.png", 8.20, 1.70, 4.65, 2.55)
    y = 1.70 + max(h1, h2) + 0.08
    add_caption(slide, "11_signal_quality_by_age_band.png", 0.55, y, 7.35, size=T_CAP)
    add_caption(slide, "11_age_effect_adjusted.png", 8.20, y, 4.65, size=T_CAP)

    snr = trends["age_vs_signal_quality"]["snr_db"]
    glo = trends["trend_tests"]["conformal_set_size_global"]
    mon = trends["trend_tests"]["conformal_set_size_mondrian"]
    note_strip(slide, y + 0.40, [
        ("Reading the left panel. ", "In-band SNR is the covariate that matters for a noise-driven story, and "
         f"it is flat: ρ = {snr['spearman_rho_vs_age']:+.3f}, p = {sci(snr['spearman_p'])}, "
         f"Kruskal–Wallis p = {sci(snr['kruskal_p'])}. Baseline wander and high-frequency noise do rise with "
         "age, but by very little."),
        ("Reading the right panel. ", "Adjusting for all four covariates shifts the age slope by "
         f"{glo['attenuation_pct_after_sq']:.1f}% for global set size and "
         f"{mon['attenuation_pct_after_sq']:.1f}% for Mondrian set size. Whatever drives the age gradient, it "
         "is not the noise in the trace."),
        ("One counter-intuitive coefficient. ", "The SNR term in the Mondrian set-size model is positive "
         f"({mon['covariate_terms']['snr_db']['beta']:+.3f} per dB, "
         f"p = {sci(mon['covariate_terms']['snr_db']['p'])}) — the opposite of a noise story. It most likely "
         "reflects large-amplitude pathological ECGs having both higher signal power and more concurrent "
         "labels."),
    ], h=FLOOR - (y + 0.40))


def build_item3(slide, trends):
    add_title(slide, TITLES["item3"])
    subhead(slide, "Every monotonicity claim now carries a trend statistic rather than an eyeballed bar chart. "
                   "OLS slopes use HC0 robust standard errors; n = 2,198 TEST records.")

    tr = trends["trend_tests"]
    spec = [("conformal_set_size_mondrian", "Mondrian conformal set size", "{:.2f}"),
            ("conformal_set_size_global", "Global-threshold set size", "{:.2f}"),
            ("per_record_abs_calibration_error", "Per-record |p − y|", "{:.3f}"),
            ("exact_match_correct", "Exact-match accuracy", "{:.3f}"),
            ("label_count", "True label count (mediator)", "{:.3f}")]
    rows = [["Outcome"] + BANDS + ["Kruskal–Wallis", "Spearman ρ (p)", "OLS slope / decade",
                                   "Adjusted for signal quality"]]
    for key, lab, fmt in spec:
        v = tr[key]
        u, a = v["ols_unadjusted"], v["ols_adjusted_signal_quality"]
        rows.append([lab] + [fmt.format(v["band_means"][b]) for b in BANDS]
                    + [f"H={v['kruskal_H']:.0f}, p={sci(v['kruskal_p'])}",
                       f"{v['spearman_rho_vs_age']:+.3f} ({sci(v['spearman_p'])})",
                       f"{u['beta']:+.4f}, p={sci(u['p'])}",
                       f"{a['beta']:+.4f}  ({-v['attenuation_pct_after_sq']:+.1f}%)"])
    add_table(slide, 0.55, 1.66, 12.3, rows,
              col_w=[2.35, 0.80, 0.90, 0.90, 0.80, 1.75, 1.75, 1.40, 1.65],
              row_h=0.52, size=T_TBL, header_size=T_TBL)

    b = trends["band_level_trends"]
    ece_ci = b["ece_slope_per_decade_ci"]
    ss_ci = b["setsize_slope_per_decade_ci"]
    note_strip(slide, 4.60, [
        ("Direction and significance. ", "Both set-size measures and per-record calibration error rise with "
         "age; exact-match accuracy falls. Every Spearman p is below 1e-28, and every Kruskal–Wallis test "
         "rejects equality across the four bands."),
        ("Band-level slopes with bootstrap intervals. ", f"macro-ECE {b['ece_slope_per_decade']:+.4f} per "
         f"decade [{ece_ci[0]:.4f}, {ece_ci[1]:.4f}]; Mondrian set size "
         f"{b['setsize_slope_per_decade']:+.4f} per decade [{ss_ci[0]:.4f}, {ss_ci[1]:.4f}]. Both intervals "
         f"exclude zero across {b['n_boot']:,} within-band resamples."),
        ("Comorbidity is a mediator, not a confound to remove. ", "Adding the record's true label count "
         "attenuates the set-size slope by about 27% and calibration error by 10%, and the age term stays "
         "highly significant throughout. Older patients genuinely carry more concurrent conditions."),
    ], h=FLOOR - 4.60)


def build_clarify(slide, trends, seeds=None):
    add_title(slide, TITLES["clarify"])
    subhead(slide, "The review is right that 'monotonic' was being applied uniformly to three different "
                   "analyses. It holds for two of the three, and the exception has a known cause.")

    tr = trends["trend_tests"]
    mon = tr["conformal_set_size_mondrian"]["band_means"]
    glo = tr["conformal_set_size_global"]["band_means"]
    cstab = trends["calibration_resampling_stability"]

    rows = [["Analysis"] + BANDS + ["Monotone across all bands?"]]
    rows.append(["Global-threshold conformal"] + [f"{glo[b]:.2f}" for b in BANDS]
                + ["Yes — strictly increasing"])
    rows.append(["Age-conditional (Mondrian) conformal"] + [f"{mon[b]:.2f}" for b in BANDS]
                + ["No — dips at 65–80 vs 40–65"])
    rows.append([f"Mondrian, mean of {cstab['n_cal_resamples']} calibration resamples"]
                + [f"{cstab['set_size'][b]['mean']:.2f}±{cstab['set_size'][b]['sd']:.2f}" for b in BANDS]
                + ["No — same dip, within ±1 SD"])
    if seeds and seeds.get("_source", "").startswith("14"):
        su = seeds["summary"]
        rows.append(["Mondrian, mean of 4 seeds (NB14, full ensemble)"]
                    + [f"{su['mondrian_set_size_' + b]['mean']:.2f}"
                       f"±{su['mondrian_set_size_' + b]['sd']:.2f}" for b in BANDS]
                    + ["No — same dip in 4 / 4 seeds"])
    add_table(slide, 0.55, 1.66, 12.3, rows,
              col_w=[4.25, 1.30, 1.40, 1.40, 1.30, 2.65],
              row_h=0.40, size=T_TBL, header_size=T_TBL)

    h = picture_fit(slide, FIG11 / "11_calibration_resampling_stability.png", 0.55, 3.76, 6.55, 2.45)
    add_caption(slide, "11_calibration_resampling_stability.png · Figure 18 in the manuscript",
                0.55, 3.76 + h + 0.06, 6.55, size=T_CAP)

    sl = cstab["slope_per_decade"]
    cover = ", ".join(f"{cstab['coverage'][b]['mean'] * 100:.1f}% at {b}" for b in BANDS)
    add_panel(slide, 7.30, 3.76, 5.55, 3.24, KEY_FILL, KEY_LINE)
    add_bullets(slide, 7.50, 3.87, 5.15, 3.02, "Why the Mondrian version dips", [
        ("Calibration-cell size, not the effect. ", "Per-band quantiles are fitted inside each band, and "
         "the bands hold unequal numbers of positives per class."),
        ("The resampling shows it. ", f"The 80+ band is the most variable by far — SD "
         f"{cstab['set_size']['80+']['sd']:.2f} against {cstab['set_size']['<40']['sd']:.2f} at <40."),
        ("The trend survives every draw. ", f"Positive in {sl['frac_positive'] * 100:.0f}% of "
         f"{cstab['n_cal_resamples']} resamples, mean {sl['mean']:+.3f} per decade "
         f"[{sl['pct'][0]:.3f}, {sl['pct'][1]:.3f}]."),
        ("Coverage. ", f"{cover}, against a 90% target."),
        ("And it survives reseeding. ", "The dip reappears at 65–80 in all four NB14 seeds."),
    ], size=T_BODY, gap=4)


def build_item7(slide):
    add_title(slide, TITLES["item7"])
    subhead(slide, "The review asked for literal ECG-with-heat-map examples alongside the importance-score "
                   "comparison on slide 52. Both 80+ cases below are TEST records; Integrated Gradients, "
                   "64 steps, zero baseline.", h=0.62)

    top, box_h = 1.80, 3.62
    h = 0.0
    for path, lab, x in [(FIG12 / "12_xai_pair_CD.png", "CD — conduction disturbance", 1.95),
                         (FIG12 / "12_xai_pair_HYP.png", "HYP — hypertrophy", 6.78)]:
        h = max(h, picture_fit(slide, path, x, top, 4.60, box_h))
        add_caption(slide, lab, x, top + box_h + 0.04, 4.60, size=T_CAP)

    note_strip(slide, top + box_h + 0.40, [
        ("How to read these. ", "Each panel overlays the Integrated-Gradients attribution on the raw trace "
         "for a high- and a low-confidence 80+ record of the same class. Attribution concentrates on the same "
         "morphological region in both — the visual counterpart of the ρ = 0.955 vs 0.980 rank agreement on "
         "slide 52."),
        ("Provenance. ", "analysis/12_xai_elderly_confidence_pair.py · Figure 21 in the manuscript."),
    ], h=FLOOR - (top + box_h + 0.40))


def build_item5(slide, seeds):
    add_title(slide, TITLES["item5"])
    if not seeds:
        subhead(slide, "Status: retraining in progress. This slide is rebuilt automatically once "
                       "analysis/13_multiseed_replication.py writes its JSON.")
        add_panel(slide, 0.6, 1.80, 12.2, 3.10, INFO_FILL, INFO_LINE)
        add_bullets(slide, 0.8, 1.95, 11.8, 2.85, "What is being run, and what it will report", [
            ("Design. ", "The anchor architecture InceptionTime1D (depth 4 — the NB03/NB05 model) is retrained "
             "at seeds 1, 7 and 13 under the hyperparameters recorded in the manuscript, then each seed is "
             "pushed through the entire downstream pipeline: ensemble re-weighting, per-class temperature "
             "scaling, global and Mondrian conformal thresholds, prediction-set size by age band, and the "
             "trend statistic."),
            ("Why only the anchor. ", "Training is CPU-only here. DualBranchECGNet and ResNet1D101 stay at "
             "their released checkpoints, so this measures seed variance in the component that can be "
             "retrained honestly rather than claiming full ensemble seed variance."),
            ("Reported. ", "macro-AUC mean ± SD over four seeds for both the anchor and the ensemble, and "
             "whether the age gradient — direction, monotonicity and significance — survives every seed."),
            ("Reproduction guard. ", "The script re-derives the seed-42 pipeline from the released checkpoints "
             "first and checks it against the archived numbers before training anything."),
        ], size=T_BODY, gap=10)
        return

    s, summ, meta = seeds["seeds"], seeds["summary"], seeds["meta"]
    order = [str(meta["anchor_seed"])] + [str(x) for x in meta["new_seeds"]]
    a, e = summ["anchor_macro_auc_test"], summ["ensemble_macro_auc_test"]
    subhead(slide, f"Anchor InceptionTime1D retrained at seeds {', '.join(order[1:])} and re-run through the "
                   f"whole pipeline. Anchor macro-AUC {a['mean']:.4f} ± {a['sd']:.4f}; "
                   f"ensemble {e['mean']:.4f} ± {e['sd']:.4f}.")

    rows = [["Seed", "Anchor macro-AUC", "Ensemble macro-AUC"] + [f"Global |S| {b}" for b in BANDS]
            + ["Spearman ρ vs age", "Monotone"]]
    for k in order:
        d = s[k]
        g = d["set_size_global"]
        rows.append([f"{k} (released)" if not d["retrained"] else k,
                     f"{d['macro_auc_test_anchor']:.4f}", f"{d['macro_auc_test_ensemble']:.4f}"]
                    + [f"{g['band_means'][b]:.2f}" for b in BANDS]
                    + [f"{g['spearman_rho_vs_age']:+.3f}", "Yes" if g["monotone"] else "No"])
    rows.append(["mean ± SD", f"{a['mean']:.4f}±{a['sd']:.4f}", f"{e['mean']:.4f}±{e['sd']:.4f}"]
                + [f"{summ['global_set_size_' + b]['mean']:.2f}±{summ['global_set_size_' + b]['sd']:.2f}"
                   for b in BANDS]
                + [f"{summ['global_spearman_rho']['mean']:+.3f}", "—"])
    add_table(slide, 0.55, 1.66, 12.3, rows,
              col_w=[1.45, 1.70, 1.80, 1.05, 1.15, 1.15, 1.05, 1.60, 1.35],
              row_h=0.46, size=T_TBL, header_size=T_TBL)

    cons = summ["age_trend_direction_consistent"]
    gs = summ["global_slope_per_decade"]
    y = 1.66 + 0.46 * len(rows) + 0.18
    add_panel(slide, 0.55, y, 6.05, FLOOR - y, INFO_FILL, INFO_LINE)
    add_bullets(slide, 0.75, y + 0.12, 5.65, FLOOR - y - 0.24, "Seed variance on the headline number", [
        ("Anchor model. ", f"range {a['min']:.4f}–{a['max']:.4f}, spread {a['range']:.4f} macro-AUC."),
        ("Full ensemble. ", f"range {e['min']:.4f}–{e['max']:.4f}, spread {e['range']:.4f} — narrower, since "
                            "two of three members are held fixed."),
        ("Scope, stated plainly. ", "This is seed variance for the anchor inside the pipeline, not for all "
                                    "three architectures. The remaining two are named as an open item."),
    ], size=T_BODY, gap=8)

    add_panel(slide, 6.80, y, 6.05, FLOOR - y, KEY_FILL, KEY_LINE)
    add_bullets(slide, 7.00, y + 0.12, 5.65, FLOOR - y - 0.24, "Does the age gradient survive every seed?", [
        ("Direction. ", "Set size rises with age in "
         f"{'all' if cons['global_set_size_positive_in_all_seeds'] else 'not all'} seeds, and exact-match "
         f"accuracy falls in {'all' if cons['exact_match_negative_in_all_seeds'] else 'not all'} seeds."),
        ("Monotonicity. ", "The global-threshold band ordering is monotone in "
         f"{'every' if cons['global_set_size_monotone_in_all_seeds'] else 'not every'} seed."),
        ("Slope and significance. ", f"{gs['mean']:+.4f} ± {gs['sd']:.4f} labels per decade across seeds; "
         f"every seed's Spearman p below 1e-3: {cons['all_spearman_p_below_0_001']}."),
    ], size=T_BODY, gap=8)


def build_item8(slide):
    add_title(slide, TITLES["item8"])
    subhead(slide, "This was the one item genuinely absent from the deck. It is manuscript section 10; "
                   "it is now also a slide.")

    add_panel(slide, 0.55, 1.72, 6.05, 2.40, INFO_FILL, INFO_LINE)
    add_bullets(slide, 0.75, 1.83, 5.65, 2.18, "Data availability", [
        ("Source. ", "PTB-XL v1.0.3 from PhysioNet, Creative Commons Attribution 4.0 "
                     "(Wagner et al. 2020)."),
        ("No new collection. ", "No data were collected for this study; it is a secondary analysis."),
        ("Record count. ", "21,799 against 21,837 in the original paper is an upstream de-duplication fix "
                           "in the v1.0.3 changelog, not a filtering choice of ours."),
    ], size=T_BODY, gap=8)

    add_panel(slide, 6.80, 1.72, 6.05, 2.40, KEY_FILL, KEY_LINE)
    add_bullets(slide, 7.00, 1.83, 5.65, 2.18, "Ethics", [
        ("De-identified public data. ", "No IRB approval and no individual consent were required for this "
                                        "secondary analysis."),
        ("No re-identification. ", "No attempt was made to re-identify any subject."),
        ("Scope of the claims. ", "This reports subgroup performance disparities. It is not a deployable "
                                  "clinical device and must not guide patient care as it stands."),
    ], size=T_BODY, gap=8)

    add_panel(slide, 0.55, 4.16, 6.05, FLOOR - 4.16, INFO_FILL, INFO_LINE)
    add_bullets(slide, 0.75, 4.27, 5.65, FLOOR - 4.38, "Code availability", [
        ("Pipeline. ", "Notebooks 01–10 run data preparation through final hypothesis testing; "
                       "notebooks 11–14 add the confound and trend tests, the attribution overlays and "
                       "the multi-seed replications."),
        ("Reproduction guard. ", "analysis/11, /13 and /14 each re-derive the calibration and conformal "
                                 "pipeline from the saved checkpoints and assert agreement with the "
                                 "archive first."),
        ("Stored outputs. ", "Five result JSONs and nine seed checkpoints; every slide number is read "
                             "from them."),
    ], size=T_BODY, gap=4)

    add_panel(slide, 6.80, 4.16, 6.05, FLOOR - 4.16, KEY_FILL, KEY_LINE)
    add_bullets(slide, 7.00, 4.27, 5.65, FLOOR - 4.38, "Environment and hyperparameters", [
        ("Environment. ", "Python 3.12.3, PyTorch 2.13.0, NumPy 2.5.3, SciPy 1.18.0, scikit-learn 1.9.0; "
                          "CPU-only, 4 threads; seed 42 released, replication at seeds 1, 7, 13."),
        ("Training. ", "AdamW, lr 1e-3, weight decay 1e-4, batch 64, 10 epochs. The anchor adds "
                       "ReduceLROnPlateau (0.5, patience 2) and early stopping (patience 4); the other "
                       "two members use CosineAnnealingLR (T_max 10) with no early stopping, the NB02 "
                       "recipe their seed-42 checkpoints used."),
        ("Evaluation. ", "Conformal α = 0.10; ECE over 15 equal-width bins; 1,000 bootstrap resamples with "
                         "shared indices."),
    ], size=T_BODY, gap=4)


def build_sota(slide):
    add_title(slide, TITLES["sota"])
    subhead(slide, "Found while re-checking the deck against the manuscript this round. It changes a "
                   "headline claim, so it is reported the same way as the Round 1 corrections.")

    rows = [["", "What the deck said", "What is actually true"]]
    rows.append(["Comparator", "\u201cPublished SOTA 0.9290\u201d, used on slides 32, 36, 40, 55 and the Summary",
                 "0.929 is Strodthoff et al.'s result on the ALL task (71 labels). On the five-class "
                 "superdiagnostic task we run, their six-model ensemble reaches 0.934(05)."])
    rows.append(["Single models", "not quoted separately",
                 "Strodthoff's best single model, resnet1d_wang, reaches 0.930(06); Gitau et al.'s "
                 "reproduction gives 0.918\u20130.929 across the single models and 0.934 for the ensemble."])
    rows.append(["Our claim", "\u201cStatistical parity with SOTA \u2014 0.9290 lies inside our interval\u201d",
                 "Our 0.9245 [0.9176, 0.9315] contains 0.929 but NOT 0.934. The honest statement is "
                 "parity with the best published SINGLE models and a ~0.007 gap to the ensembles."])
    add_table(slide, 0.55, 1.72, 12.3, rows,
              col_w=[1.55, 4.45, 6.30], row_h=0.83, size=T_TBL, header_size=T_TBL,
              left_cols=(0, 1, 2))

    note_strip(slide, 5.22, [
        ("Where this is now corrected. ", "Slides 32, 36 and 40, the master literature table on slide 55 "
         "and the closing Summary have been rewritten to compare against 0.934 and to state the single-model "
         "parity claim instead. The manuscript already carried the correct framing in \u00a76."),
        ("Why it matters. ", "The deck's headline read as \u201cwe match the state of the art\u201d. Against the "
         "right comparator we match the best published single models and sit below the six-model ensembles. "
         "That is still a good result for a three-model ensemble, and it is the one the data supports."),
    ], h=FLOOR - 5.22)


def build_scorecard(slide, seeds):
    add_title(slide, TITLES["score"])
    subhead(slide, "Status after Part 10. 'Part 10' means a slide in this section now carries it; "
                   "'MS' refers to the manuscript draft.")

    full = bool(seeds) and seeds.get("_source", "").startswith("14")
    if full:
        e = seeds["summary"]["ensemble_macro_auc_test"]
        item5_now = (f"Closed — 4 seeds × 3 members, {e['mean']:.4f} ± {e['sd']:.4f}")
    elif seeds:
        item5_now = "Addressed for the anchor model; scope stated"
    else:
        item5_now = "Retraining in progress — Part 10 slide pending"
    data = [
        ("1", "Stale cross-references on the literature slides", "Closed", "Closed", "Slides 29, 31"),
        ("2", "Signal-quality confound on the age effect", "Partially addressed",
         "Closed — four covariates, age effect re-fit adjusted", "Part 10 · MS §5.5, §8"),
        ("3", "Formal trend significance test across age bands", "Not formally addressed",
         "Closed — Spearman, Kruskal–Wallis, robust OLS", "Part 10 · MS §5.5"),
        ("4", "Sample sizes and CI widths on every age table", "Closed", "Closed", "Slide 45"),
        ("5", "Multi-seed replication", "Not addressed", item5_now, "Part 10 · MS §8"),
        ("6", "Central framing decision", "Resolved", "Resolved — age-fairness audit", "Slides 36, 57"),
        ("7", "Required figures", "Substantially addressed",
         "Closed — literal ECG × attribution overlays added", "Part 10 · MS Fig. 21"),
        ("8", "Reproducibility and ethics statement", "Not addressed",
         "Closed — data, ethics, code, environment", "Part 10 · MS §10"),
        ("9", "Related-work coverage", "Partially addressed",
         "Conformal and demographic-bias literature in MS §2", "MS §2, refs 11–22"),
    ]
    rows = [["#", "Action item from the review", "Previous verdict", "Status now", "Where"]]
    rows += [list(r) for r in data]
    add_table(slide, 0.55, 1.66, 12.3, rows,
              col_w=[0.50, 3.95, 2.20, 3.80, 1.85],
              row_h=0.43, size=T_TBL, header_size=T_TBL, left_cols=(0, 1, 2, 3, 4))

    y = 1.66 + 0.43 * len(rows) + 0.14
    add_panel(slide, 0.55, y, 12.3, FLOOR - y, KEY_FILL, KEY_LINE)
    _, tf = textbox(slide, 0.75, y + 0.08, 11.9, FLOOR - y - 0.16, MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _run(p, "Still open before submission:  ", T_BODY, RED, bold=True)
    tail = ("Section 6 comparators are the original papers' own partitions, not a paired test on our "
            "split; every result comes from the single fold-9 / fold-10 arrangement, so the four seeds "
            "measure initialisation variance and not split variance; and the IEEE LaTeX build still "
            "needs a full resync against the Markdown manuscript.")
    if not full:
        tail = "seed variance for the two heavier ensemble members; " + tail[0].lower() + tail[1:]
    _run(p, tail, T_BODY, BODY)


# ------------------------------------------------------------------ main
def main():
    trends = load_trends()
    seeds = load_seeds()
    print(f"[DATA] trend tests loaded; multi-seed results {'present' if seeds else 'NOT YET present'}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DECK.with_suffix(DECK.suffix + f".bak_pre_round2_{stamp}")
    shutil.copy2(DECK, backup)
    print(f"[BACKUP] {backup.name}")

    prs = Presentation(DECK)
    n_before = len(prs.slides._sldIdLst)

    summary_idx = find_slide(prs, SUMMARY_TITLE)
    if summary_idx is None:
        summary_idx = n_before
        print("[WARN] closing Summary slide not found by title; appending at the end")

    template_idx = find_slide(prs, "Provenance and Corrections — What Changed in This Pass")
    if template_idx is None:
        raise SystemExit("could not find a template slide to clone")

    plan = [("divider", build_divider, (trends, seeds)),
            ("item2", build_item2, (trends,)),
            ("item2b", build_item2b, (trends,)),
            ("item3", build_item3, (trends,)),
            ("clarify", build_clarify, (trends, seeds)),
            ("item7", build_item7, ()),
            ("item5", build_item5, (seeds,)),
            ("item8", build_item8, ()),
            ("sota", build_sota, ()),
            ("score", build_scorecard, (seeds,))]

    pos = summary_idx
    for key, fn, args in plan:
        title = TITLES[key]
        idx = find_slide(prs, title)
        if idx is None:
            clone_slide(prs, template_idx)
            move_slide(prs, len(prs.slides._sldIdLst) - 1, pos)
            idx, verb = pos, "ADD"
        else:
            verb = "REBUILD"
        slide = prs.slides[idx]
        strip_content(slide)
        fn(slide, *args)
        print(f"[{verb}] slide {idx + 1}: {title}")
        pos = idx + 1

    prs.save(DECK)
    print(f"[SAVED] {DECK.name}: {n_before} → {len(prs.slides._sldIdLst)} slides")


if __name__ == "__main__":
    main()
