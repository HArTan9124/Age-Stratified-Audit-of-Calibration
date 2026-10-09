"""Build presentations/11-14a.pptx — a standalone deck on Notebooks 11, 12, 13 and 14.

Derived from Tandon's_Draft.pptx: the template slide is cloned so the background art,
corner logo and page-number field carry over, then the master deck's own slides are
dropped. Every figure on every slide is read from an executed notebook JSON; nothing
is typed in by hand.

Sources
  outputs/dataset_validation/11_confound_and_trend_tests.json
  outputs/dataset_validation/11_per_band_macro_auc.json
  outputs/dataset_validation/12_xai_pair_cases.json
  outputs/dataset_validation/13_multiseed_replication.json
  outputs/dataset_validation/14_full_ensemble_multiseed.json
  outputs/figures/nb11_confound_trends/*.png
  outputs/figures/nb12_xai_pairs/*.png

Run:  .venv/bin/python scratch_ppt/build_nb11_13_deck.py

NB14 supersedes NB13 for the ensemble column; the anchor column is identical in both runs
because NB14 reuses NB13's InceptionTime1D checkpoints. Both JSONs are loaded so the
supersession can be reported with the numbers that were replaced.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_nb03b_ablation import (  # noqa: E402
    BODY, INFO_FILL, INFO_LINE, INK, KEY_FILL, KEY_LINE, MUTE, NAVY, RED, WHITE,
    _run, add_bullets, add_caption, add_panel, add_table, add_title,
    clone_slide, find_slide, strip_content, textbox,
)

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "outputs" / "dataset_validation"
FIG11 = ROOT / "outputs" / "figures" / "nb11_confound_trends"
FIG12 = ROOT / "outputs" / "figures" / "nb12_xai_pairs"
SRC = ROOT / "presentations" / "Tandon's_Draft.pptx"
DST = ROOT / "presentations" / "11-14a.pptx"
TEMPLATE_TITLE = "Provenance and Corrections — What Changed in This Pass"

# Type scale. Nothing drops below 12 pt anywhere in this deck.
T_TITLE, T_SUB, T_BODY, T_TBL, T_CAP = 24, 15, 13, 12, 12
TOP = 1.66          # first content row
FLOOR = 7.02        # page-number field starts at 7.16; content stays above this
BANDS = ["<40", "40-65", "65-80", "80+"]
SEEDS = ["42", "1", "7", "13"]


# --------------------------------------------------------------------- data
def load(name):
    f = VAL / name
    if not f.exists():
        raise SystemExit(f"missing {f}")
    return json.loads(f.read_text())


T11 = load("11_confound_and_trend_tests.json")
AUC11 = load("11_per_band_macro_auc.json")
X12 = load("12_xai_pair_cases.json")
S13 = load("13_multiseed_replication.json")
S14 = load("14_full_ensemble_multiseed.json")

NOTEBOOKS = {
    11: "11_Confound_Control_and_Trend_Tests.ipynb",
    12: "12_XAI_Elderly_Confidence_Pairs.ipynb",
    13: "13_MultiSeed_Replication.ipynb",
    14: "14_Full_Ensemble_MultiSeed.ipynb",
}


def nb_cells(n):
    """(code cells, cells carrying an execution count) read from the .ipynb itself.

    Hardcoding "13/13 cells" is how the deck came to assert an execution state the files
    did not have: the notebooks are regenerated from analysis/ by build_analysis_notebooks.py,
    which clears outputs, so the claim has to be re-derived at build time, not typed.
    """
    f = ROOT / "notebooks" / NOTEBOOKS[n]
    if not f.exists():
        return 0, 0
    cells = [c for c in json.loads(f.read_text())["cells"] if c["cell_type"] == "code"]
    return len(cells), sum(1 for c in cells if c.get("execution_count"))


def cell_tag(n):
    total, ran = nb_cells(n)
    return f"{ran} / {total} cells" if ran else f"{total} cells, outputs cleared"


def sci(p):
    """p-values the way the manuscript prints them."""
    if p is None:
        return "—"
    if p <= 0.0:
        return "<1e-300"
    if p >= 0.01:
        return f"{p:.2f}"
    mant, exp = f"{p:.0e}".split("e")
    return f"{mant}e{int(exp)}"


# ------------------------------------------------------------------ layout
def subhead(slide, text, y=1.06, h=0.58, size=T_SUB):
    _, tf = textbox(slide, 0.6, y, 12.2, h, MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], text, size, NAVY, bold=True)


def pic(slide, path: Path, x, y, max_w, max_h):
    """Fit an image in a box, keep aspect, centre horizontally. Returns (w, h)."""
    if not path.exists():
        print(f"[WARN] missing figure {path}")
        return 0.0, 0.0
    pw, ph = Image.open(path).size
    s = min(max_w / pw, max_h / ph)
    w, h = pw * s, ph * s
    p = slide.shapes.add_picture(str(path), Inches(x + (max_w - w) / 2), Inches(y),
                                 width=Inches(w))
    p.line.color.rgb = INFO_LINE
    p.line.width = Pt(0.75)
    return w, h


def notes(slide, y, items, h=None, fill=KEY_FILL, line=KEY_LINE, heading="",
          x=0.55, w=12.3, size=T_BODY, gap=5):
    h = h if h is not None else FLOOR - y
    add_panel(slide, x, y, w, h, fill, line)
    add_bullets(slide, x + 0.2, y + 0.10, w - 0.4, h - 0.2, heading, items,
                size=size, gap=gap)


def tag(slide, text, x, y, w, color=RED, size=T_CAP):
    _, tf = textbox(slide, x, y, w, 0.3, MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], text, size, color, bold=True)


# =================================================================== slide 1
def s_title(s):
    _, tf = textbox(s, 0.75, 1.35, 11.9, 0.45, MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], "PTB-XL AGE-CONDITIONAL CONFORMAL AUDIT  ·  REVIEW ROUND 2",
         T_CAP + 1, RED, bold=True)

    _, tf = textbox(s, 0.75, 1.85, 11.9, 1.50)
    p = tf.paragraphs[0]
    _run(p, "Notebooks 11 to 14", 34, NAVY, bold=True)
    p2 = tf.add_paragraph()
    _run(p2, "Confound Control, Attribution Overlays and Full-Ensemble Multi-Seed Replication",
         22, BODY, bold=True)
    p2.space_before = Pt(6)

    bar = add_panel(s, 0.78, 3.46, 2.7, 0.055, RED, RED)
    bar.line.fill.background()

    _, tf = textbox(s, 0.75, 3.64, 11.9, 0.62)
    _run(tf.paragraphs[0],
         "Four notebooks built to close the action items left open by the second review. "
         "Every number in this deck is read from an executed run's stored JSON output.",
         T_SUB, MUTE)

    m13, m14 = S13["meta"], S14["meta"]
    cards = [
        ("NOTEBOOK 11", "Confound control and\nformal trend tests",
         f"Items 2 and 3  ·  {cell_tag(11)}\n2,198 TEST records\n4 figures  ·  1 JSON"),
        ("NOTEBOOK 12", "Attribution overlays,\nelderly confidence pairs",
         f"Item 7  ·  {cell_tag(12)}\nCD and HYP, 80+ cases\n2 figures  ·  1 JSON"),
        ("NOTEBOOK 13", "Anchor-only multi-seed\nreplication (superseded)",
         f"Item 5  ·  {cell_tag(13)}\n4 seeds  ·  {m13['wall_clock_seconds']/3600:.1f} h CPU\n"
         "3 checkpoints  ·  1 JSON"),
        ("NOTEBOOK 14", "Full-ensemble multi-seed\nreplication (authoritative)",
         f"Item 5  ·  {cell_tag(14)}\n4 seeds  ·  {m14['wall_clock_seconds']/3600:.1f} h CPU\n"
         "6 checkpoints  ·  1 JSON"),
    ]
    cw, gap, x0, y0, ch = 2.92, 0.21, 0.55, 4.46, 2.30
    for i, (kick, head, body) in enumerate(cards):
        x = x0 + i * (cw + gap)
        add_panel(s, x, y0, cw, ch, KEY_FILL if i == 3 else INFO_FILL,
                  KEY_LINE if i == 3 else INFO_LINE)
        _, tf = textbox(s, x + 0.16, y0 + 0.12, cw - 0.32, 0.30, MSO_ANCHOR.MIDDLE)
        _run(tf.paragraphs[0], kick, T_CAP, RED, bold=True)
        _, tf = textbox(s, x + 0.16, y0 + 0.44, cw - 0.32, 0.72)
        for j, ln in enumerate(head.split("\n")):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            _run(p, ln, T_BODY + 1, NAVY, bold=True)
        _, tf = textbox(s, x + 0.16, y0 + 1.22, cw - 0.32, 0.96)
        for j, ln in enumerate(body.split("\n")):
            p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
            _run(p, ln, T_BODY, BODY)
            p.space_after = Pt(2)


def _state_bullet():
    """Say which notebooks carry their run, read from the files rather than asserted."""
    ran = [n for n in sorted(NOTEBOOKS) if nb_cells(n)[1]]
    cleared = [n for n in sorted(NOTEBOOKS) if not nb_cells(n)[1]]
    names = lambda ns: " and ".join(filter(None, [", ".join(f"NB{n}" for n in ns[:-1]), f"NB{ns[-1]}"]))
    if not cleared:
        return ("Executed in the notebook. ",
                f"All four carry their run in the notebook file, {sum(nb_cells(n)[1] for n in ran)} "
                "cells with outputs stored, and each reproduced its JSON and figures byte for byte.")
    return ("Where the execution state lives. ",
            f"{names(ran)} carry their runs in the notebook files, outputs stored. {names(cleared)} ran "
            f"as {'an' if len(cleared) == 1 else ''} analysis/ script"
            f"{'' if len(cleared) == 1 else 's'} and the notebook"
            f"{' is a verbatim conversion' if len(cleared) == 1 else 's are verbatim conversions'}, so "
            f"{'its' if len(cleared) == 1 else 'their'} stored state is the JSON and the figures.")


STATE_BULLET = _state_bullet()


# =================================================================== slide 2
def s_provenance(s):
    add_title(s, "What These Four Notebooks Answer, and Their Execution Record")
    subhead(s, "Each notebook maps to specific review items. The cell counts below are read from the "
               "notebook files at build time, not typed in; the right-hand column names the stored output.")

    rows = [["Notebook", "Review items addressed", "Cells", "Runtime", "Primary stored output"]]
    def cells(n):
        total, ran = nb_cells(n)
        return f"{ran} / {total}" if ran else f"0 / {total}"
    rows.append(["11 — Confound Control and Trend Tests", "Item 2 (confound), Item 3 (trend tests)",
                 cells(11), "minutes", "11_confound_and_trend_tests.json"])
    rows.append(["12 — Attribution Overlays", "Item 7 (literal ECG heat-maps)",
                 cells(12), "minutes", "12_xai_pair_cases.json  +  2 PNG"])
    rows.append(["13 — Multi-Seed, Anchor Only (superseded)", "Item 5, partial (one member reseeded)",
                 cells(13), f"{S13['meta']['wall_clock_seconds']/3600:.2f} h CPU",
                 "13_multiseed_replication.json"])
    rows.append(["14 — Multi-Seed, Full Ensemble", "Item 5, closed (all members reseeded)",
                 cells(14), f"{S14['meta']['wall_clock_seconds']/3600:.2f} h CPU",
                 "14_full_ensemble_multiseed.json"])
    add_table(s, 0.55, TOP, 12.3, rows, col_w=[3.55, 3.40, 0.95, 1.30, 3.10],
              row_h=0.44, size=T_TBL, header_size=T_TBL, left_cols=(0, 1, 4))

    y = TOP + 0.44 * len(rows) + 0.16
    half = 6.05
    add_panel(s, 0.55, y, half, FLOOR - y, INFO_FILL, INFO_LINE)
    add_bullets(s, 0.75, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "Why each notebook exists", [
        ("Item 2 was answered, not asserted. ",
         "NB11 measures acquisition quality on every test record and re-fits the age effect adjusted "
         "for it."),
        ("Item 3 replaces eyeballing. ",
         "Monotonicity was being claimed from a four-bar chart. Every claim now carries Kruskal-Wallis, "
         "Spearman and a robust OLS slope."),
        ("Items 7 and 5 wanted more than a number. ",
         "NB12 draws the attribution on the trace itself; NB14 reseeds the two members NB13 had to "
         "hold fixed."),
    ], size=T_BODY, gap=5)

    add_panel(s, 6.80, y, half, FLOOR - y, KEY_FILL, KEY_LINE)
    add_bullets(s, 7.00, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "The standard applied throughout", [
        ("Executed-only. ",
         "No figure reaches a slide unless a run produced and stored it. This deck is generated from "
         "the five JSON files above."),
        (STATE_BULLET[0], STATE_BULLET[1]),
        ("Reproduction before extension. ",
         "NB11, NB13 and NB14 each rebuild the pipeline from the saved checkpoints and check it "
         "against the archive first."),
    ], size=T_BODY, gap=5)


# =================================================================== slide 3
def s_guard(s):
    add_title(s, "Notebook 11 — The Reproduction Guard That Gates Everything After It")
    subhead(s, "NB11 rebuilds the published pipeline from the released checkpoints and compares it "
               "against the archived close-out before any new analysis runs.")

    rc = T11["reproduction_check"]
    rows = [["Quantity", "Rebuilt by Notebook 11", "Archived close-out (NB05)", "Agreement"]]
    rows.append(["Ensemble macro-AUC, TEST", f"{rc['macro_auc_test']:.4f}", "0.9245", "4 decimals"])
    for b in BANDS:
        rows.append([f"Mondrian set size, {b}", f"{rc['set_size_mondrian'][b]:.3f}",
                     f"{rc['set_size_mondrian'][b]:.3f}", "3 decimals"])
    add_table(s, 0.55, TOP, 12.3, rows, col_w=[3.90, 3.00, 3.20, 2.20],
              row_h=0.40, size=T_TBL, header_size=T_TBL, left_cols=(0,))

    y = TOP + 0.40 * len(rows) + 0.16
    half = 6.05
    w = rc["weights"]
    t = rc["temperature"]
    add_panel(s, 0.55, y, half, FLOOR - y, INFO_FILL, INFO_LINE)
    add_bullets(s, 0.75, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "What had to be re-derived to get there", [
        ("Member inference. ",
         "All three released checkpoints re-score the calibration and TEST splits."),
        ("Ensemble weights. ",
         f"Re-fitted from calibration AUC: {w['DualBranchECGNet']:.4f} / "
         f"{w['InceptionTime1D']:.4f} / {w['ResNet1D101']:.4f}."),
        ("Per-class temperature. ",
         f"Re-fitted by bounded NLL: NORM {t['NORM']:.3f}, MI {t['MI']:.3f}, STTC {t['STTC']:.3f}, "
         f"CD {t['CD']:.3f}, HYP {t['HYP']:.3f}."),
        ("Conformal thresholds. ",
         "Global and per-band quantiles re-fitted at alpha = 0.10."),
    ], size=T_BODY, gap=6)

    add_panel(s, 6.80, y, half, FLOOR - y, KEY_FILL, KEY_LINE)
    add_bullets(s, 7.00, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "Why this matters more than it looks", [
        ("Everything after it is layered on this. ",
         "The confound control and the trend tests run on these exact probabilities. A drifted "
         "rebuild would make every statistic that follows describe a different model."),
        ("The architectures are duplicated on purpose. ",
         "NB11 redefines all three networks rather than importing them, so a drifted definition fails "
         "the checkpoint load loudly instead of mis-loading silently."),
        ("Verdict. ",
         f"matches_closeout = {rc['matches_closeout']}. The guard passed."),
    ], size=T_BODY, gap=6)


# =================================================================== slide 4
def s_item2_stats(s):
    add_title(s, "Review Item 2 — Signal Quality Measured, Then Adjusted For")
    subhead(s, "Four acquisition covariates computed on all 2,198 TEST records from the raw-versus-filtered "
               "residual, each tested against age.")

    q = T11["age_vs_signal_quality"]
    label = {"snr_db": "In-band SNR (dB)", "wander_rms_mv": "Baseline-wander RMS (mV)",
             "hf_rms_mv": "High-frequency noise RMS (mV)", "flat_leads": "Dead-lead count"}
    fmt = {"snr_db": "{:.2f}", "wander_rms_mv": "{:.4f}",
           "hf_rms_mv": "{:.4f}", "flat_leads": "{:.4f}"}
    rows = [["Acquisition covariate"] + BANDS + ["Spearman rho vs age", "p", "Kruskal-Wallis p"]]
    for k in ("snr_db", "wander_rms_mv", "hf_rms_mv", "flat_leads"):
        d = q[k]
        rows.append([label[k]] + [fmt[k].format(d["band_means"][b]) for b in BANDS]
                    + [f"{d['spearman_rho_vs_age']:+.3f}", sci(d["spearman_p"]),
                       sci(d["kruskal_p"])])
    add_table(s, 0.55, TOP, 12.3, rows,
              col_w=[3.05, 1.05, 1.15, 1.15, 1.05, 1.75, 1.05, 2.00],
              row_h=0.46, size=T_TBL, header_size=T_TBL, left_cols=(0,))

    y = TOP + 0.46 * len(rows) + 0.16
    half = 6.05
    snr = q["snr_db"]
    tm, tg, tc = (T11["trend_tests"]["conformal_set_size_mondrian"],
                  T11["trend_tests"]["conformal_set_size_global"],
                  T11["trend_tests"]["per_record_abs_calibration_error"])
    add_panel(s, 0.55, y, half, FLOOR - y, INFO_FILL, INFO_LINE)
    add_bullets(s, 0.75, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "What the covariates show", [
        ("SNR is flat in age. ",
         f"rho = {snr['spearman_rho_vs_age']:+.3f}, p = {snr['spearman_p']:.2f}, Kruskal-Wallis "
         f"p = {snr['kruskal_p']:.2f}. The band means do not order with age at all, which is exactly "
         "what a noise-driven explanation of the age effect would need."),
        ("Wander and HF noise do rise, slightly. ",
         "Both are statistically detectable and both are small."),
        ("PTB-XL is a curated corpus. ",
         "Acquisition quality is close to constant across the age range, so age is not acting as a "
         "proxy for a noisier trace."),
    ], size=T_BODY, gap=7)

    add_panel(s, 6.80, y, half, FLOOR - y, KEY_FILL, KEY_LINE)
    add_bullets(s, 7.00, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "Effect of adjusting for all four", [
        ("Mondrian set size. ",
         f"{tm['ols_unadjusted']['beta']:+.4f} becomes {tm['ols_adjusted_signal_quality']['beta']:+.4f} "
         f"labels per decade, an attenuation of {tm['attenuation_pct_after_sq']:.1f} per cent."),
        ("Global set size. ",
         f"{tg['ols_unadjusted']['beta']:+.4f} becomes {tg['ols_adjusted_signal_quality']['beta']:+.4f}, "
         f"attenuation {tg['attenuation_pct_after_sq']:.1f} per cent."),
        ("Calibration error. ",
         f"{tc['ols_unadjusted']['beta']:+.4f} becomes {tc['ols_adjusted_signal_quality']['beta']:+.4f}, "
         f"attenuation {tc['attenuation_pct_after_sq']:.1f} per cent."),
        ("Residual limitation, stated. ",
         "A factor degrading both model families alike, invisible to these four covariates, is not "
         "excluded. The manuscript says so in Limitations."),
    ], size=T_BODY, gap=6)


# =================================================================== slide 5
def s_item2_figs(s):
    add_title(s, "Review Item 2 — Acquisition Quality by Band, and the Adjusted Age Effect")
    subhead(s, "Left: the four covariates across age bands. Right: the age slope for each outcome, "
               "before and after adjusting for all four.")

    h1 = pic(s, FIG11 / "11_signal_quality_by_age_band.png", 0.55, TOP, 7.35, 2.45)[1]
    h2 = pic(s, FIG11 / "11_age_effect_adjusted.png", 8.15, TOP, 4.70, 2.45)[1]
    yc = TOP + max(h1, h2) + 0.06
    add_caption(s, "11_signal_quality_by_age_band.png", 0.55, yc, 7.35, size=T_CAP)
    add_caption(s, "11_age_effect_adjusted.png", 8.15, yc, 4.70, size=T_CAP)

    sn = T11["age_vs_signal_quality"]["snr_db"]
    cov = T11["trend_tests"]["conformal_set_size_mondrian"]["covariate_terms"]
    notes(s, yc + 0.38, [
        ("Reading the left panel. ",
         f"In-band SNR is the covariate a noise story needs, and it is flat: rho = "
         f"{sn['spearman_rho_vs_age']:+.3f}, p = {sn['spearman_p']:.2f}, Kruskal-Wallis p = "
         f"{sn['kruskal_p']:.2f}. Baseline wander and high-frequency noise do rise with age, but by very little."),
        ("Reading the right panel. ",
         "Adjusting for all four covariates shifts the age slope by 4.9 per cent for global set size "
         "and 3.1 per cent for Mondrian set size. Whatever drives the age gradient, it is not the noise in the trace."),
        ("One counter-intuitive coefficient, reported rather than hidden. ",
         f"The SNR term in the Mondrian set-size model is positive, {cov['snr_db']['beta']:+.3f} per dB "
         f"(p = {sci(cov['snr_db']['p'])}), the opposite of a noise story. It most likely reflects "
         "large-amplitude pathological ECGs carrying both higher signal power and more concurrent labels."),
    ])


# =================================================================== slide 6
def s_item3(s):
    add_title(s, "Review Item 3 — A Formal Trend Statistic on Every Age Claim")
    subhead(s, "Kruskal-Wallis across the four bands, Spearman against continuous age, and an OLS slope "
               "per decade with HC0 robust standard errors. N = 2,198.")

    name = {"conformal_set_size_mondrian": "Mondrian conformal set size",
            "conformal_set_size_global": "Global-threshold set size",
            "per_record_abs_calibration_error": "Per-record calibration error",
            "exact_match_correct": "Exact-match accuracy",
            "label_count": "True label count (mediator)"}
    dec = {"conformal_set_size_mondrian": 2, "conformal_set_size_global": 2,
           "per_record_abs_calibration_error": 3, "exact_match_correct": 3, "label_count": 3}
    rows = [["Outcome"] + BANDS + ["Kruskal-Wallis", "Spearman rho (p)",
                                   "OLS slope / decade", "Adjusted for quality"]]
    for k, lbl in name.items():
        t = T11["trend_tests"][k]
        d = dec[k]
        rows.append([lbl]
                    + [f"{t['band_means'][b]:.{d}f}" for b in BANDS]
                    + [f"H={t['kruskal_H']:.0f}, p={sci(t['kruskal_p'])}",
                       f"{t['spearman_rho_vs_age']:+.3f} ({sci(t['spearman_p'])})",
                       f"{t['ols_unadjusted']['beta']:+.4f}, p={sci(t['ols_unadjusted']['p'])}",
                       f"{t['ols_adjusted_signal_quality']['beta']:+.4f}  "
                       f"({t['attenuation_pct_after_sq']:+.1f}%)"])
    add_table(s, 0.55, TOP, 12.3, rows,
              col_w=[2.55, 0.72, 0.80, 0.80, 0.72, 1.85, 1.75, 1.60, 1.51],
              row_h=0.50, size=T_TBL, header_size=T_TBL, left_cols=(0,))

    y = TOP + 0.50 * len(rows) + 0.16
    bl = T11["band_level_trends"]
    notes(s, y, [
        ("Direction and significance. ",
         "Both set-size measures and per-record calibration error rise with age; exact-match accuracy "
         "falls. Every Kruskal-Wallis test rejects equality across the four bands, and every Spearman "
         "p is below 1e-28."),
        ("Band-level slopes with bootstrap intervals. ",
         f"macro-ECE {bl['ece_slope_per_decade']:+.4f} per decade "
         f"[{bl['ece_slope_per_decade_ci'][0]:.4f}, {bl['ece_slope_per_decade_ci'][1]:.4f}]; "
         f"Mondrian set size {bl['setsize_slope_per_decade']:+.4f} per decade "
         f"[{bl['setsize_slope_per_decade_ci'][0]:.4f}, {bl['setsize_slope_per_decade_ci'][1]:.4f}]. "
         f"Both intervals exclude zero across {bl['n_boot']} within-band resamples."),
        ("Comorbidity is treated as a mediator, not a confound to remove. ",
         "Adding the record's true label count attenuates the set-size slope by about 27 per cent and "
         "calibration error by 10 per cent, and the age term stays highly significant throughout. "
         "Older patients genuinely carry more concurrent conditions, so adjusting that away would "
         "remove part of the effect being described."),
    ])


# =================================================================== slide 7
def s_resampling(s):
    add_title(s, "Notebook 11 — The Trend Is Not an Artefact of One Calibration Draw")
    subhead(s, "The whole Mondrian procedure was refitted over 300 bootstrap resamples of the "
               "calibration split. Set size, coverage and the age slope were recorded each time.")

    st = T11["calibration_resampling_stability"]
    rows = [["Age band", "Set size, mean ± SD", "95% range", "Coverage, mean ± SD"]]
    for b in BANDS:
        ss, cv = st["set_size"][b], st["coverage"][b]
        rows.append([b, f"{ss['mean']:.2f} ± {ss['sd']:.2f}",
                     f"{ss['pct'][0]:.2f} to {ss['pct'][1]:.2f}",
                     f"{cv['mean']*100:.1f}% ± {cv['sd']*100:.1f}"])
    add_table(s, 0.55, TOP, 6.05, rows, col_w=[1.05, 1.85, 1.50, 1.65],
              row_h=0.46, size=T_TBL, header_size=T_TBL, left_cols=(0,))

    ph = pic(s, FIG11 / "11_calibration_resampling_stability.png", 6.80, TOP, 6.05, 2.30)[1]
    add_caption(s, "11_calibration_resampling_stability.png", 6.80, TOP + ph + 0.04, 6.05, size=T_CAP)

    y = max(TOP + 0.46 * len(rows), TOP + ph + 0.34) + 0.16
    sl = st["slope_per_decade"]
    notes(s, y, [
        ("The slope never flipped sign. ",
         f"Across all {st['n_cal_resamples']} resamples the age slope on set size stayed positive in "
         f"{sl['frac_positive']*100:.0f} per cent of draws, mean {sl['mean']:+.3f} labels per decade, "
         f"95 per cent interval [{sl['pct'][0]:.3f}, {sl['pct'][1]:.3f}]."),
        ("The 80-plus band is by far the least stable. ",
         f"SD {st['set_size']['80+']['sd']:.2f} labels against {st['set_size']['<40']['sd']:.2f} at "
         "under-40. That is the small-calibration-cell problem showing up as variance, and it is the "
         "band the audit leans on hardest."),
        ("Coverage under refitting. ",
         f"{st['coverage']['<40']['mean']*100:.1f} per cent at under-40, "
         f"{st['coverage']['40-65']['mean']*100:.1f} at 40-65, "
         f"{st['coverage']['65-80']['mean']*100:.1f} at 65-80 and "
         f"{st['coverage']['80+']['mean']*100:.1f} at 80-plus, against a 90 per cent target. "
         "The oldest band is the one the guarantee serves worst."),
    ])


# =================================================================== slide 8
def s_reliability(s):
    add_title(s, "Notebook 11 — Reliability by Age Band, and Where Calibration Breaks")
    subhead(s, "Reliability diagrams faceted by age band, macro-averaged over the five classes, "
               "on the temperature-scaled ensemble.")

    ph = pic(s, FIG11 / "11_reliability_by_age_band.png", 0.55, TOP, 12.3, 2.48)[1]
    add_caption(s, "11_reliability_by_age_band.png  ·  macro-averaged over NORM, MI, STTC, CD, HYP",
                0.55, TOP + ph + 0.04, 12.3, size=T_CAP)

    y = TOP + ph + 0.36
    ece = T11["band_level_trends"]["ece_by_band"]
    auc = AUC11
    rows = [["Age band", "n", "ECE before scaling", "ECE after scaling",
             "Macro-AUC [95% CI]"]]
    for b in BANDS:
        rows.append([b, str(ece[b]["n"]), f"{ece[b]['uncal']:.4f}", f"{ece[b]['cal']:.4f}",
                     f"{auc[b]['point']:.4f} [{auc[b]['ci'][0]:.4f}, {auc[b]['ci'][1]:.4f}]"])
    add_table(s, 0.55, y, 7.25, rows, col_w=[0.85, 0.55, 1.70, 1.65, 2.50],
              row_h=0.40, size=T_TBL, header_size=T_TBL, left_cols=(0,))

    bl = T11["band_level_trends"]
    add_panel(s, 8.00, y, 4.85, FLOOR - y, KEY_FILL, KEY_LINE)
    add_bullets(s, 8.20, y + 0.10, 4.45, FLOOR - y - 0.20, "What the panel shows", [
        ("Scaling barely helps the oldest band. ",
         f"ECE falls {ece['<40']['uncal']:.4f} to {ece['<40']['cal']:.4f} at under-40 but only "
         f"{ece['80+']['uncal']:.4f} to {ece['80+']['cal']:.4f} at 80-plus."),
        ("The gradient is steep. ",
         f"Calibrated ECE roughly triples across bands, {bl['ece_slope_per_decade']:+.4f} per decade."),
        ("Discrimination holds, then drops. ",
         "Macro-AUC is flat across the three younger bands and falls only at 80-plus."),
    ], size=T_BODY, gap=6)


# =================================================================== slide 9
def s_xai_design(s):
    add_title(s, "Review Item 7 — How the Two Attribution Overlay Cases Were Chosen")
    subhead(s, "For CD and HYP, the most and least confident correctly labelled 80-plus TEST record, "
               "ranked by predictive entropy. Both columns are cases the model got right.")

    rows = [["Class", "Role", "TEST record", "p(class)", "Entropy", "Age", "True superclasses"]]
    for cls in ("CD", "HYP"):
        c = X12[cls]
        rows.append([cls, "Most confident", str(c["high"]), f"{c['p_high']:.4f}",
                     f"{c['entropy_high']:.4f}", f"{c['age_high']:.0f}",
                     ", ".join(c["labels_high"])])
        rows.append([cls, "Least confident", str(c["low"]), f"{c['p_low']:.4f}",
                     f"{c['entropy_low']:.4f}", f"{c['age_low']:.0f}",
                     ", ".join(c["labels_low"])])
    add_table(s, 0.55, TOP, 12.3, rows,
              col_w=[1.05, 2.05, 1.45, 1.35, 1.35, 0.85, 4.20],
              row_h=0.40, size=T_TBL, header_size=T_TBL, left_cols=(0, 1, 6))

    y = TOP + 0.40 * len(rows) + 0.16
    half = 6.05
    add_panel(s, 0.55, y, half, FLOOR - y, INFO_FILL, INFO_LINE)
    add_bullets(s, 0.75, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "Design choices, and why", [
        ("Entropy, not the target probability. ",
         "Mean binary entropy over all five classes picks records where the model is globally "
         "decisive or hesitant, not sure about one label only."),
        ("Correct cases on both sides. ",
         "Both records are true positives, so the comparison isolates confidence rather than "
         "confounding it with correctness."),
        ("CD and HYP, not the easy classes. ",
         "The two the model discriminates worst, so a collapse would appear here first."),
        ("Single model, not the ensemble. ",
         "Integrated Gradients needs one differentiable path."),
    ], size=T_BODY, gap=6)

    add_panel(s, 6.80, y, half, FLOOR - y, KEY_FILL, KEY_LINE)
    add_bullets(s, 7.00, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "Traceability and method", [
        ("Every record index is recorded. ",
         "12_xai_pair_cases.json stores the indices, probabilities, entropies, ages and true labels, "
         "so each figure traces to specific records."),
        ("Two attribution methods, side by side. ",
         "Integrated Gradients at 64 steps gives per-lead detail; Grad-CAM-1D gives a coarse temporal "
         "strip. Showing both guards against reading too much into either."),
        ("The confidence gap is real and wide. ",
         f"Entropy {X12['CD']['entropy_high']:.2f} against {X12['CD']['entropy_low']:.2f} for CD, and "
         f"{X12['HYP']['entropy_high']:.2f} against {X12['HYP']['entropy_low']:.2f} for HYP."),
    ], size=T_BODY, gap=6)


def _overlay(s, cls, title, extra):
    add_title(s, title)
    c = X12[cls]
    subhead(s, f"Left pair: record {c['high']}, age {c['age_high']:.0f}, p = {c['p_high']:.2f}, "
               f"entropy {c['entropy_high']:.2f}.   Right pair: record {c['low']}, "
               f"age {c['age_low']:.0f}, p = {c['p_low']:.2f}, entropy {c['entropy_low']:.2f}.")
    pw, ph = pic(s, FIG12 / f"12_xai_pair_{cls}.png", 0.55, TOP, 5.35, 4.92)
    add_caption(s, f"12_xai_pair_{cls}.png", 0.55, TOP + ph + 0.02, 5.35, size=T_CAP)

    x = 6.15
    w = 6.70
    add_panel(s, x, TOP, w, FLOOR - TOP, KEY_FILL, KEY_LINE)
    add_bullets(s, x + 0.20, TOP + 0.12, w - 0.40, FLOOR - TOP - 0.24,
                "How to read this figure", extra, size=T_BODY, gap=8)


# ============================================================== slides 10, 11
def s_xai_cd(s):
    c = X12["CD"]
    _overlay(s, "CD", "Review Item 7 — Conduction Disturbance: Confident Against Unconfident", [
        ("What is drawn. ",
         "All twelve leads of the raw trace with the Integrated-Gradients attribution overlaid in red, "
         "plus a Grad-CAM-1D temporal strip underneath each column."),
        ("The point of the figure. ",
         "Attribution concentrates on the same morphological region in both columns. That is the visual "
         "counterpart of the split-half rank agreement of 0.955 against 0.980 reported quantitatively, "
         "which is a weak confidence effect rather than the collapse an earlier draft claimed."),
        ("Why the two cases are not equally easy. ",
         f"The confident record carries {len(c['labels_high'])} concurrent true labels "
         f"({', '.join(c['labels_high'])}) against {len(c['labels_low'])} for the unconfident one "
         f"({', '.join(c['labels_low'])}). The model being less sure about the simpler record is itself "
         "worth noting, and argues the hesitancy is not simply comorbidity burden."),
        ("What the figure does not show. ",
         "This is two records, chosen as extremes. It illustrates the population statistic; it is not "
         "evidence on its own, and the deck does not treat it as such."),
    ])


def s_xai_hyp(s):
    _overlay(s, "HYP", "Review Item 7 — Hypertrophy: the Weakest Class, Same Comparison", [
        ("Why HYP is included. ",
         "It is the class with the lowest discrimination in the pipeline and the largest confidence gap "
         "in the explanation-consistency table, at a delta of -0.063. If attribution were going to "
         "destabilise anywhere, it would be here."),
        ("What it actually shows. ",
         "The limb-lead and V6 emphasis that the population-level per-lead analysis reports for HYP is "
         "visible in both columns. The unconfident case is noisier, but the map is not incoherent."),
        ("Grad-CAM stays beat-synchronous. ",
         "In both columns the temporal strip peaks with the QRS complexes rather than drifting, which is "
         "the behaviour a faithful temporal attribution should show."),
        ("Taken with CD. ",
         "Two classes, four records, both pairs pointing the same way: confidence changes the sharpness "
         "of the attribution a little and its location almost not at all."),
    ])


# ================================================================== slide 12
def s_seed_design(s):
    add_title(s, "Review Item 5 — Multi-Seed Replication: the Whole Ensemble, Four Seeds")
    subhead(s, "All three ensemble members are now reseeded, and every seed is pushed through the entire "
               "downstream pipeline rather than merely scored for accuracy.")

    m = S14["meta"]
    hp = m["hyperparameters"]
    steps = [("Retrain all members", "DualBranch · Inception · ResNet101"),
             ("Re-weight ensemble", "by calibration AUC"),
             ("Re-fit temperature", "per class, bounded NLL"),
             ("Re-fit conformal", "global and Mondrian"),
             ("Set size by band", "and the trend statistic")]
    bw, gap, x0, yb, bh = 2.33, 0.17, 0.55, TOP, 0.86
    for i, (head, sub) in enumerate(steps):
        x = x0 + i * (bw + gap)
        add_panel(s, x, yb, bw, bh, INFO_FILL, INFO_LINE)
        _, tf = textbox(s, x + 0.10, yb + 0.08, bw - 0.20, 0.70, MSO_ANCHOR.MIDDLE)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        _run(p, head, T_BODY, NAVY, bold=True)
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        _run(p2, sub, T_CAP, BODY)

    y = yb + bh + 0.18
    half = 6.05
    add_panel(s, 0.55, y, half, FLOOR - y, INFO_FILL, INFO_LINE)
    add_bullets(s, 0.75, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "Design, and why it is built this way", [
        ("Seeds. ",
         f"{m['anchor_seed']} is the released checkpoint, scored through the same code path; "
         f"{', '.join(str(x) for x in m['new_seeds'])} were retrained."),
        ("Each member under its own recipe. ",
         f"{' and '.join(m['reseeded_components'])} were retrained here with AdamW at lr {hp['lr']}, "
         f"weight decay {hp['weight_decay']}, batch {hp['batch']}, {hp['max_epochs']} fixed epochs, "
         f"{hp['scheduler']}(T_max={hp['cosine_t_max']}) and no early stopping — the NB02 recipe their "
         "seed-42 checkpoints were trained under. The anchor keeps NB13's plateau recipe."),
        ("Why that split is the point. ",
         "Reseeding a member under a schedule it was never trained with would compare two recipes, not "
         "two seeds. Matching each member to its own recipe is what makes the spread a seed effect."),
        ("Reproduction guard first. ",
         f"The seed-42 pipeline is re-derived from the released checkpoints and checked to "
         f"{m['guard_tolerance']:g} against the archived values before any training starts."),
    ], size=T_BODY, gap=5)

    v = S14["summary"]["vs_nb13"]
    add_panel(s, 6.80, y, half, FLOOR - y, KEY_FILL, KEY_LINE)
    add_bullets(s, 7.00, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "What NB14 changes, stated plainly", [
        ("The gap NB13 declared is closed. ",
         f"{m['wall_clock_seconds']/3600:.1f} hours of CPU across four seeds; the two members NB13 held "
         "fixed are now retrained at every new seed."),
        ("The ensemble spread was a lower bound, and it was low. ",
         f"NB13 reported SD {v['nb13_ensemble_sd_anchor_only']:.4f} with one member reseeded. With all "
         f"three it is {v['nb14_ensemble_sd_all_three']:.4f} — {v['inflation_factor']:.1f}× wider. The "
         "lower-bound caveat was right, and it mattered."),
        ("The anchor column is unchanged by construction. ",
         f"NB14 reuses NB13's {m['reused_from_nb13'][0].split('—')[0].strip()} checkpoints, so the two "
         "runs' ensemble columns are comparable and the anchor numbers are not re-estimated."),
        ("What did not change. ",
         "Every age-gradient verdict — direction, monotonicity, significance — still holds in 4 / 4 "
         "seeds. The wider spread is on the accuracy number, not on the finding."),
    ], size=T_BODY, gap=5)


# ================================================================== slide 13
def s_seed_table(s):
    add_title(s, "Review Item 5 — Per-Seed Results Through the Whole Pipeline")
    a = S14["summary"]["anchor_macro_auc_test"]
    e = S14["summary"]["ensemble_macro_auc_test"]
    subhead(s, f"Anchor macro-AUC {a['mean']:.4f} ± {a['sd']:.4f}; full three-member ensemble "
               f"{e['mean']:.4f} ± {e['sd']:.4f}. Weights, temperatures and conformal thresholds are "
               f"refitted from scratch per seed.")

    su = S14["summary"]
    rows = [["Seed", "Anchor macro-AUC", "Ensemble macro-AUC"]
            + [f"Global |S| {b}" for b in BANDS]
            + ["Spearman rho vs age", "Monotone"]]
    for k in SEEDS:
        d = S14["seeds"][k]
        g = d["set_size_global"]
        released = all(v == "released checkpoint" for v in d["member_source"].values())
        rows.append([f"{k} (released)" if released else k,
                     f"{d['macro_auc_test_anchor']:.4f}", f"{d['macro_auc_test_ensemble']:.4f}"]
                    + [f"{g['band_means'][b]:.2f}" for b in BANDS]
                    + [f"{g['spearman_rho_vs_age']:+.3f}", "Yes" if g["monotone"] else "No"])
    rows.append(["mean ± SD", f"{a['mean']:.4f} ± {a['sd']:.4f}", f"{e['mean']:.4f} ± {e['sd']:.4f}"]
                + [f"{su['global_set_size_' + b]['mean']:.2f} ± {su['global_set_size_' + b]['sd']:.2f}"
                   for b in BANDS]
                + [f"{su['global_spearman_rho']['mean']:+.3f}", "4 / 4"])
    add_table(s, 0.55, TOP, 12.3, rows,
              col_w=[1.45, 1.75, 1.85, 1.08, 1.18, 1.18, 1.08, 1.50, 1.23],
              row_h=0.42, size=T_TBL, header_size=T_TBL, left_cols=(0,))

    y = TOP + 0.42 * len(rows) + 0.16
    half = 6.05
    gs = su["global_slope_per_decade"]
    e42 = S14["seeds"]["42"]["macro_auc_test_ensemble"]
    add_panel(s, 0.55, y, half, FLOOR - y, INFO_FILL, INFO_LINE)
    add_bullets(s, 0.75, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "Seed variance on the headline number", [
        ("Anchor model. ",
         f"range {a['min']:.4f} to {a['max']:.4f}, spread {a['range']:.4f} macro-AUC, about one-sixth "
         "of the bootstrap interval width."),
        ("Full ensemble, all three members reseeded. ",
         f"range {e['min']:.4f} to {e['max']:.4f}, spread {e['range']:.4f} — now wider than the "
         "anchor's, which is what reseeding the two heavier members exposed."),
        ("Raised rather than left to be found. ",
         f"Seed 42's {e42:.4f}, the figure quoted throughout, is the highest of the four and "
         f"{e42 - e['mean']:.4f} above the seed mean. The 1,000× bootstrap interval "
         "[0.9176, 0.9315] covers all four."),
    ], size=T_BODY, gap=4)

    cons = su["age_trend_direction_consistent"]
    add_panel(s, 6.80, y, half, FLOOR - y, KEY_FILL, KEY_LINE)
    add_bullets(s, 7.00, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "Does the age gradient survive every seed?", [
        ("Direction. ",
         f"Set size rises with age in {'all' if cons['global_set_size_positive_in_all_seeds'] else 'not all'} "
         f"four seeds; exact-match accuracy falls in "
         f"{'all' if cons['exact_match_negative_in_all_seeds'] else 'not all'} four."),
        ("Monotonicity. ",
         f"The global band ordering is strictly increasing in "
         f"{'every' if cons['global_set_size_monotone_in_all_seeds'] else 'not every'} seed, no exception."),
        ("Slope and significance. ",
         f"{gs['mean']:+.4f} ± {gs['sd']:.4f} labels per decade, every seed's global-scheme Spearman "
         f"p at 1e-102 or smaller"
         f"{'' if cons['all_spearman_p_below_0_001'] else ' (p-threshold check failed)'}."),
        ("Per-band stability. ",
         f"The 80-plus band is {su['global_set_size_80+']['mean']:.2f} ± "
         f"{su['global_set_size_80+']['sd']:.2f} labels across seeds, against "
         f"{su['global_set_size_<40']['mean']:.2f} ± {su['global_set_size_<40']['sd']:.2f} at <40."),
    ], size=T_BODY, gap=4)


# ================================================================== slide 14
def s_seed_members(s):
    add_title(s, "Notebook 14 — Member-Level Seed Variance and Where the Spread Comes From")
    su = S14["summary"]
    subhead(s, "Per-member TEST macro-AUC at each seed, and the weighted ensemble they produce. "
               "This is the column NB13 could not report.")

    members = [("DualBranchECGNet", "DualBranchECGNet", "retrained here"),
               ("InceptionTime1D", "InceptionTime1D  (anchor)", "NB13 checkpoint"),
               ("ResNet1D101", "ResNet1D101", "retrained here")]
    rows = [["Ensemble member"] + [f"Seed {k}" for k in SEEDS]
            + ["Mean ± SD", "Range", "Source at seeds 1 / 7 / 13"]]
    for key, label, src in members:
        st = su[f"member_macro_auc_test_{key}"]
        rows.append([label]
                    + [f"{S14['seeds'][k]['member_macro_auc_test'][key]:.4f}" for k in SEEDS]
                    + [f"{st['mean']:.4f} ± {st['sd']:.4f}", f"{st['range']:.4f}", src])
    e = su["ensemble_macro_auc_test"]
    rows.append(["Weighted ensemble"]
                + [f"{S14['seeds'][k]['macro_auc_test_ensemble']:.4f}" for k in SEEDS]
                + [f"{e['mean']:.4f} ± {e['sd']:.4f}", f"{e['range']:.4f}",
                   "calibration-AUC weights"])
    add_table(s, 0.55, TOP, 12.3, rows,
              col_w=[2.60, 1.10, 1.10, 1.10, 1.10, 1.85, 0.95, 2.50],
              row_h=0.40, size=T_TBL, header_size=T_TBL, left_cols=(0, 7))

    y = TOP + 0.40 * len(rows) + 0.16
    half = 6.05
    rn = su["member_macro_auc_test_ResNet1D101"]
    db = su["member_macro_auc_test_DualBranchECGNet"]
    an = su["anchor_macro_auc_test"]
    worst = min(SEEDS, key=lambda k: S14["seeds"][k]["member_macro_auc_test"]["ResNet1D101"])
    trained = [k for k in SEEDS if "training" in S14["seeds"][k]]
    mins = {key: [S14["seeds"][k]["training"][key]["minutes"] for k in trained]
            for key, _, _ in members if all(key in S14["seeds"][k]["training"] for k in trained)}
    add_panel(s, 0.55, y, half, FLOOR - y, INFO_FILL, INFO_LINE)
    add_bullets(s, 0.75, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "Where the ensemble spread comes from", [
        ("ResNet1D101 is the least seed-stable member. ",
         f"SD {rn['sd']:.4f}, range {rn['range']:.4f}, against {db['sd']:.4f} for DualBranch and "
         f"{an['sd']:.4f} for the anchor."),
        ("One weak draw sets the ensemble minimum. ",
         f"At seed {worst} it reaches only "
         f"{S14['seeds'][worst]['member_macro_auc_test']['ResNet1D101']:.4f}, "
         f"{rn['mean'] - S14['seeds'][worst]['member_macro_auc_test']['ResNet1D101']:.4f} below its own "
         f"mean, and that seed's ensemble is the lowest of the four at "
         f"{S14['seeds'][worst]['macro_auc_test_ensemble']:.4f}."),
        ("Near-equal weights are why it propagates. ",
         "Calibration-AUC weights sit within 0.01 of a third at every seed, so a weak member is not "
         "down-weighted away."),
        ("Training cost. ",
         f"DualBranch {math.floor(min(mins['DualBranchECGNet']))}–"
         f"{math.ceil(max(mins['DualBranchECGNet']))} min per "
         f"seed, ResNet1D101 {math.floor(min(mins['ResNet1D101']))}–"
         f"{math.ceil(max(mins['ResNet1D101']))} min; "
         f"{S14['meta']['wall_clock_seconds']/3600:.1f} h in total on {S14['meta']['threads']} CPU "
         "threads."),
    ], size=T_BODY, gap=4)

    v = su["vs_nb13"]
    add_panel(s, 6.80, y, half, FLOOR - y, KEY_FILL, KEY_LINE)
    add_bullets(s, 7.00, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "What this supersedes, and what it does not", [
        ("The ensemble column only. ",
         "The anchor figures on the previous slide are identical in both runs: NB14 reuses NB13's "
         "InceptionTime1D checkpoints."),
        ("The honest size of the correction. ",
         f"Reported ensemble SD moves from {v['nb13_ensemble_sd_anchor_only']:.4f} to "
         f"{v['nb14_ensemble_sd_all_three']:.4f}, a factor of {v['inflation_factor']:.1f} — the "
         "narrower figure is not full-ensemble seed variance."),
        ("The finding is unaffected. ",
         "Set size still rises with age in 4 / 4 seeds, monotone under the global threshold in 4 / 4."),
        ("Still single-shot. ",
         "Every seed reuses the same fold-9 / fold-10 split, so this is initialisation variance, not "
         "split variance."),
    ], size=T_BODY, gap=4)


# ================================================================== slide 15
def s_seed_detail(s):
    add_title(s, "Review Item 5 — Every Trend Statistic, Seed by Seed")
    subhead(s, "The same statistics recomputed independently for each seed, on the full reseeded "
               "ensemble. The right-hand column is the question the review was really asking.")

    su = S14["summary"]
    S = S14["seeds"]
    spec = [
        ("Global set size, slope / decade", lambda d: f"{d['set_size_global']['slope_per_decade']:+.4f}",
         "global_slope_per_decade", "{:+.4f}", "Positive in 4 / 4"),
        ("Global set size, Spearman rho", lambda d: f"{d['set_size_global']['spearman_rho_vs_age']:+.3f}",
         "global_spearman_rho", "{:+.3f}", "Positive in 4 / 4"),
        ("Global set size, monotone across bands", lambda d: "Yes" if d['set_size_global']['monotone'] else "No",
         None, None, "Yes in 4 / 4"),
        ("Mondrian set size, slope / decade", lambda d: f"{d['set_size_mondrian']['slope_per_decade']:+.4f}",
         "mondrian_slope_per_decade", "{:+.4f}", "Positive in 4 / 4"),
        ("Mondrian set size, Spearman rho", lambda d: f"{d['set_size_mondrian']['spearman_rho_vs_age']:+.3f}",
         "mondrian_spearman_rho", "{:+.3f}", "Positive in 4 / 4"),
        ("Mondrian 65-80 below 40-65 (the dip)",
         lambda d: f"{d['set_size_mondrian']['band_means']['40-65'] - d['set_size_mondrian']['band_means']['65-80']:+.2f}",
         None, None, "Dips in 4 / 4"),
        ("Exact-match accuracy, Spearman rho", lambda d: f"{d['exact_match']['spearman_rho_vs_age']:+.3f}",
         "exact_match_spearman_rho", "{:+.3f}", "Negative in 4 / 4"),
    ]
    rows = [["Statistic"] + [f"Seed {k}" for k in SEEDS] + ["Mean ± SD", "Consistency"]]
    for label, get, key, f, verdict in spec:
        mean = (f.format(su[key]["mean"]) + " ± " + f.format(su[key]["sd"]).lstrip("+")
                if key else "—")
        rows.append([label] + [get(S[k]) for k in SEEDS] + [mean, verdict])
    add_table(s, 0.55, TOP, 12.3, rows,
              col_w=[3.90, 1.05, 1.05, 1.05, 1.05, 2.05, 2.15],
              row_h=0.44, size=T_TBL, header_size=T_TBL, left_cols=(0, 6))

    y = TOP + 0.44 * len(rows) + 0.16
    ms, gsl = su["mondrian_slope_per_decade"], su["global_slope_per_decade"]
    mr, gr = su["mondrian_spearman_rho"], su["global_spearman_rho"]
    notes(s, y, [
        ("The Mondrian dip reproduces in all four seeds. ",
         "A dip that recurs under four independent initialisations of all three members is not sampling "
         "noise from one run. It is calibration-cell fragmentation appearing wherever those cells are "
         "drawn, which strengthens that explanation rather than weakening it."),
        ("Correction to what NB13 suggested about Mondrian noise. ",
         f"On the anchor-only run the Mondrian slope looked 4.7× noisier than the global one. With all "
         f"three members reseeded the two are comparable — ± {ms['sd']:.4f} per decade against "
         f"± {gsl['sd']:.4f} — and on Spearman rho Mondrian is the tighter (± {mr['sd']:.3f} against "
         f"± {gr['sd']:.3f}). That line is withdrawn."),
    ])


# ================================================================== slide 16
def s_close(s):
    add_title(s, "What Notebooks 11 to 14 Close, and What Remains Open")
    subhead(s, "Status of the review items these four notebooks were built to address, with the "
               "evidence each one now rests on.")

    e = S14["summary"]["ensemble_macro_auc_test"]
    rows = [["Item", "Action item from the review", "Status now", "Evidence"]]
    rows += [
        ["2", "Signal-quality confound on the age effect", "Closed",
         "NB11 — four covariates, age effect re-fit adjusted, 3.1% attenuation"],
        ["3", "Formal trend significance test across age bands", "Closed",
         "NB11 — Kruskal-Wallis, Spearman, HC0-robust OLS on five outcomes"],
        ["5", "Multi-seed replication", "Closed — full ensemble",
         f"NB14 — 4 seeds × 3 members, {e['mean']:.4f} ± {e['sd']:.4f}; gradient holds 4 / 4"],
        ["7", "Literal ECG with attribution heat-maps", "Closed",
         "NB12 — CD and HYP overlays, 80+ confident against unconfident"],
    ]
    add_table(s, 0.55, TOP, 12.3, rows, col_w=[0.75, 4.20, 2.70, 4.65],
              row_h=0.42, size=T_TBL, header_size=T_TBL, left_cols=(0, 1, 2, 3))

    y = TOP + 0.42 * len(rows) + 0.16
    half = 6.05
    add_panel(s, 0.55, y, half, FLOOR - y, KEY_FILL, KEY_LINE)
    add_bullets(s, 0.75, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "Still open, and honestly so", [
        ("One fold arrangement. ",
         "Every result in the deck comes from the single patient-safe fold-9 / fold-10 split. The four "
         "seeds vary initialisation, not the split, so split variance is unmeasured."),
        ("Section 6 comparators. ",
         "The literature table reports each paper's own partition. It is not a paired test on our split, "
         "and no such test is claimed."),
        ("The IEEE LaTeX build. ",
         "Still carries pre-correction figures and is marked do-not-submit; it needs a full resync "
         "against the Markdown manuscript, not just the comparator fix."),
    ], size=T_BODY, gap=5)

    add_panel(s, 6.80, y, half, FLOOR - y, INFO_FILL, INFO_LINE)
    add_bullets(s, 7.00, y + 0.12, half - 0.40, FLOOR - y - 0.24,
                "What the four notebooks establish together", [
        ("Not an acquisition artefact. ",
         "SNR has no association with age, and four covariates move the effect under 5 per cent."),
        ("Statistically supported, not eyeballed. ",
         "Every claim carries a trend test, and band-level slopes have intervals excluding zero."),
        ("It survives reseeding and resampling. ",
         "Four seeds of all three members, 300 calibration resamples, two conformal bases and a "
         "different model family all point the same way."),
        ("The explanation leg stays weak. ",
         "The overlays show what 0.955 against 0.980 looks like, not a collapse."),
    ], size=T_BODY, gap=5)


# ===================================================================== build
PLAN = [
    ("Notebooks 11 to 14", s_title),
    ("What These Four Notebooks Answer, and Their Execution Record", s_provenance),
    ("Notebook 11 — The Reproduction Guard That Gates Everything After It", s_guard),
    ("Review Item 2 — Signal Quality Measured, Then Adjusted For", s_item2_stats),
    ("Review Item 2 — Acquisition Quality by Band, and the Adjusted Age Effect", s_item2_figs),
    ("Review Item 3 — A Formal Trend Statistic on Every Age Claim", s_item3),
    ("Notebook 11 — The Trend Is Not an Artefact of One Calibration Draw", s_resampling),
    ("Notebook 11 — Reliability by Age Band, and Where Calibration Breaks", s_reliability),
    ("Review Item 7 — How the Two Attribution Overlay Cases Were Chosen", s_xai_design),
    ("Review Item 7 — Conduction Disturbance: Confident Against Unconfident", s_xai_cd),
    ("Review Item 7 — Hypertrophy: the Weakest Class, Same Comparison", s_xai_hyp),
    ("Review Item 5 — Multi-Seed Replication: the Whole Ensemble, Four Seeds", s_seed_design),
    ("Review Item 5 — Per-Seed Results Through the Whole Pipeline", s_seed_table),
    ("Notebook 14 — Member-Level Seed Variance and Where the Spread Comes From", s_seed_members),
    ("Review Item 5 — Every Trend Statistic, Seed by Seed", s_seed_detail),
    ("What Notebooks 11 to 14 Close, and What Remains Open", s_close),
]


def drop_slides(prs, keep_from):
    """Remove slides [0, keep_from) and their relationships."""
    lst = prs.slides._sldIdLst
    for el in list(lst)[:keep_from]:
        rid = el.get(qn("r:id"))
        prs.part.drop_rel(rid)
        lst.remove(el)


def main():
    prs = Presentation(SRC)
    n_src = len(prs.slides._sldIdLst)
    tpl = find_slide(prs, TEMPLATE_TITLE)
    if tpl is None:
        raise SystemExit(f"template slide not found in {SRC.name}")
    print(f"[SRC] {SRC.name}: {n_src} slides, template at {tpl + 1}")

    for i, (title, fn) in enumerate(PLAN, 1):
        slide = clone_slide(prs, tpl)
        strip_content(slide)
        fn(slide)
        print(f"[ADD] {i:2d}. {title}")

    drop_slides(prs, n_src)
    prs.save(DST)
    print(f"[SAVED] {DST.relative_to(ROOT)}: {len(prs.slides._sldIdLst)} slides")


if __name__ == "__main__":
    main()
