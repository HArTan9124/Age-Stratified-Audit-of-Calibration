"""Insert / refresh the Notebook 03b (HYP augmentation ablation) section in the
ECG_PTBXL_NB01-10_Complete deck.

Reads the real numbers from outputs/dataset_validation/03b_hyp_augmentation_ablation.json
and the training trace from 03b_ablation_run.log, then rebuilds three slides:

  A · Notebook 03b - Ablation Design: What Was Controlled      (new)
  B · Notebook 03b - HYP Augmentation Ablation: ON vs OFF      (existing chart slide, text refreshed)
  C · Notebook 03b - Complete Ablation Summary (Notebook Output)  (new - the full SUMMARY table)

It also patches the NB03 "Six Fixes" table and the Part 4 divider so the deck's
NB03 narrative matches the ablation result, and back-fills the slide-number
placeholder on the NB07-10 slides that were built without one.

Re-runnable: slides A and C are located by title and rebuilt in place.
"""
import copy
import json
import shutil
from datetime import datetime
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "outputs" / "dataset_validation"
DECK = ROOT / "presentations" / "ECG_PTBXL_NB01-10_Complete.pptx"

# ---- deck template constants (sampled from the existing slides) -------------
TNR = "Times New Roman"
RED = RGBColor(0xA0, 0x00, 0x20)      # title / key-finding accent
NAVY = RGBColor(0x1F, 0x38, 0x64)     # sub-heading
BODY = RGBColor(0x20, 0x38, 0x64)     # body copy
INK = RGBColor(0x1A, 0x1A, 0x2E)      # table ink / header fill
MUTE = RGBColor(0x5A, 0x5A, 0x66)     # captions
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BAND = RGBColor(0xF0, 0xEF, 0xEA)     # table zebra band
INFO_FILL, INFO_LINE = RGBColor(0xF4, 0xF6, 0xFA), RGBColor(0xDA, 0xE0, 0xEC)
KEY_FILL, KEY_LINE = RGBColor(0xFB, 0xF3, 0xD5), RGBColor(0xE7, 0xD9, 0xA8)
POS = RGBColor(0x1F, 0x6F, 0x43)      # positive delta
NEG = RGBColor(0xA0, 0x00, 0x20)      # negative delta

TITLE_BOX = (Inches(0.5), Inches(0.28), Inches(12.33), Inches(0.8))

TITLE_A = "Notebook 03b — Ablation Design: What Was Controlled"
TITLE_B = "Notebook 03b — HYP Augmentation Ablation: ON vs OFF"
TITLE_C = "Notebook 03b — Complete Ablation Summary (Notebook Output)"


# ---------------------------------------------------------------- slide utils
def slide_title(slide):
    """First real text on the slide — the page-number field is not a title."""
    for sh in slide.shapes:
        if sh.is_placeholder or not sh.has_text_frame:
            continue
        txt = sh.text_frame.text.strip()
        if txt:
            return txt
    return ""


def find_slide(prs, title):
    for i, s in enumerate(prs.slides):
        if slide_title(s) == title:
            return i
    return None


def move_slide(prs, old, new):
    lst = prs.slides._sldIdLst
    sld = list(lst)[old]
    lst.remove(sld)
    lst.insert(new, sld)


def get_or_create(prs, title, template_idx, target_pos):
    """Return an emptied slide for `title`, reusing it if the deck already has one.

    Rebuilding in place (rather than delete + re-add) keeps the package free of
    orphaned slide parts, which is what makes this script safe to re-run.
    """
    idx = find_slide(prs, title)
    reused = idx is not None
    if not reused:
        clone_slide(prs, template_idx)
        move_slide(prs, len(prs.slides._sldIdLst) - 1, target_pos)
        idx = target_pos
    slide = prs.slides[idx]
    strip_content(slide)
    return idx, slide, reused


def clone_slide(prs, src_idx):
    """Append a copy of slide `src_idx` (background art, logo, page number)."""
    src = prs.slides[src_idx]
    dest = prs.slides.add_slide(src.slide_layout)
    for sh in list(dest.shapes):
        sh._element.getparent().remove(sh._element)

    rid_map = {}
    for rid, rel in src.part.rels.items():
        if rel.reltype.endswith("slideLayout"):
            continue
        if rel.is_external:
            rid_map[rid] = dest.part.rels.get_or_add_ext_rel(rel.reltype, rel.target_ref)
        else:
            rid_map[rid] = dest.part.relate_to(rel.target_part, rel.reltype)

    tree = dest.shapes._spTree
    for sh in src.shapes:
        el = copy.deepcopy(sh._element)
        for node in el.iter():
            for attr in (qn("r:embed"), qn("r:link"), qn("r:id")):
                if attr in node.attrib and node.attrib[attr] in rid_map:
                    node.attrib[attr] = rid_map[node.attrib[attr]]
        tree.append(el)
    return dest


def strip_content(slide):
    """Keep only the background art, the corner logo and the page-number field.

    The background sits above the top edge and the logo in the bottom-right
    corner; any other picture is slide content and goes.
    """
    for sh in list(slide.shapes):
        if sh.is_placeholder:
            continue
        if sh.shape_type == MSO_SHAPE_TYPE.PICTURE and (
                sh.top < 0 or sh.left > Inches(11)):
            continue
        sh._element.getparent().remove(sh._element)


# ---------------------------------------------------------------- text utils
def _run(p, text, size, color=BODY, bold=False, italic=False):
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.name = TNR
    r.font.color.rgb = color
    return r


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    return tb, tf


def add_title(slide, text):
    tb, tf = textbox(slide, 0.5, 0.28, 12.33, 0.8, MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    _run(p, text, 24, RED, bold=True)


def add_subhead(slide, text, y=1.16, size=13):
    tb, tf = textbox(slide, 0.6, y, 12.13, 0.32, MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], text, size, NAVY, bold=True)


def add_caption(slide, text, x, y, w, size=10):
    tb, tf = textbox(slide, x, y, w, 0.3, MSO_ANCHOR.MIDDLE)
    _run(tf.paragraphs[0], text, size, MUTE, italic=True)


def add_panel(slide, x, y, w, h, fill, line):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                 Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = line
    box.line.width = Pt(1)
    box.shadow.inherit = False
    box.text_frame.text = ""
    return box


def add_bullets(slide, x, y, w, h, heading, items, size=11, gap=5,
                anchor=MSO_ANCHOR.TOP):
    """items: list of (lead, rest) - lead is rendered bold."""
    tb, tf = textbox(slide, x, y, w, h, anchor)
    first = True
    if heading:
        p = tf.paragraphs[0]
        _run(p, heading, size, NAVY, bold=True)
        p.space_after = Pt(gap)
        first = False
    for lead, rest in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        _run(p, "▪  ", size, RED, bold=True)
        if lead:
            _run(p, lead, size, NAVY, bold=True)
        _run(p, rest, size, BODY)
        p.space_after = Pt(gap)
    return tb


# ---------------------------------------------------------------- table utils
def add_table(slide, x, y, w, rows, col_w, row_h=0.3, size=11,
              header_size=11, delta_col=None, left_cols=(0,)):
    n_r, n_c = len(rows), len(rows[0])
    gf = slide.shapes.add_table(n_r, n_c, Inches(x), Inches(y), Inches(w),
                                Inches(row_h * n_r))
    tbl = gf.table
    for ci, cw in enumerate(col_w):
        tbl.columns[ci].width = Inches(cw)
    for ri in range(n_r):
        tbl.rows[ri].height = Inches(row_h)
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = tbl.cell(ri, ci)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = (INK if ri == 0
                                        else (WHITE if ri % 2 else BAND))
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if ci in left_cols else PP_ALIGN.CENTER
            colour = WHITE if ri == 0 else INK
            bold = ri == 0
            if ri and delta_col is not None and ci == delta_col:
                colour = NEG if str(val).startswith("−") else POS
                bold = True
            _run(p, str(val), header_size if ri == 0 else size, colour, bold=bold)
    # neutral table style (no theme banding on top of the explicit fills)
    for el in tbl._tbl.findall(qn("a:tblPr")):
        el.set("firstRow", "1")
        el.set("bandRow", "0")
        for sid in el.findall(qn("a:tableStyleId")):
            sid.text = "{5C22544A-7EE6-4342-B048-85BDC9FD1C3A}"
    return tbl


# ---------------------------------------------------------------- data
def load_results():
    res = json.loads((VAL / "03b_hyp_augmentation_ablation.json").read_text())
    on, off = res["aug_on"], res["aug_off"]
    order = [("macro_auc", on["macro_auc"], off["macro_auc"]),
             ("macro_auprc", on["macro_auprc"], off["macro_auprc"]),
             ("fmax", on["fmax"], off["fmax"])]
    for cls in ("NORM", "MI", "STTC", "CD", "HYP"):
        order.append((f"{cls}-AUC", on["per_class"][cls]["auc"],
                      off["per_class"][cls]["auc"]))
        order.append((f"{cls}-AUPRC", on["per_class"][cls]["auprc"],
                      off["per_class"][cls]["auprc"]))
    return on, off, order


def fmt_delta(v):
    """Match the notebook's 4-dp print, with a typographic minus sign."""
    s = f"{v:+.4f}"
    return s.replace("-", "−")


# ---------------------------------------------------------------- slide A
def build_design_slide(s):
    add_title(s, TITLE_A)
    add_subhead(s, "Ablation of NB03 Fix #2 — the Gaussian-noise augmentation is the "
                   "only variable that differs between the two arms")

    add_panel(s, 0.55, 1.55, 5.95, 2.00, INFO_FILL, INFO_LINE)
    add_bullets(s, 0.72, 1.66, 5.62, 1.80, "WHY THIS NOTEBOOK EXISTS", [
        ("", "NB03 adds Fix #2 — Gaussian noise (σ = 0.01) on every HYP-positive "
             "training record — and presents it as the remedy for HYP's weak per-class AUC."),
        ("", "The fix was never ablated: no run existed without it, so the claimed "
             "benefit was asserted rather than measured."),
        ("", "Atwa et al. (2025), the source of the recipe, explicitly caution that this "
             "noise augmentation can degrade performance — which makes the missing "
             "control a substantive gap, not a formality."),
    ], size=10.5, gap=4)

    add_panel(s, 6.83, 1.55, 5.95, 2.00, INFO_FILL, INFO_LINE)
    add_bullets(s, 7.00, 1.66, 5.62, 1.80, "THE ONLY VARIABLE THAT CHANGES", [
        ("AUG-ON · ", "every HYP-positive record is duplicated with additive Gaussian "
                      "noise σ = 0.01 — TRAIN grows 17,418 → 19,537 (+2,119 HYP copies)."),
        ("AUG-OFF · ", "the identical pipeline trained on the un-augmented 17,418 "
                       "records. Nothing else differs."),
        ("Checkpoints · ", "AUG-ON reuses NB03's trained model "
                           "(05_optimized_InceptionTime1D_best.pth); AUG-OFF is a fresh "
                           "run saved to 03b_ablation_noaug_InceptionTime1D_best.pth."),
    ], size=10.5, gap=4)

    rows = [
        ("Component", "Held identical in both arms"),
        ("Architecture", "InceptionTime1D — 4 Inception blocks, 32 filters per branch, "
                         "kernels 9 / 19 / 39 + max-pool branch, GAP → dropout 0.3 → 5 logits"),
        ("Initialisation", "Global seed 42; torch.manual_seed(42) re-applied immediately "
                           "before each arm, so both runs start from identical weights"),
        ("Loss", "BCEWithLogitsLoss with per-class pos_weight = neg / pos, recomputed "
                 "from each arm's own training set"),
        ("Optimisation", "AdamW (lr 1e-3, wd 1e-4), batch 64, ≤ 10 epochs, EarlyStopping "
                         "patience 4, ReduceLROnPlateau on VAL macro-AUC (factor 0.5, patience 2)"),
        ("Splits", "Patient-disjoint NB01 folds — TRAIN 17,418 · VAL 1,080 · TEST 2,198 "
                   "(fold 10, untouched during training and model selection)"),
        ("Model selection", "Checkpoint of the best VAL macro-AUC epoch — AUG-OFF peaked "
                            "at 0.9268 (epoch 8 of 10, 5,190 s on CPU)"),
        ("Metrics", "macro-AUC · macro-AUPRC · Fmax (threshold swept 0.05–0.95 in 0.01 "
                    "steps) · per-class AUC and AUPRC — all on the TEST split"),
    ]
    add_table(s, 0.55, 3.68, 12.23, rows, [2.25, 9.98], row_h=0.32,
              size=10.5, header_size=11, left_cols=(0, 1))
    add_caption(s, "Because every other component is pinned, any TEST-set difference "
                   "between the two arms is attributable to the augmentation alone.",
                0.6, 6.34, 12.13)
    return s


# ---------------------------------------------------------------- slide C
def build_summary_slide(s, order):
    add_title(s, TITLE_C)
    add_subhead(s, "The verbatim SUMMARY block printed by "
                   "03b_hyp_augmentation_ablation.ipynb — held-out TEST split, N = 2,198")

    rows = [("Metric", "AUG-ON", "AUG-OFF", "Δ (ON − OFF)")]
    for name, v_on, v_off in order:
        rows.append((name, f"{v_on:.4f}", f"{v_off:.4f}", fmt_delta(v_on - v_off)))
    add_table(s, 0.55, 1.58, 6.05, rows, [2.05, 1.25, 1.30, 1.45],
              row_h=0.295, size=10.5, header_size=10.5, delta_col=3)

    add_panel(s, 6.90, 1.58, 5.88, 4.13, INFO_FILL, INFO_LINE)
    add_bullets(s, 7.07, 1.70, 5.54, 3.90, "HOW TO READ THIS TABLE", [
        ("Ranking quality is unchanged. ",
         "macro-AUC −0.0006 and macro-AUPRC −0.0002 are an order of magnitude "
         "smaller than NB07's bootstrap interval on the headline AUC (±0.008). "
         "On everything AUC measures, the two arms are the same model."),
        ("Only the thresholded metric favours the augmentation. ",
         "Fmax +0.0157. The extra noisy copies flatten the probability surface so a "
         "single global threshold sweep lands better — a decision-threshold effect, "
         "not better ranking."),
        ("HYP — the class the augmentation exists to rescue — is worse with it on. ",
         "HYP-AUC −0.0073 and HYP-AUPRC −0.0078. Its nearest neighbour CD also "
         "degrades (−0.0057 AUC, −0.0136 AUPRC)."),
        ("The gains land on the wrong classes. ",
         "MI (+0.0046 AUC) and STTC (+0.0055 AUC, +0.0138 AUPRC) improve — classes "
         "that were never augmented. That pattern is the signature of run-to-run "
         "variation, not of a targeted minority-class fix."),
    ], size=11.5, gap=9)

    add_panel(s, 0.55, 5.88, 12.23, 1.02, KEY_FILL, KEY_LINE)
    tb, tf = textbox(s, 0.73, 5.96, 11.87, 0.86, MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _run(p, "Verdict. ", 12, RED, bold=True)
    _run(p, "Across every ranking metric the HYP augmentation is neutral-to-harmful, "
            "so NB03 Fix #2 cannot be presented as the reason HYP works. The honest "
            "reading is weaker still: at a single seed on one split, every delta sits "
            "inside NB07's per-class bootstrap intervals (HYP-AUC 95% CI width 0.042), "
            "so this is ", 12, BODY)
    _run(p, "no evidence of benefit", 12, RED, bold=True)
    _run(p, " rather than proof of harm. The augmentation is retained only for the "
            "+0.0157 Fmax it buys at the decision-threshold stage.", 12, BODY)
    return s


# ---------------------------------------------------------------- slide B text
def refresh_chart_slide(prs, idx, on, off):
    s = prs.slides[idx]
    caption_box = bullet_box = None
    for sh in s.shapes:
        if not sh.has_text_frame:
            continue
        t = sh.text_frame.text
        if t.startswith("Ablation of NB03 Fix #2"):
            caption_box = sh
        elif t.startswith("1 · AUG-ON"):
            bullet_box = sh
    if caption_box is not None:
        tf = caption_box.text_frame
        tf.clear()
        _run(tf.paragraphs[0],
             "Per-class ROC-AUC, both arms — identical InceptionTime1D pipeline, "
             "held-out TEST split, N = 2,198", 13, NAVY, bold=True)
    if bullet_box is not None:
        tf = bullet_box.text_frame
        tf.clear()
        lines = [
            ("1 · AUG-ON (NB03 recipe — Gaussian noise σ = 0.01 on HYP-positive records):  "
             f"macro-AUC {on['macro_auc']:.4f} · macro-AUPRC {on['macro_auprc']:.4f} · "
             f"Fmax {on['fmax']:.4f} · HYP-AUC {on['per_class']['HYP']['auc']:.4f}", False),
            ("2 · AUG-OFF (identical pipeline, no augmentation):  "
             f"macro-AUC {off['macro_auc']:.4f} · macro-AUPRC {off['macro_auprc']:.4f} · "
             f"Fmax {off['fmax']:.4f} · HYP-AUC {off['per_class']['HYP']['auc']:.4f}  "
             "(same seed, architecture and epoch budget)", False),
            ("3 · Macro-AUC and macro-AUPRC are effectively identical (Δ ≤ 0.0006, far "
             "inside NB07's ±0.008 bootstrap interval); only Fmax favours AUG-ON (+0.0157)", False),
            ("4 · Key result — the augmentation does not fix HYP: HYP-AUC is 0.0073 lower "
             "and HYP-AUPRC 0.0078 lower with augmentation ON, matching Atwa et al.'s own "
             "caution that this noise recipe can hurt. Full metric table on the next slide.", True),
        ]
        for i, (text, is_key) in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            if i:
                p.space_before = Pt(4)
            _run(p, text, 12, RED if is_key else BODY, bold=is_key)


# ---------------------------------------------------------------- narrative fixes
def patch_nb03_fix_table(prs):
    idx = find_slide(prs, "Notebook 03 — Six Literature-Grounded Fixes")
    if idx is None:
        return False
    tbl = [sh for sh in prs.slides[idx].shapes
           if getattr(sh, "has_table", False) and sh.has_table][0].table
    cell = tbl.cell(2, 1)
    tf = cell.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    _run(p, "Gaussian noise (σ = 0.01) on every HYP-positive record — the rarest & "
            "weakest superclass (Atwa 2025, Table 4). ", 10.5, BODY)
    _run(p, "Ablated in NB03b: neutral on macro-AUC (Δ −0.0006) and −0.0073 on HYP-AUC "
            "— it does not fix HYP.", 10.5, RED, bold=True)
    return True


def patch_part4_divider(prs):
    idx = find_slide(prs, "Part 4 · Notebook 03 — Literature-Grounded Fixes & Model Selection")
    if idx is None:
        return False
    for sh in prs.slides[idx].shapes:
        if not sh.has_text_frame:
            continue
        paras = sh.text_frame.paragraphs
        texts = [p.text for p in paras]
        if not any(t.startswith("What this notebook does") for t in texts):
            continue
        # the extra bullet needs a taller body box than the original four lines
        sh.height = Inches(2.50)
        if any("NB03b" in t for t in texts):
            return True
        anchor = next(i for i, t in enumerate(texts)
                      if t.startswith("▪  Key result"))
        new_p = copy.deepcopy(paras[anchor]._p)
        paras[anchor]._p.addnext(new_p)
        from pptx.text.text import _Paragraph
        p = _Paragraph(new_p, sh.text_frame)
        for r in list(p.runs):
            r._r.getparent().remove(r._r)
        _run(p, "▪  Ablation follow-up — NB03b re-runs the identical pipeline with Fix #2 "
                "switched off and shows the HYP augmentation is not what makes the model work.",
             13, RGBColor(0xC0, 0x39, 0x2B), bold=True)
        return True
    return False


def backfill_page_numbers(prs, template_idx):
    """Slides built by the NB07-10 script have no page-number field; add one."""
    src = None
    for sh in prs.slides[template_idx].shapes:
        if sh.is_placeholder:
            src = sh._element
            break
    if src is None:
        return 0
    added = 0
    for s in prs.slides:
        if any(sh.is_placeholder for sh in s.shapes):
            continue
        s.shapes._spTree.append(copy.deepcopy(src))
        added += 1
    return added


# ---------------------------------------------------------------- main
def main():
    on, off, order = load_results()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DECK.with_suffix(DECK.suffix + f".bak_pre_03b_rebuild_{stamp}")
    shutil.copy2(DECK, backup)
    print(f"[BACKUP] {backup.name}")

    prs = Presentation(DECK)

    chart_idx = find_slide(prs, TITLE_B)
    if chart_idx is None:
        raise SystemExit("NB03b chart slide not found — aborting.")

    # design slide sits immediately before the chart slide
    a_idx, slide_a, reused = get_or_create(prs, TITLE_A, chart_idx, chart_idx)
    build_design_slide(slide_a)
    print(f"[{'REBUILD' if reused else 'ADD'}] design slide at position {a_idx + 1}")

    chart_idx = find_slide(prs, TITLE_B)
    refresh_chart_slide(prs, chart_idx, on, off)
    print(f"[EDIT] chart slide at position {chart_idx + 1}")

    # summary slide sits immediately after the chart slide
    c_idx, slide_c, reused = get_or_create(prs, TITLE_C, chart_idx, chart_idx + 1)
    build_summary_slide(slide_c, order)
    print(f"[{'REBUILD' if reused else 'ADD'}] summary slide at position {c_idx + 1}")

    print("[EDIT] NB03 fix table:", patch_nb03_fix_table(prs))
    print("[EDIT] Part 4 divider:", patch_part4_divider(prs))
    print("[EDIT] page numbers back-filled on",
          backfill_page_numbers(prs, chart_idx), "slides")

    prs.save(DECK)
    print(f"[SAVED] {DECK}  ({len(prs.slides._sldIdLst)} slides)")


if __name__ == "__main__":
    main()
