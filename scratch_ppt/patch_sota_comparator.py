"""Correct the deck's SOTA comparator from 0.929 to the superdiagnostic 0.934.

The deck claims parity with "published SOTA (0.9290)" on slides 32, 36, 40, the master
literature table on 55 and the closing Summary on 76. The manuscript
(papers/Draft_Paper.md, abstract and section 2) establishes that 0.929 is Strodthoff
et al.'s result on the **"all" task** (71 labels), not on the five-class
**superdiagnostic** task this project actually performs. On superdiagnostic their
six-model ensemble reaches 0.934(05) and their best single model, resnet1d_wang,
0.930(06); Gitau et al.'s independent reproduction gives 0.934 for the ensemble and
0.918-0.929 for the single models.

That changes a headline claim. Against the correct comparator our ensemble's interval,
0.9245 [0.9176, 0.9315], does **not** contain 0.934 — so "statistical parity with SOTA"
is wrong as stated. It is parity with the best published *single* models, and a real
~0.007 gap to the published ensembles. This script rewrites the deck to say that,
matching the manuscript, and leaves run-level formatting intact.

Slide 74 documents this correction and must keep quoting the wrong figure, so its five
mentions of 0.929 are deliberately left alone.

History / why this script was rewritten
---------------------------------------
The first version targeted presentations/Tandon's_Draft.pptx (67 slides, shapes named
"TextBox N") and printed "[SKIP] ... not found (already corrected?)" when an edit missed.
The live deck is "all 13.pptx" (76 slides, shapes named "Google Shape;NNN;pNN"), so every
edit missed and the skip messages read like success — slide 74 then claimed a propagation
that had never happened. Unmatched edits now raise instead of being skipped, and an edit
is only treated as already-applied when its DONE marker is actually present.

Re-runnable: each edit is located by a substring and skipped only if its DONE marker is
already in place.
"""
from __future__ import annotations

import copy
import shutil
import sys
from datetime import datetime
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parent.parent
DECK = ROOT / "presentations" / "all 13.pptx"

SINGLE_RANGE = "0.918–0.930"
ENS = "0.934"

# Paragraph rewrites.
#   slide      1-based slide number
#   shape      exact shape name
#   para       0-based paragraph index within the shape
#   needle     substring that must be present in the stale paragraph
#   done       substring that is present once the edit has been applied
#   runs       {run index: new text} — every other run keeps its text and formatting
EDITS = [
    # Slide 32 — "Part 8 · Next Steps & Roadmap", Tier 2 bullet.
    # The whole bullet is one bold 15pt run, so run 0 carries the bullet glyph too.
    dict(slide=32, shape="Google Shape;468;p35", para=3,
         needle="leaving ~0.002 to Strodthoff",
         done="six-model ensemble at " + ENS,
         runs={0: "▪  Tier 2 — CLOSED without the Gitau retrain: NB08’s optimised "
                  "weighting + TTA reaches 0.9269 [0.9199–0.9337], a +0.0025 gain whose CI "
                  "excludes zero. Against the correct superdiagnostic comparator — "
                  "Strodthoff’s six-model ensemble at " + ENS + " — about 0.007 AUC "
                  "remains; the 0.929 quoted earlier is their all-task (71-label) figure "
                  "and is not comparable."}),

    # Slide 36 — "Executive Summary", third headline finding.
    # run 0 = "▪  ", run 1 = bold lead, run 2 = body.
    dict(slide=36, shape="Google Shape;513;p39", para=3,
         needle="Statistical parity with SOTA",
         done="below the published ensembles",
         runs={1: "On par with published single models, below the published ensembles. ",
               2: "Paired bootstrap puts our ensemble at 0.9245 [0.9176–0.9315] and "
                  "NB08’s optimised blend at 0.9269 [0.9199–0.9337], a +0.0025 gain whose "
                  "CI excludes zero. That matches the best published single models ("
                  + SINGLE_RANGE + ") but sits ~0.007 below the six-model ensembles of "
                  "Strodthoff and Gitau, both " + ENS + " on superdiagnostic. The 0.929 "
                  "shown earlier is their all-task result (NB07–08)."}),

    # Slide 40 — NB08 subtitle. run 0 = bold question, run 1 = body.
    dict(slide=40, shape="Google Shape;558;p43", para=0,
         needle="close the 0.0045 AUC gap to published SOTA",
         done="published superdiagnostic ensemble",
         runs={1: "Notebook 08 deepened InceptionTime1D from depth 4 to depth 9 and added "
                  "concatenated global pooling to test whether it could close the gap to "
                  "the published superdiagnostic ensemble (" + ENS + ")."}),

    # Slide 40 — NB08 conclusion panel. run 0 = "–  ", run 1 = bold lead, run 2 = body.
    dict(slide=40, shape="Google Shape;564;p43", para=0,
         needle="The remaining ~0.002 to published SOTA",
         done="correct superdiagnostic comparator",
         runs={2: "added depth is not what closes the gap — optimised blending is. "
                  "Against the correct superdiagnostic comparator (" + ENS + ") roughly "
                  "0.007 AUC remains, though our interval does cover the best published "
                  "single models (" + SINGLE_RANGE + ")."}),

    # Slide 76 — "Summary — Final Conclusions", first conclusion.
    # run 0 = "▪  ", run 1 = bold lead, run 2 = body.
    dict(slide=76, shape="Google Shape;758;p60", para=1,
         needle="Parity on raw accuracy",
         done="On par with published single models",
         runs={1: "On par with published single models. ",
               2: "Our ensemble reaches 0.9245 [0.9176–0.9315] and the NB08 blend 0.9269 "
                  "[0.9199–0.9337] — within the range of the best published single models "
                  "(" + SINGLE_RANGE + "), and about 0.007 below the six-model ensembles "
                  "of Strodthoff and Gitau, both " + ENS + " on the superdiagnostic task."}),
]

# Slide 55 master literature table. The AUC column is 1.55in at 13pt and its rows are
# 0.63in, which fits two short lines — hence the terse "ens." / "single" split rather
# than "0.9340 (6-model ens., superdiag.)", which would wrap to four lines and push the
# table into the caption below it.
#   (row, col, [paragraph texts]) — extra paragraphs are cloned from the first so they
#   inherit its size, weight and alignment.
TABLE_CELLS = [
    (1, 1, ["6-model ens. / resnet1d_wang"]),
    (1, 3, ["0.9340 ens.", "0.9300 single"]),
    (2, 3, ["0.9340 ens.", "0.918–0.929"]),
]

CAPTION = (55, "Google Shape;737;p58",
           "matches published SOTA",
           "on par with the best published single models",
           "Our pipeline is on par with the best published single models — 0.9245 "
           "[0.918–0.932] against " + SINGLE_RANGE + " — and ~0.007 below the six-model "
           "ensembles of Strodthoff and Gitau, both " + ENS + " on superdiagnostic; the "
           "0.929 shown here previously was their all-task result. We remain the first "
           "PTB-XL benchmark to add age-conditional Mondrian conformal sets. Figures are "
           "1000× bootstrap on the held-out TEST split (N = 2,198); “—” means not "
           "reported.")


class EditMissed(RuntimeError):
    """An edit matched neither its needle nor its done marker — fail loudly."""


def find_shape(slide, name):
    for sh in slide.shapes:
        if sh.name == name:
            return sh
    raise EditMissed(f"shape {name!r} not on slide")


def set_runs(pa, runs):
    """Rewrite selected runs in place, keeping every run's own formatting."""
    for idx, text in runs.items():
        if idx >= len(pa.runs):
            raise EditMissed(f"run {idx} missing (paragraph has {len(pa.runs)} runs)")
        pa.runs[idx].text = text


def set_cell(cell, texts):
    """Set a table cell to one paragraph per line, cloning the first for extra lines."""
    tf = cell.text_frame
    first = tf.paragraphs[0]
    # Drop any paragraph past the first, then re-clone to the requested count.
    for pa in list(tf.paragraphs)[1:]:
        pa._p.getparent().remove(pa._p)
    for _ in range(len(texts) - 1):
        tf._txBody.append(copy.deepcopy(first._p))
    for pa, text in zip(tf.paragraphs, texts):
        pa.runs[0].text = text
        for r in pa.runs[1:]:
            r.text = ""


def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = DECK.with_suffix(DECK.suffix + f".bak_pre_sota_{stamp}")
    shutil.copy2(DECK, bak)
    print(f"[BACKUP] {bak.name}")

    prs = Presentation(DECK)
    applied, already, missed = 0, 0, []

    for e in EDITS:
        tag = f"slide {e['slide']} {e['shape']} p{e['para']}"
        try:
            sh = find_shape(prs.slides[e["slide"] - 1], e["shape"])
            pa = sh.text_frame.paragraphs[e["para"]]
            if e["done"] in pa.text:
                print(f"[OK   ] {tag}: already corrected")
                already += 1
                continue
            if e["needle"] not in pa.text:
                raise EditMissed(f"needle {e['needle']!r} not in {pa.text[:80]!r}")
            set_runs(pa, e["runs"])
            print(f"[EDIT ] {tag}")
            applied += 1
        except (EditMissed, IndexError) as exc:
            print(f"[MISS ] {tag}: {exc}")
            missed.append(tag)

    sl = TABLE_CELLS and 55
    table = next((sh.table for sh in prs.slides[sl - 1].shapes if sh.has_table), None)
    if table is None:
        missed.append(f"slide {sl} table")
        print(f"[MISS ] slide {sl}: no table found")
    else:
        for ri, ci, texts in TABLE_CELLS:
            cell = table.cell(ri, ci)
            if cell.text.replace("\n", " ") == " ".join(texts):
                print(f"[OK   ] slide {sl} table[{ri}][{ci}]: already corrected")
                already += 1
                continue
            old = cell.text
            set_cell(cell, texts)
            print(f"[EDIT ] slide {sl} table[{ri}][{ci}]: {old!r} -> {' / '.join(texts)!r}")
            applied += 1

    sl, shape_name, needle, done, text = CAPTION
    tag = f"slide {sl} {shape_name} caption"
    try:
        pa = find_shape(prs.slides[sl - 1], shape_name).text_frame.paragraphs[0]
        if done in pa.text:
            print(f"[OK   ] {tag}: already corrected")
            already += 1
        elif needle not in pa.text:
            raise EditMissed(f"needle {needle!r} not in {pa.text[:80]!r}")
        else:
            pa.runs[0].text = text
            for r in pa.runs[1:]:
                r.text = ""
            print(f"[EDIT ] {tag}")
            applied += 1
    except (EditMissed, IndexError) as exc:
        print(f"[MISS ] {tag}: {exc}")
        missed.append(tag)

    if missed:
        print(f"\n[ABORT] {len(missed)} edit(s) found no target; deck NOT saved:")
        for m in missed:
            print(f"         - {m}")
        print(f"         restore point: {bak.name}")
        return 1

    prs.save(DECK)
    print(f"\n[SAVED] {DECK.name}: {applied} applied, {already} already correct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
