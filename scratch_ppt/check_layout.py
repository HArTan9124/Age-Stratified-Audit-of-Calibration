"""Layout audit for a PPTX: font sizes, dead vertical space, and text that will not fit.

Three checks, reported per slide:

  SMALL      body or table text below the readable floor (default 12 pt)
  SPACE      the last content shape ends well above the content floor, i.e. the slide
             has an empty band the supervisor asked us to fill with larger type
  TIGHT      a text frame holds more text than its own box height can show, estimated
             from a character-metric model calibrated against this deck's Times New Roman

Text height is estimated, not measured — PowerPoint does the real line-breaking. The
estimate is deliberately slightly pessimistic so that TIGHT flags are worth acting on.

Usage:
    .venv/bin/python scratch_ppt/check_layout.py "presentations/Tandon's_Draft.pptx"
    .venv/bin/python scratch_ppt/check_layout.py <deck> --slides 57-66
    .venv/bin/python scratch_ppt/check_layout.py <deck> --verbose
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pptx import Presentation

EMU = 914400.0
FLOOR = 7.05            # the page-number placeholder starts at 7.16
SMALL_PT = 12.0         # readable floor for body and table text
SPACE_IN = 0.90         # dead space worth filling
CHAR_W = 0.50           # mean glyph advance as a fraction of point size (Times New Roman)
LINE_H = 1.20           # line box as a multiple of point size


def near(a, b, t=0.15):
    return abs(a - b) < t


def is_decor(sh):
    """The deck's watermark band, corner logo and page-number field are not content."""
    L, T, H = sh.left / EMU, sh.top / EMU, sh.height / EMU
    if near(T, -3.13) and near(H, 13.33):
        return True
    if near(L, 11.36) and near(T, 6.71):
        return True
    if sh.has_text_frame and sh.text_frame.text.strip() == "‹#›":
        return True
    return False


def shape_bottom(sh):
    """A table's stored height is stale in this deck; its rows are the truth."""
    if sh.has_table:
        return (sh.top + sum(r.height for r in sh.table.rows)) / EMU
    return (sh.top + sh.height) / EMU


def para_runs(tf):
    for pa in tf.paragraphs:
        runs = [r for r in pa.runs if r.text.strip()]
        if runs:
            yield pa, runs


def est_text_height(tf, width_in):
    """Estimated rendered height of a text frame, in inches."""
    usable = width_in - (tf.margin_left + tf.margin_right) / EMU
    total = (tf.margin_top + tf.margin_bottom) / EMU
    for pa, runs in para_runs(tf):
        size = max((r.font.size or pa.font.size).pt for r in runs
                   if (r.font.size or pa.font.size)) if any(
            (r.font.size or pa.font.size) for r in runs) else SMALL_PT
        chars = sum(len(r.text) for r in runs)
        per_line = max(1, int(usable / (CHAR_W * size / 72.0)))
        lines = max(1, -(-chars // per_line))
        total += lines * LINE_H * size / 72.0
        total += ((pa.space_before.pt if pa.space_before else 0)
                  + (pa.space_after.pt if pa.space_after else 0)) / 72.0
    return total


def font_sizes(sh):
    out = []
    frames = []
    if sh.has_text_frame:
        frames.append(sh.text_frame)
    if sh.has_table:
        frames += [c.text_frame for row in sh.table.rows for c in row.cells]
    for tf in frames:
        for pa, runs in para_runs(tf):
            for r in runs:
                s = r.font.size or pa.font.size
                if s:
                    out.append((s.pt, len(r.text)))
    return out


def audit(path: Path, only=None, verbose=False):
    prs = Presentation(path)
    rows = []
    print(f"{'SL':>3} {'bot':>5} {'slack':>6} {'minPt':>5} {'modePt':>6}  flags")
    for i, s in enumerate(prs.slides, 1):
        if only and not (only[0] <= i <= only[1]):
            continue
        bot, fs, tight = 0.0, [], []
        for sh in s.shapes:
            if is_decor(sh):
                continue
            bot = max(bot, shape_bottom(sh))
            fs += font_sizes(sh)
            if sh.has_text_frame and sh.text_frame.text.strip() and not sh.has_table:
                need = est_text_height(sh.text_frame, sh.width / EMU)
                have = sh.height / EMU
                if need > have + 0.04:
                    tight.append((sh.name, round(need, 2), round(have, 2),
                                  sh.text_frame.text.strip()[:48]))
        slack = FLOOR - bot
        mn = min(f[0] for f in fs) if fs else None
        w = {}
        for sz, n in fs:
            w[sz] = w.get(sz, 0) + n
        mode = max(w, key=w.get) if w else None

        flags = []
        if slack >= SPACE_IN:
            flags.append("SPACE")
        if slack < -0.15:
            flags.append("OVERFLOW")
        if mn is not None and mn < SMALL_PT:
            flags.append("SMALL")
        if tight:
            flags.append(f"TIGHT×{len(tight)}")
        rows.append((i, slack, mn, flags, tight))
        print(f"{i:>3} {bot:5.2f} {slack:6.2f} {str(mn or '-'):>5} {str(mode or '-'):>6}  "
              f"{'+'.join(flags) or '-'}")
        if verbose:
            for name, need, have, txt in tight:
                print(f"      TIGHT {name}: needs {need}in, box {have}in — {txt!r}")

    print()
    for tag in ("SPACE", "OVERFLOW", "SMALL"):
        sel = [r[0] for r in rows if tag in r[3]]
        print(f"{tag:<9} n={len(sel):<3} {sel}")
    tight = [r[0] for r in rows if any(f.startswith('TIGHT') for f in r[3])]
    print(f"{'TIGHT':<9} n={len(tight):<3} {tight}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("deck")
    ap.add_argument("--slides", help="inclusive 1-based range, e.g. 57-66")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    rng = tuple(int(x) for x in a.slides.split("-")) if a.slides else None
    audit(Path(a.deck), rng, a.verbose)


if __name__ == "__main__":
    main()
