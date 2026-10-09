"""Raise the deck's type size, filling the empty space the supervisor flagged.

The review note was: "if there is an empty space on the slide, please increase the font
size in the ppt, right now it is too small". The deck was built with an 11 pt body and
table default and a 10 pt caption, and 38 of its slides carried text below 12 pt while 18
of them ended more than an inch above the content floor.

Approach — grow only into space that is provably empty
------------------------------------------------------
A blind global scale-up would push text off the bottom of the slides that are already
full, so this works per layout unit instead:

1. Shapes are grouped into units. A background panel and the text box drawn inside it
   are one unit, because growing the text means growing the panel. Tables, standalone
   text boxes and pictures are their own units.
2. For each unit, the space below it is measured: the top of the nearest unit that
   overlaps it horizontally, or the content floor if nothing is below it.
3. The unit's font scale is then the largest that still fits in its own height plus that
   verified-empty space, searched over a descending ladder. Sizes are never reduced, are
   raised to the readable floor whenever the floor fits, and are capped so headings do
   not balloon.
4. The unit's geometry grows to match — panel and text box heights together, table row
   heights proportionally.

Text height is estimated with the same character-metric model as check_layout.py, which
is deliberately pessimistic, so a unit that this script reports as fitting has margin.

Usage:
    .venv/bin/python scratch_ppt/apply_typography.py "presentations/Tandon's_Draft.pptx"
    .venv/bin/python scratch_ppt/apply_typography.py <deck> --dry-run
    .venv/bin/python scratch_ppt/apply_typography.py <deck> --slides 1-56
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import copy

from pptx import Presentation
from pptx.util import Pt

EMU = 914400.0
FLOOR = 7.02            # the page-number placeholder starts at 7.16
TOP_GUARD = 0.20        # never let a unit grow above this
BODY_FLOOR = 12.0       # readable floor for body, table and caption text
TITLE_CAP = 26.0        # do not grow titles past this
BODY_CAP = 15.0         # do not grow body text past this
CHAR_W = 0.50           # mean glyph advance as a fraction of point size
LINE_H = 1.20
SCALES = [1.25, 1.20, 1.15, 1.10, 1.05, 1.00]


# ------------------------------------------------------------------ geometry
def near(a, b, t=0.15):
    return abs(a - b) < t


def is_decor(sh):
    L, T, H = sh.left / EMU, sh.top / EMU, sh.height / EMU
    if near(T, -3.13) and near(H, 13.33):
        return True
    if near(L, 11.36) and near(T, 6.71):
        return True
    if sh.has_text_frame and sh.text_frame.text.strip() == "‹#›":
        return True
    return False


def box(sh):
    """left, top, right, bottom in inches; a table's rows are its true height."""
    l, t = sh.left / EMU, sh.top / EMU
    r = l + sh.width / EMU
    b = t + (sum(row.height for row in sh.table.rows) / EMU if sh.has_table
             else sh.height / EMU)
    return l, t, r, b


def h_overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > 0.05


def contains(outer, inner, pad=0.12):
    return (outer[0] - pad <= inner[0] and outer[1] - pad <= inner[1]
            and outer[2] + pad >= inner[2] and outer[3] + pad >= inner[3])


# ------------------------------------------------------------------ text metrics
def para_runs(tf):
    for pa in tf.paragraphs:
        runs = [r for r in pa.runs if r.text.strip()]
        if runs:
            yield pa, runs


def frames_of(sh):
    out = []
    if sh.has_text_frame and sh.text_frame.text.strip():
        out.append((sh.text_frame, sh.width / EMU))
    if sh.has_table:
        for row in sh.table.rows:
            for c in row.cells:
                if c.text_frame.text.strip():
                    out.append((c.text_frame, None))
    return out


def est_height(tf, width_in, scale=1.0):
    usable = width_in - (tf.margin_left + tf.margin_right) / EMU
    total = (tf.margin_top + tf.margin_bottom) / EMU
    for pa, runs in para_runs(tf):
        sizes = [(r.font.size or pa.font.size) for r in runs]
        size = max((s.pt for s in sizes if s), default=BODY_FLOOR) * scale
        chars = sum(len(r.text) for r in runs)
        per_line = max(1, int(usable / (CHAR_W * size / 72.0)))
        total += max(1, -(-chars // per_line)) * LINE_H * size / 72.0
        total += ((pa.space_before.pt if pa.space_before else 0)
                  + (pa.space_after.pt if pa.space_after else 0)) / 72.0
    return total


def cell_min_height(cell, scale):
    """Row height a table cell needs: one line is the common case, wrapped cells more."""
    width = None
    return est_height(cell.text_frame, width, scale) if width else None


# ------------------------------------------------------------------ units
class Unit:
    def __init__(self, shapes):
        self.shapes = shapes
        bs = [box(s) for s in shapes]
        self.l = min(b[0] for b in bs)
        self.t = min(b[1] for b in bs)
        self.r = max(b[2] for b in bs)
        self.b = max(b[3] for b in bs)
        self.panel = next((s for s in shapes
                           if s.has_text_frame and not s.text_frame.text.strip()), None)
        self.table = next((s for s in shapes if s.has_table), None)
        self.texts = [s for s in shapes
                      if s.has_text_frame and s.text_frame.text.strip()]

    @property
    def rect(self):
        return (self.l, self.t, self.r, self.b)

    def has_text(self):
        return bool(self.texts or self.table)


def build_units(slide):
    content = [sh for sh in slide.shapes if not is_decor(sh)]
    # an empty auto-shape that encloses a text box is that text box's backing panel
    panels = [sh for sh in content
              if sh.has_text_frame and not sh.text_frame.text.strip() and not sh.has_table]
    taken, units = set(), []
    for p in panels:
        inner = [sh for sh in content
                 if sh is not p and id(sh) not in taken and contains(box(p), box(sh))]
        if inner:
            units.append(Unit([p] + inner))
            taken.add(id(p))
            taken.update(id(sh) for sh in inner)
    for sh in content:
        if id(sh) not in taken:
            units.append(Unit([sh]))
            taken.add(id(sh))
    units.sort(key=lambda u: (u.t, u.l))
    return units


def room_below(u, units):
    """Lowest the unit may extend to: the nearest unit beneath it, else the floor."""
    lim = FLOOR
    for o in units:
        if o is u or not h_overlap(u.rect, o.rect):
            continue
        if o.t >= u.b - 0.02:
            lim = min(lim, o.t - 0.06)
    return max(lim, u.b)


# ------------------------------------------------------------------ scaling
def unit_fits(u, scale, limit):
    """Would this unit, scaled, still end at or above `limit`?"""
    if u.table is not None:
        tbl = u.table.table
        need = 0.0
        for row in tbl.rows:
            cur = row.height / EMU
            # a row grows with its text, floored at the readable line height
            line = max((_scaled_pt(r.font.size or pa.font.size, scale)
                        for c in row.cells for pa, runs in para_runs(c.text_frame)
                        for r in runs), default=BODY_FLOOR)
            need += max(cur, line * LINE_H / 72.0 + 0.08)
        return u.t + need <= limit + 1e-6, need
    need = 0.0
    for sh in u.texts:
        need = max(need, est_height(sh.text_frame, sh.width / EMU, scale)
                   + (box(sh)[1] - u.t))
    return u.t + need <= limit + 1e-6, need


def _scaled_pt(size, scale):
    if size is None:
        return BODY_FLOOR
    return size.pt * scale


def pick_scale(u, limit):
    for s in SCALES:
        ok, _ = unit_fits(u, s, limit)
        if ok:
            return s
    return 1.0


def apply_scale(u, scale, title_like):
    """Resize every run in the unit, then grow the unit's geometry to match."""
    changed = []
    cap = TITLE_CAP if title_like else BODY_CAP
    for sh in u.shapes:
        for tf, _w in frames_of(sh):
            for pa, runs in para_runs(tf):
                for r in runs:
                    cur = r.font.size or pa.font.size
                    if cur is None:
                        continue
                    old = cur.pt
                    new = old * scale
                    if old < BODY_FLOOR <= new or old < BODY_FLOOR:
                        new = max(new, BODY_FLOOR)
                    new = min(round(new * 2) / 2, cap)
                    new = max(new, old)
                    if abs(new - old) > 0.01:
                        r.font.size = Pt(new)
                        changed.append((old, new))
    return changed


def grow_geometry(u, limit):
    """Give the unit the height its newly sized text needs, without passing `limit`."""
    if u.table is not None:
        tbl = u.table.table
        total = 0.0
        for row in tbl.rows:
            line = max((_scaled_pt(r.font.size or pa.font.size, 1.0)
                        for c in row.cells for pa, runs in para_runs(c.text_frame)
                        for r in runs), default=BODY_FLOOR)
            want = max(row.height / EMU, line * LINE_H / 72.0 + 0.08)
            total += want
        if u.t + total > limit:                       # never exceed the verified space
            total = max(0.1, limit - u.t)
        scale = total / max(1e-6, sum(r.height for r in tbl.rows) / EMU)
        for row in tbl.rows:
            row.height = int(row.height * scale)
        return
    need = 0.0
    for sh in u.texts:
        need = max(need, est_height(sh.text_frame, sh.width / EMU) + (box(sh)[1] - u.t))
    new_b = min(u.t + need, limit)
    if u.panel is not None and new_b > u.b:
        # the panel ends exactly at the verified limit, never past it
        u.panel.height = int((new_b - box(u.panel)[1]) * EMU)
    for sh in u.texts:
        h_need = est_height(sh.text_frame, sh.width / EMU)
        room = max(0.1, new_b - box(sh)[1])
        if h_need > sh.height / EMU:
            sh.height = int(min(h_need, room) * EMU)


# ------------------------------------------------------------------ main
def table_need(tbl, factor=1.0):
    """Row heights a table needs when its cells wrap inside their own columns.

    The first model here only compared a row's height to one line of text, which let
    tables grow to a size where narrow columns wrapped to three and four lines. This
    measures each cell against its column width, which is what actually wraps.
    """
    widths = [c.width / EMU for c in tbl.columns]
    rows = []
    for r in tbl.rows:
        need = 0.0
        for ci, cell in enumerate(r.cells):
            size = max((_scaled_pt(run.font.size or pa.font.size, factor)
                        for pa, runs in para_runs(cell.text_frame) for run in runs),
                       default=0.0)
            if size <= 0:
                continue
            per_line = max(1, int((widths[ci] - 0.12) / (CHAR_W * size / 72.0)))
            lines = max(1, -(-len(cell.text) // per_line))
            need = max(need, lines * LINE_H * size / 72.0 + 0.08)
        rows.append(max(need, 0.18))
    return rows


def fit_tables(slide):
    """Shrink a table's type just enough that no cell overflows its own row.

    Runs after scaling. Row heights grow first, into space the slide actually has;
    only if that is not enough does the font come back down, and never below the
    readable floor unless it was already there.
    """
    units = build_units(slide)
    fixed = []
    for u in units:
        if u.table is None:
            continue
        tbl = u.table.table
        # a table never extends past the content floor, even if it already does:
        # room_below() would otherwise hand back the table's own overflowing bottom
        limit = min(room_below(u, units), FLOOR)
        cur = [r.height / EMU for r in tbl.rows]
        for f in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75):
            need = table_need(tbl, f)
            if u.t + sum(need) <= limit + 1e-6:
                break
        else:
            need = table_need(tbl, 0.75)
            f = 0.75
        if f < 1.0:
            for r in tbl.rows:
                for c in r.cells:
                    for pa, runs in para_runs(c.text_frame):
                        for run in runs:
                            sz = run.font.size or pa.font.size
                            if sz is None:
                                continue
                            new = max(round(sz.pt * f * 2) / 2, min(sz.pt, BODY_FLOOR))
                            if new < sz.pt:
                                run.font.size = Pt(new)
            need = table_need(tbl, 1.0)
        grown = [max(a, b) for a, b in zip(cur, need)]
        if u.t + sum(grown) > limit:
            grown = need
        if u.t + sum(grown) > limit:                  # still over: share the deficit out
            k = (limit - u.t) / max(1e-6, sum(grown))
            grown = [h * k for h in grown]
        if any(abs(a - b) > 0.01 for a, b in zip(cur, grown)) or f < 1.0:
            for r, h in zip(tbl.rows, grown):
                r.height = int(h * EMU)
            fixed.append((round(f, 2), round(sum(grown), 2)))
    return fixed


def fill_slack(slide, min_slack=0.60):
    """Spend a slide's leftover empty band on its content rather than leaving it blank.

    The band is given to the slide's table (taller rows, which is also what lets the
    larger type breathe) and everything below the table shifts down by the same amount,
    so the lowest shape lands on the content floor. Slides without a table are left
    alone: stretching a text panel only makes a bigger empty box, and the section
    dividers are meant to be sparse. Returns the inches consumed.
    """
    units = [u for u in build_units(slide) if u.has_text() or u.table is not None]
    if not units:
        return 0.0
    bottom = max(u.b for u in units)
    extra = FLOOR - bottom
    if extra < min_slack:
        return 0.0

    tables = [u for u in units if u.table is not None]
    if not tables:
        return 0.0
    u = tables[0]
    tbl = u.table.table
    if True:
        cur = sum(r.height for r in tbl.rows) / EMU
        if cur <= 0.01:
            return 0.0
        factor = (cur + extra) / cur
        for row in tbl.rows:
            row.height = int(row.height * factor)
        for o in units:
            if o is u or o.t < u.b - 0.02:
                continue
            for sh in o.shapes:
                sh.top = int(sh.top + extra * EMU)
        return extra


def slide_bottom(slide):
    """Lowest point of real content on the slide, in inches."""
    bs = [box(sh)[3] for sh in slide.shapes if not is_decor(sh)]
    return max(bs) if bs else 0.0


def snapshot(slide):
    return [copy.deepcopy(el) for el in list(slide.shapes._spTree)]


def restore(slide, snap):
    tree = slide.shapes._spTree
    for el in list(tree):
        tree.remove(el)
    for el in snap:
        tree.append(el)


def process(prs, only=None, dry=False, fill=True):
    """Resize slide by slide, reverting any slide the pass would make worse.

    The text-height model is an estimate, so instead of trusting it we check the
    result: a slide whose content now reaches lower than both the content floor and
    where it started is rolled back untouched.
    """
    stats = []
    for i, slide in enumerate(prs.slides, 1):
        if only and not (only[0] <= i <= only[1]):
            continue
        before_bottom = slide_bottom(slide)
        snap = None if dry else snapshot(slide)

        units = build_units(slide)
        raised, scales = 0, []
        for u in units:
            if not u.has_text():
                continue
            limit = room_below(u, units)
            title_like = u.t < 1.10
            s = pick_scale(u, limit)
            if s <= 1.0:
                ok, _ = unit_fits(u, 1.0, limit)
                if not ok:
                    continue
            scales.append(round(s, 2))
            if dry:
                continue
            ch = apply_scale(u, s, title_like)
            if ch:
                raised += len(ch)
                grow_geometry(u, limit)
            else:
                # already at the cap, but the box may still be too small for the text
                # it holds — give it the verified space below rather than letting it spill
                need = max((est_height(sh.text_frame, sh.width / EMU) + (box(sh)[1] - u.t)
                            for sh in u.texts), default=0.0)
                if need > (u.b - u.t) + 0.04 and limit > u.b + 0.04:
                    grow_geometry(u, limit)

        if not dry:
            for f, h in fit_tables(slide):
                print(f"[TABLE]  slide {i}: refit at x{f} of scaled size, height {h}in")

        if not dry and fill:
            used = fill_slack(slide)
            if used:
                print(f"[FILL]   slide {i}: {used:.2f}in of empty band given to content")

        if dry:
            print(f"{i:>3}  scales {scales}")
            stats.append((i, 0, scales))
            continue

        after_bottom = slide_bottom(slide)
        if after_bottom > max(FLOOR, before_bottom) + 0.02:
            restore(slide, snap)
            print(f"[REVERT] slide {i}: content would reach {after_bottom:.2f}in "
                  f"(was {before_bottom:.2f}in, floor {FLOOR})")
            raised = 0
        stats.append((i, raised, scales))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("deck")
    ap.add_argument("--slides")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-fill", action="store_true",
                    help="raise type only; leave empty bands as they are")
    a = ap.parse_args()
    deck = Path(a.deck)
    rng = tuple(int(x) for x in a.slides.split("-")) if a.slides else None

    prs = Presentation(deck)
    if not a.dry_run:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bak = deck.with_suffix(deck.suffix + f".bak_pre_typography_{stamp}")
        shutil.copy2(deck, bak)
        print(f"[BACKUP] {bak.name}")

    stats = process(prs, rng, a.dry_run, not a.no_fill)
    if not a.dry_run:
        prs.save(deck)
        total = sum(s[1] for s in stats)
        touched = sum(1 for s in stats if s[1])
        print(f"[SAVED] {deck.name}: {total} runs resized across {touched} slides")


if __name__ == "__main__":
    main()
