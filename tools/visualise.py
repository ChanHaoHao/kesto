#!/usr/bin/env python3
"""Draw each stage of `vision.py` to a numbered PNG. Imported by `--vis`.

This is explanatory only -- nothing here feeds a board back into a solver, and
`vision.py` never imports it except when the flag is passed. It exists because
the pipeline's two load-bearing claims are both visual ones: that the gutters
between cells are unmistakable dips in the projection profile, and that the four
cell types fall into colour clusters nowhere near each other. Stages 3 and 6 are
those two claims drawn out.

The stages run as far as the read got. A rejected image still gets 01-03, which
is where a rejection is nearly always visible.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vision import CELL_FLOOR, GUTTER_DIP, N, profile_axis

PAPER = (16, 16, 18)
INK = (238, 238, 240)
MUTED = (138, 138, 148)
WARM = (235, 163, 81)
COOL = (84, 157, 211)
WALL = (168, 168, 172)
EMPTY = (96, 96, 104)
MARK = (232, 74, 106)  # the lattice, and anything the code decided

CHART = 150  # px given to a profile chart
GAP = 14
HEAD = 52  # caption bar height

CHAR_COLOUR = {"o": WARM, "*": WARM, ".": COOL, "#": WALL, "-": EMPTY}
CHAR_NAME = {"o": "block", "*": "block on goal", ".": "goal", "#": "wall", "-": "empty"}


def _font(size):
    return ImageFont.load_default(size=size)


def _captioned(img, title, note=""):
    """Put a titled bar above a panel, widening it if the caption needs room."""
    tf, nf = _font(19), _font(14)
    need = max(tf.getbbox(title)[2], nf.getbbox(note)[2] if note else 0) + 28
    out = Image.new("RGB", (max(img.width, need), img.height + HEAD), PAPER)
    out.paste(img, (0, HEAD))
    d = ImageDraw.Draw(out)
    d.text((14, 10), title, font=tf, fill=INK)
    if note:
        d.text((14, 31), note, font=nf, fill=MUTED)
    return out


def _mask_image(mask, on=INK, off=PAPER):
    """A bool array as a two-tone image."""
    rgb = np.where(mask[..., None], np.array(on), np.array(off))
    return Image.fromarray(rgb.astype(np.uint8))


def _dim(rgb, factor=0.35):
    return Image.fromarray((rgb * factor).astype(np.uint8))


def _dashed(d, a, b, fill, dash=7):
    """A dashed line, which PIL has no primitive for."""
    (x0, y0), (x1, y1) = a, b
    span = max(abs(x1 - x0), abs(y1 - y0))
    steps = max(1, int(span // dash))
    for i in range(0, steps, 2):
        t0, t1 = i / steps, min(1.0, (i + 1) / steps)
        d.line(
            [x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0, x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1],
            fill=fill,
            width=2,
        )


# --- the stages ---------------------------------------------------------------


def step_input(rgb):
    return _captioned(
        Image.fromarray(rgb.astype(np.uint8)),
        "1. the capture",
        f"{rgb.shape[1]}x{rgb.shape[0]}, {len(np.unique(rgb.reshape(-1, 3), axis=0))} distinct colours",
    )


def step_mask(mask):
    lit = int(mask.sum())
    return _captioned(
        _mask_image(mask),
        "2. cell mask -- luminance above the background",
        f"{lit:,} of {mask.size:,} px ({100 * lit / mask.size:.0f}%) belong to a cell, not a gutter",
    )


def _chart(counts, prof, length, thickness, vertical):
    """One projection profile: the curve, the gutter floor, and the gutters found.

    `vertical=True` draws the curve growing upward from a baseline under an
    image; otherwise it grows rightward beside one. Either way index tracks the
    image axis, so a dip lines up with the gutter that produced it.
    """
    pad_t, pad_b = 34, 16  # room for the annotation, and off the panel edge
    depth = thickness - pad_t - pad_b
    size = (length, thickness) if vertical else (thickness, length)
    img = Image.new("RGB", size, PAPER)
    d = ImageDraw.Draw(img)
    top = int(np.max(counts)) or 1

    def pos(v):  # value -> the across-axis pixel it reaches
        return thickness - pad_b - v / top * depth if vertical else pad_b + v / top * depth

    for a, b in prof.gutters:  # shade first, so the curve draws over the band
        lo, hi = prof.lo + a, prof.lo + b + 1
        d.rectangle((lo, 0, hi, thickness) if vertical else (0, lo, thickness, hi), fill=(92, 34, 46))

    base = pos(0)
    for i, v in enumerate(counts):
        colour = MARK if v < prof.floor else (120, 178, 232)
        d.line([i, base, i, pos(v)] if vertical else [base, i, pos(v), i], fill=colour)
    d.line([0, base, length, base] if vertical else [base, 0, base, length], fill=MUTED)

    f = pos(prof.floor)
    _dashed(d, *(((0, f), (length, f)) if vertical else ((f, 0), (f, length))), fill=(250, 210, 90))
    if vertical:
        dip = min(int(counts[prof.lo + a : prof.lo + b + 1].min()) for a, b in prof.gutters) if prof.gutters else top
        d.text((8, 6), f"plateau {top} px, deepest gutter {dip} px", font=_font(14), fill=(120, 178, 232))
        d.text((8, 20), f"floor = {GUTTER_DIP:g} x plateau = {prof.floor:.0f}", font=_font(13), fill=(250, 210, 90))
    return img


def step_profiles(mask, rprof, cprof):
    """The claim the grid detection rests on, drawn."""
    h, w = mask.shape
    panel = Image.new("RGB", (w + GAP + CHART, h + GAP + CHART), PAPER)
    panel.paste(_mask_image(mask, on=(72, 76, 86)), (0, 0))
    panel.paste(_chart(mask.sum(0), cprof, w, CHART, vertical=True), (0, h + GAP))
    panel.paste(_chart(mask.sum(1), rprof, h, CHART, vertical=False), (w + GAP, 0))

    d = ImageDraw.Draw(panel)
    d.text((w + GAP + 8, h + GAP + 30), "row\nprofile\n(right)", font=_font(14), fill=MUTED)

    ok = len(rprof.gutters) == N - 1 and len(cprof.gutters) == N - 1
    note = f"{len(rprof.gutters)} row and {len(cprof.gutters)} column gutters; {N - 1} of each means an {N}x{N}"
    return _captioned(
        panel,
        "3. projection profiles -- summing the mask down each axis",
        note + ("" if ok else "   <-- REJECTED HERE"),
    )


def step_lattice(a):
    """Where the cell boundaries landed: the midpoint of every gutter."""
    img = _dim(a.rgb, 0.55)
    d = ImageDraw.Draw(img)
    for prof, vertical in ((a.cprof, True), (a.rprof, False)):
        for ga, gb in prof.gutters:
            box = (prof.lo + ga, 0, prof.lo + gb, img.height) if vertical else (0, prof.lo + ga, img.width, prof.lo + gb)
            d.rectangle(box, fill=(70, 30, 40))
    for x in a.cols:
        d.line([x, 0, x, img.height], fill=MARK, width=1)
    for y in a.rows:
        d.line([0, y, img.width, y], fill=MARK, width=1)

    f = _font(13)
    for i in range(N):
        d.text((a.cols[i] + 4, a.rows[0] + 3), str(i), font=f, fill=MARK)
        d.text((a.cols[0] + 4, a.rows[i] + 3), str(i), font=f, fill=MARK)
    pitch = (a.cols[-1] - a.cols[0]) / N
    return _captioned(img, "4. the lattice", f"cell pitch {pitch:.1f} px; boundaries at each gutter's midpoint")


def step_channels(a):
    """The three colour channels the classifier actually reads."""
    # The greys are stretched off their own range. Walls and empties sit at 85
    # and 42 on a 26 background, which is a real separation but an invisible one
    # at display brightness; stage 6 carries the untouched numbers.
    hot = a.rgb[..., 0][a.grey]
    lo, hi = (float(hot.min()), float(hot.max())) if hot.size else (0.0, 1.0)
    stretched = np.where(a.grey, 40 + (a.rgb[..., 0] - lo) / max(1.0, hi - lo) * 215, 0)
    panels = [
        (_mask_image(a.warm, on=WARM), "warm -- block fill"),
        (_mask_image(a.cool, on=COOL), "cool -- goal outline"),
        (Image.fromarray(np.repeat(stretched[..., None], 3, 2).astype(np.uint8)), "grey -- wall or empty (stretched)"),
    ]
    h, w = a.mask.shape
    out = Image.new("RGB", (3 * w + 2 * GAP, h + 26), PAPER)
    d = ImageDraw.Draw(out)
    for i, (img, label) in enumerate(panels):
        out.paste(img, (i * (w + GAP), 0))
        d.text((i * (w + GAP) + 6, h + 6), label, font=_font(15), fill=INK)
    return _captioned(
        out,
        "5. colour channels -- saturation splits coloured from grey",
        "a goal's interior is identical to an empty cell, so cells are scored over their whole footprint",
    )


def step_greys(a):
    """The wall/empty cut, derived from the levels the board actually has."""
    vals = sorted(v for v in a.levels if v is not None)
    w, h = 780, 320
    pad, bar, floor = 40, 26, 250  # margins, colourbar height, baseline
    img = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(img)
    if not vals:
        d.text((14, 14), "no grey cells on this board", font=_font(16), fill=MUTED)
        return _captioned(img, "6. the wall/empty cut", "every cell on this board is coloured")

    lo = max(0, min(vals + [a.cut]) - 25)
    hi = min(255, max(vals + [a.cut]) + 25)
    span = max(1.0, hi - lo)

    def px(v):
        return pad + (v - lo) / span * (w - 2 * pad)

    # A colourbar in the actual greys, so "walls render lighter" is literal
    # rather than asserted, and the histogram sits directly above its own axis.
    for x in range(pad, w - pad):
        lv = int(lo + (x - pad) / (w - 2 * pad) * span)
        d.line([x, floor + 6, x, floor + 6 + bar], fill=(lv, lv, lv))

    counts = {}
    for v in vals:
        counts[round(v)] = counts.get(round(v), 0) + 1
    tallest = max(counts.values())
    for lv, n in counts.items():
        x, bh = px(lv), int(n / tallest * (floor - 90))
        d.rectangle([x - 7, floor - bh, x + 7, floor], fill=WALL if lv > a.cut else EMPTY)
        d.text((x, floor - bh - 16), f"x{n}", font=_font(14), fill=INK, anchor="mm")
        d.text((x, floor + 6 + bar + 12), str(lv), font=_font(13), fill=MUTED, anchor="mm")

    x = px(a.cut)
    _dashed(d, (x, 56), (x, floor + 6 + bar), MARK)
    d.text((x, 44), f"cut {a.cut:.1f}", font=_font(16), fill=MARK, anchor="mm")

    walls = [v for v in vals if v > a.cut]
    empties = [v for v in vals if v <= a.cut]
    if empties:
        d.text((px(np.mean(empties)), 24), f"empty  {len(empties)} cells", font=_font(15), fill=EMPTY, anchor="mm")
    if walls:
        d.text((px(np.mean(walls)), 24), f"wall  {len(walls)} cells", font=_font(15), fill=WALL, anchor="mm")
    return _captioned(
        img,
        "6. the wall/empty cut -- 2-means over the greys present",
        "derived, not hardcoded: the levels are a styling choice, but walls being lighter is structural",
    )


def step_board(a):
    """The read, laid back over the picture it came from."""
    img = _dim(a.rgb, 0.3)
    d = ImageDraw.Draw(img)
    pitch = (a.cols[-1] - a.cols[0]) / N
    f = _font(max(12, int(pitch * 0.6)))
    for y in range(N):
        for x in range(N):
            ch = a.grid[y][x]
            cx = (a.cols[x] + a.cols[x + 1]) / 2
            cy = (a.rows[y] + a.rows[y + 1]) / 2
            d.text((cx, cy), ch, font=f, fill=CHAR_COLOUR[ch], anchor="mm")

    legend = Image.new("RGB", (img.width, 34), PAPER)
    ld = ImageDraw.Draw(legend)
    x = 14
    for ch, name in CHAR_NAME.items():
        ld.text((x, 8), ch, font=_font(18), fill=CHAR_COLOUR[ch])
        ld.text((x + 16, 10), name, font=_font(14), fill=MUTED)
        x += 30 + 8 * len(name)
    out = Image.new("RGB", (img.width, img.height + 34), PAPER)
    out.paste(img, (0, 0))
    out.paste(legend, (0, img.height))

    blocks = sum(r.count("o") + r.count("*") for r in a.grid)
    goals = sum(r.count(".") + r.count("*") for r in a.grid)
    note = f"{blocks} blocks, {goals} goals" + (" -- match" if blocks == goals else " -- MISMATCH, rejected")
    return _captioned(out, "7. the board", note)


def draw_steps(rgb, a, out_dir):
    """Write every stage the read reached. Returns the paths written, in order.

    `a` is None when the image was rejected before classification, in which case
    only the stages that got computed are drawn -- which is the useful half, as
    a rejection is a gutter-count failure nearly every time.
    """
    os.makedirs(out_dir, exist_ok=True)
    mask = a.mask if a is not None else rgb.mean(2) > CELL_FLOOR
    rprof = a.rprof if a is not None else profile_axis(mask, 1)
    cprof = a.cprof if a is not None else profile_axis(mask, 0)

    stages = [
        ("01_input", lambda: step_input(rgb)),
        ("02_cellmask", lambda: step_mask(mask)),
        ("03_profiles", lambda: step_profiles(mask, rprof, cprof)),
    ]
    if a is not None:
        stages += [
            ("04_lattice", lambda: step_lattice(a)),
            ("05_channels", lambda: step_channels(a)),
            ("06_greysplit", lambda: step_greys(a)),
            ("07_board", lambda: step_board(a)),
        ]

    written = []
    for name, build in stages:
        path = os.path.join(out_dir, f"{name}.png")
        build().save(path)
        written.append(path)
    return written
