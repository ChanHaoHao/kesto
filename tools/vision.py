#!/usr/bin/env python3
"""Read a Kesto board off a screenshot of the site.

Usage:
    python tools/vision.py board.png --solve
    python tools/vision.py board.png
    python tools/vision.py board.png -o board_today.txt
    python tools/vision.py board.png --debug
    python tools/vision.py board.png --vis steps/

Emits the same grid text `solver.py` and everything in `tools/` already parse
(`#` wall, `o` block, `.` goal, `*` block on goal, `-` empty), so a screenshot
drops into any of them. `--solve` hands the board it just read straight to
`solve.py` instead, which is the whole daily in one command and needs no file
in between.

Why there is no real computer vision here
-----------------------------------------
The site renders the board to a canvas as flat fills on an exact pixel grid --
no camera, no lighting, no perspective. A capture holds around 220 distinct
colours and 93% of its pixels are one of six of them. So the grid comes from
projection profiles (the gutters between cells are deep dips in the row and
column sums) and the cell type comes from a colour lookup. Contours, template
matching and Hough transforms would all be answering a harder question than the
one being asked. `--vis` draws every stage of that if you want to watch it work.

This does mean the input must be a screenshot. A phone photo of a monitor needs
the board quad detected and a homography applied before any of this holds, and
`find_grid` will refuse it rather than guess.

The one trap
------------
A goal cell's interior is byte-identical to an empty cell -- the goal is drawn
only as a thin rounded outline. Sampling the centre of each cell, which is the
obvious first thing to write, silently reads every goal as empty. `classify`
therefore scores whole cell footprints, never centre pixels.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import NamedTuple

import numpy as np
from PIL import Image

N = 8

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# --- classification thresholds -----------------------------------------------
#
# Every one of these sits in the middle of a gap at least an order of magnitude
# wider than the tolerance needed, measured over a real capture:
#
#   feature            blocks       goals       walls       empty
#   warm fraction      0.83-0.85    0.00        0.00        0.00
#   cool fraction      0.00         0.27        0.00        0.00
#   median grey        --           42          85          42
#
# They are deliberately not tuned tight. If the site restyles, the numbers that
# move are the greys -- which is why `_split_greys` derives its own cut from the
# levels present instead of naming them here.

SATURATED = 40  # chroma above which a pixel counts as coloured, not grey
BLOCK_FRAC = 0.30  # coloured-warm coverage that makes a cell a block
GOAL_FRAC = 0.05  # coloured-cool coverage that makes a cell a goal
CELL_FLOOR = 36  # luminance above which a pixel belongs to a cell, not a gutter
GUTTER_DIP = 0.5  # gutters fall to this fraction of the profile's cell plateau


class BoardNotFound(Exception):
    """The image does not look like a board screenshot."""


def _sibling(name):
    """Import a tools/ or repo-root script on demand.

    Those are scripts rather than a package, and both of the modules reached
    this way are optional: `visualise` only when `--vis` is passed, `solve` only
    when `--solve` is, which keeps a plain read off the numpy search stack.
    """
    for d in (HERE, ROOT):
        if d not in sys.path:
            sys.path.insert(0, d)
    return importlib.import_module(name)


class Profile(NamedTuple):
    """One axis of the projection step, kept whole so `--vis` can draw it."""

    axis: int  # 1 for the row profile, 0 for the column profile
    lo: int  # first coordinate holding cell pixels
    hi: int  # last
    span: np.ndarray  # cell-pixel count per coordinate, over [lo, hi]
    floor: float  # below this a coordinate reads as gutter
    gutters: list  # (start, stop) index pairs into `span`

    @property
    def name(self):
        return "row" if self.axis == 1 else "column"

    @property
    def edges(self):
        """The N+1 cell boundaries, at the centre of each gutter."""
        return [self.lo, *(self.lo + (a + b) // 2 for a, b in self.gutters), self.hi + 1]


class Analysis(NamedTuple):
    """Every intermediate of one read, in the order the pipeline makes them."""

    rgb: np.ndarray
    mask: np.ndarray  # pixels belonging to a cell rather than a gutter
    rprof: Profile
    cprof: Profile
    warm: np.ndarray  # coloured and red-dominant -- block fill
    cool: np.ndarray  # coloured and blue-dominant -- goal outline
    grey: np.ndarray  # unsaturated -- wall or empty
    kinds: list  # per cell: a settled char, or None while still grey
    levels: list  # per cell: median grey level, or None if already settled
    cut: float  # the derived wall/empty boundary
    grid: list  # per row: list of chars

    @property
    def rows(self):
        return self.rprof.edges

    @property
    def cols(self):
        return self.cprof.edges

    @property
    def text(self):
        return "\n".join(" ".join(row) for row in self.grid)

    def cell(self, y, x):
        """Pixel slice of one cell."""
        return slice(self.rows[y], self.rows[y + 1]), slice(self.cols[x], self.cols[x + 1])


def load_rgb(path):
    """Decode any screenshot flavour down to an (H, W, 3) int array."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.int16).astype(int)


def _runs(flat, is_gap):
    """Start/stop pairs of consecutive True values in `is_gap(flat[i])`."""
    out, run = [], None
    for i in range(len(flat) + 1):
        gap = is_gap(flat[i]) if i < len(flat) else True
        if gap and run is None:
            run = i
        elif not gap and run is not None:
            out.append((run, i - 1))
            run = None
    return out


def profile_axis(mask, axis):
    """Project the cell mask onto one axis and find the gutters in it.

    Deliberately does no validation -- `--vis` needs the profile of an image
    that is about to be rejected more than it needs one that parses.
    """
    counts = mask.sum(axis=axis)
    on = np.where(counts > 0)[0]
    if not len(on):
        raise BoardNotFound("no board-coloured pixels in the image")
    lo, hi = int(on[0]), int(on[-1])
    span = counts[lo : hi + 1]

    # Relative, not absolute: a downscaled or JPEG-recompressed capture blurs
    # the gutter until it is a single pixel that never reaches zero, but it
    # stays a deep dip against the plateau a row of cells holds. Thresholding
    # on a fixed count reads such an image as a 4x4 board, and silently.
    floor = span.max() * GUTTER_DIP
    edge = len(span) - 1
    gutters = [g for g in _runs(span, lambda v: v < floor) if g[0] > 0 and g[1] < edge]
    return Profile(axis, lo, hi, span, floor, gutters)


def find_grid(rgb):
    """Locate the 8x8 cell lattice. Returns the row and column `Profile`s.

    The gutter count is the check that matters: an 8x8 board has exactly seven
    per axis, and anything else means the crop caught page chrome, the board is
    cut off, or the image is not a screenshot at all.
    """
    mask = rgb.mean(2) > CELL_FLOOR
    profiles = (profile_axis(mask, 1), profile_axis(mask, 0))
    for p in profiles:
        if len(p.gutters) != N - 1:
            raise BoardNotFound(
                f"found {len(p.gutters)} {p.name} gutters, need {N - 1} -- "
                "crop the image down to the board, or it is not a screenshot"
            )
    return mask, profiles


def _split_greys(greys):
    """Cut between wall grey and empty grey, from the values actually present.

    The absolute levels are a styling choice and can change; that walls render
    lighter than empty cells is the structural fact, so the cut is derived. A
    board with no walls at all leaves one cluster, and everything grey is empty.
    """
    vals = np.array(sorted(v for v in greys if v is not None))
    if len(vals) < 2 or vals[-1] - vals[0] < 8:
        return vals[-1] + 1 if len(vals) else 0  # one cluster: nothing is a wall
    lo, hi = vals[0], vals[-1]
    for _ in range(20):  # 2-means, seeded at the extremes
        near_lo = vals[np.abs(vals - lo) <= np.abs(vals - hi)]
        near_hi = vals[np.abs(vals - lo) > np.abs(vals - hi)]
        if not len(near_lo) or not len(near_hi):
            break
        lo, hi = near_lo.mean(), near_hi.mean()
    return (lo + hi) / 2


def analyse(rgb):
    """Run the whole pipeline over a decoded image, keeping every stage."""
    mask, (rprof, cprof) = find_grid(rgb)
    rows, cols = rprof.edges, cprof.edges

    r, b = rgb[..., 0], rgb[..., 2]
    chroma = rgb.max(2) - rgb.min(2)
    coloured = chroma > SATURATED
    warm, cool, grey = coloured & (r > b), coloured & (b > r), ~coloured

    # Pass one: the coloured cells settle themselves; grey cells park their
    # level so the wall/empty cut can be drawn from the whole board at once.
    kinds, levels = [], []
    for y in range(N):
        for x in range(N):
            cell = slice(rows[y], rows[y + 1]), slice(cols[x], cols[x + 1])
            block, goal = warm[cell].mean(), cool[cell].mean()
            if block > BLOCK_FRAC:
                # A block sitting on a goal keeps the goal's outline around it.
                kinds.append("*" if goal > GOAL_FRAC else "o")
                levels.append(None)
            elif goal > GOAL_FRAC:
                kinds.append(".")
                levels.append(None)
            else:
                kinds.append(None)
                lv = float(np.median(r[cell][grey[cell]])) if grey[cell].any() else 0.0
                levels.append(lv)

    # Pass two: one cut for the board, applied to every cell still undecided.
    cut = _split_greys(levels)
    chars = [k if k else ("#" if lv > cut else "-") for k, lv in zip(kinds, levels)]
    grid = [chars[y * N : (y + 1) * N] for y in range(N)]
    return Analysis(rgb, mask, rprof, cprof, warm, cool, grey, kinds, levels, cut, grid)


def check_counts(grid):
    """Reject a read whose blocks and goals disagree."""
    blocks = sum(row.count("o") + row.count("*") for row in grid)
    goals = sum(row.count(".") + row.count("*") for row in grid)
    if blocks != goals:
        # Invariant of the puzzle, and a swipe preserves it -- so a mismatch is
        # a misread here, not a hard board, and every solver would reject it.
        raise BoardNotFound(
            f"read {blocks} blocks against {goals} goals; they must match. "
            "Re-run with --debug or --vis to see which cells are in doubt."
        )


def read_board(path):
    """Screenshot path -> grid text, in the charset `kesto.board.render` emits."""
    a = analyse(load_rgb(path))
    check_counts(a.grid)
    return a.text


def debug_report(a):
    """The measurements behind each cell, for when a read looks wrong."""
    print(f"grid rows {a.rows}\ngrid cols {a.cols}")
    print(f"wall/empty cut at grey {a.cut:.1f}")
    print("\nper cell: warm/cool coverage, median grey")
    for y in range(N):
        line = []
        for x in range(N):
            cell = a.cell(y, x)
            lv = np.median(a.rgb[..., 0][cell][a.grey[cell]]) if a.grey[cell].any() else -1
            line.append(f"{a.warm[cell].mean():.2f}/{a.cool[cell].mean():.2f}/{int(lv):3d}")
        print("  " + " ".join(line))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("image", help="screenshot of the board")
    ap.add_argument("-o", "--out", help="write the grid here instead of stdout")
    ap.add_argument("--debug", action="store_true", help="dump per-cell measurements")
    ap.add_argument(
        "--solve", action="store_true", help="solve the board it read, via tools/solve.py"
    )
    ap.add_argument("--mem-gb", type=float, default=None, help="--solve: RLIMIT_AS ceiling")
    ap.add_argument(
        "--max-len", type=int, default=200, help="--solve: give up beyond this length"
    )
    ap.add_argument(
        "--vis",
        nargs="?",
        const="vis",
        metavar="DIR",
        help="draw each pipeline stage to numbered PNGs in DIR (default: vis/)",
    )
    args = ap.parse_args()

    try:
        rgb = load_rgb(args.image)
    except OSError as e:
        print(f"vision: cannot read {args.image}: {e}", file=sys.stderr)
        return 1

    a = err = None
    try:
        a = analyse(rgb)
        check_counts(a.grid)
    except BoardNotFound as e:
        err = e

    if args.vis:
        # Drawing lives in a sibling module so the method above stays readable,
        # and nothing else uses it.
        for path in _sibling("visualise").draw_steps(rgb, a, args.vis):
            print(f"  {path}", file=sys.stderr)
    if args.debug and a is not None:
        debug_report(a)
    if err is not None:
        print(f"vision: {err}", file=sys.stderr)
        return 1

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(a.text + "\n")
        print(f"wrote {args.out}", file=sys.stderr)
    # Flushed because `solve.run` logs its plies to stderr, which is unbuffered:
    # without this the board turns up after the search that was run on it as
    # soon as stdout is a pipe rather than a terminal.
    print(a.text, flush=True)

    if args.solve:
        # `solve.run` owns the memory cap, the scratch directory and its
        # cleanup, so the board goes to exactly the search `solve.py board.txt`
        # would have run -- the file in between was the only thing dropped.
        puzzle = _sibling("solver").parse_grid(a.text)
        tag = os.path.splitext(os.path.basename(args.image))[0]
        return _sibling("solve").run(puzzle, tag, args.mem_gb, max_len=args.max_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
