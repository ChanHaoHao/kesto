"""Grid text and puzzle arguments -- the ways a board gets into the solver.

One charset throughout, the one `kesto.board.render` emits, so a rendered board
pastes straight back in and `kesto.vision` has one obvious thing to produce:

    # wall   o block   . goal   * block already on a goal   - empty
"""

from __future__ import annotations

import os

from .board import N, Puzzle, parse
from .puzzles import by_slug

# Which bitboards each grid character contributes to.
_CHARS = {
    "#": ("walls",),
    "o": ("blocks",),
    ".": ("goals",),
    "*": ("blocks", "goals"),
    "-": (),
}


def parse_grid(text: str) -> Puzzle:
    """Build a Puzzle from a transcribed 8x8 board. No published solution."""
    rows = [r for r in (line.split() or line.strip() for line in text.splitlines()) if r]
    rows = [list("".join(r)) for r in rows]
    if len(rows) != N or any(len(r) != N for r in rows):
        got = f"{len(rows)} rows of {sorted({len(r) for r in rows})}"
        raise ValueError(f"need an {N}x{N} grid, got {got}")

    bits = {"walls": 0, "blocks": 0, "goals": 0}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch not in _CHARS:
                raise ValueError(f"bad char {ch!r} at ({x}, {y}); use {' '.join(_CHARS)}")
            for field in _CHARS[ch]:
                bits[field] |= 1 << (y * N + x)
    return Puzzle(solution="", **bits)


def load_puzzle(arg: str) -> Puzzle:
    """Accept a grid file, an encoded puzzle string, or a bundled slug."""
    if os.path.isfile(arg):
        with open(arg) as fh:
            return parse_grid(fh.read())
    return parse(arg) if "." in arg or len(arg) > 12 else by_slug(arg)

