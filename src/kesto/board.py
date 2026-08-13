"""Kesto board representation and move engine.

Kesto is played on an 8x8 grid holding blocks, walls and goal squares. A swipe
moves *every* block one cell in the swiped direction; a block stays put if its
destination is off the board, is a wall, or is occupied by a block that is
itself stuck. The puzzle is solved when the blocks exactly cover the goals.

Everything here is a 64-bit bitboard. Bit ``y * 8 + x`` is cell ``(x, y)``,
with ``x`` increasing rightward and ``y`` increasing downward. Blocks are
interchangeable, so the block bitboard alone is a canonical state key -- that
is what makes graph search on this puzzle tractable.
"""

from __future__ import annotations

import base64
from typing import NamedTuple

N = 8
FULL = (1 << 64) - 1

FILE0 = 0x0101010101010101  # x == 0
FILE7 = 0x8080808080808080  # x == 7
RANK0 = 0x00000000000000FF  # y == 0
RANK7 = 0xFF00000000000000  # y == 7

MOVES = ("U", "D", "L", "R")

# Per direction: cells that cannot move (already against the far edge),
# `forward` shifts a bitboard one cell along the move, and `back` maps the
# contents of cell c+delta onto cell c (i.e. "what is directly ahead of me").
_DIRS = {
    "R": (FILE7, lambda v: (v << 1) & ~FILE0 & FULL, lambda v: (v >> 1) & ~FILE7),
    "L": (FILE0, lambda v: (v >> 1) & ~FILE7, lambda v: (v << 1) & ~FILE0 & FULL),
    "D": (RANK7, lambda v: (v << 8) & FULL, lambda v: v >> 8),
    "U": (RANK0, lambda v: v >> 8, lambda v: (v << 8) & FULL),
}


def move(blocks: int, walls: int, direction: str) -> int:
    """Apply one swipe, returning the new block bitboard.

    A block is stuck iff the cell ahead of it is off-board or a wall, or holds
    a block that is itself stuck. That is a closure, so it is computed to a
    fixpoint; everything not in the stuck set advances exactly one cell.
    """
    edge, forward, back = _DIRS[direction]
    stuck = (blocks & edge) | (blocks & back(walls))
    while True:
        grown = stuck | (blocks & back(stuck))
        if grown == stuck:
            break
        stuck = grown
    return stuck | forward(blocks & ~stuck)


def cells(bitboard: int) -> list[tuple[int, int]]:
    """Expand a bitboard into a list of ``(x, y)`` coordinates."""
    return [(i % N, i // N) for i in range(64) if (bitboard >> i) & 1]


def popcount(bitboard: int) -> int:
    """Number of set cells."""
    return bin(bitboard).count("1")


def render(blocks: int, walls: int, goals: int) -> str:
    """Draw a board as text. ``#`` wall, ``*`` block on goal, ``o`` block,
    ``.`` empty goal, ``-`` empty cell."""
    rows = []
    for y in range(N):
        row = []
        for x in range(N):
            bit = 1 << (y * N + x)
            if walls & bit:
                row.append("#")
            elif blocks & bit:
                row.append("*" if goals & bit else "o")
            elif goals & bit:
                row.append(".")
            else:
                row.append("-")
        rows.append(" ".join(row))
    return "\n".join(rows)


class Puzzle(NamedTuple):
    """A Kesto puzzle plus the published solution, when one is known."""

    walls: int
    blocks: int
    goals: int
    solution: str
    slug: str = ""
    title: str = ""

    @property
    def n_blocks(self) -> int:
        return popcount(self.blocks)

    @property
    def n_walls(self) -> int:
        return popcount(self.walls)

    def apply(self, path: str) -> int:
        """Play a move string from the start position, returning the result."""
        b = self.blocks
        for m in path:
            b = move(b, self.walls, m)
        return b

    def solves(self, path: str) -> bool:
        """True if ``path`` lands every block exactly on the goal set."""
        return self.apply(path) == self.goals

    def __str__(self) -> str:
        return render(self.blocks, self.walls, self.goals)


def _bitboard(raw: bytes, offset: int) -> int:
    v = 0
    for i in range(8):
        v = (v << 8) | raw[offset + i]
    return v


def parse(encoded: str) -> Puzzle:
    """Decode the site's ``<base64url>.<moves>`` puzzle format.

    The base64 payload is 24 bytes: 8 each of walls, blocks and goals, most
    significant byte first. The suffix is the published solution as ``UDLR``.
    """
    b64, _, solution = encoded.partition(".")
    raw = base64.b64decode(b64.replace("-", "+").replace("_", "/") + "==")
    return Puzzle(
        walls=_bitboard(raw, 0),
        blocks=_bitboard(raw, 8),
        goals=_bitboard(raw, 16),
        solution=solution,
    )
