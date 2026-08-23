"""Does the move engine match the real game?

The published solutions are the oracle: each one is a known-optimal path the
site accepted, so replaying it through `kesto.board.move` and landing on every
goal is a sharp check rather than a smoke test.
"""

from __future__ import annotations

from .board import Puzzle
from .puzzles import load


def check_engine(puzzles: list[Puzzle] | None = None) -> list[str]:
    """Replay each published solution; return the slugs that fail to solve.

    An empty list means the move engine matches the real game on every
    bundled puzzle.
    """
    return [p.slug for p in (puzzles or load()) if not p.solves(p.solution)]
