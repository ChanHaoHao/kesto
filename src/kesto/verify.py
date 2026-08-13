"""Correctness harnesses for the engine and for candidate heuristics."""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

from .board import Puzzle, move
from .puzzles import load


class Violation(NamedTuple):
    """A point where a heuristic overestimated the true remaining distance."""

    slug: str
    step: int  # index along the known-optimal path
    estimate: int  # what the heuristic returned
    actual: int  # true moves remaining from here


def check_engine(puzzles: list[Puzzle] | None = None) -> list[str]:
    """Replay each published solution; return the slugs that fail to solve.

    An empty list means the move engine matches the real game on every
    bundled puzzle.
    """
    return [p.slug for p in (puzzles or load()) if not p.solves(p.solution)]


def check_admissible(
    h: Callable[[int, int], int], puzzles: list[Puzzle] | None = None
) -> list[Violation]:
    """Check a heuristic against the bundled known-optimal solutions.

    Walks each optimal path and asserts ``h(state, goals) <= moves remaining``
    at every step. An empty result is necessary for admissibility -- though not
    sufficient, since it only samples states lying on optimal paths.
    """
    out = []
    for p in puzzles or load():
        state = p.blocks
        for i, m in enumerate(p.solution):
            remaining = len(p.solution) - i
            est = h(state, p.goals)
            if est > remaining:
                out.append(Violation(p.slug, i, est, remaining))
            state = move(state, p.walls, m)
    return out
