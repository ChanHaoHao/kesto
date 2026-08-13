"""Solvers for Kesto, the daily 8x8 sliding-block puzzle.

One swipe moves every block one cell; the puzzle is solved when the blocks
exactly cover the goal squares. See :mod:`kesto.board` for the engine,
:mod:`kesto.bfs` for a working optimal solver, and :mod:`kesto.astar` for the
heuristic search.
"""

from .board import MOVES, Puzzle, cells, move, parse, popcount, render
from .puzzles import by_slug, load

__all__ = [
    "MOVES",
    "Puzzle",
    "by_slug",
    "cells",
    "load",
    "move",
    "parse",
    "popcount",
    "render",
]
