"""Optimal solver for Kesto, the daily 8x8 sliding-block puzzle.

One swipe moves every block one cell; the puzzle is solved when the blocks
exactly cover the goal squares. :mod:`kesto.board` is the reference engine,
:mod:`kesto.vision` reads a board off a screenshot of the site, and
:mod:`kesto.search` finds the shortest solution for one.
"""

from .board import MOVES, Puzzle, cells, move, parse, popcount, render
from .grid import load_puzzle, parse_grid
from .puzzles import by_slug, load

__all__ = [
    "MOVES",
    "Puzzle",
    "by_slug",
    "cells",
    "load",
    "load_puzzle",
    "move",
    "parse",
    "parse_grid",
    "popcount",
    "render",
]
