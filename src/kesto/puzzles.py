"""The bundled corpus of solved Kesto puzzles.

These are the fifteen published case studies from kestopuzzle.com, each with
the site's own solution. They double as the test set: a correct engine replays
every one of them onto the goal squares, and a correct optimal solver returns a
path of exactly the published length (all fifteen published solutions are
optimal, confirmed by breadth-first search).
"""

from __future__ import annotations

import json
from importlib import resources

from .board import Puzzle, parse

__all__ = ["load", "by_slug"]


def load() -> list[Puzzle]:
    """Return all bundled puzzles, with slug and title attached."""
    text = resources.files(__package__).joinpath("data/puzzles.json").read_text()
    out = []
    for row in json.loads(text):
        p = parse(row["encoded"])
        out.append(p._replace(slug=row["slug"], title=row["title"]))
    return out


def by_slug(slug: str) -> Puzzle:
    """Look up a single puzzle by its date slug, e.g. ``"20260608"``."""
    for p in load():
        if p.slug == slug:
            return p
    raise KeyError(f"no bundled puzzle with slug {slug!r}")
