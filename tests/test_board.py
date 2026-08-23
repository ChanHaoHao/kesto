"""The reference engine, against the fifteen published solutions.

`kesto.board.move` is the definition of a swipe everything else is checked
against, so these run first and cheaply. The published solutions are known
optimal and were accepted by the site, which makes replaying them a sharp test
rather than a smoke test.
"""

from __future__ import annotations

import pytest

from kesto import cells, load, move, parse, popcount
from kesto.verify import check_engine

PUZZLES = load()


def test_corpus_loaded():
    assert len(PUZZLES) == 15
    assert all(p.slug and p.solution for p in PUZZLES)


def test_engine_matches_published_solutions():
    assert check_engine(PUZZLES) == []


@pytest.mark.parametrize("p", PUZZLES, ids=lambda p: p.slug)
def test_block_count_matches_goal_count(p):
    assert popcount(p.blocks) == popcount(p.goals)


def test_parse_roundtrip():
    p = parse("gQAIACAAAIEAAAAMEhIMAABwUHAAAAAA.URRDRDRDDLDU")
    assert p.solution == "URRDRDRDDLDU"
    assert p.solves(p.solution)


def test_move_is_at_most_one_cell():
    """A swipe steps each block one cell -- it is not a slide-to-the-wall."""
    p = PUZZLES[0]
    before = set(cells(p.blocks))
    after = set(cells(move(p.blocks, p.walls, "R")))
    for x, y in after - before:
        assert (x - 1, y) in before
