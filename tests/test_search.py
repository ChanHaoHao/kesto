"""The numpy search, and the equivalence that lets it stand in for the engine.

`kesto.search` never calls `kesto.board.move`; it runs `vmove`, the same rule
vectorised over an array of bitboards. Every answer the solver gives rests on
those two agreeing, so that is pinned here directly rather than inferred from
the solver happening to return the right lengths.
"""

from __future__ import annotations

import numpy as np
import pytest

from kesto import load, move
from kesto.grid import parse_grid
from kesto.search.backward import predecessors
from kesto.search.engine import U64, vmove

PUZZLES = load()
DIRS = "UDLR"


def _reachable(p, plies):
    """A spread of real states: everything within `plies` swipes of the start."""
    seen = {p.blocks}
    frontier = [p.blocks]
    for _ in range(plies):
        nxt = [move(s, p.walls, d) for s in frontier for d in DIRS]
        frontier = [s for s in nxt if s not in seen]
        seen.update(frontier)
    return sorted(seen)


@pytest.mark.parametrize("p", PUZZLES, ids=lambda p: p.slug)
def test_vmove_matches_the_reference_engine(p):
    """The bridge the whole package stands on, over every corpus board."""
    states = _reachable(p, 3)
    arr = np.array(states, dtype=U64)
    walls = U64(p.walls)
    for d in DIRS:
        got = vmove(arr, walls, d)
        want = [move(s, p.walls, d) for s in states]
        assert [int(v) for v in got] == want, f"{p.slug} disagrees on {d}"


@pytest.mark.parametrize("p", PUZZLES[:5], ids=lambda p: p.slug)
def test_predecessors_are_sound(p):
    """Pruning may cost recall, but every state returned must really step here."""
    target = np.array([p.goals], dtype=U64)
    walls = U64(p.walls)
    preds = predecessors(target, walls, p.n_blocks)
    for s in preds:
        assert any(int(vmove(np.array([s], U64), walls, d)[0]) == p.goals for d in DIRS)


def test_predecessors_find_the_real_one():
    """The state one swipe before the goal is a predecessor of it."""
    p = PUZZLES[0]
    before = p.apply(p.solution[:-1])
    walls = U64(p.walls)
    preds = predecessors(np.array([p.goals], dtype=U64), walls, p.n_blocks)
    assert before in {int(s) for s in preds}


def test_grid_roundtrips_through_render():
    """A rendered board pastes straight back in -- one charset, both ways."""
    from kesto import render

    for p in PUZZLES:
        back = parse_grid(render(p.blocks, p.walls, p.goals))
        assert (back.blocks, back.walls, back.goals) == (p.blocks, p.walls, p.goals)
