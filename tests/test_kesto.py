"""Tests for the Kesto engine and for the search modules you implement.

The engine tests run always. The BFS and A* tests auto-skip while their modules
still raise NotImplementedError, so the suite stays green as you work and starts
enforcing each solver the moment it exists.
"""

from __future__ import annotations

import pytest

from kesto import cells, load, move, parse, popcount
from kesto.astar import heuristic
from kesto.astar import solve as astar_solve
from kesto.bfs import solve as bfs_solve
from kesto.verify import check_admissible, check_engine

PUZZLES = load()
# Puzzles BFS clears in roughly a second. Search cost grows with both depth and
# block count, so both are bounded here; the rest are exercised by hand with a
# much larger cap (see README).
FAST = [p for p in PUZZLES if len(p.solution) <= 18 and p.n_blocks <= 8]
CAP = 6_000_000


def _implemented(fn, *args) -> bool:
    """True unless the function is still a NotImplementedError stub."""
    try:
        fn(*args)
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


needs_bfs = pytest.mark.skipif(
    not _implemented(bfs_solve, PUZZLES[0], 1), reason="kesto.bfs not implemented yet"
)
needs_astar = pytest.mark.skipif(
    not _implemented(heuristic, 0, 0), reason="kesto.astar not implemented yet"
)


# --- engine -----------------------------------------------------------------


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


# --- bfs --------------------------------------------------------------------


@needs_bfs
def test_bfs_returns_empty_path_when_already_solved():
    p = PUZZLES[0]._replace(blocks=PUZZLES[0].goals)
    r = bfs_solve(p, max_states=CAP)
    assert r.found and r.path == ""


@needs_bfs
@pytest.mark.parametrize("p", FAST, ids=lambda p: p.slug)
def test_bfs_finds_published_optimum(p):
    r = bfs_solve(p, max_states=CAP)
    assert r.found, f"{p.slug} exceeded the state cap"
    assert p.solves(r.path), "returned path must actually solve the puzzle"
    assert r.length == len(p.solution), "published solutions are optimal"


@needs_bfs
def test_bfs_respects_state_cap():
    hard = max(PUZZLES, key=lambda p: len(p.solution))
    r = bfs_solve(hard, max_states=1000)
    assert not r.found and not r.exhausted and r.states <= 1000 + 4


# --- astar ------------------------------------------------------------------


@needs_astar
def test_heuristic_is_zero_at_goal():
    for p in PUZZLES:
        assert heuristic(p.goals, p.goals) == 0


@needs_astar
def test_heuristic_admissible_on_optimal_paths():
    assert check_admissible(heuristic, PUZZLES) == []


@needs_astar
@pytest.mark.parametrize("p", FAST, ids=lambda p: p.slug)
def test_astar_finds_published_optimum(p):
    r = astar_solve(p, max_states=CAP)
    assert r.found, f"{p.slug} exceeded the state cap"
    assert p.solves(r.path)
    assert r.length == len(p.solution)


@needs_bfs
@needs_astar
@pytest.mark.parametrize("p", FAST, ids=lambda p: p.slug)
def test_astar_explores_no_more_than_bfs(p):
    """The whole point of the heuristic: fewer states for the same answer."""
    assert astar_solve(p, max_states=CAP).states <= bfs_solve(p, max_states=CAP).states
