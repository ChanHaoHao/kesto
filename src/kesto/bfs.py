"""Breadth-first search over Kesto block configurations -- to be implemented.

Why this works at all: blocks are interchangeable, so a state is fully described
by the block bitboard. The visited set is therefore a plain set of 64-bit ints
with no permutation blow-up, and every edge costs one move -- which is exactly
the condition under which BFS returns an optimal solution.

What to expect once it runs. Cost is exponential in *solution depth*, not block
count: an 11-block puzzle at depth 16 finishes in ~13M states, while an 8-block
one at depth 36 does not finish at all under a 100M cap. Effective branching
after deduplication is roughly 3-4x. On the bundled corpus a correct
implementation solves 11 of 15 under a 20M cap, and in each of those the depth
it finds equals the published solution length -- the published solutions are
optimal, so that equality is the strongest correctness signal available here.

See :mod:`kesto.astar` for the heuristic search that is meant to close the gap
on the remaining puzzles.
"""

from __future__ import annotations

import collections
from typing import NamedTuple

from .board import MOVES, Puzzle, move

__all__ = ["Result", "solve", "reachable"]


class Result(NamedTuple):
    """Outcome of a search. Shared by :mod:`kesto.bfs` and :mod:`kesto.astar`."""

    path: str | None  # optimal move string, or None if no solution was found
    states: int  # states recorded before stopping
    exhausted: bool  # True if the whole reachable space was searched

    @property
    def found(self) -> bool:
        return self.path is not None

    @property
    def length(self) -> int | None:
        return None if self.path is None else len(self.path)


def solve(puzzle: Puzzle, max_states: int = 20_000_000) -> Result:
    """Find a shortest solution by breadth-first search.

    Args:
        puzzle: the puzzle to solve. The search runs from ``puzzle.blocks`` to
            ``puzzle.goals`` over the wall layout ``puzzle.walls``.
        max_states: stop and give up once this many states have been recorded.
            Guards against the 20+ move puzzles eating all available memory.

    Returns:
        A :class:`Result`. ``path`` is a shortest move string over ``UDLR``, or
        ``None`` if the cap was hit or the goal is unreachable -- ``exhausted``
        distinguishes those two cases (``True`` means the frontier emptied, so
        the goal genuinely cannot be reached).

    Implementation notes:
        - FIFO frontier; ``collections.deque`` with ``popleft`` is what you
          want. A list with ``pop(0)`` is O(n) per step and will crawl.
        - Keep one dict mapping ``state -> (parent_state, move)`` and let it
          serve as the visited set too, so membership and reconstruction share
          storage. Seed it with ``{start: None}`` to mark the root.
        - Generate successors with ``move(state, puzzle.walls, d)`` for ``d`` in
          ``kesto.board.MOVES``.
        - A swipe can leave the board unchanged (everything already jammed).
          That is not a special case -- the visited check absorbs it.
        - Because every edge costs one, you may test for the goal when a state
          is *generated* rather than when it is popped. That saves close to a
          full ply and is still optimal. (This is exactly the shortcut that is
          NOT safe in A* with an inconsistent heuristic.)
        - Handle ``puzzle.blocks == puzzle.goals`` up front and return an empty
          path.
        - Reconstruct by walking parent links back from the goal, collecting the
          moves, then reversing.
    """

    start = puzzle.blocks
    if start == puzzle.goals:
        return Result("", 1, False)

    q = collections.deque([start])
    visited = {start: None}

    def backtrack(state):
        path = []
        while visited[state] is not None:
            state, m = visited[state]
            path.append(m)
        return "".join(reversed(path))

    while q:
        state = q.popleft()
        for m in MOVES:
            next_state = move(state, puzzle.walls, m)
            if next_state not in visited:
                visited[next_state] = (state, m)
                # Tested here rather than at pop. Every edge costs one, so the
                # first time the goal is generated it already sits at minimum
                # depth -- and stopping now saves close to a full ply, which is
                # the difference between finishing under the cap and not.
                if next_state == puzzle.goals:
                    return Result(backtrack(next_state), len(visited), False)
                q.append(next_state)

        if len(visited) > max_states:
            # Giving up at the cap, not a proof of unreachability -- the
            # frontier still holds work, so exhausted stays False.
            return Result(None, len(visited), False)

    return Result(None, len(visited), True)


def reachable(puzzle: Puzzle, max_states: int = 20_000_000) -> int:
    """Count the states reachable from the start, ignoring the goal.

    Useful for sizing a search before committing to it: this is the true ceiling
    on what any exhaustive method must handle, and it is far smaller than the
    naive ``C(64 - walls, n_blocks)`` bound -- lockstep movement makes most
    configurations unreachable.

    Args:
        puzzle: the puzzle whose reachable space should be measured.
        max_states: stop early once this many distinct states have been seen.

    Returns:
        The number of distinct states seen, which is the full reachable count
        only if it came in under ``max_states``.

    Implementation note: the same traversal as :func:`solve` with the goal test
    removed and no need for parent links -- a plain ``set`` suffices.
    """
    
    start = puzzle.blocks
    q = collections.deque([start])
    visited = {start}

    while q:
        state = q.popleft()
        for m in MOVES:
            next_state = move(state, puzzle.walls, m)
            if next_state not in visited:
                visited.add(next_state)
                q.append(next_state)

        if len(visited) > max_states:
            break

    return len(visited)
