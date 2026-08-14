"""A* search over Kesto block configurations -- to be implemented.

BFS is optimal but explores every state within the solution radius, which gets
expensive past ~20 moves. A* keeps the optimality guarantee while skipping
states whose ``f = g + h`` already exceeds the best possible solution, provided
``h`` never overestimates the moves remaining.

The admissible bound this module is built around
--------------------------------------------------
Count the swipes by direction: ``R`` rights, ``L`` lefts, ``U`` ups, ``D``
downs, so the solution length is ``R + L + U + D``. A single right-swipe
increases any given block's ``x`` by at most one (blocking only ever reduces
movement), so no block's net ``+x`` displacement can exceed ``R``.

Now pick a threshold ``t`` and look at the region ``x >= t``. The final
position must hold exactly ``g = #goals with x >= t`` blocks there, while the
current position holds ``b = #blocks with x >= t``. So at least ``need = g - b``
blocks must cross the boundary rightward. The cheapest possible way for that to
happen is for the ``need`` nearest blocks to cross; the furthest of those sits
at some ``x = xv`` and needs ``t - xv`` right-swipes, and since that block's net
displacement is bounded by ``R``, we get ``R >= t - xv``. Maximise over all
``t`` for the tightest bound, and repeat the argument for left, up and down.
Summing the four independent bounds gives an admissible ``h``.

This bound has been checked against all fifteen bundled puzzles: it never
exceeded the true remaining distance at any point along their known-optimal
paths. It is, however, *loose* -- it recovers only about 25-40% of the true
depth on the hard puzzles -- so expect real but not dramatic pruning. Tightening
it (or replacing it outright) is the interesting part of the work.

Use :func:`kesto.verify.check_admissible` to test any candidate heuristic
against the bundled optimal paths before trusting it.
"""

from __future__ import annotations

import heapq
from collections.abc import Callable

from .bfs import Result
from .board import MOVES, Puzzle, cells, move

__all__ = ["axis_bound", "heuristic", "solve"]


def axis_bound(block_coords: list[int], goal_coords: list[int], increasing: bool) -> int:
    """Lower-bound the swipes needed in ONE direction along ONE axis.

    Implements the threshold-crossing argument from the module docstring for a
    single axis and a single sign.

    Args:
        block_coords: the current blocks' coordinates along this axis (one
            entry per block; order is irrelevant, duplicates are meaningful).
        goal_coords: the goal squares' coordinates along the same axis.
        increasing: ``True`` to bound the direction that increases the
            coordinate (right on x, down on y); ``False`` for the decreasing
            direction (left on x, up on y).

    Returns:
        The largest provable number of swipes in that direction, or ``0`` if no
        threshold yields a positive bound.

    Implementation notes:
        - Sweep thresholds ``t`` over ``range(1, 8)``.
        - For the increasing direction, the region is ``coord >= t`` and a
          block at ``v < t`` costs ``t - v`` to cross; take candidates in
          descending order so the nearest blocks are considered first.
        - For the decreasing direction, mirror it: the region is
          ``coord <= t - 1`` and a block at ``v > t - 1`` costs ``v - (t - 1)``;
          take candidates in ascending order.
        - Only draw a bound when ``need > 0`` and enough candidate blocks exist
          to supply it; the relevant candidate is the ``need``-th one.
    """

    best = 0
    for t in range(1, 8):
        if increasing:
            b = sum(1 for v in block_coords if v >= t)
            g = sum(1 for v in goal_coords if v >= t)
            need = g - b
            candidates = sorted((v for v in block_coords if v < t), reverse=True)
            if need > 0 and len(candidates) >= need:
                best = max(best, t - candidates[need - 1])
        else:
            b = sum(1 for v in block_coords if v <= t - 1)
            g = sum(1 for v in goal_coords if v <= t - 1)
            need = g - b
            candidates = sorted(v for v in block_coords if v > t - 1)
            if need > 0 and len(candidates) >= need:
                best = max(best, candidates[need - 1] - (t - 1))
    return best


def heuristic(blocks: int, goals: int) -> int:
    """Admissible estimate of the swipes still needed to reach ``goals``.

    Sum four independent :func:`axis_bound` results -- x-increasing,
    x-decreasing, y-increasing, y-decreasing. They are independent because each
    counts swipes of a different direction, and the four counts add up to the
    solution length.

    Args:
        blocks: current block bitboard.
        goals: goal bitboard (fixed for a given puzzle).

    Returns:
        A lower bound on remaining moves. Must return ``0`` when
        ``blocks == goals``, and must never exceed the true distance.

    Use :func:`kesto.board.cells` to expand the bitboards into ``(x, y)`` pairs.
    """

    if blocks == goals:
        return 0

    block_cells, goal_cells = cells(blocks), cells(goals)
    bx = [x for x, _ in block_cells]
    gx = [x for x, _ in goal_cells]
    by = [y for _, y in block_cells]
    gy = [y for _, y in goal_cells]

    return (
        axis_bound(bx, gx, True)
        + axis_bound(bx, gx, False)
        + axis_bound(by, gy, True)
        + axis_bound(by, gy, False)
    )


def solve(
    puzzle: Puzzle,
    h: Callable[[int, int], int] | None = None,
    max_states: int = 20_000_000,
) -> Result:
    """Find a shortest solution with A*, returning the same :class:`Result`
    shape as :func:`kesto.bfs.solve`.

    Args:
        puzzle: the puzzle to solve.
        h: heuristic taking ``(blocks, goals)``; defaults to :func:`heuristic`.
        max_states: cap on states recorded before giving up.

    Returns:
        A :class:`Result`. ``path`` is ``None`` if the cap was hit or the goal
        is unreachable, with ``exhausted`` distinguishing the two.

    Implementation notes:
        - Priority queue ordered by ``f = g + h``; ``heapq`` is enough. Push
          ``(f, g, state)`` and break ties toward larger ``g`` (deeper nodes)
          for a useful speed-up.
        - Track the best known ``g`` per state and skip a popped entry whose
          ``g`` is worse than the recorded one -- cheaper than decrease-key.
        - The bound in this module is admissible but not obviously consistent,
          so allow a state to be re-opened if you later reach it with a smaller
          ``g``, otherwise optimality is not guaranteed.
        - Test the goal on *pop*, not on generation. With an inconsistent
          heuristic, testing on generation can return a non-optimal path.
        - Successors come from ``move(state, puzzle.walls, d)`` for ``d`` in
          ``kesto.board.MOVES``; note a swipe may be a no-op, which naturally
          dedupes against the visited map.
    """

    h = heuristic if h is None else h
    start, goals = puzzle.blocks, puzzle.goals

    # Best g reached so far per state; doubles as the visited set. `parent`
    # carries the move that produced each state, for reconstruction.
    dist = {start: 0}
    parent: dict[int, tuple[int, str] | None] = {start: None}
    pq = [(h(start, goals), 0, start)]

    while pq:
        _, neg_g, state = heapq.heappop(pq)
        g = -neg_g
        if g > dist[state]:
            continue  # stale entry left behind by a re-open

        if state == goals:
            path = []
            while parent[state] is not None:
                state, m = parent[state]
                path.append(m)
            return Result("".join(reversed(path)), len(dist), False)

        for m in MOVES:
            nxt = move(state, puzzle.walls, m)
            if nxt not in dist or g + 1 < dist[nxt]:
                dist[nxt] = g + 1
                parent[nxt] = (state, m)
                heapq.heappush(pq, (g + 1 + h(nxt, goals), -(g + 1), nxt))

        if len(dist) > max_states:
            return Result(None, len(dist), False)

    return Result(None, len(dist), True)
