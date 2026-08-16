"""Vectorised form of kesto.astar.heuristic, for scoring a whole frontier.

The scalar version expands each state into (x, y) pairs and sorts, which is
hopeless at millions of states per cull. The same bound falls out of bitboard
popcounts instead.

For the increasing direction, axis_bound looks for the `need`-th nearest block
below a threshold t. Because count(v <= coord < t) is non-increasing in v, the
set of v satisfying `count >= need` is a prefix [0..c], so the block it wants is
recovered by counting how many v qualify rather than by sorting:

    c = (#v in [0, t) with popcount(blocks & mask[v, t)) >= need) - 1
    bound_t = t - c = t - count + 1

The decreasing direction mirrors it over a suffix. `selftest` checks the result
against the scalar heuristic on random reachable states.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the Kesto repo, for `import kesto` and solver.py
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

N = 8
U64 = np.uint64


def _masks():
    """Half-open coordinate-range masks along each axis."""
    x_in = np.zeros((N + 1, N + 1), U64)  # x_in[a, b] : a <= x < b
    y_in = np.zeros((N + 1, N + 1), U64)
    for a in range(N + 1):
        for b in range(a, N + 1):
            mx = my = 0
            for i in range(64):
                x, y = i % N, i // N
                if a <= x < b:
                    mx |= 1 << i
                if a <= y < b:
                    my |= 1 << i
            x_in[a, b], y_in[a, b] = np.uint64(mx), np.uint64(my)
    return x_in, y_in


X_IN, Y_IN = _masks()


def _axis(blocks, goals, span, increasing):
    """Vectorised axis_bound over an array of block bitboards."""
    best = np.zeros(blocks.shape, np.int8)
    for t in range(1, N):
        if increasing:
            # region is coord >= t, i.e. the complement of [0, t)
            b = np.bitwise_count(blocks & span[0, t])
            g = int(np.bitwise_count(goals & span[0, t]))
            # need blocks to cross into the region, so compare the outside
            need = (int(np.bitwise_count(goals)) - g) - (
                np.bitwise_count(blocks).astype(np.int16) - b.astype(np.int16)
            )
            count = np.zeros(blocks.shape, np.int16)
            for v in range(t):
                count += np.bitwise_count(blocks & span[v, t]).astype(np.int16) >= need
            bound = t - count + 1
        else:
            b = np.bitwise_count(blocks & span[0, t]).astype(np.int16)
            g = int(np.bitwise_count(goals & span[0, t]))
            need = g - b
            count = np.zeros(blocks.shape, np.int16)
            for v in range(N, t - 1, -1):
                count += np.bitwise_count(blocks & span[t, v]).astype(np.int16) >= need
            # qualifying v form a suffix, so the need-th nearest block sits at
            # c = 8 - count and the bound is c - (t - 1).
            bound = N + 1 - count - t
        ok = (need > 0) & (count > 0)
        best = np.maximum(best, np.where(ok, bound, 0).astype(np.int8))
    return best


def heuristic(blocks, goals):
    """Admissible lower bound on moves remaining, for a uint64 array."""
    blocks = np.asarray(blocks, U64)
    g = U64(goals)
    h = (
        _axis(blocks, g, X_IN, True).astype(np.int16)
        + _axis(blocks, g, X_IN, False)
        + _axis(blocks, g, Y_IN, True)
        + _axis(blocks, g, Y_IN, False)
    )
    return np.where(blocks == g, 0, h).astype(np.int16)


def selftest(n=4000):
    import random

    from kesto.astar import heuristic as scalar
    from kesto.board import move
    from kesto.puzzles import load

    bad = 0
    for p in load():
        states = [p.blocks]
        for _ in range(n // len(load())):
            s = random.choice(states)
            states.append(move(s, p.walls, random.choice("UDLR")))
        got = heuristic(np.array(states, U64), p.goals)
        want = np.array([scalar(s, p.goals) for s in states], np.int16)
        bad += int((got != want).sum())
    print(f"vheur selftest: {bad} mismatches")
    return bad == 0


if __name__ == "__main__":
    raise SystemExit(0 if selftest() else 1)
