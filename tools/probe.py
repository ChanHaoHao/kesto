"""Level-synchronous BFS over uint64 bitboards, to size board.txt's state space.

Diagnostic only: no parent links, so it answers "how big / is the goal in there"
without the ~180 bytes/state that kesto.bfs.solve's dict costs.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the Kesto repo, for `import kesto` and solver.py
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from solver import load_puzzle

U64 = np.uint64
FULL = np.uint64(0xFFFFFFFFFFFFFFFF)
FILE0 = np.uint64(0x0101010101010101)
FILE7 = np.uint64(0x8080808080808080)
RANK0 = np.uint64(0x00000000000000FF)
RANK7 = np.uint64(0xFF00000000000000)
ONE, EIGHT = np.uint64(1), np.uint64(8)

_DIRS = {
    "R": (FILE7, lambda v: (v << ONE) & ~FILE0, lambda v: (v >> ONE) & ~FILE7),
    "L": (FILE0, lambda v: (v >> ONE) & ~FILE7, lambda v: (v << ONE) & ~FILE0),
    "D": (RANK7, lambda v: v << EIGHT, lambda v: v >> EIGHT),
    "U": (RANK0, lambda v: v >> EIGHT, lambda v: v << EIGHT),
}


def vmove(blocks, walls, direction):
    """kesto.board.move, vectorised over an array of block bitboards."""
    edge, forward, back = _DIRS[direction]
    stuck = (blocks & edge) | (blocks & back(walls))
    for _ in range(8):  # chains are at most 8 long, so this is the fixpoint
        stuck |= blocks & back(stuck)
    return stuck | forward(blocks & ~stuck)


def bfs(puzzle, max_states):
    walls = np.uint64(puzzle.walls)
    goal = np.uint64(puzzle.goals)
    visited = np.array([puzzle.blocks], dtype=U64)
    frontier = visited.copy()
    depth = 0
    while frontier.size:
        nxt = np.concatenate([vmove(frontier, walls, d) for d in "UDLR"])
        nxt = np.unique(nxt)
        frontier = nxt[np.isin(nxt, visited, assume_unique=True, invert=True)]
        depth += 1
        visited = np.union1d(visited, frontier)
        print(
            f"depth {depth:3d}  new {frontier.size:>12,}  total {visited.size:>13,}"
            f"  {time.perf_counter() - t0:8.1f}s",
            flush=True,
        )
        if (frontier == goal).any():
            print(f"\nGOAL REACHED at depth {depth}")
            return
        if visited.size > max_states:
            print(f"\ngave up: over cap at depth {depth}")
            return
    print(f"\nEXHAUSTED at depth {depth}: {visited.size:,} states, goal NOT reachable")


t0 = time.perf_counter()
if __name__ == "__main__":
    p = load_puzzle(sys.argv[1] if len(sys.argv) > 1 else "board.txt")
    bfs(p, int(sys.argv[2]) if len(sys.argv) > 2 else 120_000_000)
