"""Level-synchronous BFS over uint64 bitboards, to size board.txt's state space."""
import argparse
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the Kesto repo, for `import kesto` and solver.py
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from solver import load_puzzle

t0 = time.perf_counter()

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


def log(msg):
    """Progress, on stderr so that stdout carries only the result.

    These runs are long and usually redirected; keeping the two streams apart is
    what lets `solve.py board.txt > answer.txt` still show its plies live.
    """
    print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", file=sys.stderr, flush=True)


def cap_memory(gb):
    """Hard address-space ceiling.

    Without this a bad size estimate lets the kernel OOM-killer pick a victim
    globally -- which is how an earlier run took the whole session down. With
    it, overshooting raises MemoryError inside this process and nothing else on
    the machine is touched.
    """
    limit = int(gb * 1024**3)
    resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    log(f"address space capped at {gb:.1f} GiB")


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


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
            f"  rss={rss_gb():5.2f}G  {time.perf_counter() - t0:8.1f}s",
            flush=True,
        )
        if (frontier == goal).any():
            print(f"\nGOAL REACHED at depth {depth}")
            return
        if visited.size > max_states:
            print(f"\ngave up: over cap at depth {depth}")
            return
    print(f"\nEXHAUSTED at depth {depth}: {visited.size:,} states, goal NOT reachable")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", nargs="?", default="board.txt")
    ap.add_argument("max_states", nargs="?", type=int, default=120_000_000)
    ap.add_argument("--mem-gb", type=float, default=4.0)
    args = ap.parse_args()

    # Before any allocation: the state cap alone does not bound memory, because
    # union1d holds the old and new visited arrays at once, so a single ply can
    # double the footprint between two checks of it.
    cap_memory(args.mem_gb)
    try:
        bfs(load_puzzle(args.puzzle), args.max_states)
    except MemoryError:
        print(f"\nout of memory under the {args.mem_gb:.1f} GiB cap", flush=True)
        sys.exit(1)
