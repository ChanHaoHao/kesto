"""Vectorised move engine over uint64 bitboards, plus the run's memory guards.

The leaf of `kesto.search`: everything else here is built on `vmove`, which is
`kesto.board.move` applied to a whole numpy array of block bitboards at once.
That equivalence is what lets the searches in this package stand in for the
reference engine the puzzle corpus validates, and `tests/test_search.py` pins it.
"""

from __future__ import annotations

import resource
import sys
import time

import numpy as np

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

