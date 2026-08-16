"""Bidirectional search: does board.txt's start meet its goal's backward basin?

Backward branching (~3.1x) is worse than forward (~2.3x), so the budget is spent
asymmetrically: a shallow backward basin B, then a deep forward sweep that
intersects each new level against B. A hit at forward depth df on a state whose
backward depth is db proves a solution of length df + db, which is then replayed
through the forward engine to confirm.

No hit is not a proof of unsolvability -- it only rules out solutions up to the
combined depth reached.
"""
from __future__ import annotations

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
sys.path.insert(0, HERE)

from probe import U64, vmove
from vpred import predecessors

from solver import load_puzzle

t0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)


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


def expand(layer, walls):
    """All successors of a layer, sorted and deduplicated, allocating once.

    np.unique would concatenate into one buffer and sort into another; filling
    a preallocated block and sorting it in place halves peak memory, which is
    what decides whether the deep plies survive.
    """
    n = layer.size
    out = np.empty(n * 4, U64)
    for i, m in enumerate("UDLR"):
        out[i * n : (i + 1) * n] = vmove(layer, walls, m)
    out.sort()
    keep = np.empty(out.size, bool)
    keep[0] = True
    np.not_equal(out[1:], out[:-1], out=keep[1:])
    return out[keep]


CHUNK_ROWS = 8 << 20


def mask_not_in(cand, seen):
    """Boolean mask of `cand` entries absent from sorted `seen`.

    np.isin argsorts the concatenation of both inputs, which at ply 18 means a
    2.2 GiB int64 index array. Both sides are already sorted here, so a chunked
    searchsorted answers the same question with bounded temporaries.
    """
    out = np.empty(cand.size, bool)
    if seen.size == 0:
        out[:] = True
        return out
    for lo in range(0, cand.size, CHUNK_ROWS):
        c = cand[lo : lo + CHUNK_ROWS]
        idx = np.searchsorted(seen, c)
        np.clip(idx, 0, seen.size - 1, out=idx)
        out[lo : lo + CHUNK_ROWS] = seen[idx] != c
    return out


def intersect_sorted(a, b):
    """Values present in both sorted arrays, without a full concatenate+sort."""
    if a.size == 0 or b.size == 0:
        return np.empty(0, U64)
    parts = []
    for lo in range(0, a.size, CHUNK_ROWS):
        c = a[lo : lo + CHUNK_ROWS]
        idx = np.searchsorted(b, c)
        np.clip(idx, 0, b.size - 1, out=idx)
        hit = b[idx] == c
        if hit.any():
            parts.append(c[hit])
    return np.concatenate(parts) if parts else np.empty(0, U64)


def merge_sorted(seen, layer):
    """Union of two sorted arrays with no overlap, allocating once."""
    merged = np.empty(seen.size + layer.size, U64)
    merged[: seen.size] = seen
    merged[seen.size :] = layer
    merged.sort()
    return merged


def backward_basin(p, walls, popcnt, max_depth, cap):
    """States within `max_depth` of the goal, sorted, with per-state depth."""
    seen = np.array([p.goals], U64)
    depth_of = np.zeros(1, np.uint8)
    layer = seen.copy()
    for d in range(1, max_depth + 1):
        nxt = predecessors(layer, walls, popcnt)
        layer = nxt[np.isin(nxt, seen, assume_unique=True, invert=True)]
        if not layer.size:
            log(f"backward basin CLOSED at depth {d - 1}: {seen.size:,} states")
            return seen, depth_of, True
        both = np.concatenate([seen, layer])
        depth_of = np.concatenate([depth_of, np.full(layer.size, d, np.uint8)])
        order = np.argsort(both, kind="stable")
        seen, depth_of = both[order], depth_of[order]
        del both, order
        log(f"backward depth {d:3d}: +{layer.size:,} -> {seen.size:,}  rss={rss_gb():.2f}GiB")
        if seen.size > cap:
            break
    return seen, depth_of, False


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def forward_meet(p, walls, basin, bdepth, max_depth, cap, levels=None, ckpt=None):
    """BFS from the start, intersecting each new level against the basin.

    Pass `levels` (an empty list) to retain each ply for reconstruction; that
    roughly doubles peak memory, so it is off by default.
    """
    seen = np.array([p.blocks], U64)
    layer = seen.copy()
    if levels is not None:
        levels.append(layer.copy())
    hit = np.intersect1d(layer, basin)
    if hit.size:
        return 0, hit, bdepth[np.searchsorted(basin, hit)]
    for d in range(1, max_depth + 1):
        nxt = expand(layer, walls)
        layer = nxt[np.isin(nxt, seen, assume_unique=True, invert=True)]
        del nxt
        if not layer.size:
            log(f"forward space EXHAUSTED at depth {d - 1}")
            return None, None, None
        seen = merge_sorted(seen, layer)
        if levels is not None:
            levels.append(layer.copy())
        if ckpt:
            np.save(os.path.join(ckpt, f"fwd_{d:02d}.npy"), layer)
        hit = np.intersect1d(layer, basin)
        log(
            f"forward depth {d:3d}: +{layer.size:,} -> {seen.size:,}"
            f"  meets={hit.size:,}  rss={rss_gb():.2f}GiB"
        )
        if hit.size:
            return d, hit, bdepth[np.searchsorted(basin, hit)]
        if seen.size > cap:
            log(f"forward gave up at depth {d}")
            return None, None, None
    return None, None, None


def descend(state, basin, bdepth, walls):
    """Walk a basin state to the goal, one decreasing backward depth at a time."""
    moves = []
    cur = np.uint64(state)
    k = int(bdepth[np.searchsorted(basin, cur)])
    while k:
        for m in "UDLR":
            nxt = vmove(np.array([cur], U64), walls, m)[0]
            j = np.searchsorted(basin, nxt)
            if j < basin.size and basin[j] == nxt and int(bdepth[j]) == k - 1:
                moves.append(m)
                cur, k = nxt, k - 1
                break
        else:
            raise RuntimeError(f"stuck at backward depth {k}")
    return "".join(moves)


def climb(state, levels, walls, popcnt):
    """Walk a forward state back to the start through the stored levels."""
    moves = []
    cur = np.uint64(state)
    for d in range(len(levels) - 1, 0, -1):
        prev = predecessors(np.array([cur], U64), walls, popcnt)
        cands = prev[np.isin(prev, levels[d - 1], assume_unique=True)]
        if not cands.size:
            raise RuntimeError(f"no predecessor of {cur} in level {d - 1}")
        pick = cands[0]
        for m in "UDLR":
            if vmove(np.array([pick], U64), walls, m)[0] == cur:
                moves.append(m)
                break
        cur = pick
    return "".join(reversed(moves))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", nargs="?", default=os.path.join(ROOT, "board.txt"))
    ap.add_argument("--back-depth", type=int, default=10)
    ap.add_argument("--back-cap", type=int, default=14_000_000)
    ap.add_argument("--fwd-depth", type=int, default=17)
    ap.add_argument("--fwd-cap", type=int, default=110_000_000)
    ap.add_argument(
        "--reconstruct",
        action="store_true",
        help="retain forward levels so a meet yields a replayable move string",
    )
    ap.add_argument("--mem-gb", type=float, default=3.0, help="hard RLIMIT_AS ceiling")
    ap.add_argument("--ckpt", help="directory to write each forward level into")
    args = ap.parse_args()

    cap_memory(args.mem_gb)
    if args.ckpt:
        os.makedirs(args.ckpt, exist_ok=True)
    p = load_puzzle(args.puzzle)
    walls = np.uint64(p.walls)
    popcnt = p.goals.bit_count()

    basin, bdepth, closed = backward_basin(p, walls, popcnt, args.back_depth, args.back_cap)
    if closed:
        solvable = np.uint64(p.blocks) in set(basin.tolist())
        print(f"\nbasin is complete -- board is {'SOLVABLE' if solvable else 'UNSOLVABLE'}")
        return

    levels = [] if args.reconstruct else None
    df, hits, dbs = forward_meet(
        p, walls, basin, bdepth, args.fwd_depth, args.fwd_cap, levels, args.ckpt
    )
    if df is None:
        print(f"\nno meeting state up to forward {args.fwd_depth} + backward {args.back_depth}")
        print(f"=> no solution of length <= {args.fwd_depth + args.back_depth}; longer ones not ruled out")
        return

    db = int(dbs.min())
    total = int(df) + db
    print(f"\nMEET at forward depth {df}, backward depth {db}")
    print(f"=> board is SOLVABLE in {total} moves")

    if levels is not None:
        state = int(hits[int(np.argmin(dbs))])
        path = climb(state, levels, walls, popcnt) + descend(state, basin, bdepth, walls)
        print(f"path: {path} ({len(path)} moves)")
        print(f"replay through the forward engine: {'OK' if p.solves(path) else 'FAILS'}")


if __name__ == "__main__":
    main()
