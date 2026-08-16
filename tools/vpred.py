"""Vectorised backward BFS: which states can reach the goal?

Predecessors are still found by enumeration rather than inversion -- `move`
preserves block count and each block in T either stayed put (stuck) or advanced
one cell, so every predecessor is one "stayed/advanced" assignment -- but two
things make it cheap enough to run deep:

  * Per-block pruning. A block at cell c can only have stayed if the cell ahead
    of it is a wall/edge or holds another block of T (a stuck block stays, so it
    must still be there); it can only have advanced if the cell behind it exists
    and is not a wall. Most blocks fail one test, which is forced, so the free
    choices k are far below the block count and enumeration is 2^k, not 2^16.
  * Batching. States sharing a k are enumerated as one (n, 2^k) array and
    validated with a single vectorised `move`, so the whole layer is a handful
    of numpy passes.

The forward engine is the referee throughout: a candidate is a predecessor only
if vmove(candidate) == T, so pruning can only ever cost recall, never
soundness.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the Kesto repo, for `import kesto` and solver.py
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from probe import U64, vmove  # the validated forward engine

from solver import load_puzzle

N = 8
# Cell offsets: `ahead` is where a block is heading, `behind` is where a block
# that lands on this cell must have come from.
_STEP = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}

# Candidate budget per batch, in (state, assignment) pairs.
CHUNK = 1 << 21


def _tables(direction, walls):
    """Per-cell ahead/behind indices and the static half of the two options."""
    dx, dy = _STEP[direction]
    ahead = np.full(64, -1, np.int64)
    behind = np.full(64, -1, np.int64)
    for i in range(64):
        x, y = i % N, i // N
        ax, ay = x + dx, y + dy
        bx, by = x - dx, y - dy
        if 0 <= ax < N and 0 <= ay < N:
            ahead[i] = ay * N + ax
        if 0 <= bx < N and 0 <= by < N:
            behind[i] = by * N + bx
    wall = np.array([(walls >> i) & 1 for i in range(64)], bool)
    # Advancing is possible only from a real, non-wall cell.
    can_move = (behind >= 0) & ~wall[np.maximum(behind, 0)]
    can_move &= behind >= 0
    # Staying requires being stuck: blocked ahead by the edge or a wall. The
    # remaining case (blocked by another stuck block) depends on T, so it is
    # resolved per state in _options.
    blocked = (ahead < 0) | wall[np.maximum(ahead, 0)]
    # Plain Python scalars: _options shifts 64-bit ints, which numpy scalars
    # cannot index into without overflowing a C long.
    return (
        ahead.tolist(),
        behind.tolist(),
        can_move.tolist(),
        blocked.tolist(),
    )


def _options(state, ahead, behind, can_move, blocked):
    """Split one state's blocks into forced bits and free (stay, move) pairs.

    Returns (base, pairs) or None if some block admits neither option, which
    means this state has no predecessor in this direction at all.
    """
    base = 0
    pairs = []
    s = state
    while s:
        i = (s & -s).bit_length() - 1
        s &= s - 1
        a = ahead[i]
        stay = blocked[i] or (a >= 0 and (state >> a) & 1)
        mv = can_move[i]
        if stay and mv:
            pairs.append((1 << i, 1 << behind[i]))
        elif stay:
            base |= 1 << i
        elif mv:
            base |= 1 << behind[i]
        else:
            return None
    return base, pairs


# Fold partial results into the running set once this many candidates pile up.
# Accumulating every hit and deduping once at the end costs peak memory
# proportional to the raw predecessor count (with multiplicity), which is what
# put backward ply 12 out of reach.
FLUSH = 16 << 20
# States whose Python option tuples are materialised at once.
STATE_CHUNK = 1 << 18


def _not_in(cand, seen):
    """Mask of `cand` absent from sorted `seen`, in bounded slices.

    Duplicated from meet.mask_not_in rather than imported: meet imports this
    module, so the reverse import would be circular.
    """
    out = np.empty(cand.size, bool)
    if seen is None or seen.size == 0:
        out[:] = True
        return out
    for lo in range(0, cand.size, 8 << 20):
        c = cand[lo : lo + (8 << 20)]
        idx = np.searchsorted(seen, c)
        np.clip(idx, 0, seen.size - 1, out=idx)
        out[lo : lo + (8 << 20)] = seen[idx] != c
    return out


def _dedupe_inplace(a, chunk=8 << 20):
    """Compact a sorted array in place, returning a view of the filled prefix.

    `a[mask]` would allocate a second full-size array -- 1.28 GiB at ply 14, and
    exactly the allocation that failed. Deduplication of a sorted array only
    moves elements leftward, so each slice can be written back over the buffer
    it came from, provided the previous slice's original last value is carried
    across the boundary before being overwritten.
    """
    n, prev = 0, None
    for lo in range(0, a.size, chunk):
        block = a[lo : lo + chunk].copy()  # bounded: the only extra memory here
        mask = np.empty(block.size, bool)
        mask[0] = True if prev is None else block[0] != prev
        np.not_equal(block[1:], block[:-1], out=mask[1:])
        prev = block[-1]
        vals = block[mask]
        a[n : n + vals.size] = vals
        n += vals.size
    return a[:n]


def _fold(acc, buf, exclude=None):
    """Merge buffered hits into the running set, allocating one array.

    np.unique(np.concatenate(...)) needs several full-size temporaries, which is
    what ran the accumulator out of memory at ply 13. Filling one block and
    sorting it in place costs one. Dropping states already in `exclude` here
    keeps the accumulator down to genuinely new states rather than every
    predecessor rediscovered from earlier plies.
    """
    if not buf:
        return acc
    new = np.concatenate(buf)
    if exclude is not None:
        new = new[_not_in(new, exclude)]
    if acc is None or acc.size == 0:
        merged = new
    else:
        merged = np.empty(acc.size + new.size, U64)
        merged[: acc.size] = acc
        merged[acc.size :] = new
        del new
    if merged.size == 0:
        return merged
    merged.sort()
    return _dedupe_inplace(merged)


def predecessors(states, walls, popcnt, exclude=None):
    """All predecessors of every state in `states` (a uint64 array).

    `exclude` (sorted) is dropped as results are folded, so callers doing BFS
    can hand in their visited set and keep the accumulator small.
    """
    acc, out, pending = None, [], 0
    wall_i = int(walls)
    for direction in "UDLR":
        ahead, behind, can_move, blocked = _tables(direction, wall_i)
        # `by_k` holds Python option tuples, which cost far more per state than
        # the bitboards do. Building it for a whole 15M-state layer at once is
        # what exhausted memory, so the input is walked in slices.
        for s0 in range(0, states.size, STATE_CHUNK):
            by_k = {}
            for st in states[s0 : s0 + STATE_CHUNK].tolist():
                got = _options(int(st), ahead, behind, can_move, blocked)
                if got is None:
                    continue
                base, pairs = got
                by_k.setdefault(len(pairs), []).append((base, pairs, int(st)))

            for k, rows in by_k.items():
                width = 1 << k
                idx = np.arange(width, dtype=U64)
                # Enumerate in chunks so the (n, 2^k) block stays bounded.
                for lo in range(0, len(rows), max(1, CHUNK // width)):
                    batch = rows[lo : lo + max(1, CHUNK // width)]
                    cand = np.array([b for b, _, _ in batch], dtype=U64)[:, None]
                    cand = np.repeat(cand, width, axis=1)
                    for j in range(k):
                        take = ((idx >> U64(j)) & U64(1)).astype(bool)
                        stay = np.array([p[j][0] for _, p, _ in batch], dtype=U64)
                        mvto = np.array([p[j][1] for _, p, _ in batch], dtype=U64)
                        cand |= np.where(take[None, :], mvto[:, None], stay[:, None])
                    target = np.array([t for _, _, t in batch], dtype=U64)[:, None]
                    ok = np.bitwise_count(cand) == popcnt
                    ok &= vmove(cand, walls, direction) == target
                    keep = cand[ok]
                    out.append(keep)
                    pending += keep.size
                    if pending >= FLUSH:
                        acc, out, pending = _fold(acc, out, exclude), [], 0
            del by_k
    acc = _fold(acc, out, exclude)
    return np.empty(0, U64) if acc is None else acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", nargs="?", default=os.path.join(ROOT, "board.txt"))
    ap.add_argument("--depth", type=int, default=40)
    ap.add_argument("--cap", type=int, default=60_000_000)
    args = ap.parse_args()

    p = load_puzzle(args.puzzle)

    walls = np.uint64(p.walls)
    popcnt = p.goals.bit_count()
    if popcnt != p.blocks.bit_count():
        print(f"block/goal count mismatch: {p.blocks.bit_count()} vs {popcnt}")
        return

    start = np.uint64(p.blocks)
    seen = np.array([p.goals], U64)
    layer = seen.copy()
    t0 = time.perf_counter()
    for depth in range(1, args.depth + 1):
        nxt = predecessors(layer, walls, popcnt)
        layer = nxt[np.isin(nxt, seen, assume_unique=True, invert=True)]
        seen = np.union1d(seen, layer)
        print(
            f"back-depth {depth:3d}  new {layer.size:>12,}  total {seen.size:>13,}"
            f"  {time.perf_counter() - t0:8.1f}s",
            flush=True,
        )
        if (layer == start).any():
            print(f"\nSTART REACHED -- board is SOLVABLE, optimal depth {depth}")
            return
        if not layer.size:
            print(f"\nbackward space CLOSED at {seen.size:,} states")
            print("the start is not among them -- board is UNSOLVABLE")
            return
        if seen.size > args.cap:
            print(f"\ngave up: over cap at depth {depth}")
            return


if __name__ == "__main__":
    main()
