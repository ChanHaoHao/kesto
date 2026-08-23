"""Ply bookkeeping for a bidirectional search: sorted-array sets on disk.

Visited sets are kept as sorted uint64 arrays rather than a dict, which is the
whole reason boards past ply 17 are reachable -- 8 bytes a state against the
~149 a Python dict entry costs. Everything here is the set algebra that
representation needs, plus the scratch-directory handling for levels too large
to hold resident.
"""

from __future__ import annotations

import os
import re

import numpy as np

from .engine import U64, vmove

LEVEL_FILES = re.compile(r"(?:fwd|back)_\d{3}\.npy|puzzle\.json")
CHUNK_ROWS = 8 << 20

def check_counts(p):
    """Reject a board whose block and goal counts differ."""
    goals = p.goals.bit_count()
    if p.n_blocks != goals:
        raise SystemExit(f"unsolvable: {p.n_blocks} blocks but {goals} goals")


def clear_levels(out_dir):
    """Remove the levels and manifest from a directory, leaving anything else."""
    try:
        names = os.listdir(out_dir)
    except OSError:
        return 0, []
    freed = 0
    for n in names:
        if LEVEL_FILES.fullmatch(n):
            f = os.path.join(out_dir, n)
            freed += os.path.getsize(f)
            os.remove(f)
    return freed, os.listdir(out_dir)


def expand(layer, walls):
    """All successors of a layer, sorted and deduplicated, allocating once."""
    n = layer.size
    out = np.empty(n * 4, U64)
    for i, m in enumerate("UDLR"):
        out[i * n : (i + 1) * n] = vmove(layer, walls, m)
    out.sort()
    keep = np.empty(out.size, bool)
    keep[0] = True
    np.not_equal(out[1:], out[:-1], out=keep[1:])
    return out[keep]


def mask_not_in(cand, seen):
    """Boolean mask of `cand` entries absent from sorted `seen`."""
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

