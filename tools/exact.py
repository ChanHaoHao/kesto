"""Exact optimal-length search by matched BFS levels, checkpointed to disk.

A solution of length L has, at forward distance df, a state whose backward
distance is exactly L - df. So deciding L needs one intersection per split, not
a whole search: L is achievable iff fwd[df] meets back[L - df] for some split.

Levels are built once and written to disk, so extending the reachable L later
costs only the new plies. Only `seen` (for dedup) and the working layer are ever
resident, which is what keeps the deep plies inside the memory cap.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the Kesto repo, for `import kesto` and solver.py
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)

from meet import (
    cap_memory,
    expand,
    intersect_sorted,
    log,
    mask_not_in,
    merge_sorted,
    rss_gb,
)
from probe import U64, vmove
from solve import claim_dir
from vpred import predecessors

from solver import load_puzzle


def build(kind, seed, step, out_dir, max_depth):
    """BFS levels in one direction, resuming from whatever is on disk."""
    os.makedirs(out_dir, exist_ok=True)

    def path(d):
        return os.path.join(out_dir, f"{kind}_{d:03d}.npy")

    have = [d for d in range(max_depth + 1) if os.path.exists(path(d))]
    if have and max(have) >= max_depth:
        log(f"{kind}: levels 0..{max_depth} already on disk")
        return
    if have:
        depth = max(have)
        layer = np.load(path(depth))
        seen = layer.copy()
        for d in range(depth):
            seen = merge_sorted(seen, np.load(path(d)))
        log(f"{kind}: resuming from ply {depth} ({seen.size:,} seen)")
    else:
        depth = 0
        layer = np.array([seed], U64)
        seen = layer.copy()
        np.save(path(0), layer)

    while depth < max_depth:
        depth += 1
        nxt = step(layer, seen)
        layer = nxt[mask_not_in(nxt, seen)]
        del nxt
        if not layer.size:
            log(f"{kind}: space CLOSED at ply {depth - 1}")
            return
        # Nothing is expanded from the final ply, so folding it into `seen`
        # would allocate a full extra copy for no benefit -- and at ply 18 that
        # copy is what decides whether the run fits in memory.
        if depth < max_depth:
            seen = merge_sorted(seen, layer)
            total = seen.size
        else:
            total = seen.size + layer.size  # `seen` deliberately not updated
        np.save(path(depth), layer)
        log(f"{kind} ply {depth:3d}: +{layer.size:,} -> {total:,}  rss={rss_gb():.2f}GiB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", nargs="?", default=os.path.join(ROOT, "board.txt"))
    ap.add_argument("--fwd-depth", type=int, default=16)
    ap.add_argument("--back-depth", type=int, default=11)
    ap.add_argument("--known-lower", type=int, default=27, help="L values already ruled out")
    ap.add_argument("--mem-gb", type=float, default=4.0)
    ap.add_argument("--dir", default=os.path.join(ROOT, "work", "levels"))
    args = ap.parse_args()

    cap_memory(args.mem_gb)
    p = load_puzzle(args.puzzle)
    claim_dir(args.dir, p, explicit=True)
    walls = np.uint64(p.walls)
    popcnt = p.goals.bit_count()

    build(
        "back",
        p.goals,
        lambda L, seen: predecessors(L, walls, popcnt, exclude=seen),
        args.dir,
        args.back_depth,
    )
    build("fwd", p.blocks, lambda L, seen: expand(L, walls), args.dir, args.fwd_depth)

    def load(kind, d):
        f = os.path.join(args.dir, f"{kind}_{d:03d}.npy")
        return np.load(f) if os.path.exists(f) else None

    reach = args.fwd_depth + args.back_depth
    log(f"levels cover L up to {reach}")
    for L in range(args.known_lower, reach + 1):
        for df in range(max(0, L - args.back_depth), min(args.fwd_depth, L) + 1):
            f, b = load("fwd", df), load("back", L - df)
            if f is None or b is None:
                continue
            hit = intersect_sorted(f, b)
            if hit.size:
                print(f"\nOPTIMUM IS {L}  (split {df} + {L - df}, {hit.size:,} meeting state(s))")
                np.save(os.path.join(args.dir, "meet_state.npy"), hit[:1])
                print(f"meeting state saved for reconstruction")
                return
        log(f"L={L}: no meet -- optimum is at least {L + 1}")

    print(f"\nno solution of length <= {reach}; optimum is at least {reach + 1}")


if __name__ == "__main__":
    main()
