"""Staged beam search: expand a few plies, cull to the closest survivors, repeat.

Exact BFS dies at ply 17 because every ply must be kept. Culling caps the
frontier instead, so depth costs linear memory rather than exponential -- the
trade is completeness: if the solution path ever falls outside the kept
fraction, it is gone for good. A hit is therefore proof of solvability, but a
miss proves nothing.

Two things make the cull worth trusting:
  * `vheur.heuristic` is the admissible bound from kesto.astar -- a genuine
    lower bound on moves remaining, not a guess -- with blocks-already-on-goal
    as a fine-grained tiebreak, since the bound itself is coarse.
  * The backward basin from `meet` gives an exact distance-to-goal for any state
    within its radius, so touching it ends the search immediately with a
    complete, replayable path.

Plies are checkpointed to disk, so reconstruction never needs them all resident.
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
sys.path.insert(0, HERE)

import vheur
from meet import (
    backward_basin,
    cap_memory,
    descend,
    expand,
    intersect_sorted,
    log,
    mask_not_in,
    merge_sorted,
    rss_gb,
)
from probe import U64, vmove
from vpred import predecessors

from solver import load_puzzle


def cull(cand, goals, keep):
    """Keep the `keep` states closest to the goal by the admissible bound."""
    h = vheur.heuristic(cand, goals).astype(np.int32)
    on_goal = np.bitwise_count(cand & U64(goals)).astype(np.int32)
    key = h * 17 - on_goal  # bound dominates; on-goal breaks its ties
    idx = np.argpartition(key, keep)[:keep]
    return np.sort(cand[idx]), int(h.min())


def climb_disk(state, depth, ckpt, walls, popcnt):
    """Walk a state back to the start through the checkpointed plies."""
    moves = []
    cur = np.uint64(state)
    for d in range(depth, 0, -1):
        prev = np.load(os.path.join(ckpt, f"ply_{d - 1:03d}.npy"))
        cands = predecessors(np.array([cur], U64), walls, popcnt)
        cands = cands[np.isin(cands, prev, assume_unique=True)]
        if not cands.size:
            raise RuntimeError(f"no predecessor of {cur} in ply {d - 1}")
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
    ap.add_argument("--keep", type=int, default=300_000, help="frontier size after a cull")
    ap.add_argument("--stride", type=int, default=3, help="plies expanded between culls")
    ap.add_argument("--max-depth", type=int, default=60)
    ap.add_argument("--visited-cap", type=int, default=40_000_000)
    ap.add_argument("--mem-gb", type=float, default=4.0)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "work", "beam_levels"))
    ap.add_argument(
        "--levels-dir",
        help="reuse back_*.npy from exact.py instead of rebuilding the basin",
    )
    args = ap.parse_args()

    cap_memory(args.mem_gb)
    os.makedirs(args.ckpt, exist_ok=True)
    for f in os.listdir(args.ckpt):
        os.remove(os.path.join(args.ckpt, f))

    p = load_puzzle(args.puzzle)
    walls = np.uint64(p.walls)
    popcnt = p.goals.bit_count()

    if args.levels_dir:
        # Levels from exact.py are already exact BFS shells, so depth is implied
        # by which file a state came from -- no need to rebuild the basin.
        parts, depths = [], []
        for d in range(args.back_depth + 1):
            f = os.path.join(args.levels_dir, f"back_{d:03d}.npy")
            if not os.path.exists(f):
                break
            lvl = np.load(f)
            parts.append(lvl)
            depths.append(np.full(lvl.size, d, np.uint8))
        both = np.concatenate(parts)
        bd = np.concatenate(depths)
        order = np.argsort(both, kind="stable")
        basin, bdepth, closed = both[order], bd[order], False
        deepest = len(parts) - 1
        del both, bd, order, parts, depths
        log(f"basin loaded from disk: {basin.size:,} states to depth {deepest}")
    else:
        basin, bdepth, closed = backward_basin(
            p, walls, popcnt, args.back_depth, args.back_cap
        )
    if closed:
        solvable = np.uint64(p.blocks) in set(basin.tolist())
        print(f"\nbasin complete -- board is {'SOLVABLE' if solvable else 'UNSOLVABLE'}")
        return

    frontier = np.array([p.blocks], U64)
    np.save(os.path.join(args.ckpt, "ply_000.npy"), frontier)
    visited = frontier.copy()
    recent = [frontier]

    for d in range(1, args.max_depth + 1):
        cand = expand(frontier, walls)
        cand = cand[mask_not_in(cand, visited)]
        if not cand.size:
            print(f"\nfrontier died at ply {d} -- nothing new reachable")
            return

        hit = intersect_sorted(cand, basin)
        if hit.size:
            db = bdepth[np.searchsorted(basin, hit)]
            best = int(np.argmin(db))
            state, db = int(hit[best]), int(db[best])
            np.save(os.path.join(args.ckpt, f"ply_{d:03d}.npy"), cand)
            log(f"ply {d}: MEET on {hit.size:,} state(s), backward depth {db}")
            path = climb_disk(state, d, args.ckpt, walls, popcnt) + descend(
                state, basin, bdepth, walls
            )
            print(f"\nSOLVED in {len(path)} moves: {path}")
            print(f"replay through the forward engine: {'OK' if p.solves(path) else 'FAILS'}")
            return

        hmin = None
        if d % args.stride == 0 and cand.size > args.keep:
            cand, hmin = cull(cand, p.goals, args.keep)
        note = f"  h_min={hmin}" if hmin is not None else ""
        log(f"ply {d:3d}: {cand.size:>10,} states  rss={rss_gb():.2f}GiB{note}")

        np.save(os.path.join(args.ckpt, f"ply_{d:03d}.npy"), cand)
        frontier = cand
        recent.append(cand)
        visited = merge_sorted(visited, cand[mask_not_in(cand, visited)])
        while visited.size > args.visited_cap and len(recent) > args.stride + 2:
            recent.pop(0)
            visited = recent[0]
            for r in recent[1:]:
                visited = merge_sorted(visited, r[mask_not_in(r, visited)])

    print(f"\nno meet within {args.max_depth} plies")


if __name__ == "__main__":
    main()
