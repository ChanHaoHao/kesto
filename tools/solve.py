"""Solve a Kesto board optimally, choosing depths and directions automatically.

Usage:
    python tools/solve.py board.txt
    python tools/solve.py 20260613
    python tools/solve.py board.txt --mem-gb 5

How it decides
--------------
Both ends are grown as exact BFS shells. A solution of length L exists iff
fwd[df] meets back[L - df] for some split, so lengths are tested in increasing
order and the first one that hits is optimal by construction -- there is no
heuristic anywhere in the answer, and no depth to guess.

Which side grows next is decided by whichever frontier is currently smaller,
since that is the cheaper ply to expand. Boards differ wildly in which direction
is cheap: some have tiny backward branching and huge forward branching, some the
reverse, so committing to one direction in advance is what makes a board look
unsolvable when it merely needed the other end.

Levels are checkpointed under --dir, so a re-run resumes rather than restarts,
and reconstruction streams them back one ply at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from meet import cap_memory, expand, intersect_sorted, mask_not_in, merge_sorted
from probe import U64, vmove
from vpred import predecessors

from solver import load_puzzle

t0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - t0:7.1f}s] {msg}", flush=True)


def default_mem_gb():
    """Most of currently-available RAM, leaving the rest of the machine alone."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return max(1.0, round(int(line.split()[1]) / 1024**2 * 0.6, 1))
    except OSError:
        pass
    return 4.0


def contains(sorted_arr, values):
    """Mask of `values` present in `sorted_arr`."""
    return ~mask_not_in(values, sorted_arr)


def puzzle_id(p):
    """Short digest of the board itself -- walls included, not just the name."""
    raw = f"{p.walls}:{p.blocks}:{p.goals}".encode()
    return hashlib.sha1(raw).hexdigest()[:8]


def claim_dir(out_dir, p, explicit):
    """Bind a checkpoint directory to one board, refusing a mismatched reuse.

    Levels are only meaningful for the board that produced them. Editing a
    board file while old levels sit under the same name silently resumes the
    wrong search -- which surfaced here as a bogus meet and a reconstruction
    that could not find its own parent.
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "puzzle.json")
    want = {"walls": p.walls, "blocks": p.blocks, "goals": p.goals}

    if os.path.exists(manifest):
        with open(manifest) as fh:
            got = json.load(fh)
        if got != want:
            raise SystemExit(
                f"checkpoint mismatch in {out_dir}\n"
                f"  those levels belong to a different board.\n"
                f"  delete the directory, or pass a different --dir."
            )
        return

    stale = [f for f in os.listdir(out_dir) if f.endswith(".npy")]
    if stale:
        # Pre-manifest directory: verify what can be verified before trusting it.
        f0, b0 = os.path.join(out_dir, "fwd_000.npy"), os.path.join(out_dir, "back_000.npy")
        ok = os.path.exists(f0) and os.path.exists(b0)
        if ok:
            ok = int(np.load(f0)[0]) == p.blocks and int(np.load(b0)[0]) == p.goals
        if not ok:
            raise SystemExit(
                f"checkpoint mismatch in {out_dir}\n"
                f"  those levels belong to a different board.\n"
                f"  delete the directory, or pass a different --dir."
            )
        log(f"adopting existing levels in {out_dir} (walls unverifiable, pre-manifest)")
        if explicit:
            log("  if that board's walls differ, delete the directory and re-run")

    with open(manifest, "w") as fh:
        json.dump(want, fh)


class Side:
    """One end of the search: exact BFS shells, checkpointed to disk."""

    def __init__(self, name, seed, step, out_dir):
        self.name, self.step, self.dir = name, step, out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.depth = 0
        self.frontier = np.array([seed], U64)
        self.seen = self.frontier.copy()
        self.exhausted = False
        self._sizes = {}
        if not os.path.exists(self.path(0)):
            np.save(self.path(0), self.frontier)
        self._resume()

    def path(self, d):
        return os.path.join(self.dir, f"{self.name}_{d:03d}.npy")

    def _resume(self):
        # Only the frontier is loaded here. `seen` costs gigabytes on a deep
        # board and is needed solely to extend, which a resumed run often never
        # does -- so it is rebuilt lazily on the first extend() instead.
        while os.path.exists(self.path(self.depth + 1)):
            self.depth += 1
        if self.depth:
            self.frontier = np.load(self.path(self.depth))
            self.seen = None
            log(f"{self.name}: resumed at ply {self.depth}")

    def _load_seen(self):
        if self.seen is not None:
            return
        self.seen = np.load(self.path(0))
        for d in range(1, self.depth + 1):
            self.seen = merge_sorted(self.seen, np.load(self.path(d)))
        log(f"{self.name}: seen rebuilt ({self.seen.size:,} states)")

    def extend(self):
        """Build the next ply. False if the space is exhausted."""
        self._load_seen()
        nxt = self.step(self.frontier, self.seen)
        layer = nxt[mask_not_in(nxt, self.seen)]
        del nxt
        if not layer.size:
            self.exhausted = True
            log(f"{self.name}: space exhausted at ply {self.depth}")
            return False
        self.depth += 1
        self.frontier = layer
        self.seen = merge_sorted(self.seen, layer)
        np.save(self.path(self.depth), layer)
        log(f"{self.name} ply {self.depth:3d}: +{layer.size:,} -> {self.seen.size:,}")
        return True

    def level(self, d):
        f = self.path(d)
        return np.load(f) if os.path.exists(f) else None

    def size(self, d):
        """States at ply d, read from the header without loading the array."""
        if d not in self._sizes:
            f = self.path(d)
            if not os.path.exists(f):
                return None
            self._sizes[d] = int(np.load(f, mmap_mode="r").shape[0])
        return self._sizes[d]


def climb(state, df, fwd, walls, popcnt):
    """Walk a meeting state back to the start through the forward levels."""
    moves, cur = [], np.uint64(state)
    for d in range(df, 0, -1):
        prev = fwd.level(d - 1)
        cands = predecessors(np.array([cur], U64), walls, popcnt)
        cands = cands[contains(prev, cands)]
        if not cands.size:
            raise RuntimeError(f"no predecessor of {cur} in fwd ply {d - 1}")
        pick = cands[0]
        for m in "UDLR":
            if vmove(np.array([pick], U64), walls, m)[0] == cur:
                moves.append(m)
                break
        cur = pick
    return "".join(reversed(moves))


def descend(state, db, back, walls):
    """Walk a meeting state forward to the goal through the backward levels."""
    moves, cur = [], np.uint64(state)
    for d in range(db, 0, -1):
        nxt_level = back.level(d - 1)
        for m in "UDLR":
            nxt = vmove(np.array([cur], U64), walls, m)[0]
            if contains(nxt_level, np.array([nxt], U64))[0]:
                moves.append(m)
                cur = nxt
                break
        else:
            raise RuntimeError(f"no successor of {cur} in back ply {d - 1}")
    return "".join(moves)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("puzzle", help="grid file, bundled slug, or encoded string")
    ap.add_argument("--mem-gb", type=float, default=None, help="RLIMIT_AS ceiling")
    ap.add_argument("--dir", default=None, help="checkpoint directory")
    ap.add_argument("--max-len", type=int, default=200, help="give up beyond this length")
    args = ap.parse_args()

    mem = args.mem_gb if args.mem_gb is not None else default_mem_gb()
    cap_memory(mem)

    p = load_puzzle(args.puzzle)
    if p.n_blocks != bin(p.goals).count("1"):
        print(f"unsolvable: {p.n_blocks} blocks but {bin(p.goals).count('1')} goals")
        return 1

    # Default directory carries the board digest, so editing a board file
    # lands in a fresh directory instead of resuming someone else's levels.
    tag = os.path.splitext(os.path.basename(args.puzzle))[0]
    out_dir = args.dir or os.path.join(ROOT, "work", f"{tag}-{puzzle_id(p)}")
    claim_dir(out_dir, p, explicit=args.dir is not None)
    walls, popcnt = np.uint64(p.walls), p.n_blocks

    if p.blocks == p.goals:
        print("\nOPTIMAL: 0 moves (already solved)")
        return 0

    fwd = Side("fwd", p.blocks, lambda L, seen: expand(L, walls), out_dir)
    back = Side("back", p.goals, lambda L, seen: predecessors(L, walls, popcnt, exclude=seen), out_dir)

    L = 1
    while L <= args.max_len:
        # Grow until L is decidable, always extending the cheaper frontier.
        while fwd.depth + back.depth < L:
            if fwd.exhausted and back.exhausted:
                print(f"\nUNSOLVABLE: both spaces exhausted, no meet up to {L - 1}")
                return 2
            grow_fwd = not fwd.exhausted and (
                back.exhausted or fwd.frontier.size <= back.frontier.size
            )
            (fwd if grow_fwd else back).extend()

        # Any one split decides L. Every shorter length has already been ruled
        # out, so if a length-L path exists it is optimal, and its state at
        # forward distance df must sit at backward distance exactly L - df --
        # otherwise the two halves would splice into something shorter. Since
        # every valid split is equally conclusive, take the cheapest: the cost
        # is driven by the smaller side, and shallow plies are millions of times
        # smaller than deep ones.
        best = None
        for df in range(max(0, L - back.depth), min(fwd.depth, L) + 1):
            sf, sb = fwd.size(df), back.size(L - df)
            if sf is None or sb is None:
                continue
            cost = min(sf, sb)
            if best is None or cost < best[0]:
                best = (cost, df)
        if best is not None:
            df = best[1]
            a, b = fwd.level(df), back.level(L - df)
            if a.size > b.size:  # scan the smaller, search the larger
                a, b = b, a
            hit = intersect_sorted(a, b)
            if hit.size:
                state = int(hit[0])
                path = climb(state, df, fwd, walls, popcnt) + descend(
                    state, L - df, back, walls
                )
                ok = p.solves(path)
                print(f"\nOPTIMAL: {len(path)} moves")
                print(f"path   : {path}")
                print(f"verify : {'OK -- replays onto every goal' if ok else 'FAILED'}")
                return 0 if ok else 3
        log(f"L={L}: no meet (fwd {fwd.depth} + back {back.depth})")
        L += 1

    print(f"\nno solution up to {args.max_len} moves")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
