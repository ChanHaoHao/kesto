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
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..grid import load_puzzle
from .backward import predecessors
from .engine import U64, cap_memory, log, vmove
from .levels import (
    check_counts,
    clear_levels,
    expand,
    intersect_sorted,
    mask_not_in,
    merge_sorted,
)

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


def discard_levels(out_dir):
    """Delete this run's levels."""
    try:
        freed, leftover = clear_levels(out_dir)
        if not leftover:
            os.rmdir(out_dir)
    except OSError as e:
        log(f"could not clear {out_dir}: {e}")
        return
    log(
        f"levels discarded ({freed / 1024**3:.2f} GiB freed)"
        + (f"; {len(leftover)} foreign file(s) left in place" if leftover else "")
    )


class Side:
    """One end of the search: exact BFS shells."""
    def __init__(self, name, seed, step, out_dir):
        self.name, self.step, self.dir = name, step, out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.depth = 0
        self.frontier = np.array([seed], U64)
        self.seen = self.frontier.copy()
        self.exhausted = False
        self._sizes = {}
        self._save(0, self.frontier)

    def path(self, d):
        return os.path.join(self.dir, f"{self.name}_{d:03d}.npy")

    def _save(self, d, layer):
        """Write a ply, stopping the run if the scratch directory vanished.

        Recreating it would be worse than failing: the earlier plies are gone
        with it, so reconstruction could not walk back to the start, and the
        search would carry on and report a length it can no longer justify.
        """
        if not os.path.isdir(self.dir):
            raise SystemExit(
                f"{self.dir} disappeared mid-run.\n"
                f"  The plies written so far went with it, so no path could be\n"
                f"  reconstructed. Nothing else should touch this directory while\n"
                f"  a solve is running -- check for a second solve.py on this board."
            )
        np.save(self.path(d), layer)

    def extend(self):
        """Build the next ply. False if the space is exhausted."""
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
        self._save(self.depth, layer)
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


def search(p, out_dir, max_len):
    """Run the bidirectional search. Returns the process exit code."""
    walls, popcnt = np.uint64(p.walls), p.n_blocks

    if p.blocks == p.goals:
        print("\nOPTIMAL: 0 moves (already solved)")
        return 0

    fwd = Side("fwd", p.blocks, lambda L, seen: expand(L, walls), out_dir)
    back = Side("back", p.goals, lambda L, seen: predecessors(L, walls, popcnt, exclude=seen), out_dir)

    L = 1
    while L <= max_len:
        # Grow until L is decidable, always extending the cheaper frontier.
        while fwd.depth + back.depth < L:
            if fwd.exhausted and back.exhausted:
                print(f"\nUNSOLVABLE: both spaces exhausted, no meet up to {L - 1}")
                return 2
            grow_fwd = not fwd.exhausted and (
                back.exhausted or fwd.frontier.size <= back.frontier.size
            )
            (fwd if grow_fwd else back).extend()

        best = None
        for df in range(max(0, L - back.depth), min(fwd.depth, L) + 1):
            sf, sb = fwd.size(df), back.size(L - df)
            if sf is None or sb is None:
                continue
            cost = sf + sb
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

    print(f"\nno solution up to {max_len} moves")
    return 2


def run(p, tag, mem_gb=None, out_dir=None, max_len=200):
    """Solve one puzzle under a memory cap, in scratch cleaned up either way.

    The whole of `main` below the argument parsing, so a caller holding a
    `Puzzle` -- `vision.py --solve`, with a board it just read off a screenshot
    -- gets the same managed run rather than a second copy of the bookkeeping.
    Returns the process exit code.
    """
    cap_memory(mem_gb if mem_gb is not None else default_mem_gb())
    check_counts(p)

    # Scratch goes under the working directory, not the package: this is an
    # installed library now, and a run's levels belong beside the board it was
    # given. One fixed directory per board name, holding this run's scratch only.
    # Every run is a new board: anything already in there is a previous search's
    # leftovers, never resumed, so it goes before this one starts.
    out_dir = out_dir or os.path.join(os.getcwd(), "work", tag)
    os.makedirs(out_dir, exist_ok=True)
    clear_levels(out_dir)

    # And it goes again on the way out, however that happens -- solved,
    # unsolvable, gave up, crashed, Ctrl-C. Nothing is kept for a later run
    # because no later run would read it.
    try:
        return search(p, out_dir, max_len)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    finally:
        discard_levels(out_dir)



def main(argv=None):
    ap = argparse.ArgumentParser(prog="kesto solve", description=__doc__.splitlines()[0])
    ap.add_argument("puzzle", help="grid file, bundled slug, or encoded string")
    ap.add_argument("--mem-gb", type=float, default=None, help="RLIMIT_AS ceiling")
    ap.add_argument("--dir", default=None, help="checkpoint directory")
    ap.add_argument("--max-len", type=int, default=200, help="give up beyond this length")
    args = ap.parse_args(argv)

    tag = os.path.splitext(os.path.basename(args.puzzle))[0]
    return run(load_puzzle(args.puzzle), tag, args.mem_gb, args.dir, args.max_len)
