#!/usr/bin/env python3
"""Ad-hoc checker for kesto.bfs.solve / kesto.astar.solve on a single puzzle.

Usage:
    python check_bfs.py '<base64url>.<moves>'   # an encoded puzzle from the site
    python check_bfs.py 20260608                # a bundled puzzle, by slug
    python check_bfs.py board.txt               # a transcribed 8x8 grid
    python check_bfs.py 20260608 --cap 2000000  # smaller cap for a quick probe
    python check_bfs.py 20260608 --solver astar # A* instead of BFS

The published solution suffix is optional. Without it you still get the "does
this path actually reach the goals" check, just not the optimality comparison.

Grid files use the same charset kesto.board.render emits, so a rendered board
can be pasted straight back in. Whitespace between cells is optional:

    # wall   o block   . goal   * block already on a goal   - empty
"""

from __future__ import annotations

import argparse
import os
import time

from kesto import astar, bfs
from kesto.board import N, Puzzle, parse, render
from kesto.puzzles import by_slug

# Both take (puzzle, max_states=...) and return a kesto.bfs.Result.
SOLVERS = {"bfs": bfs.solve, "astar": astar.solve}

# Which bitboards each grid character contributes to.
_CHARS = {
    "#": ("walls",),
    "o": ("blocks",),
    ".": ("goals",),
    "*": ("blocks", "goals"),
    "-": (),
}


def parse_grid(text: str) -> Puzzle:
    """Build a Puzzle from a transcribed 8x8 board. No published solution."""
    rows = [r for r in (line.split() or line.strip() for line in text.splitlines()) if r]
    rows = [list("".join(r)) for r in rows]
    if len(rows) != N or any(len(r) != N for r in rows):
        got = f"{len(rows)} rows of {sorted({len(r) for r in rows})}"
        raise ValueError(f"need an {N}x{N} grid, got {got}")

    bits = {"walls": 0, "blocks": 0, "goals": 0}
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            if ch not in _CHARS:
                raise ValueError(f"bad char {ch!r} at ({x}, {y}); use {' '.join(_CHARS)}")
            for field in _CHARS[ch]:
                bits[field] |= 1 << (y * N + x)
    return Puzzle(solution="", **bits)


def load_puzzle(arg: str) -> Puzzle:
    """Accept a grid file, an encoded puzzle string, or a bundled slug."""
    if os.path.isfile(arg):
        with open(arg) as fh:
            return parse_grid(fh.read())
    return parse(arg) if "." in arg or len(arg) > 12 else by_slug(arg)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("puzzle", help="encoded puzzle string, or a bundled slug")
    ap.add_argument("--cap", type=int, default=20_000_000, help="max_states")
    ap.add_argument("--quiet", action="store_true", help="skip the board render")
    ap.add_argument(
        "--solver", choices=sorted(SOLVERS), default="bfs", help="search to run"
    )
    args = ap.parse_args()

    p = load_puzzle(args.puzzle)
    if not args.quiet:
        print(render(p.blocks, p.walls, p.goals))
    print(f"solver={args.solver} blocks={p.n_blocks} walls={p.n_walls} cap={args.cap:,}")
    if p.solution:
        print(f"published: {p.solution} ({len(p.solution)} moves)")

    t0 = time.perf_counter()
    r = SOLVERS[args.solver](p, max_states=args.cap)
    dt = time.perf_counter() - t0
    print(f"\nsearched {r.states:,} states in {dt:.2f}s  exhausted={r.exhausted}")

    ok = True
    if not r.found:
        # Not automatically a bug: deep puzzles legitimately blow the cap.
        # exhausted=True is the real failure, since it claims unreachable.
        verdict = "UNREACHABLE (frontier emptied)" if r.exhausted else "gave up at cap"
        print(f"no solution -- {verdict}")
        ok = not r.exhausted or not p.solution
        if p.solution and r.exhausted:
            print("  ^ but a published solution exists, so the search is wrong")
    else:
        print(f"found:     {r.path} ({r.length} moves)")

        # 1. Does the path actually work? Replays it through the move engine.
        if p.solves(r.path):
            print("  [ok] path lands every block on the goals")
        else:
            ok = False
            print("  [FAIL] path does NOT reach the goals")
            if not args.quiet:
                print(render(p.apply(r.path), p.walls, p.goals))

        # 2. Is it optimal? The published solutions are known optimal, so a
        #    shorter result means the engine or the search is cheating, and a
        #    longer one means the search is not returning a shortest path (for
        #    A*, usually an inadmissible heuristic).
        if p.solution:
            d = r.length - len(p.solution)
            if d == 0:
                print("  [ok] length matches the published optimum")
            elif d > 0:
                ok = False
                print(f"  [FAIL] {d} move(s) longer than optimal -- not a shortest path")
            else:
                ok = False
                print(f"  [FAIL] {-d} move(s) SHORTER than optimal -- engine bug?")

        # 3. Cheap sanity checks on the Result fields themselves.
        if r.states <= 0:
            ok = False
            print("  [FAIL] states counter never incremented")
        if r.exhausted:
            print("  [note] exhausted=True on a hit; fine only if the goal was")
            print("         the last state expanded, otherwise your flag is wrong")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
