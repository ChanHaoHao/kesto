"""Brute-force backward BFS: which states can reach board.txt's goal?

Predecessors are found by enumeration, not inversion: `move` preserves block
count and each block in T either stayed put or advanced one cell, so every
predecessor of T is one of the 2^16 "stayed / came from behind" assignments.
Checking each with the forward engine keeps this honest.
"""
import os
import sys
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the Kesto repo, for `import kesto` and solver.py
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from kesto.board import N, cells, move
from solver import load_puzzle

DELTA = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


def predecessors(state, walls):
    """Every S with move(S, walls, d) == state, for some d."""
    out = set()
    for d, (dx, dy) in DELTA.items():
        origins = []  # per block: the cells it could have occupied
        for x, y in cells(state):
            opts = [1 << (y * N + x)]  # stayed put (stuck)
            ox, oy = x - dx, y - dy  # or advanced from behind
            if 0 <= ox < N and 0 <= oy < N and not walls >> (oy * N + ox) & 1:
                opts.append(1 << (oy * N + ox))
            origins.append(opts)
        for combo in product(*origins):
            s = 0
            for bit in combo:
                s |= bit
            if s.bit_count() == state.bit_count() and move(s, walls, d) == state:
                out.add(s)
    return out


p = load_puzzle(os.path.join(ROOT, "board.txt"))
layer, seen = {p.goals}, {p.goals}
for depth in range(1, int(sys.argv[1]) + 1):
    nxt = set()
    for s in layer:
        nxt |= predecessors(s, p.walls) - seen
    seen |= nxt
    layer = nxt
    hit = "  <-- START IS HERE" if p.blocks in seen else ""
    print(f"back-depth {depth:2d}  new {len(nxt):>9,}  total {len(seen):>10,}{hit}", flush=True)
    if not layer:
        print(f"\nbackward space CLOSED at {len(seen):,} states -- start unreachable")
        break
    if p.blocks in seen:
        break
