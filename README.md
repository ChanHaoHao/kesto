# kesto

Optimal solvers for [Kesto](https://kestopuzzle.com/), the daily 8x8 sliding-block puzzle.

## The rules

An 8x8 grid holds **blocks**, **walls** and **goal squares**, always with exactly as
many blocks as goals (4-11 in the published puzzles). You swipe up/down/left/right and
**every block moves together**. You solve it by covering every goal square at once.

What a swipe actually does — this is the part that surprises people:

1. Each block moves **exactly one cell**. It is *not* a slide-until-you-hit-something
   like 2048 or Threes.
2. A block stays put if its destination is off the board, is a wall, or is occupied by a
   block that already resolved this turn.
3. Resolution runs from the leading edge backward, so when a lead block jams, everything
   queued behind it jams against it too.

Point 3 is the whole game. Blocks move in lockstep, so the *only* way to change their
relative arrangement is to park some against an obstruction while the rest keep moving.

The engine in `kesto.board` was reverse-engineered from the site's own JS bundle and
replays all fifteen published solutions exactly.

## Layout

| Module | Status | What it does |
| --- | --- | --- |
| `kesto.board` | done | Bitboard representation, `move()`, puzzle parsing, `render()` |
| `kesto.puzzles` | done | The 15 published case-study puzzles as a test corpus |
| `kesto.verify` | done | Engine check and an admissibility harness for heuristics |
| `kesto.bfs` | **stubs** | `solve`, `reachable` — plus the shared `Result` type |
| `kesto.astar` | **stubs** | `axis_bound`, `heuristic`, `solve` |

Both solvers are yours to write. Every stub carries a full spec in its docstring —
arguments, return contract, implementation notes and the pitfalls worth knowing. The
engine, the corpus and the verification harnesses are done, so you can check your work
from the first line.

## Getting started

```bash
uv sync
uv run pytest    # solver tests auto-skip until each module is implemented
```

The engine works today, so you can explore puzzles before writing any search:

```python
from kesto import load, move

p = load()[0]
print(p.title, "-", p.n_blocks, "blocks")
print(p)                       # '#' wall, 'o' block, '.' goal, '*' block on goal

print(p.solves(p.solution))    # True: the published solution, replayed
after = move(p.blocks, p.walls, "R")   # one swipe, every block at once
```

Once `kesto.bfs.solve` exists:

```python
from kesto.bfs import solve

r = solve(p)
print(r.length, r.path, r.states)
assert p.solves(r.path)
```

## Why the state space is manageable

Blocks are interchangeable, so a state is fully described by the 64-bit block occupancy
bitboard. That is the visited key: no permutation blow-up, and the successor function is
a handful of shifts and masks. The "stuck" set has a clean closure form — a block is
stuck iff the cell ahead is a wall/edge or holds a block that is itself stuck — computed
to a fixpoint.

## What BFS should achieve

Measured from a throwaway reference implementation, so you have targets to hit rather
than a mystery. Python BFS at a 20M-state cap, on all 15 published puzzles. **In every
puzzle it solved, the BFS depth equalled the published solution length** — the site's
solutions are optimal, which makes that equality a sharp correctness check.

```
blocks walls  pub  bfs       states      sec
     6    10   18   18    1,502,295     3.73
     6     2   20   20    4,859,285    14.81
     8     2   15   15      812,086     1.20
     8     2   22    -   >20,000,000   52.56   cap
     5     4   22   22      833,194     2.11
     4     4   27   27      210,967     0.59
     8     2   23    -   >20,000,000   46.61   cap
     8     6   14   14    1,706,496     2.80
     8     8   21    -   >20,000,000   57.75   cap
     4     4   29   29      329,211     1.15
     8     4   14   14    3,249,211     6.45
     8     4   16   16    3,312,361     7.18
    11     4   16   16   13,196,828    27.94
     8     4   36    -   >20,000,000   47.33   cap
     8     6   12   12      234,304     0.31
```

11/15. Cost is exponential in **solution depth**, not block count — an 11-block puzzle at
depth 16 solves fine, while an 8-block one at depth 36 does not. Effective branching after
dedup is roughly 3-4x. If your state counts land in this ballpark, your implementation is
behaving; if they are far larger, suspect the visited check.

A throwaway C prototype (same algorithm, open-addressed hash table, 100M cap) got this to
**13/15**, which says the remaining gap is memory and constant factors rather than
anything algorithmic:

```
 8 blocks,  2 walls, depth 22   67,418,147 states   12.7s
 8 blocks,  8 walls, depth 21   21,431,705 states    3.8s
 8 blocks,  2 walls, depth 23   >100,000,000        cap
 8 blocks,  4 walls, depth 36   >100,000,000        cap
```

Every returned path was verified against the Python engine.

## The A* bound

The two survivors need better pruning. `kesto/astar.py` has the full derivation in its
module docstring; the short version:

Count swipes by direction, so length is `R + L + U + D`. A right-swipe moves any block's
`x` by at most one, so no block's net `+x` displacement can exceed `R`. Pick a threshold
`t`: the final position must hold `#goals with x >= t` blocks in that region while the
current one holds `#blocks with x >= t`, so the difference must cross the boundary. The
cheapest crossing set is the nearest blocks, and the furthest of those pins a lower bound
on `R`. Maximise over `t`, repeat per direction, sum the four.

Verified admissible against all fifteen known-optimal paths — but **loose**, recovering
only ~25-40% of true depth on the hard puzzles, so expect real but not dramatic pruning.
Tightening it is the interesting part.

```python
from kesto.verify import check_admissible
from kesto.astar import heuristic

assert check_admissible(heuristic) == []   # empty means no overestimates
```

Note that an empty result is necessary but not sufficient — it only samples states lying
on optimal paths.
