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

| Module | What it does |
| --- | --- |
| `kesto.board` | Bitboard representation, `move()`, puzzle parsing, `render()` |
| `kesto.puzzles` | The 15 published case-study puzzles as a test corpus |
| `kesto.verify` | Engine check and an admissibility harness for heuristics |
| `kesto.bfs` | `solve`, `reachable` — plus the shared `Result` type |
| `kesto.astar` | `axis_bound`, `heuristic`, `solve` |
| `tools/` | numpy search engines for boards the above cannot reach, plus `vision.py` — see [tools/README.md](tools/README.md) |

Everything is implemented. The in-package solvers are the readable reference: plain
Python, a dict for the visited set, exact answers. `tools/` is the same mathematics
rewritten for boards where that representation runs out of memory.

## Getting started

```bash
uv sync
uv run pytest
```

```python
from kesto import load, move

p = load()[0]
print(p.title, "-", p.n_blocks, "blocks")
print(p)                       # '#' wall, 'o' block, '.' goal, '*' block on goal

print(p.solves(p.solution))    # True: the published solution, replayed
after = move(p.blocks, p.walls, "R")   # one swipe, every block at once
```

```python
from kesto.bfs import solve

r = solve(p)
print(r.length, r.path, r.states)
assert p.solves(r.path)
```

Single boards from the command line, including one you transcribed yourself:

```bash
uv run python solver.py 20260608           # a bundled slug
uv run python solver.py board.txt          # '#' wall, 'o' block, '.' goal, '-' empty
uv run python solver.py 20260608 --solver astar
```

**For a real daily board, run `tools/solve.py` instead.** `solver.py` drives the
in-package solvers, which is what you want for reading the algorithms or checking them
against the corpus; it caps out well short of a hard board. `tools/solve.py` picks its
own depths and directions and prints the optimal length, the move string and a replay
check:

```bash
uv run python tools/solve.py board_today.txt
```

See [past where the dict runs out](#tools--past-where-the-dict-runs-out) for why the two
exist separately.

## Why the state space is manageable

Blocks are interchangeable, so a state is fully described by the 64-bit block occupancy
bitboard. That is the visited key: no permutation blow-up, and the successor function is
a handful of shifts and masks. The "stuck" set has a clean closure form — a block is
stuck iff the cell ahead is a wall/edge or holds a block that is itself stuck — computed
to a fixpoint.

Effective branching after deduplication starts near 4 and decays as the space fills. A
representative daily board, ply by ply: 4.0, 3.5, 3.1, 2.8, 2.9, 2.8, 2.7, 2.6, 2.5,
2.4, 2.3, 2.2, 2.1, 2.0, 1.9. Cost is therefore exponential in **solution depth**, not
block count — an 11-block puzzle at depth 16 is easy, an 8-block one at depth 36 is not.

## What the in-package solvers reach

Both solvers on all 15 published puzzles, 5M-state cap. `h0` is the A* heuristic
evaluated at the start position, so `h0` vs `pub` shows how much of the true depth the
bound recovers. **Every puzzle either matched the published length or hit the cap — no
solver ever returned a non-optimal path**, and the published solutions are known
optimal, so that equality is the sharpest correctness check available here.

```
slug      blk wall  pub   h0 |  bfs      states    sec |   A*      states    sec
20260608    8    6   12    6 |   12     234,304   0.32 |   12       7,057   0.29
20260602    8    6   14    6 |   14   1,706,496   2.66 |   14      12,742   0.53
20260624    8    4   14    4 |   14   3,249,211   5.63 |   14      71,432   2.98
20260620    8    2   15   11 |   15     812,086   1.20 |   15       2,918   0.12
20260605    8    4   16   10 |   16   3,312,361   6.85 |   16      27,373   1.14
20260625   11    4   16    8 |    -   >5,000,000  8.55 |   16      71,478   3.41
20260617    6   10   18    6 |   18   1,502,295   3.73 |   18      64,543   2.54
20260527    6    2   20    6 |   20   4,859,285  14.97 |   20   1,306,603  55.52
20260523    8    8   21   10 |    -   >5,000,000 10.56 |   21     286,972  12.34
20260601    5    4   22    6 |   22     833,194   2.06 |   22     106,018   4.15
20260607    8    2   22    8 |    -   >5,000,000  9.53 |    -   >5,000,000 230.48
20260627    8    2   23    6 |    -   >5,000,000  8.92 |    -   >5,000,000 232.78
20260528    4    4   27    6 |   27     210,967   0.58 |   27     154,747   6.37
20260524    4    4   29    8 |   29     329,211   0.97 |   29     328,060  14.12
20260613    8    4   36    9 |    -   >5,000,000  9.72 |    -   >5,000,000 229.96
```

**BFS 10/15, A* 12/15.** A* explores 22% of BFS's states on average where both finish,
but the average hides the shape of it: on the shallow boards the bound is worth two
orders of magnitude (234k states down to 7k), while on `20260524` it saves nothing at
all (329,211 down to 328,060). The heuristic helps exactly where the goal is far in
*displacement*, and stops helping where the depth comes from rearrangement instead.

Note the seconds column. A* pays ~24x per state for the priority queue and the bound —
at the cap, 230s against BFS's 9.7s. Two extra puzzles is a real gain, but on a board
neither can finish, A* just burns the same cap far more slowly.

## The A* bound

`kesto/astar.py` has the full derivation in its module docstring; the short version:

Count swipes by direction, so length is `R + L + U + D`. A right-swipe moves any block's
`x` by at most one, so no block's net `+x` displacement can exceed `R`. Pick a threshold
`t`: the final position must hold `#goals with x >= t` blocks in that region while the
current one holds `#blocks with x >= t`, so the difference must cross the boundary. The
cheapest crossing set is the nearest blocks, and the furthest of those pins a lower bound
on `R`. Maximise over `t`, repeat per direction, sum the four.

```python
from kesto.verify import check_admissible
from kesto.astar import heuristic

assert check_admissible(heuristic) == []   # empty means no overestimates
```

That passes, but an empty result is necessary and not sufficient — it only samples
states lying on optimal paths.

The bound is admissible and **loose**: across the corpus it recovers 39% of true depth
on average, ranging from 73% on `20260620` down to 22% on `20260528`. It is worst
precisely where it would pay most — on the three boards nobody finishes it recovers
36%, 26% and 25%. Tightening it is the interesting part, and the corpus makes the
tightening measurable.

## tools/ — past where the dict runs out

The in-package solvers keep every visited state in a Python dict at ~149 bytes/state, so
they exhaust memory long before they exhaust the search. At the 5M cap above, three
published boards sit beyond both of them; raising the cap moves the line but does not
remove it.

`tools/` stores states as 8-byte bitboards in numpy arrays and grows exact BFS shells
from *both* ends, extending whichever frontier is currently smaller. Lengths are tested
in increasing order, so the first hit is optimal by construction — no heuristic anywhere
in the answer.

```bash
uv run python tools/solve.py board_today.txt
uv run python tools/solve.py 20260613 --mem-gb 5
```

This settles all fifteen published puzzles at the published optimum, `20260613` at
depth 36 included — the three the in-package solvers cannot reach among them, each
with its path replayed through `kesto.board` as a check. See
[tools/README.md](tools/README.md) for the other three engines and the memory rules.

If you have a screenshot of the board rather than a transcription, `tools/vision.py`
reads one into the same grid format, and the whole daily is two commands:

```bash
uv run python tools/vision.py board.png -o board_today.txt
uv run python tools/solve.py board_today.txt
```

There is no real computer vision in it — the site draws flat fills on an exact
pixel grid, so the lattice comes from projection profiles and the cell type from
a colour lookup. `--vis DIR` draws every stage of that to numbered PNGs if you
want to watch it work, or to see why a board would not parse.
