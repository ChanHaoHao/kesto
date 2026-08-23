# tools/ — search engines for hard Kesto boards

Analysis tooling, not part of `kesto`. These exist because `kesto.bfs` and
`kesto.astar` keep every visited state in a Python dict (~149 bytes/state), which
tops out around ply 17 on a 16-block board. Everything here stores states as
8-byte bitboards in numpy arrays instead, which is what makes depth-30+ boards
reachable.

Every search script takes the same puzzle argument as `solver.py`: a grid file,
a bundled slug, or an encoded puzzle string. `vision.py` is the exception --
it takes a screenshot and writes one of those grid files.

## Start here

```bash
.venv/bin/python tools/vision.py board.png -o board_today.txt   # if you have a screenshot
.venv/bin/python tools/solve.py board_today.txt
```

`solve.py` is the one you want in almost every case. It picks depths and
directions itself and prints the optimal length, the move string, and a
verification that the path replays onto every goal.

The other three are the layers underneath it, usable on their own:

| I want to… | use |
| --- | --- |
| **solve a board optimally** | **`solve.py`** |
| turn a screenshot into a grid file | `vision.py` |
| see how it read that screenshot | `vision.py --vis` |
| know how big the board's state space is | `probe.py` |
| a simpler one-shot bidirectional search | `meet.py` |
| see the backward levels alone | `vpred.py` |

`vision.py` stands apart from the rest and imports none of them; it is a way
into the grid format, not a search. `visualise.py` is its `--vis` drawing half
and nothing else imports it. The dependency order of the other four is
`probe.py` → `vpred.py` → `meet.py` → `solve.py`:
`probe.py` is the leaf, owning the vectorised forward engine `vmove` plus
`cap_memory`/`log`/`rss_gb`; `vpred.py` owns `predecessors`; `meet.py` owns the
sorted-array set operations. `solve.py` imports from all three.

## solve.py — optimal, automatic

```bash
.venv/bin/python tools/solve.py board_today.txt
.venv/bin/python tools/solve.py 20260613 --mem-gb 5
```

Grows exact BFS shells from both ends, always extending whichever frontier is
currently smaller, and tests lengths in increasing order — so the first hit is
optimal by construction, with no heuristic in the answer.

Choosing the direction is the part that matters and the part you should not do
by hand. Boards differ wildly in which end is cheap: some have tiny backward
branching and huge forward branching, some the reverse, and committing to one
direction in advance is what makes a board look unsolvable when it merely
needed the other end.

**Every run is a new board, and leaves nothing behind.** The default directory
is `work/<boardname>/`, and it is scratch for a single run: cleared on the way
in, cleared again on the way out — solved, `UNSOLVABLE`, gave up at
`--max-len`, failed verify, crashed, or Ctrl-C. There is no resume, no flag,
and no state carried between runs, so editing a board in place needs no
thought: whatever sits in the directory is a previous search's leftovers and is
deleted, never adopted.

Levels are written at all only because reconstruction streams them back one ply
at a time; holding every ply resident is what runs a deep board out of memory.

Only files matching the level naming (`fwd_NNN.npy`, `back_NNN.npy`,
`puzzle.json`) are removed, so anything else in the directory survives.

**Limit:** it needs both shells resident to extend. Boards whose next required
ply exceeds RAM can't be finished this way.

## vision.py — screenshot to grid

```bash
.venv/bin/python tools/vision.py board.png
.venv/bin/python tools/vision.py board.png -o board_today.txt
.venv/bin/python tools/vision.py board.png --debug
.venv/bin/python tools/vision.py board.png --vis steps/
```

Prints the board in the charset every other entry point reads, so transcribing
the daily puzzle by hand is optional.

There is deliberately no computer vision in it. The site draws the board to a
canvas as flat fills on an exact pixel grid: a real capture holds 220 distinct
colours, 93% of its pixels are one of six of them, and there is no noise,
lighting or perspective to model. So the lattice comes from projection profiles
-- the gutters between cells are deep dips in the row and column sums -- and the
cell type comes from a colour lookup. Every threshold sits in the middle of a
gap an order of magnitude wider than it needs.

Two things are worth knowing before you change it:

- **A goal's interior is byte-identical to an empty cell.** The goal is drawn
  only as a thin rounded outline, so sampling the centre pixel of each cell --
  the obvious first implementation -- reads every goal as empty. Cells are
  scored over their whole footprint instead.
- **The wall/empty cut is derived, not hardcoded.** Absolute greys are a styling
  choice and can change; that walls render lighter than empty cells is
  structural, so `_split_greys` clusters the greys actually present. A board
  with no walls leaves one cluster and stays wall-free.

It requires a screenshot. A photo of a monitor needs the board quad found and a
homography applied first, and `find_grid` refuses such an image rather than
guessing at it -- as it does for a crop that caught page chrome, or one that cut
the board off mid-grid. A read whose block count does not match its goal count
is rejected too, since a swipe preserves that count and no solver could use the
result.

### Watching it work

`--debug` prints the lattice and the per-cell measurements behind a disputed
read. `--vis DIR` draws the whole pipeline instead, one numbered PNG per stage
(`vis/` if you name no directory):

| | |
| --- | --- |
| `01_input` | the capture, with its colour count |
| `02_cellmask` | which pixels belong to a cell rather than a gutter |
| `03_profiles` | that mask summed down each axis, gutters and floor marked |
| `04_lattice` | where the cell boundaries landed |
| `05_channels` | the warm, cool and grey masks the classifier reads |
| `06_greysplit` | the grey levels present, and the cut 2-means drew through them |
| `07_board` | the read, laid back over the picture it came from |

Stages 3 and 6 are the two claims the method rests on, drawn out: that the
gutters are unmistakable dips, and that the cell types cluster nowhere near each
other. Stage 5 is where the goal-versus-empty trap is visible -- a goal's
interior is the same grey as an empty cell, and only the ring separates them.

A **rejected** image still gets stages 1-3, which is where a rejection is
visible nearly every time: the caption on `03_profiles` reads back the gutter
count it found against the seven it needed. That is the first thing to look at
when a board will not parse.

The drawing lives in `visualise.py`, imported only when the flag is passed.
Nothing there feeds a board back into a solver.

## probe.py — size the space

```bash
.venv/bin/python tools/probe.py board.txt 20000000 --mem-gb 4
```

Level-by-level forward BFS with no parent links. Prints states per ply and stops
at the cap. Read the growth factor: if it decays toward 1.0 the space is finite
and a one-directional search can settle the board outright; if it holds above ~2
you want `solve.py`.

Useful as an independent check on `solve.py`, since it shares none of the
bidirectional machinery — only `vmove`.

## meet.py — bidirectional search

```bash
.venv/bin/python tools/meet.py board.txt --back-depth 10 --fwd-depth 17 --reconstruct
```

One-shot bidirectional search: build a backward basin of a fixed radius, then
BFS forward into it. `--reconstruct` retains forward levels so a meet yields a
replayable move string. `solve.py` supersedes this for optimality work — it
picks the depths itself and proves the answer is shortest — but `meet.py` is
simpler when you want a single answer at depths you choose.

Also the library `solve.py` builds on: `expand`, `merge_sorted`,
`intersect_sorted`, `mask_not_in`, `clear_levels`, `check_counts`.

## vpred.py — backward levels

```bash
.venv/bin/python tools/vpred.py board.txt --depth 20
```

Prints backward level sizes from the goal and stops when it reaches the start.
Its `predecessors` is what the backward half of `solve.py` and `meet.py` run on:
predecessors are found by enumeration rather than inversion, pruned per block,
and every candidate is confirmed with `vmove(candidate) == T` — so pruning can
cost recall but never soundness.

## Memory

**Always pass `--mem-gb`.** It sets `RLIMIT_AS`, so overshooting raises
`MemoryError` inside the search instead of letting the kernel OOM-killer pick a
victim across the whole machine. Set it below free RAM, not below total RAM.
Watch the `rss=` column; a ply roughly doubles it.

`solve.py` defaults to 60% of `MemAvailable` when you don't pass one. Every
engine takes the flag, `probe.py` included — its state cap bounds the *count* of
visited states, not the bytes, and a single ply's `union1d` holds two copies of
the visited array at once. `cap_memory`, `log` and `rss_gb` live in `probe.py`,
the leaf module; `meet.py` re-exports them for everything else.

## Validation

The bundled corpus is the end-to-end check, and the published solutions are
known optimal, so matching their length is a sharp test rather than a smoke
test. `solve.py` returns the published optimum on all fifteen, from
`20260608` at 12 moves through `20260613` at 36, each with `verify: OK`:

```bash
for s in 20260608 20260602 20260624 20260620 20260605 \
         20260625 20260617 20260527 20260523 20260601 \
         20260607 20260627 20260528 20260524 20260613; do
    .venv/bin/python tools/solve.py $s --mem-gb 6 2>/dev/null | grep OPTIMAL
done
```

`vmove` is checked against `kesto.board.move`, the reference engine, by running
both over the same states — they agree everywhere, which is what lets the numpy
engines stand in for the one the corpus validates.

`vision.py` is covered by `tests/test_vision.py`, which pins a real capture
(`tests/fixtures/board.png`) against the board transcribed from it by hand, and
then round-trips all fifteen corpus puzzles through a renderer built in the
site's palette. That covers what one screenshot cannot: `*` cells, wall-free
boards, every block count from 4 to 11, and the resampled and JPEG-recompressed
captures that first exposed the gutter threshold as too strict. `--vis` is
covered too -- that it draws every stage on a good image, only the stages it
reached on a rejected one, and that it stays a side effect of a normal read.

A board whose block count differs from its goal count is rejected up front by
every entry point — the count is invariant under a swipe, so such a board cannot
be solved and any bound a search reported for it would be meaningless.
