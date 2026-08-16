# tools/ — search engines for hard Kesto boards

Analysis tooling, not part of `kesto`. These exist because `kesto.bfs` and
`kesto.astar` keep every visited state in a Python dict (~149 bytes/state), which
tops out around ply 17 on a 16-block board. Everything here stores states as
8-byte bitboards in numpy arrays instead, which is what makes depth-30+ boards
reachable.

Every script takes the same puzzle argument as `solver.py`: a grid file, a
bundled slug, or an encoded puzzle string.

## Start here

```bash
.venv/bin/python tools/solve.py board.txt
```

`solve.py` is the one you want in almost every case. It picks depths and
directions itself and prints the optimal length, the move string, and a
verification that the path replays onto every goal.

The other three are the layers underneath it, usable on their own:

| I want to… | use |
| --- | --- |
| **solve a board optimally** | **`solve.py`** |
| know how big the board's state space is | `probe.py` |
| a simpler one-shot bidirectional search | `meet.py` |
| see the backward levels alone | `vpred.py` |

The dependency order is `probe.py` → `vpred.py` → `meet.py` → `solve.py`:
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

A board whose block count differs from its goal count is rejected up front by
every entry point — the count is invariant under a swipe, so such a board cannot
be solved and any bound a search reported for it would be meaningless.
