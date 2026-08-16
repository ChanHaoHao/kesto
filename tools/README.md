# tools/ — search probes for hard Kesto boards

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
verification that the path replays onto every goal. The others are for when you
want to see or control the machinery.

| I want to… | use |
| --- | --- |
| **solve a board optimally** | **`solve.py`** |
| know how big the board's state space is | `probe.py` |
| find *a* solution when optimality is out of reach | `beam.py` |
| drive the level building by hand | `exact.py` |
| a simpler one-shot bidirectional search | `meet.py` |

`vpred.py`, `vheur.py` and `back.py` are libraries the above import; run them
directly only for their selftests.

## solve.py — optimal, automatic

```bash
.venv/bin/python tools/solve.py board_today.txt
.venv/bin/python tools/solve.py 20260613 --mem-gb 5
```

Grows exact BFS shells from both ends, always extending whichever frontier is
currently smaller, and tests lengths in increasing order — so the first hit is
optimal by construction, with no heuristic in the answer.

Choosing the direction is the part that matters and the part you should not do
by hand: `board.txt` needed forward 18 + backward 14, while `board_today.txt`
needed forward 22 + backward 15 and would have been hopeless one-directionally
(53M states just to reach the goal, versus 11.2M bidirectionally).

Levels are checkpointed under `--dir` so an interrupted run resumes where it
stopped rather than restarting. The default directory is
`work/<boardname>-<digest>/`, where the digest covers walls, blocks and goals —
**editing a board file therefore starts a fresh directory rather than resuming
the old board's levels.** Every directory also carries a `puzzle.json`, and a
mismatch is refused outright:

```
checkpoint mismatch in work/board_today
  those levels belong to a different board.
  delete the directory, or pass a different --dir.
```

That guard exists because the earlier version keyed only on the filename, so
editing a board silently resumed the wrong search. `exact.py` enforces the same
check on its `--dir`.

**Limit:** it needs both shells resident to extend. Boards whose next required
ply exceeds RAM can't be finished this way — `board.txt` at length 33 is past
that line on a 16 GB machine, which is what `exact.py`'s incremental,
one-ply-at-a-time mode is for.

## probe.py — size the space

```bash
.venv/bin/python tools/probe.py board.txt 20000000
```

Level-by-level BFS with no parent links. Prints states per ply and stops at the
cap. Read the growth factor: if it decays toward 1.0 the space is finite and
`exact.py` can settle the board outright; if it holds above ~2 you need
`beam.py`.

## beam.py — find a solution

```bash
.venv/bin/python tools/beam.py board.txt --back-depth 10 --keep 400000 --mem-gb 4
```

Expands `--stride` plies, then culls to the `--keep` states closest to the goal
by the admissible bound, and repeats. Builds a backward basin first so any state
landing inside it finishes immediately with a complete path.

- `--keep` — frontier size after each cull. More is better and slower.
- `--back-depth` — basin radius. Deeper basin = shorter solutions found.
- `--levels-dir work/levels` — reuse `back_*.npy` that `exact.py` already built
  instead of recomputing the basin. Much faster, and gives a deeper target.

**Incomplete by design.** A result is a verified solution; silence proves
nothing, and the length is an upper bound, not the optimum.

## exact.py — prove the optimum

```bash
.venv/bin/python tools/exact.py board.txt --fwd-depth 16 --back-depth 11 \
    --known-lower 1 --mem-gb 4 --dir work/levels
```

Builds exact BFS shells from both ends and intersects matched pairs: a solution
of length L exists iff `fwd[df]` meets `back[L-df]` for some split. Reports the
first achievable L, or the best lower bound proven.

Levels are checkpointed to `--dir`, so raising `--fwd-depth`/`--back-depth`
later only builds the new plies. Coverage is `fwd-depth + back-depth`; push
whichever side is growing more slowly.

## meet.py — bidirectional search

```bash
.venv/bin/python tools/meet.py board.txt --back-depth 10 --fwd-depth 17 --reconstruct
```

One-shot bidirectional search. `--reconstruct` retains forward levels so a meet
yields a replayable move string. `exact.py` supersedes this for optimality work;
`meet.py` is simpler when you just want a single answer.

## Memory

**Always pass `--mem-gb`.** It sets `RLIMIT_AS`, so overshooting raises
`MemoryError` inside the probe instead of letting the kernel OOM-killer pick a
victim across the whole machine. Set it below free RAM, not below total RAM.
Watch the `rss=` column; a ply roughly doubles it.

## Selftests

```bash
.venv/bin/python tools/vheur.py            # vs kesto.astar.heuristic
.venv/bin/python tools/vpred.py --selftest # vs brute-force 2^16 enumeration
```

Both should report zero mismatches. The vectorised engines are also checked
end-to-end against the bundled corpus — `probe.py 20260601` reaches depth 22,
`exact.py 20260624 --known-lower 1` reports 14, `beam.py 20260613` returns 36,
each matching the published optimum.
