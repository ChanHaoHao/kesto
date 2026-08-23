# kesto

Reads a [Kesto](https://kestopuzzle.com/) board off a screenshot and solves it
optimally. Kesto is a daily 8x8 sliding-block puzzle.

```bash
uv sync
uv run kesto read board.png --solve
```

```
o - - - - - - -        OPTIMAL: 26 moves
- o - - - - - -        path   : DDUUUURRRRRLUDLLDDRRUDDLLL
- - o - - # - -        verify : OK -- replays onto every goal
- - - o - - - -
- . - . o - - -
. - . - . o - -
- . - . - - o -
- - . - - - - o
```

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

## The two commands

```bash
uv run kesto read board.png --solve     # screenshot straight to an answer
uv run kesto solve board.txt            # a board you transcribed
uv run kesto solve 20260613             # a bundled puzzle, by slug
```

`read` turns a capture of the site into grid text; `--solve` hands that board to
the search without writing a file. Without it you get the grid on stdout, and
`-o FILE` writes it. `solve` takes a grid file, a bundled slug, or an encoded
puzzle string from the site.

Grid files use the charset `kesto.board.render` emits, so a rendered board pastes
straight back in:

```
# wall   o block   . goal   * block already on a goal   - empty
```

## Layout

| Module | What it does |
| --- | --- |
| `kesto.board` | Bitboard representation, `move()`, puzzle parsing, `render()` |
| `kesto.grid` | Grid text and puzzle arguments — `parse_grid`, `load_puzzle` |
| `kesto.vision` | Reads a board off a screenshot — see [the method](#kestovision--no-computer-vision) |
| `kesto.visualise` | Draws each vision stage for `--vis` |
| `kesto.search` | Bidirectional BFS over uint64 bitboards |
| `kesto.puzzles` | The 15 published puzzles as a test corpus |
| `kesto.verify` | Replays the published solutions through the engine |

`kesto.search` is four modules in dependency order — `engine` → `backward` →
`levels` → `bidirectional`. `engine` owns the vectorised `vmove` plus the memory
guards, `backward` owns `predecessors`, `levels` owns the sorted-array set
algebra and the scratch directory, and `bidirectional` grows both shells and
meets them in the middle.

## kesto.vision — no computer vision

The site renders the board to a canvas as flat fills on an exact pixel grid: a
capture holds around 220 distinct colours, 93% of its pixels are one of six of
them, and there is no noise, lighting or perspective to model. So the lattice
comes from **projection profiles** — the gutters between cells are deep dips in
the row and column sums — and the cell type comes from a **colour lookup**.
Contours, template matching and Hough transforms would all be answering a harder
question than the one being asked.

Two things the method turns on, both easy to get wrong:

- **A goal's interior is byte-identical to an empty cell.** The goal is drawn only
  as a thin rounded outline, so sampling the centre pixel of each cell — the
  obvious first implementation — reads every goal as empty. Cells are scored over
  their whole footprint instead.
- **The wall/empty cut is derived, not hardcoded.** Absolute greys are a styling
  choice and can change; that walls render lighter than empty cells is structural,
  so `_split_greys` clusters the greys actually present. A board with no walls
  leaves one cluster and stays wall-free.

Gutters are found by a threshold **relative** to the profile's plateau, not an
absolute pixel count. A downscaled or JPEG-recompressed capture blurs the gutter
to a single pixel that never reaches zero, and an absolute threshold read those
as a 4x4 board — silently.

It requires a screenshot. A photo of a monitor needs the board quad found and a
homography applied first, and `find_grid` refuses such an image rather than
guessing — as it does for a crop that caught page chrome, or one that cut the
board off mid-grid. A read whose block count does not match its goal count is
rejected too, since a swipe preserves that count and no solver could use the
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

Stages 3 and 6 are the two claims the method rests on, drawn out. Stage 5 is
where the goal-versus-empty trap is visible.

A **rejected** image still gets stages 1-3, which is where a rejection is visible
nearly every time: the caption on `03_profiles` reads back the gutter count it
found against the seven it needed. That is the first thing to look at when a
board will not parse.

## kesto.search — how it decides

Both ends are grown as exact BFS shells. A solution of length L exists iff
`fwd[df]` meets `back[L - df]` for some split, so lengths are tested in
increasing order and the first one that hits is optimal by construction — there
is no heuristic anywhere in the answer, and no depth to guess.

Which side grows next is decided by whichever frontier is currently smaller,
since that is the cheaper ply to expand. Boards differ wildly in which direction
is cheap: some have tiny backward branching and huge forward branching, some the
reverse, so committing to one direction in advance is what makes a board look
unsolvable when it merely needed the other end.

States are 8-byte bitboards in sorted numpy arrays rather than dict keys (~149
bytes each), which is what puts depth-30 boards in reach at all.

Predecessors are found by enumeration rather than inversion, pruned per block,
and every candidate is confirmed with `vmove(candidate) == T` — so pruning can
cost recall but never soundness.

### Memory

**Always pass `--mem-gb`.** It sets `RLIMIT_AS`, so overshooting raises
`MemoryError` inside the search instead of letting the kernel OOM-killer pick a
victim across the whole machine. Set it below free RAM, not below total RAM. A
ply roughly doubles resident size. Without the flag it defaults to 60% of
`MemAvailable`.

**Every run is a new board, and leaves nothing behind.** Scratch goes to
`work/<boardname>/` under the working directory, cleared on the way in and again
on the way out — solved, `UNSOLVABLE`, gave up at `--max-len`, failed verify,
crashed, or Ctrl-C. There is no resume and no state carried between runs. Only
files matching the level naming are removed, so anything else there survives.

**Limit:** it needs both shells resident to extend. Boards whose next required
ply exceeds RAM can't be finished this way.

## Validation

```bash
uv run pytest
```

The bundled corpus is the end-to-end check, and the published solutions are known
optimal, so matching their length is a sharp test rather than a smoke test:

```bash
for s in 20260608 20260602 20260624 20260620 20260605 \
         20260625 20260617 20260527 20260523 20260601 \
         20260607 20260627 20260528 20260524 20260613; do
    uv run kesto solve $s --mem-gb 6 2>/dev/null | grep OPTIMAL
done
```

This returns the published optimum on all fifteen, from `20260608` at 12 moves
through `20260613` at 36, each with `verify: OK`.

The suite pins three things in particular:

- **`vmove` against `kesto.board.move`**, over every reachable state within three
  plies of each corpus board, in all four directions. Nothing in `kesto.search`
  calls the reference engine, so every answer rests on those two agreeing.
- **`predecessors` soundness** — every state it returns really does step onto the
  target.
- **`kesto.vision`** against a real capture, plus all fifteen corpus boards
  round-tripped through a renderer built in the site's palette. That covers what
  one screenshot cannot: `*` cells, wall-free boards, every block count, and the
  resampled and JPEG-recompressed captures that first exposed the gutter
  threshold as too strict.

A board whose block count differs from its goal count is rejected up front by
every entry point — the count is invariant under a swipe, so such a board cannot
be solved and any bound reported for it would be meaningless.
