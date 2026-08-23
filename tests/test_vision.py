"""Round-trip tests for kesto.vision.

`tests/fixtures/board.png` is a real capture off the site and the ground truth
for the real thing; `test_real_screenshot` pins it against the grid transcribed
from it by hand. It lives here rather than at the repo root because a
root copy is a scratch file the daily workflow overwrites.

The rest render the published corpus in the site's palette and read it back,
which covers the cases one screenshot cannot: `*` (a block already sitting on a
goal), boards with no walls at all, every block count from 4 to 11, and the
resampled and recompressed captures a real screenshot pipeline produces.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from kesto import load
from kesto.board import render
from kesto.vision import BoardNotFound, analyse, load_rgb, read_board

Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
PUZZLES = load()

# The board in tests/fixtures/board.png, read off the image by eye.
REAL_BOARD = """\
o o o o o o o o
# # # # # # - -
- - - - - - - -
- - # # # # - -
- # . . . . - #
- # . . . . - #
- - # # # # - -
- - - - - - - -"""

# Sampled off board.png, so the fixtures exercise the same thresholds the real
# screenshot does rather than a palette invented for the test.
BACKGROUND = (26, 26, 26)
EMPTY = (42, 42, 42)
WALL = (85, 85, 85)
BLOCK = (235, 163, 81)
GOAL_RING = (84, 157, 211)

PITCH, GUTTER, PAD, RADIUS = 58, 4, 20, 12


def draw_board(text, path, scale=1.0):
    """Render a grid the way the site does: flat fills on rounded cells."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    n = len(rows)
    size = PAD * 2 + n * PITCH - GUTTER
    img = Image.new("RGB", (size, size), BACKGROUND)
    d = ImageDraw.Draw(img)
    for y, row in enumerate(rows):
        for x, ch in enumerate(row):
            x0, y0 = PAD + x * PITCH, PAD + y * PITCH
            box = (x0, y0, x0 + PITCH - GUTTER - 1, y0 + PITCH - GUTTER - 1)
            fill = {"#": WALL, "o": BLOCK, "*": BLOCK}.get(ch, EMPTY)
            d.rounded_rectangle(box, RADIUS, fill=fill)
            if ch in ".*":
                d.rounded_rectangle(box, RADIUS, outline=GOAL_RING, width=3)
    if scale != 1.0:
        wh = int(size * scale)
        img = img.resize((wh, wh), Image.LANCZOS)
    img.save(path)
    return path


def test_real_screenshot():
    """The actual capture, against the board transcribed from it by hand."""
    assert read_board(os.path.join(FIXTURES, "board.png")) == REAL_BOARD


@pytest.mark.parametrize("p", PUZZLES, ids=[p.slug for p in PUZZLES])
def test_corpus_round_trip(p, tmp_path):
    want = render(p.blocks, p.walls, p.goals)
    assert read_board(draw_board(want, tmp_path / f"{p.slug}.png")) == want


def test_blocks_on_goals(tmp_path):
    """`*` must survive: the goal ring is still drawn under the block."""
    want = render(PUZZLES[0].goals, PUZZLES[0].walls, PUZZLES[0].goals)
    assert "*" in want
    assert read_board(draw_board(want, tmp_path / "starred.png")) == want


def test_board_without_walls(tmp_path):
    """One grey cluster instead of two -- nothing may be promoted to a wall."""
    want = render(PUZZLES[0].blocks, 0, PUZZLES[0].goals)
    assert "#" not in want
    assert read_board(draw_board(want, tmp_path / "nowalls.png")) == want


@pytest.mark.parametrize("scale", [0.6, 0.75, 1.5, 2.0])
def test_resampled(scale, tmp_path):
    """Retina captures and downscales both land on the same grid."""
    want = render(PUZZLES[0].blocks, PUZZLES[0].walls, PUZZLES[0].goals)
    got = read_board(draw_board(want, tmp_path / f"s{scale}.png", scale=scale))
    assert got == want


def test_jpeg_artifacts(tmp_path):
    """Lossy recompression blurs the edges but not the fills."""
    want = render(PUZZLES[0].blocks, PUZZLES[0].walls, PUZZLES[0].goals)
    png = draw_board(want, tmp_path / "q.png")
    jpg = tmp_path / "q.jpg"
    Image.open(png).save(jpg, quality=80)
    assert read_board(str(jpg)) == want


def test_rejects_non_board(tmp_path):
    """No lattice means refuse, not guess."""
    path = tmp_path / "blank.png"
    Image.new("RGB", (400, 400), (200, 200, 200)).save(path)
    with pytest.raises(BoardNotFound, match="gutters"):
        read_board(str(path))


def test_rejects_partial_crop(tmp_path):
    """A board cut off mid-grid has too few gutters to be an 8x8."""
    want = render(PUZZLES[0].blocks, PUZZLES[0].walls, PUZZLES[0].goals)
    png = draw_board(want, tmp_path / "full.png")
    img = Image.open(png)
    cropped = tmp_path / "cropped.png"
    img.crop((0, 0, img.width // 2, img.height)).save(cropped)
    with pytest.raises(BoardNotFound, match="gutters"):
        read_board(str(cropped))


# --- --vis ---------------------------------------------------------------------

STAGES = ["01_input", "02_cellmask", "03_profiles", "04_lattice", "05_channels",
          "06_greysplit", "07_board"]


def _drawn(out_dir):
    """Stage names actually written, in order, each checked to be a real image."""
    names = []
    for name in sorted(os.listdir(out_dir)):
        with Image.open(os.path.join(out_dir, name)) as im:
            im.load()
            assert im.width > 100 and im.height > 100, f"{name} is a stub"
        names.append(os.path.splitext(name)[0])
    return names


def test_vis_draws_every_stage(tmp_path):
    from kesto.visualise import draw_steps

    rgb = load_rgb(os.path.join(FIXTURES, "board.png"))
    out = tmp_path / "vis"
    written = draw_steps(rgb, analyse(rgb), str(out))
    assert len(written) == len(STAGES)
    assert _drawn(out) == STAGES


def test_vis_draws_what_it_reached_on_a_rejected_image(tmp_path):
    """The stages a rejection needs are exactly the ones computed before it."""
    from kesto.visualise import draw_steps

    want = render(PUZZLES[0].blocks, PUZZLES[0].walls, PUZZLES[0].goals)
    png = draw_board(want, tmp_path / "full.png")
    img = Image.open(png)
    half = tmp_path / "half.png"
    img.crop((0, 0, img.width // 2, img.height)).save(half)

    rgb = load_rgb(str(half))
    with pytest.raises(BoardNotFound):
        analyse(rgb)

    out = tmp_path / "vis"
    draw_steps(rgb, None, str(out))
    assert _drawn(out) == STAGES[:3]


def test_vis_flag_still_prints_the_board(tmp_path):
    """--vis is a side effect; it must not disturb stdout or the exit code."""
    out = tmp_path / "steps"
    proc = _cli(os.path.join(FIXTURES, "board.png"), "--vis", str(out))
    assert proc.returncode == 0
    assert proc.stdout.strip() == REAL_BOARD
    assert _drawn(out) == STAGES


# --- --solve -------------------------------------------------------------------


def _cli(*args, cmd="read", cwd=None):
    """Run the CLI as the user does. Subprocess because --solve caps RLIMIT_AS."""
    return subprocess.run(
        [sys.executable, "-m", "kesto", cmd, *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_solve_reaches_the_optimum():
    """A screenshot straight to an answer, with no grid file in between."""
    proc = _cli(os.path.join(FIXTURES, "board.png"), "--solve", "--mem-gb", "2",
                "--max-len", "40")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "OPTIMAL: 23 moves" in proc.stdout
    assert "verify : OK" in proc.stdout


def test_solve_agrees_with_the_file_route(tmp_path):
    """--solve must be the same search, not a second implementation of it."""
    grid = tmp_path / "board.txt"
    direct = _cli(os.path.join(FIXTURES, "board.png"), "--solve", "--mem-gb", "2")
    assert direct.returncode == 0, direct.stderr[-2000:]

    assert _cli(os.path.join(FIXTURES, "board.png"), "-o", str(grid)).returncode == 0
    viafile = _cli(str(grid), "--mem-gb", "2", cmd="solve")
    assert viafile.returncode == 0, viafile.stderr[-2000:]

    def answer(out):
        return [ln for ln in out.splitlines() if ln.startswith(("OPTIMAL", "path"))]

    assert answer(direct.stdout) == answer(viafile.stdout)


def test_solve_prints_the_board_before_solving_it():
    """stdout is block-buffered off a terminal; the board must not arrive late."""
    proc = _cli(os.path.join(FIXTURES, "board.png"), "--solve", "--mem-gb", "2")
    assert proc.stdout.index(REAL_BOARD) < proc.stdout.index("OPTIMAL")


def test_solve_refuses_an_unreadable_image(tmp_path):
    """A rejected read must not reach the solver."""
    path = tmp_path / "blank.png"
    Image.new("RGB", (400, 400), (200, 200, 200)).save(path)
    proc = _cli(str(path), "--solve")
    assert proc.returncode == 1
    assert "gutters" in proc.stderr
    assert "OPTIMAL" not in proc.stdout
