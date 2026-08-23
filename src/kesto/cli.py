"""The two entry points: read a board off a screenshot, or solve one.

    kesto read board.png --solve
    kesto solve board.txt

Both subcommands are imported lazily, so `kesto solve` never pays to load
Pillow and `kesto read` only pulls in the numpy search stack if `--solve` sends
a board there.
"""

from __future__ import annotations

import sys

USAGE = """usage: kesto <command> [options]

  read <image>    read a board off a screenshot; --solve to solve it outright
  solve <puzzle>  solve a grid file, a bundled slug, or an encoded string

`kesto <command> --help` for the rest."""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0 if argv else 2

    cmd, rest = argv[0], argv[1:]
    if cmd == "read":
        from .vision import main as run
    elif cmd == "solve":
        from .search.bidirectional import main as run
    else:
        print(f"kesto: unknown command {cmd!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    return run(rest)
