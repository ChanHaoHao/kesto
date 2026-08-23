"""`python -m kesto` -- the same entry point as the `kesto` command."""

from .cli import main

raise SystemExit(main())
