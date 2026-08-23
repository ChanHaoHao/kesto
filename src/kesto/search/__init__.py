"""Bidirectional BFS over uint64 bitboards, and the pieces it is built from.

The dependency order is `engine` -> `backward` -> `levels` -> `bidirectional`:
`engine` owns the vectorised `vmove` plus the memory guards, `backward` owns
`predecessors`, `levels` owns the sorted-array set algebra and the scratch
directory, and `bidirectional` grows both shells and meets them in the middle.

States are 8-byte bitboards in numpy arrays rather than dict keys, which is what
puts depth-30 boards in reach at all.
"""

from .bidirectional import run, search

__all__ = ["run", "search"]
