#!/usr/bin/env python3
"""Clear Numba cache and recompile all @njit kernels.

Run this whenever you change a constant or logic inside kernels.py:

    python -m src.utils.numba_tools
"""

import glob
import os


def clear_numba_cache(root: str = ".") -> int:
    """Delete all .nbi and .nbc files under *root*. Returns count of deleted files."""
    patterns = [
        os.path.join(root, "**", "*.nbi"),
        os.path.join(root, "**", "*.nbc"),
    ]
    deleted = 0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            os.remove(path)
            print(f"  removed {path}")
            deleted += 1
    return deleted


def recompile() -> None:
    """Clear Numba cache and trigger warmup recompilation."""
    print("=== Clearing Numba cache ===")
    n = clear_numba_cache()
    print(f"Deleted {n} cache file(s).\n")

    print("=== Recompiling (warmup) ===")
    from src.algorithms.mcts import warmup
    warmup()
    print("Done. All @njit functions recompiled and cached.")


if __name__ == "__main__":
    recompile()
