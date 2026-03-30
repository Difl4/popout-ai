"""Pure-integer @njit kernels for PopOut game rules — no Python objects cross the JIT boundary.

Single Responsibility: this module owns only JIT-compiled game rule functions.
Mirrors the logic in src/engine/standard/rules.py but in Numba-compiled form.

Board-state operations (apply_move, legal_moves, is_full) live in
src/engine/optimized/numba_bitboard.py.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from src.engine.optimized.numba_bitboard import nb_is_full


@njit(cache=True)
def nb_has_won(bitmask: np.int64) -> bool:
    """4-in-a-row check via bitwise shifts. Mirrors rules.has_won."""
    # Vertical
    m = bitmask & (bitmask >> 1)
    if m & (m >> 2):
        return True
    # Horizontal
    m = bitmask & (bitmask >> 7)
    if m & (m >> 14):
        return True
    # Diagonal '\'
    m = bitmask & (bitmask >> 6)
    if m & (m >> 12):
        return True
    # Diagonal '/'
    m = bitmask & (bitmask >> 8)
    if m & (m >> 16):
        return True
    return False


@njit(cache=True)
def nb_check_winner_for_player(
    mask_p1: np.int64, mask_p2: np.int64, player: np.int32
) -> bool:
    """Mirrors rules.check_winner_for_player."""
    return nb_has_won(mask_p1 if player == 1 else mask_p2)


@njit(cache=True)
def nb_evaluate_after_move(
    mask_p1: np.int64, mask_p2: np.int64, mover: np.int32
) -> np.int32:
    """Mirrors rules.evaluate_after_move — PopOut tiebreak rule included."""
    p1_wins = nb_has_won(mask_p1)
    p2_wins = nb_has_won(mask_p2)
    if p1_wins and p2_wins:
        return mover
    if p1_wins:
        return np.int32(1)
    if p2_wins:
        return np.int32(2)
    return np.int32(0)


@njit(cache=True)
def nb_is_threefold_repetition(
    hist_p1: np.ndarray, hist_p2: np.ndarray, hist_size: np.int32
) -> bool:
    """Mirrors rules.is_threefold_repetition.

    hist_p1/hist_p2 are parallel arrays of board mask snapshots (length hist_size).
    Returns True if any board state appears 3 or more times in the history.
    O(n²) scan — negligible for rollout_depth <= 150.
    """
    for i in range(hist_size):
        count = np.int32(0)
        for j in range(hist_size):
            if hist_p1[j] == hist_p1[i] and hist_p2[j] == hist_p2[i]:
                count += np.int32(1)
        if count >= 3:
            return True
    return False


@njit(cache=True)
def nb_is_draw(
    mask_p1: np.int64, mask_p2: np.int64,
    hist_p1: np.ndarray, hist_p2: np.ndarray, hist_size: np.int32,
) -> bool:
    """Mirrors rules.is_draw — full board or threefold repetition."""
    return nb_is_full(mask_p1, mask_p2) or nb_is_threefold_repetition(hist_p1, hist_p2, hist_size)
