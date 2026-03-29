"""Pure-integer @njit kernels for PopOut game rules — no Python objects cross the JIT boundary.

Single Responsibility: this module owns only JIT-compiled game mechanic functions.
Mirrors the logic in src/engine/standard/rules.py but in Numba-compiled form.

Constants
---------
_CLEAN_MASK : int
    Keeps only the 6 valid row bits per column (removes bit 6 of each column
    that leaks during pop).  Precomputed at import time:
    sum(0b111111 << (i*7) for i in range(7)) == 279258638311359
_TOP_ROW_MASK : int
    Bits at positions 5, 12, 19, 26, 33, 40, 47 — the topmost row of each
    of the 7 columns. Used to detect a full board.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Module-level constants — captured as compile-time literals by Numba
# ---------------------------------------------------------------------------
_CLEAN_MASK: int = sum(0b111111 << (i * 7) for i in range(7))  # 279258638311359
_TOP_ROW_MASK: int = 0x810204081020                              # bits 5,12,19,26,33,40,47


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
def nb_legal_moves(mask_p1: np.int64, mask_p2: np.int64, current_player: np.int32):
    """Return (moves_array, n) with legal move ints in moves_array[:n].

    Moves 0-6  -> drop into column.
    Moves 7-13 -> pop from column.
    """
    full_mask = mask_p1 | mask_p2
    target_mask = mask_p1 if current_player == 1 else mask_p2

    moves = np.empty(14, dtype=np.int32)
    n = np.int32(0)

    for c in range(7):
        if not (full_mask & (np.int64(1) << np.int64(c * 7 + 5))):
            moves[n] = np.int32(c)
            n += np.int32(1)

    for c in range(7):
        if target_mask & (np.int64(1) << np.int64(c * 7)):
            moves[n] = np.int32(c + 7)
            n += np.int32(1)

    return moves, n


@njit(cache=True)
def nb_apply_move(
    mask_p1: np.int64,
    mask_p2: np.int64,
    current_player: np.int32,
    move: np.int32,
):
    """Apply move (0-13) and return (new_mask_p1, new_mask_p2, next_player).

    Pure function — no mutation of inputs.
    """
    CLEAN = np.int64(279258638311359)  # _CLEAN_MASK precomputed

    if move < 7:
        col = np.int64(move)
        full_mask = mask_p1 | mask_p2
        col_bits = (full_mask >> (col * np.int64(7))) & np.int64(0b111111)
        new_piece = ((col_bits + np.int64(1)) & ~col_bits) << (col * np.int64(7))
        if current_player == 1:
            mask_p1 = mask_p1 | new_piece
        else:
            mask_p2 = mask_p2 | new_piece
    else:
        col = np.int64(move - 7)
        shift = col * np.int64(7)
        col_bit_mask = np.int64(0b111111) << shift
        mask_p1 = (mask_p1 & ~col_bit_mask) | ((mask_p1 & col_bit_mask) >> np.int64(1)) & col_bit_mask
        mask_p2 = (mask_p2 & ~col_bit_mask) | ((mask_p2 & col_bit_mask) >> np.int64(1)) & col_bit_mask
        mask_p1 &= CLEAN
        mask_p2 &= CLEAN

    return mask_p1, mask_p2, np.int32(3 - current_player)


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
def nb_is_full(mask_p1: np.int64, mask_p2: np.int64) -> bool:
    """True when every column's top row bit is occupied."""
    TOP = np.int64(0x810204081020)
    return ((mask_p1 | mask_p2) & TOP) == TOP
