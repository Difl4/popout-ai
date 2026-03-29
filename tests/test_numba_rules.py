"""Unit tests for the @njit kernel functions in src/engine/optimized/numba_rules.py.

Each kernel is tested in isolation against the Python reference implementation
(standard/rules.py / standard/bitboard.py) to guarantee mathematical equivalence.

A session-scoped fixture pre-compiles all JIT functions so the individual
tests don't pay the one-time compilation cost.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import has_won, evaluate_after_move
from src.engine.optimized.numba_rules import (
    nb_has_won,
    nb_legal_moves,
    nb_apply_move,
    nb_evaluate_after_move,
    nb_is_full,
)


@pytest.fixture(scope="session", autouse=True)
def _warm_kernels():
    """Trigger JIT compilation once for the entire test session."""
    from src.algorithms.mcts.optimized.numba_mcts import warmup
    warmup()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _h_mask(*col_row_pairs: tuple[int, int]) -> np.int64:
    """Build a bitmask from (col, row) pairs using the 7-bits-per-col layout."""
    mask = 0
    for col, row in col_row_pairs:
        mask |= 1 << (col * 7 + row)
    return np.int64(mask)


def _board_ints(board: PopOutBoard):
    return np.int64(board.mask_p1), np.int64(board.mask_p2), np.int32(board.current_player)


# ── nb_has_won ────────────────────────────────────────────────────────────────

class TestNbHasWon:
    def test_empty_board_no_win(self):
        assert nb_has_won(np.int64(0)) is False

    def test_horizontal_4_in_a_row(self):
        # cols 0-3, row 0 → horizontal win
        mask = _h_mask((0, 0), (1, 0), (2, 0), (3, 0))
        assert nb_has_won(mask) is True

    def test_horizontal_3_not_win(self):
        mask = _h_mask((0, 0), (1, 0), (2, 0))
        assert nb_has_won(mask) is False

    def test_vertical_4_in_a_row(self):
        # col 0, rows 0-3
        mask = _h_mask((0, 0), (0, 1), (0, 2), (0, 3))
        assert nb_has_won(mask) is True

    def test_vertical_3_not_win(self):
        mask = _h_mask((0, 0), (0, 1), (0, 2))
        assert nb_has_won(mask) is False

    def test_diagonal_down_right(self):
        # (col, row): (0,0),(1,1),(2,2),(3,3)
        mask = _h_mask((0, 0), (1, 1), (2, 2), (3, 3))
        assert nb_has_won(mask) is True

    def test_diagonal_down_left(self):
        # (col, row): (3,0),(2,1),(1,2),(0,3)
        mask = _h_mask((3, 0), (2, 1), (1, 2), (0, 3))
        assert nb_has_won(mask) is True

    def test_agrees_with_python_reference(self):
        """nb_has_won must match rules.has_won for arbitrary masks."""
        for mask_int in [0, 0x3F, 0x810204081020, 0x1041041]:
            mask = np.int64(mask_int)
            assert bool(nb_has_won(mask)) == has_won(mask_int)


# ── nb_legal_moves ────────────────────────────────────────────────────────────

class TestNbLegalMoves:
    def test_empty_board_only_7_drops(self):
        board = PopOutBoard()
        moves, n = nb_legal_moves(*_board_ints(board))
        assert int(n) == 7
        assert all(0 <= int(moves[i]) < 7 for i in range(n))

    def test_no_pops_for_player_without_pieces(self):
        board = PopOutBoard()
        board.apply_move(0)  # P1 drops col 0 → P2 has no pieces
        moves, n = nb_legal_moves(*_board_ints(board))
        legal = {int(moves[i]) for i in range(n)}
        assert not any(m >= 7 for m in legal)

    def test_pop_available_when_player_has_bottom_piece(self):
        board = PopOutBoard()
        board.apply_move(0)  # P1 col 0
        board.apply_move(1)  # P2 col 1 — now P1's turn again
        moves, n = nb_legal_moves(*_board_ints(board))
        legal = {int(moves[i]) for i in range(n)}
        assert 7 in legal  # P1 can pop col 0 (move = 7+0)

    def test_full_column_not_in_drops(self):
        board = PopOutBoard()
        for _ in range(6):
            board.apply_move(0)
        moves, n = nb_legal_moves(*_board_ints(board))
        legal = {int(moves[i]) for i in range(n)}
        assert 0 not in legal

    def test_matches_python_legal_moves(self):
        board = PopOutBoard()
        board.apply_move(3)
        board.apply_move(3)
        board.apply_move(0)
        py_legal = set(board.legal_moves())
        nb_moves, nb_n = nb_legal_moves(*_board_ints(board))
        nb_legal = {int(nb_moves[i]) for i in range(nb_n)}
        assert nb_legal == py_legal


# ── nb_apply_move ─────────────────────────────────────────────────────────────

class TestNbApplyMove:
    def test_drop_sets_bit_at_bottom(self):
        m1, m2, cp = nb_apply_move(np.int64(0), np.int64(0), np.int32(1), np.int32(0))
        assert m1 & (1 << 0)   # bit 0 = col 0, row 0
        assert int(cp) == 2

    def test_drop_stacks_second_piece_above(self):
        m1_a, m2_a, cp_a = nb_apply_move(np.int64(0), np.int64(0), np.int32(1), np.int32(0))
        m1_b, m2_b, cp_b = nb_apply_move(m1_a, m2_a, cp_a, np.int32(0))
        assert m1_b & (1 << 0)   # P1 at row 0
        assert m2_b & (1 << 1)   # P2 at row 1

    def test_pop_clears_bottom_bit(self):
        board = PopOutBoard()
        board.apply_move(0)   # P1 drops col 0
        board.apply_move(1)   # P2 drops col 1 → P1's turn
        m1, m2, cp = nb_apply_move(
            np.int64(board.mask_p1), np.int64(board.mask_p2),
            np.int32(board.current_player), np.int32(7),  # pop col 0
        )
        assert not (m1 & (1 << 0))   # bottom piece removed

    def test_player_alternates(self):
        m1, m2, cp = nb_apply_move(np.int64(0), np.int64(0), np.int32(1), np.int32(0))
        assert int(cp) == 2
        m1, m2, cp = nb_apply_move(m1, m2, cp, np.int32(1))
        assert int(cp) == 1

    def test_matches_python_apply_move(self):
        """Kernel output must equal Python reference for every step."""
        board = PopOutBoard()
        board.apply_move(3)
        for move in [2, 5, 9]:   # drop, drop, pop
            if move not in board.legal_moves():
                continue
            m1_nb, m2_nb, cp_nb = nb_apply_move(
                np.int64(board.mask_p1), np.int64(board.mask_p2),
                np.int32(board.current_player), np.int32(move),
            )
            ref = board.clone()
            ref.apply_move(move)
            assert int(m1_nb) == ref.mask_p1
            assert int(m2_nb) == ref.mask_p2
            assert int(cp_nb) == ref.current_player
            board = ref   # advance state for next iteration


# ── nb_evaluate_after_move ────────────────────────────────────────────────────

class TestNbEvaluateAfterMove:
    def test_no_winner_empty_board(self):
        result = nb_evaluate_after_move(np.int64(0), np.int64(0), np.int32(1))
        assert int(result) == 0

    def test_player1_horizontal_win(self):
        mask_p1 = _h_mask((0, 0), (1, 0), (2, 0), (3, 0))
        result = nb_evaluate_after_move(mask_p1, np.int64(0), np.int32(1))
        assert int(result) == 1

    def test_player2_vertical_win(self):
        mask_p2 = _h_mask((0, 0), (0, 1), (0, 2), (0, 3))
        result = nb_evaluate_after_move(np.int64(0), mask_p2, np.int32(2))
        assert int(result) == 2

    def test_both_win_mover_takes_precedence(self):
        mask_p1 = _h_mask((0, 0), (1, 0), (2, 0), (3, 0))
        mask_p2 = _h_mask((0, 0), (0, 1), (0, 2), (0, 3))
        result_p1 = nb_evaluate_after_move(mask_p1, mask_p2, np.int32(1))
        result_p2 = nb_evaluate_after_move(mask_p1, mask_p2, np.int32(2))
        assert int(result_p1) == 1
        assert int(result_p2) == 2

    def test_agrees_with_python_reference(self):
        board = PopOutBoard()
        for move in [3, 3, 3, 3, 4, 4, 4, 4]:
            if move in board.legal_moves():
                mover = board.current_player
                board.apply_move(move)
                py_result = evaluate_after_move(board, mover=mover)
                nb_result = nb_evaluate_after_move(
                    np.int64(board.mask_p1), np.int64(board.mask_p2), np.int32(mover)
                )
                assert int(nb_result) == py_result


# ── nb_is_full ────────────────────────────────────────────────────────────────

class TestNbIsFull:
    def test_empty_board_not_full(self):
        assert nb_is_full(np.int64(0), np.int64(0)) is False

    def test_agrees_with_python_is_full(self):
        board = PopOutBoard()
        # partially fill a few columns
        for c in [0, 1, 2]:
            for _ in range(6):
                board.apply_move(c)
        m1, m2 = np.int64(board.mask_p1), np.int64(board.mask_p2)
        assert bool(nb_is_full(m1, m2)) == board.is_full()
