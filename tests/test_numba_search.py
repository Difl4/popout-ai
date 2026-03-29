"""Unit tests for the @njit search kernels in src/algorithms/mcts/optimized/numba_search.py.

A session-scoped fixture pre-compiles all JIT functions so the individual
tests don't pay the one-time compilation cost.
"""

from __future__ import annotations

import pytest
import numpy as np

from src.engine.standard.bitboard import PopOutBoard
from src.algorithms.mcts.optimized.numba_search import (
    nb_simulate,
    nb_expand_step,
)


@pytest.fixture(scope="session", autouse=True)
def _warm_kernels():
    """Trigger JIT compilation once for the entire test session."""
    from src.algorithms.mcts.optimized.numba_mcts import warmup
    warmup()


# ── nb_simulate ───────────────────────────────────────────────────────────────

class TestNbSimulate:
    def test_returns_valid_reward_on_fresh_board(self):
        board = PopOutBoard()
        reward = nb_simulate(
            np.int64(board.mask_p1), np.int64(board.mask_p2),
            np.int32(board.current_player), np.int32(1),
            np.int32(0), np.int32(30),
        )
        assert 0.0 <= float(reward) <= 1.0

    def test_terminal_win_returns_one(self):
        board = PopOutBoard()
        reward = nb_simulate(
            np.int64(board.mask_p1), np.int64(board.mask_p2),
            np.int32(board.current_player),
            np.int32(1),   # initial_mover
            np.int32(1),   # terminal_winner == initial_mover
            np.int32(30),
        )
        assert float(reward) == pytest.approx(1.0)

    def test_terminal_loss_returns_zero(self):
        board = PopOutBoard()
        reward = nb_simulate(
            np.int64(board.mask_p1), np.int64(board.mask_p2),
            np.int32(board.current_player),
            np.int32(1),   # initial_mover = 1
            np.int32(2),   # terminal_winner = 2 ≠ 1
            np.int32(30),
        )
        assert float(reward) == pytest.approx(0.0)

    def test_reward_in_range_across_many_rollouts(self):
        board = PopOutBoard()
        for _ in range(20):
            r = nb_simulate(
                np.int64(board.mask_p1), np.int64(board.mask_p2),
                np.int32(board.current_player), np.int32(1),
                np.int32(0), np.int32(20),
            )
            assert float(r) in (0.0, 0.5, 1.0)


# ── nb_expand_step ────────────────────────────────────────────────────────────

class TestNbExpandStep:
    def test_returns_correct_board_state(self):
        board = PopOutBoard()
        m1, m2, cp, winner, moves, n = nb_expand_step(
            np.int64(board.mask_p1), np.int64(board.mask_p2),
            np.int32(board.current_player), np.int32(3),
        )
        ref = board.clone()
        ref.apply_move(3)
        assert int(m1) == ref.mask_p1
        assert int(m2) == ref.mask_p2
        assert int(cp) == ref.current_player

    def test_legal_moves_match_python(self):
        board = PopOutBoard()
        m1, m2, cp, winner, moves, n = nb_expand_step(
            np.int64(board.mask_p1), np.int64(board.mask_p2),
            np.int32(board.current_player), np.int32(0),
        )
        ref = board.clone()
        ref.apply_move(0)
        assert {int(moves[i]) for i in range(n)} == set(ref.legal_moves())
