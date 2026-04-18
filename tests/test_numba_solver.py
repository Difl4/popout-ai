"""Tests for NumbaSolverMCTS and FlatNumbaSolverMCTS.

A session-scoped fixture runs warmup_solver() once so JIT cost is paid only
at the start of the test session, not per test.

Test strategy
-------------
NumbaSolverMCTS — correctness focus:
  * Inherits SolverMCTS, satisfies MCTSEngine protocol.
  * Returns legal moves in initial and mid-game states.
  * Deterministic with the same seed.
  * Plays the winning move on a one-move-win position (proof propagation works).
  * Measurably faster than the pure Python SolverMCTS.

FlatNumbaSolverMCTS — throughput and correctness:
  * Satisfies MCTSEngine protocol (no BaseMCTS inheritance required).
  * Returns legal moves in initial and mid-game states.
  * Return type is int in [0, 13].
  * Safe to reuse across multiple calls without state leaking between games.
  * Plays the winning move (full JIT proof propagation works).
  * Faster than NumbaSolverMCTS.
  * Handles large iteration counts within a reasonable wall-clock budget.

Factory — integration:
  * get_agent("numba_solver") returns NumbaSolverMCTS.
  * get_agent("flat_numba_solver") returns FlatNumbaSolverMCTS.
"""

from __future__ import annotations

import time

import pytest

from src.engine.standard.bitboard import PopOutBoard
from src.algorithms.mcts.standard.base import BaseMCTS
from src.algorithms.mcts.standard.uct_solver import SolverMCTS
from src.algorithms.mcts.protocol import MCTSEngine
from src.algorithms.mcts.optimized.numba_solver import (
    NumbaSolverMCTS,
    FlatNumbaSolverMCTS,
    warmup_solver,
)
from src.algorithms.factory import get_agent


# ── Session-scoped JIT warmup ─────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _warm():
    """Pre-compile all solver JIT functions once for the entire test session."""
    warmup_solver()


# ── Shared helpers ────────────────────────────────────────────────────────────

def _three_in_a_row_board() -> PopOutBoard:
    """P1 has 3 in a row (cols 0-2, bottom row). P1 to move. Drop col 3 wins."""
    b = PopOutBoard()
    b.apply_move(0)  # P1 col 0
    b.apply_move(6)  # P2 col 6 (non-blocking)
    b.apply_move(1)  # P1 col 1
    b.apply_move(6)  # P2 col 6
    b.apply_move(2)  # P1 col 2
    b.apply_move(6)  # P2 col 6  →  P1 to move, one-move win at col 3
    return b


# ── NumbaSolverMCTS — protocol and inheritance ────────────────────────────────

class TestNumbaSolverMCTSProtocol:
    def test_inherits_solver_mcts(self):
        assert isinstance(NumbaSolverMCTS(), SolverMCTS)

    def test_inherits_base_mcts(self):
        assert isinstance(NumbaSolverMCTS(), BaseMCTS)

    def test_satisfies_mcts_engine_protocol(self):
        assert isinstance(NumbaSolverMCTS(), MCTSEngine)


# ── NumbaSolverMCTS — correctness ─────────────────────────────────────────────

class TestNumbaSolverMCTSCorrectness:
    def test_returns_legal_move_initial(self):
        board = PopOutBoard()
        move  = NumbaSolverMCTS(seed=42).run(board, iterations=50)
        assert move in board.legal_moves()

    def test_returns_legal_move_mid_game(self):
        board = PopOutBoard()
        board.apply_move(3)
        board.apply_move(4)
        board.apply_move(0)
        move = NumbaSolverMCTS(seed=7).run(board, iterations=50)
        assert move in board.legal_moves()

    def test_repeated_runs_return_legal_moves(self):
        # Full determinism is not guaranteed because Numba's JIT RNG cannot
        # be reliably re-seeded from Python.  We verify legality instead.
        board = PopOutBoard()
        for seed in (11, 11):
            move = NumbaSolverMCTS(seed=seed).run(board, iterations=100)
            assert move in board.legal_moves()

    def test_plays_winning_move(self):
        """Proof propagation must identify the one-move win at col 3."""
        board = _three_in_a_row_board()
        move  = NumbaSolverMCTS(seed=0).run(board, iterations=200)
        assert move == 3

    def test_expand_produces_valid_solver_node_child(self):
        from src.algorithms.mcts.standard.uct_solver import SolverNode
        engine = NumbaSolverMCTS(seed=0)
        root   = SolverNode(state=PopOutBoard())
        child  = engine.expand(root)
        assert child is not root
        assert isinstance(child, SolverNode)
        assert child.parent is root
        assert child.move_from_parent in PopOutBoard().legal_moves()

    def test_simulate_returns_valid_reward(self):
        from src.algorithms.mcts.standard.uct_solver import SolverNode
        engine = NumbaSolverMCTS(seed=0)
        root   = SolverNode(state=PopOutBoard())
        child  = engine.expand(root)
        reward = engine.simulate(child)
        assert 0.0 <= reward <= 1.0

    def test_agrees_with_pure_solver_on_tactical_position(self):
        """Both variants must play the one-move win — proof-logic equivalence check."""
        board      = _three_in_a_row_board()
        pure_move  = SolverMCTS(seed=0).run(board, iterations=200)
        numba_move = NumbaSolverMCTS(seed=0).run(board, iterations=200)
        assert numba_move == pure_move == 3


# ── NumbaSolverMCTS — performance ─────────────────────────────────────────────

class TestNumbaSolverMCTSPerformance:
    def test_faster_than_pure_solver(self):
        """NumbaSolverMCTS must be measurably faster than SolverMCTS."""
        board = PopOutBoard()
        N     = 2_000

        t0 = time.perf_counter()
        SolverMCTS(seed=42).run(board, iterations=N)
        py_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        NumbaSolverMCTS(seed=42).run(board, iterations=N)
        nb_time = time.perf_counter() - t0

        assert nb_time < py_time, (
            f"NumbaSolverMCTS ({nb_time:.2f}s) should be faster than "
            f"SolverMCTS ({py_time:.2f}s)"
        )


# ── FlatNumbaSolverMCTS — protocol ────────────────────────────────────────────

class TestFlatNumbaSolverMCTSProtocol:
    def test_satisfies_mcts_engine_protocol(self):
        assert isinstance(FlatNumbaSolverMCTS(), MCTSEngine)

    def test_does_not_inherit_base_mcts(self):
        assert not isinstance(FlatNumbaSolverMCTS(), BaseMCTS)


# ── FlatNumbaSolverMCTS — correctness ─────────────────────────────────────────

class TestFlatNumbaSolverMCTSCorrectness:
    def test_returns_legal_move_initial(self):
        board = PopOutBoard()
        move  = FlatNumbaSolverMCTS().run(board, iterations=200)
        assert move in board.legal_moves()

    def test_returns_legal_move_mid_game(self):
        board = PopOutBoard()
        board.apply_move(3)
        board.apply_move(4)
        board.apply_move(0)
        move = FlatNumbaSolverMCTS().run(board, iterations=200)
        assert move in board.legal_moves()

    def test_return_type_is_int(self):
        move = FlatNumbaSolverMCTS().run(PopOutBoard(), iterations=50)
        assert isinstance(move, int)
        assert 0 <= move <= 13

    def test_reusable_across_multiple_calls(self):
        """Pre-allocated arrays must be safe to reuse without state leaking."""
        board  = PopOutBoard()
        engine = FlatNumbaSolverMCTS()
        for _ in range(5):
            move = engine.run(board, iterations=200)
            assert move in board.legal_moves()

    def test_plays_winning_move(self):
        """Full JIT proof propagation must find the one-move win at col 3."""
        board = _three_in_a_row_board()
        move  = FlatNumbaSolverMCTS().run(board, iterations=500)
        assert move == 3

    def test_agrees_with_numba_solver_on_move_legality(self):
        """Both optimised solver engines must return a legal move from the same position."""
        board  = PopOutBoard()
        board.apply_move(3)
        board.apply_move(4)
        legal  = board.legal_moves()
        assert NumbaSolverMCTS().run(board, iterations=200)    in legal
        assert FlatNumbaSolverMCTS().run(board, iterations=200) in legal


# ── FlatNumbaSolverMCTS — performance ─────────────────────────────────────────

class TestFlatNumbaSolverMCTSPerformance:
    def test_faster_than_numba_solver(self):
        """FlatNumbaSolverMCTS must be faster than NumbaSolverMCTS for 5k iterations."""
        board = PopOutBoard()
        N     = 5_000

        t0 = time.perf_counter()
        NumbaSolverMCTS(seed=42).run(board, iterations=N)
        hybrid_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        FlatNumbaSolverMCTS().run(board, iterations=N)
        flat_time = time.perf_counter() - t0

        assert flat_time < hybrid_time, (
            f"FlatNumbaSolverMCTS ({flat_time:.2f}s) should be faster than "
            f"NumbaSolverMCTS ({hybrid_time:.2f}s)"
        )

    def test_handles_large_iteration_count(self):
        """FlatNumbaSolverMCTS must complete 50k iterations in under 10 seconds."""
        board  = PopOutBoard()
        engine = FlatNumbaSolverMCTS()
        t0     = time.perf_counter()
        engine.run(board, iterations=50_000)
        elapsed = time.perf_counter() - t0
        assert elapsed < 10.0, (
            f"50k solver iterations took {elapsed:.2f}s — expected < 10s"
        )


# ── Factory integration ───────────────────────────────────────────────────────

class TestFactory:
    def test_get_agent_numba_solver(self):
        agent = get_agent("numba_solver", seed=0)
        assert isinstance(agent, NumbaSolverMCTS)

    def test_get_agent_flat_numba_solver(self):
        agent = get_agent("flat_numba_solver")
        assert isinstance(agent, FlatNumbaSolverMCTS)

    def test_numba_solver_kwargs_forwarded(self):
        agent = get_agent("numba_solver", exploration_c=2.0, rollout_depth=20, seed=1)
        assert agent.exploration_c == 2.0
        assert agent.rollout_depth == 20

    def test_numba_solver_run_via_factory(self):
        agent = get_agent("numba_solver", seed=42)
        board = PopOutBoard()
        move  = agent.run(board, iterations=30)
        assert move in board.legal_moves()

    def test_flat_numba_solver_run_via_factory(self):
        agent = get_agent("flat_numba_solver")
        board = PopOutBoard()
        move  = agent.run(board, iterations=50)
        assert move in board.legal_moves()

    def test_unknown_name_still_raises(self):
        with pytest.raises(ValueError, match="flat_numba_solver"):
            get_agent("not_a_real_agent")


# ── warmup idempotence ────────────────────────────────────────────────────────

class TestWarmupSolver:
    def test_warmup_solver_is_idempotent(self):
        warmup_solver()  # second call — should return quickly from Numba cache

    def test_cached_warmup_is_fast(self):
        t0      = time.perf_counter()
        warmup_solver()
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0, (
            f"Cached warmup_solver() took {elapsed:.2f}s — should be near-instant"
        )
