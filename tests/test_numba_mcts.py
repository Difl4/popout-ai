"""Tests for NumbaMCTS, FlatNumbaMCTS, warmup() and the MCTSEngine protocol.

A session-scoped fixture runs warmup() once so JIT cost is paid only at the
start of the test session, not per test.
"""

from __future__ import annotations

import time

import pytest

from src.engine.standard.bitboard import PopOutBoard
from src.algorithms.mcts.standard.base import BaseMCTS, MCTSNode
from src.algorithms.mcts.protocol import MCTSEngine
from src.algorithms.mcts.optimized.numba_mcts import FlatNumbaMCTS, NumbaMCTS, warmup


@pytest.fixture(scope="session", autouse=True)
def _warm():
    """Pre-compile all JIT functions once for the entire test session."""
    warmup()


# ── MCTSEngine Protocol ───────────────────────────────────────────────────────

class TestMCTSEngineProtocol:
    """All engines must satisfy the MCTSEngine structural interface."""

    def test_base_mcts_satisfies_protocol(self):
        assert isinstance(BaseMCTS(), MCTSEngine)

    def test_numba_mcts_satisfies_protocol(self):
        assert isinstance(NumbaMCTS(), MCTSEngine)

    def test_flat_numba_mcts_satisfies_protocol(self):
        assert isinstance(FlatNumbaMCTS(), MCTSEngine)

    def test_object_without_run_does_not_satisfy(self):
        class NoRun:
            pass
        assert not isinstance(NoRun(), MCTSEngine)


# ── NumbaMCTS ─────────────────────────────────────────────────────────────────

class TestNumbaMCTS:
    def test_returns_legal_move_initial_state(self):
        board = PopOutBoard()
        move = NumbaMCTS(seed=42).run(board, iterations=50)
        assert move in board.legal_moves()

    def test_returns_legal_move_mid_game(self):
        board = PopOutBoard()
        board.apply_move(3)
        board.apply_move(3)
        board.apply_move(0)
        move = NumbaMCTS(seed=42).run(board, iterations=50)
        assert move in board.legal_moves()

    def test_inherits_base_mcts(self):
        assert isinstance(NumbaMCTS(), BaseMCTS)

    def test_simulate_returns_valid_reward(self):
        board = PopOutBoard()
        engine = NumbaMCTS(seed=42)
        root = MCTSNode(state=board)
        child = engine.expand(root)
        reward = engine.simulate(child)
        assert 0.0 <= reward <= 1.0

    def test_expand_builds_valid_child(self):
        board = PopOutBoard()
        engine = NumbaMCTS(seed=42)
        root = MCTSNode(state=board)
        child = engine.expand(root)
        assert child is not root
        assert child.move_from_parent in board.legal_moves()
        assert child.parent is root

    def test_expand_child_has_valid_untried_moves(self):
        board = PopOutBoard()
        engine = NumbaMCTS(seed=42)
        root = MCTSNode(state=board)
        child = engine.expand(root)
        # Child's untried moves must all be legal from child's state
        legal_from_child = set(child.state.legal_moves())
        for m in child.untried_moves:
            assert m in legal_from_child

    def test_best_child_returns_child_of_node(self):
        board = PopOutBoard()
        engine = NumbaMCTS(seed=42)
        root = MCTSNode(state=board)
        c1 = engine.expand(root)
        c2 = engine.expand(root)
        c1.visits, c1.value_sum = 5, 3.0
        c2.visits, c2.value_sum = 3, 2.0
        root.visits = 8
        chosen = engine.best_child(root)
        assert chosen in root.children.values()

    def test_is_faster_than_base_mcts(self):
        """NumbaMCTS must be measurably faster for 5k iterations."""
        board = PopOutBoard()
        N = 5_000

        t0 = time.perf_counter()
        BaseMCTS(seed=42).run(board, iterations=N)
        py_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        NumbaMCTS(seed=42).run(board, iterations=N)
        nb_time = time.perf_counter() - t0

        assert nb_time < py_time, (
            f"NumbaMCTS ({nb_time:.2f}s) should be faster than BaseMCTS ({py_time:.2f}s)"
        )


# ── FlatNumbaMCTS ─────────────────────────────────────────────────────────────

class TestFlatNumbaMCTS:
    def test_returns_legal_move_initial_state(self):
        board = PopOutBoard()
        move = FlatNumbaMCTS().run(board, iterations=200)
        assert move in board.legal_moves()

    def test_returns_legal_move_mid_game(self):
        board = PopOutBoard()
        board.apply_move(3)
        board.apply_move(3)
        board.apply_move(0)
        move = FlatNumbaMCTS().run(board, iterations=200)
        assert move in board.legal_moves()

    def test_return_type_is_int(self):
        move = FlatNumbaMCTS().run(PopOutBoard(), iterations=50)
        assert isinstance(move, int)
        assert 0 <= move <= 13

    def test_reusable_across_multiple_calls(self):
        """The pre-allocated arrays must be safe to reuse without state leak."""
        board = PopOutBoard()
        engine = FlatNumbaMCTS()
        for _ in range(5):
            move = engine.run(board, iterations=200)
            assert move in board.legal_moves()

    def test_hits_100k_iterations_under_5_seconds(self):
        """FlatNumbaMCTS must complete 100k iterations in < 5 s."""
        board = PopOutBoard()
        engine = FlatNumbaMCTS()
        t0 = time.perf_counter()
        engine.run(board, iterations=100_000)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"100k iterations took {elapsed:.2f}s — expected < 5s"

    def test_at_least_5x_faster_than_base_mcts(self):
        """FlatNumbaMCTS must be at least 5× faster than BaseMCTS."""
        board = PopOutBoard()
        N = 10_000

        t0 = time.perf_counter()
        BaseMCTS(seed=42).run(board, iterations=N)
        py_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        FlatNumbaMCTS().run(board, iterations=N)
        flat_time = time.perf_counter() - t0

        assert flat_time < py_time / 5, (
            f"FlatNumbaMCTS ({flat_time:.2f}s) should be 5x faster "
            f"than BaseMCTS ({py_time:.2f}s)"
        )

    def test_agrees_with_numba_mcts_on_move_legality(self):
        """Both optimised engines must return a legal move from the same position."""
        board = PopOutBoard()
        board.apply_move(3)
        board.apply_move(4)
        legal = board.legal_moves()
        assert NumbaMCTS().run(board, iterations=200) in legal
        assert FlatNumbaMCTS().run(board, iterations=200) in legal


# ── warmup ────────────────────────────────────────────────────────────────────

class TestWarmup:
    def test_warmup_runs_without_error(self):
        warmup()   # idempotent — safe to call again

    def test_cached_warmup_is_fast(self):
        """Second warmup() call must finish well under 2 seconds (cache hit)."""
        t0 = time.perf_counter()
        warmup()
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"Cached warmup took {elapsed:.2f}s — should be near-instant"

    def test_engines_work_immediately_after_warmup(self):
        warmup()
        board = PopOutBoard()
        assert FlatNumbaMCTS().run(board, iterations=100) in board.legal_moves()
