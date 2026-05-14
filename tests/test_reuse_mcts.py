"""Tests for MCTS tree-reuse variants.

Covers:
  - ReuseFlatNumbaSolverMCTS  (flat arrays + Solver, ~30-60 k iter/s)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.standard.bitboard import PopOutBoard
from src.mcts.protocol import MCTSEngine


# ── module-level Numba warmup ─────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _warm_numba_all():
    """Pre-compile all Numba kernels once for the whole module."""
    from src.mcts.optimized.numba_mcts import warmup
    from src.mcts.optimized.numba_solver import warmup_solver
    warmup()
    warmup_solver()


# ── ReuseFlatNumbaSolverMCTS ──────────────────────────────────────────────────

class TestReuseFlatNumbaSolverMCTS:
    """Tests for flat-array solver tree reuse.

    Warmup is handled by the module-level _warm_numba_all fixture.
    """

    def _agent(self, **kwargs):
        from src.mcts.optimized.numba_solver import ReuseFlatNumbaSolverMCTS
        return ReuseFlatNumbaSolverMCTS(**kwargs)

    # basic correctness ────────────────────────────────────────────────────────

    def test_returns_legal_move_fresh(self):
        agent = self._agent(seed=0)
        board = PopOutBoard()
        move = agent.run(board, 200)
        assert move in board.legal_moves()

    def test_returns_legal_move_with_opponent_move(self):
        agent = self._agent(seed=0)
        board = PopOutBoard()

        move = agent.run(board, 200)
        assert move in board.legal_moves()
        board.apply_move(move)

        legal = board.legal_moves()
        assert legal
        opp = legal[0]
        board.apply_move(opp)

        move2 = agent.run(board, 200, opponent_move=opp)
        assert move2 in board.legal_moves()

    # compaction correctness ───────────────────────────────────────────────────

    def test_compaction_preserves_proven_status(self):
        """After compaction, status/distance of inherited nodes must be intact."""
        from src.mcts.optimized.numba_solver import ReuseFlatNumbaSolverMCTS
        agent = ReuseFlatNumbaSolverMCTS(seed=0)
        board = PopOutBoard()

        move = agent.run(board, 500)
        board.apply_move(move)

        # Find an opponent move that was explored (in the best-child subtree)
        bc = agent._best_child_idx
        assert bc >= 0, "best_child_idx should be set after run()"

        nc = int(agent._n_children[bc])
        if nc == 0:
            pytest.skip("best-child has no explored children yet")

        opp_idx = int(agent._children[bc, 0])
        opp_move = int(agent._move_fp[opp_idx])
        board.apply_move(opp_move)

        # Remember status/distance before compaction
        old_status   = int(agent._status[opp_idx])
        old_distance = int(agent._distance[opp_idx])
        old_visits   = int(agent._visits[opp_idx])

        # Trigger compaction via run() with 50 extra iterations on top
        agent.run(board, 50, opponent_move=opp_move)

        # After compaction + 50 new iterations:
        # - visits must be >= old_visits (compaction preserved them; new iters add more)
        assert int(agent._visits[0]) >= old_visits, "compaction must preserve visits"
        # - status can only transition UNKNOWN → proven, never backwards
        if old_status != 0:   # 0 = STATUS_UNKNOWN
            assert int(agent._status[0]) == old_status,   "proven status must survive compaction"
            assert int(agent._distance[0]) == old_distance, "distance must survive compaction"

    def test_best_child_idx_set_after_run(self):
        agent = self._agent(seed=0)
        board = PopOutBoard()
        assert agent._best_child_idx == -1
        agent.run(board, 100)
        assert agent._best_child_idx >= 0

    def test_reset_clears_cache(self):
        agent = self._agent(seed=0)
        agent.run(PopOutBoard(), 100)
        assert agent._best_child_idx >= 0
        agent.reset()
        assert agent._best_child_idx == -1

    def test_fresh_start_after_unexpected_opponent_move(self):
        """If opponent plays a move we never expanded, run() starts fresh."""
        agent = self._agent(seed=0)
        board = PopOutBoard()
        move = agent.run(board, 10)   # few iters → probably not all children
        board.apply_move(move)

        # Force best_child to have no children (empty tree)
        agent._best_child_idx = 0
        agent._n_children[0]  = np.int32(0)

        board2 = board.clone()
        legal  = board2.legal_moves()
        assert legal
        board2.apply_move(legal[0])

        move2 = agent.run(board2, 50, opponent_move=legal[0])
        assert move2 in board2.legal_moves()

    # multi-turn sequence ──────────────────────────────────────────────────────

    def test_full_turn_sequence(self):
        """Several alternating turns with reuse; every move must be legal."""
        from src.mcts.optimized.numba_solver import ReuseFlatNumbaSolverMCTS
        board    = PopOutBoard()
        agent    = ReuseFlatNumbaSolverMCTS(seed=3)
        opponent = ReuseFlatNumbaSolverMCTS(seed=19)
        agent.reset()
        opponent.reset()

        last_opp_move:   int | None = None
        last_agent_move: int | None = None

        for _ in range(8):
            if not board.legal_moves():
                break
            agent_move = agent.run(board, 150, opponent_move=last_opp_move)
            assert agent_move in board.legal_moves()
            board.apply_move(agent_move)
            last_agent_move = agent_move

            if not board.legal_moves():
                break
            opp_move = opponent.run(board, 150, opponent_move=last_agent_move)
            assert opp_move in board.legal_moves()
            board.apply_move(opp_move)
            last_opp_move = opp_move

    # factory & protocol ───────────────────────────────────────────────────────

    def test_factory(self):
        from src.mcts.factory import get_agent
        from src.mcts.optimized.numba_solver import ReuseFlatNumbaSolverMCTS
        agent = get_agent("reuse_flat_numba_solver", seed=0)
        assert isinstance(agent, ReuseFlatNumbaSolverMCTS)
        move = agent.run(PopOutBoard(), 100)
        assert 0 <= move <= 13

    def test_satisfies_mcts_engine_protocol(self):
        assert isinstance(self._agent(), MCTSEngine)
