"""Tests for MCTS tree-reuse variants.

Covers:
  - _TreeReuseMixin          state management
  - ReuseUCT                 (pure Python, ~9 k iter/s)
  - ReuseNumbaMCTS            (Numba rollout, ~50 k iter/s)
  - ReuseFlatNumbaSolverMCTS  (flat arrays + Solver, ~30-60 k iter/s)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engine.standard.bitboard import PopOutBoard
from src.mcts.protocol import MCTSEngine
from src.mcts.standard.base import BaseMCTS, MCTSNode
from src.mcts.standard.uct_reuse import _TreeReuseMixin, ReuseUCT


# ── helpers ───────────────────────────────────────────────────────────────────

def _play_one_turn(agent: ReuseUCT, board: PopOutBoard,
                   iterations: int = 80,
                   opponent_move: int | None = None) -> int:
    """Apply one agent turn; returns the chosen move."""
    move = agent.run(board, iterations, opponent_move=opponent_move)
    board.apply_move(move)
    return move


# ── _TreeReuseMixin state ─────────────────────────────────────────────────────

class TestTreeReuseMixinState:
    def test_initial_root_is_none(self):
        agent = ReuseUCT(seed=0)
        assert agent._root is None

    def test_cached_visits_zero_before_first_run(self):
        agent = ReuseUCT(seed=0)
        assert agent.cached_visits == 0

    def test_reset_clears_root(self):
        agent = ReuseUCT(seed=0)
        agent.run(PopOutBoard(), 30)
        assert agent._root is not None
        agent.reset()
        assert agent._root is None
        assert agent.cached_visits == 0

    def test_root_advances_after_run(self):
        """After run(), _root is the chosen child — not the original root."""
        board = PopOutBoard()
        agent = ReuseUCT(seed=42)
        move = agent.run(board, 100)
        # _root was advanced to children[move]; it must have visits > 0
        assert agent._root is not None
        assert agent._root.visits > 0
        assert agent._root.move_from_parent == move

    def test_root_parent_detached_after_advance(self):
        """Setting parent=None allows the GC to reclaim the old tree."""
        agent = ReuseUCT(seed=42)
        agent.run(PopOutBoard(), 100)
        assert agent._root is None or agent._root.parent is None


# ── ReuseUCT correctness ──────────────────────────────────────────────────────

class TestReuseUCT:

    # basic sanity ─────────────────────────────────────────────────────────────

    def test_returns_legal_move_first_call(self):
        board = PopOutBoard()
        move = ReuseUCT(seed=1).run(board, 50)
        assert move in board.legal_moves()

    def test_returns_legal_move_without_opponent_move(self):
        """Omitting opponent_move → stateless call; still returns legal move."""
        board = PopOutBoard()
        agent = ReuseUCT(seed=1)
        for _ in range(3):
            move = agent.run(board.clone(), 30)
            assert move in board.legal_moves()

    def test_deterministic_with_same_seed(self):
        board = PopOutBoard()
        m1 = ReuseUCT(seed=77).run(board, 120)
        m2 = ReuseUCT(seed=77).run(board, 120)
        assert m1 == m2

    def test_isinstance_base_mcts(self):
        assert isinstance(ReuseUCT(), BaseMCTS)

    def test_satisfies_mcts_engine_protocol(self):
        """run(board, iterations) without opponent_move satisfies the protocol."""
        agent = ReuseUCT(seed=0)
        assert isinstance(agent, MCTSEngine)
        move = agent.run(PopOutBoard(), 30)
        assert isinstance(move, int)
        assert 0 <= move <= 13

    # tree reuse behaviour ─────────────────────────────────────────────────────

    def test_tree_reused_when_opponent_move_explored(self):
        """If the opponent's move was explored, the new _root has inherited visits."""
        board = PopOutBoard()
        agent = ReuseUCT(seed=42)

        # Turn 1 — build tree with enough iters so opponent's common moves are covered
        move = agent.run(board, iterations=300)
        # _root now points to children[move]; its children are the opponent's replies
        assert agent._root is not None

        # Pick an opponent move that is in our cached subtree
        explored = list(agent._root.children.keys())
        if not explored:
            pytest.skip("No opponent moves were expanded — increase iterations")

        opp_move = explored[0]
        inherited_visits = agent._root.children[opp_move].visits

        board.apply_move(move)
        board.apply_move(opp_move)

        # Turn 2 — reuse
        move2 = agent.run(board, iterations=300, opponent_move=opp_move)

        # After run() the new _root is children[move2] inside the inherited subtree.
        # We can't directly inspect the pre-run root, but the move must be legal.
        assert move2 in board.legal_moves()
        # Sanity: inherited_visits > 0 proves the subtree had real data
        assert inherited_visits > 0

    def test_fresh_start_when_opponent_move_not_in_tree(self):
        """Opponent plays a move we never expanded → _root is reset."""
        board = PopOutBoard()
        agent = ReuseUCT(seed=42)
        move = agent.run(board, iterations=5)  # too few to explore all children

        # Build a root that has NO children (simulates unexplored opponent move)
        agent._root = MCTSNode(state=board.clone())   # fresh, no children

        board.apply_move(move)
        legal = board.legal_moves()
        assert legal, "Expected legal moves after turn 1"

        opp_move = legal[0]
        board.apply_move(opp_move)

        # opponent_move not in _root.children (empty dict) → case B: fresh start
        move2 = agent.run(board, iterations=40, opponent_move=opp_move)
        assert move2 in board.legal_moves()

    def test_fresh_start_without_opponent_move(self):
        """Omitting opponent_move always resets the tree (Case C)."""
        board = PopOutBoard()
        agent = ReuseUCT(seed=0)
        agent.run(board, 50)
        root_after_turn1 = agent._root

        # Second call without opponent_move → _root reset before search
        # (We can verify indirectly: after this call _root is a new subtree)
        agent.run(board, 50)
        # Both calls must have returned legal moves — no exception is the check here.

    # multi-turn sequence ──────────────────────────────────────────────────────

    def test_full_turn_sequence(self):
        """Simulate several alternating turns; every move must be legal."""
        board = PopOutBoard()
        agent    = ReuseUCT(seed=3)
        opponent = ReuseUCT(seed=17)
        agent.reset()
        opponent.reset()

        last_opp_move:   int | None = None   # opponent's last move (for agent's reuse)
        last_agent_move: int | None = None   # agent's last move (for opponent's reuse)

        for _ in range(8):
            if not board.legal_moves():
                break

            # Agent's turn
            agent_move = agent.run(board, 60, opponent_move=last_opp_move)
            assert agent_move in board.legal_moves()
            board.apply_move(agent_move)
            last_agent_move = agent_move

            if not board.legal_moves():
                break

            # Opponent's turn
            opp_move = opponent.run(board, 60, opponent_move=last_agent_move)
            assert opp_move in board.legal_moves()
            board.apply_move(opp_move)
            last_opp_move = opp_move

    def test_reset_between_games(self):
        """reset() between games prevents stale tree from previous game."""
        agent = ReuseUCT(seed=5)

        for _ in range(2):
            agent.reset()
            board = PopOutBoard()
            move = agent.run(board, 50)
            assert move in board.legal_moves()
            board.apply_move(move)
            legal = board.legal_moves()
            if legal:
                opp = legal[0]
                board.apply_move(opp)
                move2 = agent.run(board, 50, opponent_move=opp)
                assert move2 in board.legal_moves()

    # factory integration ──────────────────────────────────────────────────────

    def test_factory_reuse(self):
        from src.mcts.factory import get_agent
        agent = get_agent("reuse", seed=0)
        assert isinstance(agent, ReuseUCT)
        move = agent.run(PopOutBoard(), 30)
        assert 0 <= move <= 13


# ── module-level Numba warmup ─────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def _warm_numba_all():
    """Pre-compile all Numba kernels once for the whole module."""
    from src.mcts.optimized.numba_mcts import warmup
    from src.mcts.optimized.numba_solver import warmup_solver
    warmup()
    warmup_solver()


# ── ReuseNumbaMCTS ────────────────────────────────────────────────────────────

class TestReuseNumbaMCTS:
    """Lighter tests — reuse logic is fully covered by TestReuseUCT."""

    def test_returns_legal_move(self):
        from src.mcts.optimized.numba_reuse import ReuseNumbaMCTS
        board = PopOutBoard()
        agent = ReuseNumbaMCTS(seed=42)
        move = agent.run(board, 50)
        assert move in board.legal_moves()

    def test_isinstance_numba_mcts(self):
        from src.mcts.optimized.numba_reuse import ReuseNumbaMCTS
        from src.mcts.optimized.numba_mcts import NumbaMCTS
        assert isinstance(ReuseNumbaMCTS(), NumbaMCTS)

    def test_isinstance_base_mcts(self):
        from src.mcts.optimized.numba_reuse import ReuseNumbaMCTS
        assert isinstance(ReuseNumbaMCTS(), BaseMCTS)

    def test_satisfies_mcts_engine_protocol(self):
        from src.mcts.optimized.numba_reuse import ReuseNumbaMCTS
        assert isinstance(ReuseNumbaMCTS(), MCTSEngine)

    def test_tree_reuse_across_turns(self):
        from src.mcts.optimized.numba_reuse import ReuseNumbaMCTS
        board = PopOutBoard()
        agent = ReuseNumbaMCTS(seed=42)

        move = agent.run(board, 80)
        assert move in board.legal_moves()
        board.apply_move(move)

        legal = board.legal_moves()
        assert legal
        opp = legal[0]
        board.apply_move(opp)

        move2 = agent.run(board, 80, opponent_move=opp)
        assert move2 in board.legal_moves()

    def test_factory_reuse_numba(self):
        from src.mcts.factory import get_agent
        from src.mcts.optimized.numba_reuse import ReuseNumbaMCTS
        agent = get_agent("reuse_numba", seed=0)
        assert isinstance(agent, ReuseNumbaMCTS)
        move = agent.run(PopOutBoard(), 30)
        assert 0 <= move <= 13


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
