"""Tests for SolverNode and SolverMCTS (src/algorithms/mcts/standard/uct_solver.py)."""

import pytest

from src.engine.standard.bitboard import PopOutBoard
from src.algorithms.mcts.standard.base import BaseMCTS
from src.algorithms.mcts.standard.uct_solver import (
    SolverMCTS,
    SolverNode,
    STATUS_UNKNOWN,
    STATUS_WIN,
    STATUS_LOSS,
    STATUS_DRAW,
)
from src.algorithms.mcts.protocol import MCTSEngine
from src.algorithms.factory import get_agent


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_solver_node(board=None, **kwargs) -> SolverNode:
    if board is None:
        board = PopOutBoard()
    return SolverNode(state=board, **kwargs)


def _three_in_a_row_board() -> PopOutBoard:
    """P1 has 3 in a row (cols 0-2, row 0). P1 to move. Drop col 3 wins."""
    b = PopOutBoard()
    b.apply_move(0)  # P1 col 0
    b.apply_move(6)  # P2 col 6 (non-blocking)
    b.apply_move(1)  # P1 col 1
    b.apply_move(6)  # P2 col 6
    b.apply_move(2)  # P1 col 2
    b.apply_move(6)  # P2 col 6 (P1 to move next)
    return b


# ── SolverNode ────────────────────────────────────────────────────────────────

class TestSolverNode:
    def test_default_status_unknown(self):
        node = _make_solver_node()
        assert node.status == STATUS_UNKNOWN

    def test_default_distance_zero(self):
        node = _make_solver_node()
        assert node.distance == 0

    def test_untried_moves_populated(self):
        node = _make_solver_node()
        assert len(node.untried_moves) == 7  # empty board: 7 drops only

    def test_terminal_node_sets_loss_status(self):
        """A node reached after the opponent won is a forced loss for current player."""
        node = _make_solver_node(terminal_winner=1)
        assert node.status == STATUS_LOSS
        assert node.distance == 0
        assert node.is_terminal is True

    def test_non_terminal_is_not_terminal(self):
        node = _make_solver_node()
        assert node.is_terminal is False

    def test_q_zero_when_unvisited(self):
        node = _make_solver_node()
        assert node.q() == 0.0

    def test_q_after_updates(self):
        node = _make_solver_node()
        node.visits = 4
        node.value_sum = 3.0
        assert node.q() == pytest.approx(0.75)


# ── SolverMCTS.expand ─────────────────────────────────────────────────────────

class TestSolverMCTSExpand:
    def test_creates_solver_node_child(self):
        solver = SolverMCTS(seed=0)
        root = SolverNode(state=PopOutBoard())
        child = solver.expand(root)
        assert child is not root
        assert isinstance(child, SolverNode)
        assert child.parent is root

    def test_terminal_node_returns_self(self):
        solver = SolverMCTS(seed=0)
        node = SolverNode(state=PopOutBoard(), terminal_winner=1)
        node.untried_moves = []
        result = solver.expand(node)
        assert result is node

    def test_no_untried_moves_returns_self(self):
        solver = SolverMCTS(seed=0)
        node = SolverNode(state=PopOutBoard())
        node.untried_moves = []
        result = solver.expand(node)
        assert result is node


# ── SolverMCTS._update_node_status ───────────────────────────────────────────

class TestUpdateNodeStatus:
    def _parent_with_children(self, child_statuses, child_distances):
        """Build a parent SolverNode with pre-configured children."""
        solver = SolverMCTS(seed=0)
        board = PopOutBoard()
        parent = SolverNode(state=board)
        parent.untried_moves = []  # mark as fully explored

        for i, (status, dist) in enumerate(zip(child_statuses, child_distances)):
            child = SolverNode(state=board.clone(), parent=parent, move_from_parent=i)
            child.status = status
            child.distance = dist
            parent.children[i] = child

        return solver, parent

    def test_win_when_any_child_is_loss(self):
        solver, parent = self._parent_with_children(
            [STATUS_LOSS, STATUS_UNKNOWN], [2, 0]
        )
        changed = solver._update_node_status(parent)
        assert changed is True
        assert parent.status == STATUS_WIN
        assert parent.distance == 3  # min(2) + 1

    def test_win_picks_min_distance_loss_child(self):
        solver, parent = self._parent_with_children(
            [STATUS_LOSS, STATUS_LOSS], [5, 1]
        )
        solver._update_node_status(parent)
        assert parent.distance == 2  # min(5, 1) + 1

    def test_loss_when_all_children_win(self):
        solver, parent = self._parent_with_children(
            [STATUS_WIN, STATUS_WIN], [3, 7]
        )
        solver._update_node_status(parent)
        assert parent.status == STATUS_LOSS
        assert parent.distance == 8  # max(3, 7) + 1

    def test_draw_when_draw_children_present(self):
        solver, parent = self._parent_with_children(
            [STATUS_WIN, STATUS_DRAW], [3, 0]
        )
        solver._update_node_status(parent)
        assert parent.status == STATUS_DRAW

    def test_no_change_when_untried_moves_remain(self):
        """Cannot conclude Loss/Draw while unexplored moves exist."""
        solver = SolverMCTS(seed=0)
        board = PopOutBoard()
        parent = SolverNode(state=board)
        # Leave untried_moves populated (default from __post_init__)

        child = SolverNode(state=board.clone(), parent=parent, move_from_parent=0)
        child.status = STATUS_WIN
        child.distance = 2
        parent.children[0] = child

        changed = solver._update_node_status(parent)
        assert changed is False
        assert parent.status == STATUS_UNKNOWN

    def test_no_change_when_already_current(self):
        solver, parent = self._parent_with_children(
            [STATUS_LOSS], [2]
        )
        solver._update_node_status(parent)
        # Call again — should report no change
        changed = solver._update_node_status(parent)
        assert changed is False


# ── SolverMCTS.best_child ─────────────────────────────────────────────────────

class TestBestChild:
    def _root_with_children(self, statuses, distances, visits=None):
        solver = SolverMCTS(seed=0)
        root = SolverNode(state=PopOutBoard())
        root.untried_moves = []
        root.visits = 10

        for i, (st, d) in enumerate(zip(statuses, distances)):
            child = SolverNode(state=PopOutBoard(), parent=root, move_from_parent=i)
            child.status = st
            child.distance = d
            child.visits = (visits[i] if visits else 5)
            child.value_sum = child.visits * 0.5
            root.children[i] = child

        return solver, root

    def test_prefers_loss_child_over_unknown(self):
        solver, root = self._root_with_children(
            [STATUS_UNKNOWN, STATUS_LOSS], [0, 3]
        )
        chosen = solver.best_child(root)
        assert chosen.status == STATUS_LOSS

    def test_picks_min_distance_among_loss_children(self):
        solver, root = self._root_with_children(
            [STATUS_LOSS, STATUS_LOSS], [5, 2]
        )
        chosen = solver.best_child(root)
        assert chosen.distance == 2

    def test_falls_back_to_ucb1_for_unknown(self):
        solver, root = self._root_with_children(
            [STATUS_UNKNOWN, STATUS_UNKNOWN], [0, 0], visits=[1, 100]
        )
        # Low-visit child should score higher via UCB1
        chosen = solver.best_child(root)
        assert chosen.visits == 1


# ── SolverMCTS.get_best_move ──────────────────────────────────────────────────

class TestGetBestMove:
    def test_prefers_proven_win_over_visits(self):
        solver = SolverMCTS(seed=0)
        root = SolverNode(state=PopOutBoard())
        root.untried_moves = []

        # Move 0: very visited but unknown
        c0 = SolverNode(state=PopOutBoard(), parent=root, move_from_parent=0)
        c0.visits = 1000
        c0.value_sum = 500.0
        c0.status = STATUS_UNKNOWN

        # Move 1: barely visited but proven win (d=2)
        c1 = SolverNode(state=PopOutBoard(), parent=root, move_from_parent=1)
        c1.visits = 5
        c1.value_sum = 4.0
        c1.status = STATUS_LOSS  # opponent is in forced loss → this is our win
        c1.distance = 2

        root.children[0] = c0
        root.children[1] = c1

        assert solver.get_best_move(root) == 1

    def test_picks_fastest_proven_win(self):
        solver = SolverMCTS(seed=0)
        root = SolverNode(state=PopOutBoard())
        root.untried_moves = []

        for move, dist in [(0, 5), (1, 1), (2, 3)]:
            c = SolverNode(state=PopOutBoard(), parent=root, move_from_parent=move)
            c.status = STATUS_LOSS
            c.distance = dist
            root.children[move] = c

        assert solver.get_best_move(root) == 1  # distance 1 is fastest

    def test_falls_back_to_most_visited(self):
        solver = SolverMCTS(seed=0)
        root = SolverNode(state=PopOutBoard())
        root.untried_moves = []

        for move, visits in [(0, 50), (1, 200), (2, 10)]:
            c = SolverNode(state=PopOutBoard(), parent=root, move_from_parent=move)
            c.visits = visits
            c.status = STATUS_UNKNOWN
            root.children[move] = c

        assert solver.get_best_move(root) == 1  # most visited


# ── SolverMCTS.run ────────────────────────────────────────────────────────────

class TestSolverMCTSRun:
    def test_returns_legal_move_initial(self):
        solver = SolverMCTS(seed=42)
        b = PopOutBoard()
        move = solver.run(b, iterations=50)
        assert move in b.legal_moves()

    def test_returns_legal_move_mid_game(self):
        solver = SolverMCTS(seed=7)
        b = PopOutBoard()
        b.apply_move(3)
        b.apply_move(4)
        move = solver.run(b, iterations=50)
        assert move in b.legal_moves()

    def test_deterministic_with_same_seed(self):
        b = PopOutBoard()
        m1 = SolverMCTS(seed=11).run(b, iterations=100)
        m2 = SolverMCTS(seed=11).run(b, iterations=100)
        assert m1 == m2

    def test_wins_immediately_with_three_in_a_row(self):
        """Solver should play col 3 to complete 4-in-a-row."""
        b = _three_in_a_row_board()
        solver = SolverMCTS(seed=0)
        move = solver.run(b, iterations=200)
        assert move == 3

    def test_raises_on_no_legal_moves(self):
        import unittest.mock as mock
        b = PopOutBoard()
        # Patch legal_moves on the class so the clone also returns no moves
        with mock.patch.object(PopOutBoard, "legal_moves", return_value=[]):
            with pytest.raises(ValueError):
                SolverMCTS(seed=0).run(b, iterations=1)

    def test_inherits_base_mcts(self):
        assert isinstance(SolverMCTS(), BaseMCTS)

    def test_satisfies_mcts_engine_protocol(self):
        assert isinstance(SolverMCTS(), MCTSEngine)


# ── Factory integration ───────────────────────────────────────────────────────

class TestFactory:
    def test_get_agent_solver(self):
        agent = get_agent("solver", seed=0)
        assert isinstance(agent, SolverMCTS)

    def test_get_agent_solver_kwargs(self):
        agent = get_agent("solver", exploration_c=2.0, rollout_depth=10, seed=1)
        assert agent.exploration_c == 2.0
        assert agent.rollout_depth == 10

    def test_solver_run_via_factory(self):
        agent = get_agent("solver", seed=42)
        b = PopOutBoard()
        move = agent.run(b, iterations=30)
        assert move in b.legal_moves()


# ── SolverMCTS vs FlatNumbaMCTS competition ───────────────────────────────────

class TestSolverVsFlatNumba:
    """SolverMCTS should win at least 1 game out of 5 against FlatNumbaMCTS
    with both agents using 10 000 iterations — a fair head-to-head."""

    def test_wins_at_least_one_game(self):
        from src.algorithms.mcts.optimized.numba_mcts import warmup
        from src.interfaces.cli import run_cvc

        warmup()  # JIT-compile numba kernels before timing starts

        ITERS = 10_000
        N_GAMES = 5
        solver_wins = 0

        for i in range(N_GAMES):
            solver = get_agent("solver", seed=i * 7)
            flat = get_agent("flat_numba", seed=i * 7 + 1)

            if i % 2 == 0:
                # Solver plays as Player 1 (X)
                result = run_cvc(
                    agent1=solver, agent2=flat,
                    agent1_name="SolverMCTS", agent2_name="FlatNumba",
                    iterations1=ITERS, iterations2=ITERS,
                    verbose=False,
                )
                if result == 1:
                    solver_wins += 1
            else:
                # Solver plays as Player 2 (O) — alternate colours for fairness
                result = run_cvc(
                    agent1=flat, agent2=solver,
                    agent1_name="FlatNumba", agent2_name="SolverMCTS",
                    iterations1=ITERS, iterations2=ITERS,
                    verbose=False,
                )
                if result == 2:
                    solver_wins += 1

        assert solver_wins >= 1, (
            f"SolverMCTS won {solver_wins}/{N_GAMES} games — expected at least 1"
        )
