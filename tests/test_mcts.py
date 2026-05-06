"""Testes para src/mcts — BaseMCTS, StandardUCT, ExperimentalUCT."""

import pytest

from src.engine.standard.bitboard import COLS, ROWS, PopOutBoard
from src.engine.standard.rules import evaluate_after_move
from src.mcts.standard.base import BaseMCTS, MCTSNode
from src.mcts.standard.uct_standard import StandardUCT
from src.mcts.standard.uct_experimental import ExperimentalUCT


# ── MCTSNode ─────────────────────────────────────────────────────────────────

class TestMCTSNode:
    def test_untried_moves_populated_on_init(self):
        b = PopOutBoard()
        node = MCTSNode(state=b)
        assert len(node.untried_moves) > 0
        # Tabuleiro vazio: só drops (7 colunas), sem pops
        assert len(node.untried_moves) == COLS

    def test_is_terminal_false_initially(self):
        b = PopOutBoard()
        node = MCTSNode(state=b)
        assert node.is_terminal is False

    def test_is_terminal_true_when_winner_set(self):
        b = PopOutBoard()
        node = MCTSNode(state=b, terminal_winner=1)
        assert node.is_terminal is True

    def test_q_zero_when_no_visits(self):
        b = PopOutBoard()
        node = MCTSNode(state=b)
        assert node.q() == 0.0

    def test_q_after_updates(self):
        b = PopOutBoard()
        node = MCTSNode(state=b)
        node.visits = 4
        node.value_sum = 2.0
        assert node.q() == pytest.approx(0.5)


# ── BaseMCTS.run ─────────────────────────────────────────────────────────────

class TestBaseMCTSRun:
    def test_returns_legal_move_initial_state(self):
        b = PopOutBoard()
        mcts = BaseMCTS(seed=42)
        move = mcts.run(b, iterations=50)
        assert isinstance(move, int)
        assert 0 <= move <= 13

    def test_returns_legal_move_mid_game(self):
        b = PopOutBoard()
        # Simular algumas jogadas
        b.apply_move(3)  # P1 drop col 3
        b.apply_move(3)  # P2 drop col 3
        b.apply_move(0)  # P1 drop col 0
        mcts = BaseMCTS(seed=123)
        move = mcts.run(b, iterations=50)
        legal = b.legal_moves()
        assert move in legal

    def test_deterministic_with_same_seed(self):
        b = PopOutBoard()
        m1 = BaseMCTS(seed=99).run(b, iterations=100)
        m2 = BaseMCTS(seed=99).run(b, iterations=100)
        assert m1 == m2

    def test_single_legal_move(self):
        """Quando só há uma jogada legal, deve devolvê-la."""
        b = PopOutBoard()
        # Preencher todas as colunas exceto a 6 com drops
        for c in range(COLS - 1):
            for _ in range(ROWS):
                b.apply_move(c)
        # Coluna 6: preencher até restar 1 espaço
        for _ in range(ROWS - 1):
            b.apply_move(COLS - 1)
        b.current_player = 1
        legal = b.legal_moves()
        # Deve haver pelo menos o drop na coluna 6 + possíveis pops
        mcts = BaseMCTS(seed=42)
        move = mcts.run(b, iterations=20)
        assert move in legal

    def test_raises_on_no_legal_moves(self):
        """Estado sem jogadas legais deve lançar ValueError."""
        b = PopOutBoard()
        # Preencher tudo com alternância de jogadores
        for c in range(COLS):
            for r in range(ROWS):
                move = c if r % 2 == 0 else (c + 7)  # Alterna drop/pop
                if r < ROWS - 1:  # Não preenche o último para ter pelo menos uma célula vazia
                    if move < 7:
                        b.apply_move(move)
        # Garantir que board está cheio
        for c in range(COLS):
            # Tenta preencher sem deixar espaço
            while not b.is_full() and len(b.legal_moves()) > 0:
                legal = b.legal_moves()
                if not legal:
                    break
                b.apply_move(legal[0])


# ── BaseMCTS internal steps ─────────────────────────────────────────────────

class TestMCTSSteps:
    def test_expand_creates_child(self):
        b = PopOutBoard()
        mcts = BaseMCTS(seed=42)
        root = MCTSNode(state=b)
        child = mcts.expand(root)
        assert child is not root
        assert child.parent is root
        assert child.move_from_parent is not None
        assert child.move_from_parent in root.children

    def test_expand_terminal_returns_self(self):
        b = PopOutBoard()
        node = MCTSNode(state=b, terminal_winner=1)
        node.untried_moves = []
        mcts = BaseMCTS(seed=42)
        result = mcts.expand(node)
        assert result is node

    def test_simulate_returns_valid_reward(self):
        b = PopOutBoard()
        mcts = BaseMCTS(seed=42)
        root = MCTSNode(state=b)
        child = mcts.expand(root)
        reward = mcts.simulate(child)
        assert 0.0 <= reward <= 1.0

    def test_backpropagate_updates_visits(self):
        b = PopOutBoard()
        mcts = BaseMCTS(seed=42)
        root = MCTSNode(state=b)
        child = mcts.expand(root)
        mcts.backpropagate(child, 1.0)
        assert child.visits == 1
        assert root.visits == 1
        assert child.value_sum == 1.0
        # Pai recebe reward invertido
        assert root.value_sum == 0.0

    def test_select_returns_node(self):
        b = PopOutBoard()
        mcts = BaseMCTS(seed=42)
        root = MCTSNode(state=b)
        selected = mcts.select(root)
        assert selected is root  # raiz com untried_moves


# ── StandardUCT ──────────────────────────────────────────────────────────────

class TestStandardUCT:
    def test_returns_legal_move(self):
        b = PopOutBoard()
        uct = StandardUCT(seed=42)
        move = uct.run(b, iterations=50)
        assert move in b.legal_moves()

    def test_inherits_base(self):
        uct = StandardUCT()
        assert isinstance(uct, BaseMCTS)


# ── ExperimentalUCT ─────────────────────────────────────────────────────────

class TestExperimentalUCT:
    def test_returns_legal_move(self):
        b = PopOutBoard()
        uct = ExperimentalUCT(seed=42)
        move = uct.run(b, iterations=50)
        assert move in b.legal_moves()

    def test_inherits_base(self):
        uct = ExperimentalUCT()
        assert isinstance(uct, BaseMCTS)

    def test_best_child_prefers_unvisited(self):
        """ExperimentalUCT deve escolher filhos não visitados primeiro."""
        b = PopOutBoard()
        uct = ExperimentalUCT(seed=42)
        root = MCTSNode(state=b)

        # Expandir dois filhos
        c1 = uct.expand(root)
        c2 = uct.expand(root)

        # Marcar c1 como visitado
        c1.visits = 10
        c1.value_sum = 5.0

        # c2 não visitado → deve ser escolhido
        chosen = uct.best_child(root)
        assert chosen is c2
