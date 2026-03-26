"""Testes para src/engine/bitboard.py — PopOutBoard."""

import pytest

from src.engine.bitboard import COLS, ROWS, PopOutBoard


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fill_column(board: PopOutBoard, col: int, player: int, count: int) -> None:
    """Preenche *count* células de uma coluna com peças do jogador."""
    for _ in range(count):
        board.apply_drop(col, player=player)


# ── Drop ─────────────────────────────────────────────────────────────────────

class TestDrop:
    def test_drop_empty_column(self):
        b = PopOutBoard()
        assert b.apply_drop(3, player=1) is True
        assert b.board[ROWS - 1][3] == 1  # peça cai para o fundo

    def test_drop_stacks_pieces(self):
        b = PopOutBoard()
        b.apply_drop(0, player=1)
        b.apply_drop(0, player=2)
        assert b.board[ROWS - 1][0] == 1
        assert b.board[ROWS - 2][0] == 2

    def test_drop_full_column_returns_false(self):
        b = PopOutBoard()
        for i in range(ROWS):
            assert b.apply_drop(0, player=(i % 2) + 1) is True
        assert b.apply_drop(0, player=1) is False  # coluna cheia

    def test_drop_invalid_column_negative(self):
        b = PopOutBoard()
        assert b.apply_drop(-1, player=1) is False

    def test_drop_invalid_column_too_large(self):
        b = PopOutBoard()
        assert b.apply_drop(COLS, player=1) is False


# ── Pop ──────────────────────────────────────────────────────────────────────

class TestPop:
    def test_pop_own_piece(self):
        b = PopOutBoard()
        b.apply_drop(2, player=1)
        assert b.board[ROWS - 1][2] == 1
        assert b.apply_pop(2, player=1) is True
        assert b.board[ROWS - 1][2] == 0  # peça removida

    def test_pop_opponent_piece_returns_false(self):
        b = PopOutBoard()
        b.apply_drop(2, player=1)
        assert b.apply_pop(2, player=2) is False

    def test_pop_empty_column_returns_false(self):
        b = PopOutBoard()
        assert b.apply_pop(0, player=1) is False

    def test_pop_invalid_column(self):
        b = PopOutBoard()
        assert b.apply_pop(-1, player=1) is False
        assert b.apply_pop(COLS, player=1) is False

    def test_gravity_after_pop(self):
        """Após pop, todas as peças acima descem uma posição."""
        b = PopOutBoard()
        # Empilhar: fundo=1, cima=2, cima=1
        b.apply_drop(3, player=1)
        b.apply_drop(3, player=2)
        b.apply_drop(3, player=1)
        # Estado antes do pop: row5=1, row4=2, row3=1
        assert b.board[ROWS - 1][3] == 1
        assert b.board[ROWS - 2][3] == 2
        assert b.board[ROWS - 3][3] == 1

        b.apply_pop(3, player=1)
        # Após pop: row5=2, row4=1, row3=0
        assert b.board[ROWS - 1][3] == 2
        assert b.board[ROWS - 2][3] == 1
        assert b.board[ROWS - 3][3] == 0


# ── Legal Moves ──────────────────────────────────────────────────────────────

class TestLegalMoves:
    def test_legal_drop_moves_initial(self):
        b = PopOutBoard()
        assert b.legal_drop_moves() == list(range(COLS))

    def test_legal_drop_moves_full_column_excluded(self):
        b = PopOutBoard()
        _fill_column(b, 0, player=1, count=ROWS)
        drops = b.legal_drop_moves()
        assert 0 not in drops
        assert len(drops) == COLS - 1

    def test_legal_pop_moves_empty_board(self):
        b = PopOutBoard()
        assert b.legal_pop_moves(player=1) == []

    def test_legal_pop_moves_with_own_pieces(self):
        b = PopOutBoard()
        b.apply_drop(4, player=1)
        assert 4 in b.legal_pop_moves(player=1)
        assert 4 not in b.legal_pop_moves(player=2)

    def test_legal_moves_combines_drop_and_pop(self):
        b = PopOutBoard()
        b.apply_drop(0, player=1)
        b.current_player = 1
        moves = b.legal_moves(player=1)
        drop_moves = [m for m in moves if m[0] == "drop"]
        pop_moves = [m for m in moves if m[0] == "pop"]
        assert len(drop_moves) == COLS  # todas as colunas livres no topo
        assert ("pop", 0) in pop_moves

    def test_legal_moves_uses_current_player_default(self):
        b = PopOutBoard()
        b.apply_drop(5, player=1)
        b.current_player = 1
        moves = b.legal_moves()  # sem argumento
        assert ("pop", 5) in moves


# ── apply_move ───────────────────────────────────────────────────────────────

class TestApplyMove:
    def test_apply_move_drop_switches_player(self):
        b = PopOutBoard()
        b.current_player = 1
        assert b.apply_move(("drop", 3), switch_player=True) is True
        assert b.current_player == 2

    def test_apply_move_pop_switches_player(self):
        b = PopOutBoard()
        b.apply_drop(0, player=1)
        b.current_player = 1
        assert b.apply_move(("pop", 0), switch_player=True) is True
        assert b.current_player == 2

    def test_apply_move_no_switch(self):
        b = PopOutBoard()
        b.current_player = 1
        b.apply_move(("drop", 0), switch_player=False)
        assert b.current_player == 1

    def test_apply_move_invalid_returns_false(self):
        b = PopOutBoard()
        b.current_player = 2
        # pop numa coluna vazia
        assert b.apply_move(("pop", 0), switch_player=True) is False
        assert b.current_player == 2  # não muda


# ── Clone ────────────────────────────────────────────────────────────────────

class TestClone:
    def test_clone_is_independent(self):
        b = PopOutBoard()
        b.apply_drop(0, player=1)
        c = b.clone()
        c.apply_drop(0, player=2)
        # Original não deve ser afetado
        assert b.board[ROWS - 2][0] == 0
        assert c.board[ROWS - 2][0] == 2

    def test_clone_preserves_state(self):
        b = PopOutBoard()
        b.apply_drop(3, player=1)
        b.current_player = 2
        c = b.clone()
        assert c.board == b.board
        assert c.current_player == 2


# ── is_full ──────────────────────────────────────────────────────────────────

class TestIsFull:
    def test_empty_board_not_full(self):
        b = PopOutBoard()
        assert b.is_full() is False

    def test_full_board(self):
        b = PopOutBoard()
        for c in range(COLS):
            for r in range(ROWS):
                b.apply_drop(c, player=(r % 2) + 1)
        assert b.is_full() is True


# ── to_feature_dict ──────────────────────────────────────────────────────────

class TestFeatureDict:
    def test_feature_dict_keys(self):
        b = PopOutBoard()
        d = b.to_feature_dict()
        assert "current_player" in d
        assert "cell_0_0" in d
        assert f"cell_{ROWS-1}_{COLS-1}" in d
        assert len(d) == ROWS * COLS + 1

    def test_feature_dict_values_empty(self):
        b = PopOutBoard()
        d = b.to_feature_dict()
        for r in range(ROWS):
            for c in range(COLS):
                assert d[f"cell_{r}_{c}"] == 0


# ── __str__ ──────────────────────────────────────────────────────────────────

class TestStr:
    def test_str_contains_column_numbers(self):
        b = PopOutBoard()
        s = str(b)
        assert "0 1 2 3 4 5 6" in s

    def test_str_shows_pieces(self):
        b = PopOutBoard()
        b.apply_drop(0, player=1)
        s = str(b)
        assert "X" in s
