"""Testes para src/engine/rules.py — vitória, conflito, empate, repetição."""

import pytest

from src.engine.bitboard import COLS, ROWS, PopOutBoard
from src.engine.rules import (
    board_signature,
    check_winner_for_player,
    evaluate_after_move,
    is_draw,
    is_threefold_repetition,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _set_cell(board: PopOutBoard, row: int, col: int, val: int) -> None:
    """Define diretamente uma célula (atalho para testes)."""
    board.board[row][col] = val


def _make_board_with(cells: list[tuple[int, int, int]]) -> PopOutBoard:
    """Cria tabuleiro com células específicas preenchidas: [(row, col, player), ...]."""
    b = PopOutBoard()
    for r, c, p in cells:
        _set_cell(b, r, c, p)
    return b


# ── Horizontal Win ───────────────────────────────────────────────────────────

class TestHorizontalWin:
    def test_horizontal_bottom_row(self):
        b = _make_board_with([(ROWS - 1, c, 1) for c in range(4)])
        assert check_winner_for_player(b, 1) is True
        assert check_winner_for_player(b, 2) is False

    def test_horizontal_middle_row(self):
        b = _make_board_with([(3, c, 2) for c in range(2, 6)])
        assert check_winner_for_player(b, 2) is True

    def test_horizontal_right_edge(self):
        b = _make_board_with([(0, c, 1) for c in range(3, 7)])
        assert check_winner_for_player(b, 1) is True

    def test_three_in_a_row_not_win(self):
        b = _make_board_with([(ROWS - 1, c, 1) for c in range(3)])
        assert check_winner_for_player(b, 1) is False


# ── Vertical Win ─────────────────────────────────────────────────────────────

class TestVerticalWin:
    def test_vertical_bottom(self):
        b = _make_board_with([(ROWS - 1 - i, 0, 1) for i in range(4)])
        assert check_winner_for_player(b, 1) is True

    def test_vertical_top(self):
        b = _make_board_with([(i, 3, 2) for i in range(4)])
        assert check_winner_for_player(b, 2) is True

    def test_three_vertical_not_win(self):
        b = _make_board_with([(ROWS - 1 - i, 2, 1) for i in range(3)])
        assert check_winner_for_player(b, 1) is False


# ── Diagonal Win ─────────────────────────────────────────────────────────────

class TestDiagonalWin:
    def test_diagonal_down_right(self):
        """Diagonal ↘ (r+k, c+k)."""
        b = _make_board_with([(2 + k, 1 + k, 1) for k in range(4)])
        assert check_winner_for_player(b, 1) is True

    def test_diagonal_down_left(self):
        """Diagonal ↙ (r+k, c-k)."""
        b = _make_board_with([(1 + k, 5 - k, 2) for k in range(4)])
        assert check_winner_for_player(b, 2) is True

    def test_diagonal_at_corner(self):
        """Diagonal começando no canto superior esquerdo."""
        b = _make_board_with([(k, k, 1) for k in range(4)])
        assert check_winner_for_player(b, 1) is True

    def test_three_diagonal_not_win(self):
        b = _make_board_with([(k, k, 1) for k in range(3)])
        assert check_winner_for_player(b, 1) is False


# ── Conflict Rule ────────────────────────────────────────────────────────────

class TestConflictRule:
    def test_both_win_mover_wins(self):
        """Se ambos têm 4-em-linha, quem jogou (mover) vence."""
        b = PopOutBoard()
        # Jogador 1: horizontal na linha de fundo
        for c in range(4):
            _set_cell(b, ROWS - 1, c, 1)
        # Jogador 2: horizontal na penúltima linha
        for c in range(4):
            _set_cell(b, ROWS - 2, c, 2)

        assert check_winner_for_player(b, 1) is True
        assert check_winner_for_player(b, 2) is True
        # Mover = 1 → jogador 1 vence
        assert evaluate_after_move(b, mover=1) == 1
        # Mover = 2 → jogador 2 vence
        assert evaluate_after_move(b, mover=2) == 2

    def test_only_player1_wins(self):
        b = _make_board_with([(ROWS - 1, c, 1) for c in range(4)])
        assert evaluate_after_move(b, mover=1) == 1
        assert evaluate_after_move(b, mover=2) == 1  # P1 vence independentemente

    def test_no_winner(self):
        b = PopOutBoard()
        assert evaluate_after_move(b, mover=1) == 0


# ── Board Signature ──────────────────────────────────────────────────────────

class TestBoardSignature:
    def test_signature_is_hashable(self):
        b = PopOutBoard()
        sig = board_signature(b)
        assert isinstance(sig, tuple)
        # Deve ser usável como chave de dicionário
        d = {sig: True}
        assert d[sig] is True

    def test_different_states_different_signatures(self):
        b1 = PopOutBoard()
        b2 = PopOutBoard()
        b2.apply_drop(0, player=1)
        assert board_signature(b1) != board_signature(b2)

    def test_same_state_same_signature(self):
        b1 = PopOutBoard()
        b1.apply_drop(3, player=1)
        b2 = PopOutBoard()
        b2.apply_drop(3, player=1)
        assert board_signature(b1) == board_signature(b2)


# ── Threefold Repetition ────────────────────────────────────────────────────

class TestThreefoldRepetition:
    def test_no_repetition(self):
        sigs = [((0,),), ((1,),), ((2,),)]
        assert is_threefold_repetition(sigs) is False

    def test_exactly_three_repetitions(self):
        sig = ((0, 0), (0, 0))
        sigs = [sig, ((1,),), sig, ((2,),), sig]
        assert is_threefold_repetition(sigs) is True

    def test_two_repetitions_not_enough(self):
        sig = ((0,),)
        sigs = [sig, sig, ((1,),)]
        assert is_threefold_repetition(sigs) is False


# ── is_draw ──────────────────────────────────────────────────────────────────

class TestIsDraw:
    def test_draw_full_board(self):
        b = PopOutBoard()
        for c in range(COLS):
            for r in range(ROWS):
                b.apply_drop(c, player=(r % 2) + 1)
        assert is_draw(b, []) is True

    def test_draw_threefold(self):
        b = PopOutBoard()
        sig = board_signature(b)
        history = [sig, sig, sig]
        assert is_draw(b, history) is True

    def test_no_draw_normal_state(self):
        b = PopOutBoard()
        assert is_draw(b, []) is False
