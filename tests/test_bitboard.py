import pytest
from src.engine.bitboard import COLS, ROWS, COL_SIZE, PopOutBoard

# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_cell(board: PopOutBoard, row: int, col: int) -> int:
    """
    Lê célula: row 0 é o fundo, row 5 é o topo.
    """
    bit_index = (col * COL_SIZE) + row
    bit = 1 << bit_index
    if board.mask_p1 & bit: return 1
    if board.mask_p2 & bit: return 2
    return 0

# ── Drop (Moves 0-6) ─────────────────────────────────────────────────────────

class TestDrop:
    def test_drop_empty_column(self):
        b = PopOutBoard()
        b.current_player = 1
        b.apply_move(3) # Drop na coluna 3
        assert _get_cell(b, 0, 3) == 1  # Cai para o fundo (row 0)

    def test_drop_stacks_pieces(self):
        b = PopOutBoard()
        b.current_player = 1
        b.apply_move(0) # P1 drop col 0
        b.apply_move(0) # P2 drop col 0
        assert _get_cell(b, 0, 0) == 1 # Fundo
        assert _get_cell(b, 1, 0) == 2 # Acima

    def test_is_full_detection(self):
        b = PopOutBoard()
        for c in range(COLS):
            for _ in range(ROWS):
                b.apply_move(c)
        # Quando todas as colunas têm 6 peças, a linha de topo está cheia
        # is_full() verifica se drops não são possíveis (linha de topo completa)
        # Mas ainda há pops disponíveis!
        assert b.is_full() is True
        # Mesmo assim, pops ainda são legais
        legal = b.legal_moves()
        assert all(move >= 7 for move in legal)  # Apenas pops (7-13)

# ── Pop (Moves 7-13) ────────────────────────────────────────────────────────

class TestPop:
    def test_pop_removes_bottom(self):
        b = PopOutBoard()
        b.apply_move(2) # P1 drop col 2 (Move 2)
        assert _get_cell(b, 0, 2) == 1
        
        b.current_player = 1 # Garante que P1 é quem faz o pop
        b.apply_move(2 + 7) # P1 pop col 2 (Move 9)
        assert _get_cell(b, 0, 2) == 0

    def test_gravity_after_pop(self):
        b = PopOutBoard()
        b.apply_move(3) # P1 (fundo)
        b.apply_move(3) # P2 (meio)
        b.apply_move(3) # P1 (topo)
        
        # Pop na coluna 3 (Move 10)
        # Como o P1 está na base, ele pode dar pop
        b.current_player = 1 
        b.apply_move(10)
        
        # O P2 que estava no meio (row 1) desce para o fundo (row 0)
        assert _get_cell(b, 0, 3) == 2
        assert _get_cell(b, 1, 3) == 1
        assert _get_cell(b, 2, 3) == 0

# ── Legal Moves ──────────────────────────────────────────────────────────────

class TestLegalMoves:
    def test_legal_moves_initial(self):
        b = PopOutBoard()
        # Apenas drops (0-6) são possíveis num tabuleiro vazio
        assert b.legal_moves() == [0, 1, 2, 3, 4, 5, 6]

    def test_legal_moves_pop_inclusion(self):
        b = PopOutBoard()
        b.apply_move(0) # P1 drop col 0
        # Agora P2 joga. P2 não tem peças na base, então só pode dar drop.
        assert 7 not in b.legal_moves()
        
        b.apply_move(1) # P2 drop col 1
        # Agora P1 joga. P1 tem peça na base da col 0.
        assert 7 in b.legal_moves() # Pop col 0 disponível