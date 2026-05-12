"""Regras do jogo PopOut: vitória, empate e repetição de estados."""

from __future__ import annotations
from collections import Counter
from typing import Iterable, List, Tuple

from src.engine.standard.bitboard import PopOutBoard, ROWS, COLS
from src.config import COL_SIZE


def has_won(bitmask: int) -> bool:
    """Checks for 4-in-a-row using bitwise shifts (Connect 4 logic)."""
    # Vertical
    m = bitmask & (bitmask >> 1)
    if m & (m >> 2): return True
    # Horizontal
    m = bitmask & (bitmask >> 7)
    if m & (m >> 14): return True
    # Diagonal 1 (\)
    m = bitmask & (bitmask >> 6)
    if m & (m >> 12): return True
    # Diagonal 2 (/)
    m = bitmask & (bitmask >> 8)
    if m & (m >> 16): return True
    return False


def find_winning_cells(mask: int) -> list[tuple[int, int]]:
    """Encontra as 4 células que formam a linha vencedora.

    Usa os mesmos deslocamentos de has_won() para identificar a direção
    e posição inicial da sequência de 4 peças.  Retorna coordenadas no
    sistema do bitboard (row=0 é a linha inferior, col=0 é a coluna da
    esquerda), compatível com o loop de renderização do renderer.

    Args:
        mask (int): Bitmask do jogador vencedor.

    Returns:
        list[tuple[int, int]]: Lista de (row, col) das 4 células vencedoras,
                               onde row=0 é a linha inferior do tabuleiro.
                               Lista vazia se não houver linha vencedora.
    """
    for shift in (1, 7, 6, 8):   # vertical, horizontal, diag\, diag/
        m = mask & (mask >> shift)
        m2 = m & (m >> (2 * shift))
        if m2:
            # lowest-set-bit trick: m2 & -m2 isolates the lowest bit
            start = (m2 & -m2).bit_length() - 1
            return [
                ((start + i * shift) % 7, (start + i * shift) // 7)
                for i in range(4)
            ]
    return []

def check_winner_for_player(board: PopOutBoard, player: int) -> bool:
    """Verifica se o jogador especificado venceu."""
    mask = board.mask_p1 if player == 1 else board.mask_p2
    return has_won(mask)

def evaluate_after_move(board: PopOutBoard, mover: int) -> int:
    """
    Resolve resultado após uma jogada.
    Regra PopOut: Se ambos têm 4-em-linha, o 'mover' (quem jogou) vence.
    """
    p1_wins = has_won(board.mask_p1)
    p2_wins = has_won(board.mask_p2)

    if p1_wins and p2_wins:
        return mover
    if p1_wins: return 1
    if p2_wins: return 2
    return 0

def board_signature(board: PopOutBoard) -> Tuple[int, int]:
    """Assinatura única imutável para o estado do tabuleiro."""
    return (board.mask_p1, board.mask_p2)

def is_threefold_repetition(history_signatures: Iterable[Tuple[int, int]]) -> bool:
    """Verifica empate por repetição tripla de estado."""
    if not history_signatures: return False
    counts = Counter(history_signatures)
    return any(v >= 3 for v in counts.values())

def is_draw(board: PopOutBoard, history_signatures: List[Tuple[int, int]]) -> bool:
    """Verifica condições de empate (cheio ou repetição)."""
    return board.is_full() or is_threefold_repetition(history_signatures)


# ── Features de alto nível para o ID3 ────────────────────────────────────────

# Máscara das colunas centrais (2, 3, 4) — pré-computada uma vez
_CENTER_MASK: int = 0
for _c in (2, 3, 4):
    for _r in range(ROWS):
        _CENTER_MASK |= (1 << (_c * COL_SIZE + _r))


def _count_threats(mask: int) -> int:
    """Conta sequências de 3 peças consecutivas (ameaças de vitória).

    Usa os mesmos deslocamentos de bits que has_won(), mas para comprimento 3.

    Args:
        mask (int): Bitmask do jogador.

    Returns:
        int: Número de sequências de 3-em-linha encontradas.
    """
    h  = mask & (mask >> 7)
    v  = mask & (mask >> 1)
    d1 = mask & (mask >> 6)
    d2 = mask & (mask >> 8)
    return (bin(h  & (mask >> 14)).count('1') +
            bin(v  & (mask >>  2)).count('1') +
            bin(d1 & (mask >> 12)).count('1') +
            bin(d2 & (mask >> 16)).count('1'))


def _can_win_in_one(board: PopOutBoard, player: int) -> bool:
    """Verifica se o jogador consegue vencer com uma única jogada.

    Args:
        board (PopOutBoard): Estado atual do jogo.
        player (int): Jogador a verificar (1 ou 2).

    Returns:
        bool: True se existe pelo menos uma jogada vencedora imediata.
    """
    b = board.clone()
    b.current_player = player
    for move in b.legal_moves():
        c = b.clone()
        c.apply_move(move)
        new_mask = c.mask_p1 if player == 1 else c.mask_p2
        if has_won(new_mask):
            return True
    return False


def extended_features(board: PopOutBoard) -> dict:
    """Dicionário de features com células + features táticas de alto nível.

    Estende to_feature_dict() com 7 features adicionais, todas discretas:

    * threats_me    — ameaças de 3-em-linha do jogador atual (0 a 3, saturado)
    * threats_opp   — ameaças de 3-em-linha do adversário (0 a 3, saturado)
    * center_me     — peças do jogador atual nas colunas centrais 2-4 (0-6)
    * center_opp    — peças do adversário nas colunas centrais 2-4 (0-6)
    * phase         — fase do jogo: 0=abertura (≤8 peças), 1=meio, 2=fim (>24)
    * can_win       — 1 se o jogador atual vence em 1 jogada, 0 caso contrário
    * opp_wins_next — 1 se o adversário vence na sua próxima jogada, 0 c.c.

    Args:
        board (PopOutBoard): Estado atual do jogo.

    Returns:
        dict: Feature dict completo (células + features táticas).
    """
    features = board.to_feature_dict()

    me  = board.current_player
    opp = 3 - me
    my_mask  = board.mask_p1 if me == 1 else board.mask_p2
    opp_mask = board.mask_p2 if me == 1 else board.mask_p1

    # Ameaças (saturadas em 3 para limitar categorias)
    features["threats_me"]  = min(_count_threats(my_mask),  3)
    features["threats_opp"] = min(_count_threats(opp_mask), 3)

    # Controlo do centro
    features["center_me"]  = bin(my_mask  & _CENTER_MASK).count('1')
    features["center_opp"] = bin(opp_mask & _CENTER_MASK).count('1')

    # Fase do jogo
    total = bin(board.merged_mask).count('1')
    features["phase"] = 0 if total <= 8 else (1 if total <= 24 else 2)

    # Flags táticas
    features["can_win"]       = int(_can_win_in_one(board, me))
    features["opp_wins_next"] = int(_can_win_in_one(board, opp))

    return features
