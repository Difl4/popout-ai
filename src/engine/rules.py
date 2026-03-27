"""Regras do jogo PopOut: vitória, empate e repetição de estados."""

from __future__ import annotations
from collections import Counter
from typing import Iterable, List, Tuple
from .bitboard import PopOutBoard

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