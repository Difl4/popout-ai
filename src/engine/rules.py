"""Regras do jogo PopOut: vitória, empate e repetição de estados."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple

from .bitboard import CONNECT_N, PopOutBoard


def check_winner_for_player(board: PopOutBoard, player: int) -> bool:
    """
    Verifica se um jogador tem 4-em-linha no estado atual.

    Args:
        board (PopOutBoard): Estado do jogo.
        player (int): Jogador a testar (1 ou 2).

    Returns:
        bool: True se houver sequência vencedora.
    """
    grid = board.board
    rows, cols = board.rows, board.cols
    n = CONNECT_N

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != player:
                continue

            if c + n <= cols and all(grid[r][c + k] == player for k in range(n)):
                return True
            if r + n <= rows and all(grid[r + k][c] == player for k in range(n)):
                return True
            if r + n <= rows and c + n <= cols and all(grid[r + k][c + k] == player for k in range(n)):
                return True
            if r + n <= rows and c - n + 1 >= 0 and all(grid[r + k][c - k] == player for k in range(n)):
                return True

    return False


def evaluate_after_move(board: PopOutBoard, mover: int) -> int:
    """
    Resolve resultado após uma jogada, considerando regra de conflito.

    Regra de conflito PopOut:
    - Se ambos têm 4-em-linha após a jogada, quem jogou vence.

    Args:
        board (PopOutBoard): Estado após jogada.
        mover (int): Jogador que acabou de jogar.

    Returns:
        int: 0 sem vencedor, ou 1/2 para vencedor.
    """
    p1 = check_winner_for_player(board, 1)
    p2 = check_winner_for_player(board, 2)

    if p1 and p2:
        return mover
    if p1:
        return 1
    if p2:
        return 2
    return 0


def board_signature(board: PopOutBoard) -> Tuple[Tuple[int, ...], ...]:
    """
    Gera assinatura imutável para deteção de repetição de estado.

    Args:
        board (PopOutBoard): Estado atual.

    Returns:
        Tuple[Tuple[int, ...], ...]: Assinatura hashable.
    """
    return tuple(tuple(row) for row in board.board)


def is_threefold_repetition(history_signatures: Iterable[Tuple[Tuple[int, ...], ...]]) -> bool:
    """
    Verifica empate por repetição tripla de estado.

    Args:
        history_signatures (Iterable): Assinaturas de estados visitados.

    Returns:
        bool: True se algum estado ocorre 3 ou mais vezes.
    """
    counts = Counter(history_signatures)
    return any(v >= 3 for v in counts.values())


def is_draw(board: PopOutBoard, history_signatures: List[Tuple[Tuple[int, ...], ...]]) -> bool:
    """
    Verifica condições de empate.

    Condições:
    - Tabuleiro cheio (sem drops possíveis).
    - Repetição tripla de estado.

    Args:
        board (PopOutBoard): Estado atual.
        history_signatures (List): Histórico de assinaturas.

    Returns:
        bool: True se houver empate.
    """
    return board.is_full() or is_threefold_repetition(history_signatures)
