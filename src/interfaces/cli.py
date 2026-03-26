"""Interface de linha de comandos para jogar PopOut contra MCTS."""

from __future__ import annotations

from ..algorithms.mcts.uct_standard import StandardUCT
from ..engine.bitboard import PopOutBoard
from ..engine.rules import evaluate_after_move


def parse_move(raw: str):
    """
    Converte texto do utilizador para formato de jogada.

    Formatos aceites:
    - d3  -> drop coluna 3
    - p5  -> pop coluna 5

    Args:
        raw (str): Texto inserido pelo utilizador.

    Returns:
        tuple[str, int]: Jogada (tipo, coluna).

    Raises:
        ValueError: Se formato for inválido.
    """
    raw = raw.strip().lower()
    if len(raw) < 2:
        raise ValueError("Formato inválido. Usa d<col> ou p<col>.")
    typ = raw[0]
    if typ not in {"d", "p"}:
        raise ValueError("Tipo inválido. Usa d para drop ou p para pop.")
    col = int(raw[1:])
    if col < 0 or col > 6:
        raise ValueError("Coluna inválida. Intervalo: 0..6.")
    return ("drop" if typ == "d" else "pop", col)


def run_cli_game(iterations: int = 400) -> None:
    """
    Corre um jogo humano (P1) vs MCTS (P2) em terminal.

    Args:
        iterations (int): Iterações do MCTS por jogada.
    """
    board = PopOutBoard()
    ai = StandardUCT(seed=42)

    print("=== PopOut CLI ===")
    print("Tu és X (Jogador 1). IA é O (Jogador 2).")
    print("Comandos: d<col> (drop), p<col> (pop). Ex.: d3, p0")

    while True:
        print("\n" + str(board))
        if board.current_player == 1:
            try:
                user_move = parse_move(input("A tua jogada: "))
            except Exception as exc:
                print(f"Erro: {exc}")
                continue

            mover = board.current_player
            if not board.apply_move(user_move, switch_player=True):
                print("Jogada inválida.")
                continue

            winner = evaluate_after_move(board, mover=mover)
            if winner:
                print(str(board))
                print(f"Vitória do jogador {winner}.")
                return
        else:
            ai_move = ai.run(board, iterations=iterations)
            print(f"IA joga: {ai_move}")
            mover = board.current_player
            board.apply_move(ai_move, switch_player=True)

            winner = evaluate_after_move(board, mover=mover)
            if winner:
                print(str(board))
                print(f"Vitória do jogador {winner}.")
                return


if __name__ == "__main__":
    run_cli_game()
