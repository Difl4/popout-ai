"""Interface de linha de comandos para jogar PopOut contra MCTS."""

from __future__ import annotations
from ..algorithms.mcts.uct_standard import StandardUCT
from ..engine.bitboard import PopOutBoard
from ..engine.rules import evaluate_after_move

def parse_move(raw: str) -> int:
    """
    Converte d<col> ou p<col> no formato inteiro do motor (0-13).
    """
    raw = raw.strip().lower()
    if len(raw) < 2: raise ValueError("Formato inválido. Usa d<col> ou p<col>.")
    typ = raw[0]
    if typ not in {"d", "p"}: raise ValueError("Tipo inválido. Usa d para drop ou p para pop.")
    col = int(raw[1:])
    if col < 0 or col > 6: raise ValueError("Coluna inválida. Intervalo: 0..6.")
    
    # Encode logic: Drop = 0-6, Pop = 7-13
    return col if typ == "d" else col + 7

def decode_move(move: int) -> str:
    """Converte o inteiro do motor de volta para texto legível."""
    if move < 7: return f"drop {move}"
    return f"pop {move - 7}"

def run_cli_game(iterations: int = 10000) -> None:
    board = PopOutBoard()
    ai = StandardUCT(seed=42)

    print("=== PopOut CLI ===")
    print("Tu és X (Jogador 1). IA é O (Jogador 2).")
    print("Comandos: d<col> (drop), p<col> (pop). Ex.: d3, p0")

    while True:
        print("\n" + str(board))
        if board.current_player == 1:
            try:
                user_move_raw = input("A tua jogada: ")
                engine_move = parse_move(user_move_raw)
            except Exception as exc:
                print(f"Erro: {exc}")
                continue

            # Valida contra as jogadas legais geradas pelo motor
            if engine_move not in board.legal_moves():
                print("Jogada inválida para o estado atual.")
                continue

            mover = board.current_player
            board.apply_move(engine_move)

            winner = evaluate_after_move(board, mover=mover)
            if winner:
                print(str(board))
                print(f"Vitória do jogador {winner}.")
                return
        else:
            ai_move_int = ai.run(board, iterations=iterations)
            print(f"IA joga: {decode_move(ai_move_int)}")
            mover = board.current_player
            board.apply_move(ai_move_int)

            winner = evaluate_after_move(board, mover=mover)
            if winner:
                print(str(board))
                print(f"Vitória do jogador {winner}.")
                return

if __name__ == "__main__":
    run_cli_game()