"""Interface de linha de comandos para PopOut — 3 modos de jogo.

Modos suportados:
    hvh  — Humano vs Humano
    hvc  — Humano vs Computador (MCTS)
    cvc  — Computador vs Computador (2 algoritmos diferentes)
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

from src.mcts.factory import get_agent
from src.mcts.protocol import MCTSEngine
from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import board_signature, evaluate_after_move, is_threefold_repetition


# ── Utilitários de parsing e display ─────────────────────────────────────────

def parse_move(raw: str) -> int:
    """Converte 'd<col>' ou 'p<col>' para inteiro do motor (0-13).

    Args:
        raw (str): Texto introduzido pelo utilizador (ex: 'd3', 'p0').

    Returns:
        int: Move codificado (0-6 drop, 7-13 pop).

    Raises:
        ValueError: Se o formato, tipo ou coluna forem inválidos.
    """
    raw = raw.strip().lower()
    if len(raw) < 2:
        raise ValueError("Formato inválido. Usa d<col> ou p<col>.")
    typ = raw[0]
    if typ not in {"d", "p"}:
        raise ValueError("Tipo inválido. Usa 'd' para drop ou 'p' para pop.")
    try:
        col = int(raw[1:])
    except ValueError:
        raise ValueError(f"Coluna não é um número: '{raw[1:]}'.")
    if col < 0 or col > 6:
        raise ValueError(f"Coluna inválida ({col}). Intervalo: 0..6.")
    return col if typ == "d" else col + 7


def decode_move(move: int) -> str:
    """Converte inteiro do motor para texto legível.

    Args:
        move (int): Move codificado (0-13).

    Returns:
        str: Descrição legível (ex: 'drop 3', 'pop 0').
    """
    if move < 7:
        return f"drop {move}"
    return f"pop {move - 7}"


def _print_board(board: PopOutBoard) -> None:
    """Imprime o tabuleiro com indicador do jogador atual."""
    player_label = "X (Jogador 1)" if board.current_player == 1 else "O (Jogador 2)"
    print(f"\n{str(board)}")
    print(f"Turno: {player_label}")


def _check_winner(board: PopOutBoard, mover: int) -> Optional[int]:
    """Verifica vitória imediata após uma jogada.

    NÃO verifica empates por tabuleiro cheio ou repetição — essas condições
    são opcionais e tratadas em _handle_draw_rules (Regras 2 e 3 do PopOut).

    Returns:
        Vencedor (1 ou 2), ou None se o jogo continua.
    """
    winner = evaluate_after_move(board, mover=mover)
    return winner if winner else None


def _handle_draw_rules(
    board: PopOutBoard,
    history: list,
    is_human: bool,
    player_label: str,
) -> Optional[int]:
    """Aplica as Regras 2 e 3 do PopOut sobre empate.

    Regra 2 (tabuleiro cheio): o jogador decide fazer pop ou declarar empate.
    Regra 3 (repetição tripla): qualquer jogador PODE (não é obrigado) declarar empate.

    Para jogadores humanos, a decisão é pedida via input.
    Para a IA, o empate é declarado automaticamente (estratégia conservadora).

    Returns:
        -1 se empate declarado ou forçado, None se o jogo deve continuar.
    """
    # Regra 3: repetição tripla
    if is_threefold_repetition(history):
        if is_human:
            try:
                ans = input(
                    f"  Estado repetido 3 vezes! {player_label}, declaras empate? [s/n]: "
                ).strip().lower()
            except EOFError:
                ans = "s"
            if ans == "s":
                return -1
        else:
            print(f"  {player_label} declara empate por repetição tripla.")
            return -1

    # Regra 2: tabuleiro cheio
    if board.is_full():
        pop_moves = [m for m in board.legal_moves() if m >= 7]
        if not pop_moves:
            print(f"  Tabuleiro cheio! {player_label} sem jogadas pop. Empate forçado.")
            return -1
        if is_human:
            try:
                ans = input(
                    f"  Tabuleiro cheio! {player_label}: faz [p]op ou aceita [e]mpate? "
                ).strip().lower()
            except EOFError:
                ans = "e"
            if ans == "e":
                return -1
        # IA com tabuleiro cheio: joga normalmente (vai escolher um pop)

    return None


# ── Modo 1: Humano vs Humano ──────────────────────────────────────────────────

def run_hvh() -> None:
    """Jogo Humano vs Humano via linha de comandos."""
    board = PopOutBoard()
    history: list = []

    print("=" * 40)
    print("  POPOUT — Humano vs Humano")
    print("=" * 40)
    print("Jogador 1 = X  |  Jogador 2 = O")
    print("Comandos: d<col> (drop), p<col> (pop)")
    print("Exemplos: d3 (drop coluna 3), p0 (pop coluna 0)\n")

    while True:
        history.append(board_signature(board))
        _print_board(board)

        player_label = "X (Jogador 1)" if board.current_player == 1 else "O (Jogador 2)"

        # Regras 2 e 3: empate opcional (tabuleiro cheio ou repetição)
        draw = _handle_draw_rules(board, history, True, player_label)
        if draw is not None:
            print("Resultado: EMPATE.")
            return

        try:
            raw = input(f"{player_label} — a tua jogada: ")
            move = parse_move(raw)
        except (ValueError, EOFError) as exc:
            print(f"  ⚠  Erro: {exc}")
            continue

        if move not in board.legal_moves():
            print("  ⚠  Jogada inválida para o estado atual.")
            continue

        mover = board.current_player
        board.apply_move(move)

        result = _check_winner(board, mover)
        if result is not None:
            _print_board(board)
            print(f"Resultado: Jogador {result} ('{'X' if result == 1 else 'O'}') VENCEU!")
            return


# ── Modo 2: Humano vs Computador ─────────────────────────────────────────────

def run_hvc(iterations: int = 1000, human_player: int = 1, agent_name: str = "standard") -> None:
    """Jogo Humano vs Computador (MCTS).

    Args:
        iterations (int): Iterações MCTS por jogada da IA.
        human_player (int): Qual jogador é humano (1 ou 2).
        agent_name (str): Nome do agente (standard, solver, experimental, numba, flat_numba).
    """
    board = PopOutBoard()
    ai = get_agent(agent_name, seed=42)
    history: list = []

    human_symbol = "X" if human_player == 1 else "O"
    ai_symbol    = "O" if human_player == 1 else "X"

    print("=" * 40)
    print("  POPOUT — Humano vs Computador")
    print("=" * 40)
    print(f"Tu és {human_symbol} (Jogador {human_player})  |  IA é {ai_symbol} [{agent_name}]  |  {iterations} iterações MCTS")
    print("Comandos: d<col> (drop), p<col> (pop). Ex.: d3, p0\n")

    while True:
        history.append(board_signature(board))
        _print_board(board)

        is_human = board.current_player == human_player
        symbol = "X" if board.current_player == 1 else "O"
        player_label = f"{symbol} (Jogador {board.current_player})"

        # Regras 2 e 3: empate opcional (tabuleiro cheio ou repetição)
        draw = _handle_draw_rules(board, history, is_human, player_label)
        if draw is not None:
            print("Resultado: EMPATE.")
            return

        if is_human:
            # --- Jogada humana ---
            try:
                raw = input("A tua jogada: ")
                move = parse_move(raw)
            except (ValueError, EOFError) as exc:
                print(f"  ⚠  Erro: {exc}")
                continue

            if move not in board.legal_moves():
                print("  ⚠  Jogada inválida para o estado atual.")
                continue
        else:
            # --- Jogada da IA ---
            print(f"  IA a pensar ({iterations} iter)...")
            move = ai.run(board, iterations=iterations)
            print(f"  IA joga: {decode_move(move)}")

        mover = board.current_player
        board.apply_move(move)

        result = _check_winner(board, mover)
        if result is not None:
            _print_board(board)
            if result == human_player:
                print("Resultado: GANHOU! Parabéns!")
            else:
                print("Resultado: A IA venceu. Tenta de novo!")
            return


# ── Modo 3: Computador vs Computador ─────────────────────────────────────────

def run_cvc(
    agent1: MCTSEngine,
    agent2: MCTSEngine,
    agent1_name: str = "Agente 1",
    agent2_name: str = "Agente 2",
    iterations1: int = 500,
    iterations2: int = 500,
    verbose: bool = True,
    max_moves: int = 200,
) -> int:
    """Jogo Computador vs Computador entre dois agentes MCTS.

    Args:
        agent1 (MCTSEngine): Agente do Jogador 1 (X).
        agent2 (MCTSEngine): Agente do Jogador 2 (O).
        agent1_name (str): Nome descritivo do agente 1.
        agent2_name (str): Nome descritivo do agente 2.
        iterations1 (int): Iterações MCTS do agente 1.
        iterations2 (int): Iterações MCTS do agente 2.
        verbose (bool): Se True, imprime o tabuleiro a cada jogada.
        max_moves (int): Limite de movimentos para evitar ciclos infinitos.

    Returns:
        int: Vencedor (1 ou 2) ou -1 em caso de empate.
    """
    board = PopOutBoard()
    history: list = []
    agents = {1: (agent1, agent1_name, iterations1), 2: (agent2, agent2_name, iterations2)}

    if verbose:
        print("=" * 45)
        print(f"  POPOUT — Computador vs Computador")
        print("=" * 45)
        print(f"  X (J1): {agent1_name} ({iterations1} iter)")
        print(f"  O (J2): {agent2_name} ({iterations2} iter)")
        print()

    for move_num in range(max_moves):
        history.append(board_signature(board))

        if verbose:
            _print_board(board)

        agent, name, iters = agents[board.current_player]
        symbol = "X" if board.current_player == 1 else "O"
        player_label = f"{symbol} ({name})"

        # Regras 2 e 3: IA declara empate automaticamente
        draw = _handle_draw_rules(board, history, False, player_label)
        if draw is not None:
            if verbose:
                print(f"Resultado: EMPATE após {move_num + 1} jogadas.")
            return -1

        move = agent.run(board, iterations=iters)

        if verbose:
            print(f"  {player_label}: {decode_move(move)}")

        mover = board.current_player
        board.apply_move(move)

        result = _check_winner(board, mover)
        if result is not None:
            if verbose:
                _print_board(board)
                winner_name = agent1_name if result == 1 else agent2_name
                print(f"Resultado: {winner_name} ({'X' if result == 1 else 'O'}) VENCEU em {move_num + 1} jogadas!")
            return result

    if verbose:
        print(f"  ⚠  Limite de {max_moves} jogadas atingido. Empate forçado.")
    return -1


_AGENTS = {
    "1": ("standard",          "StandardUCT"),
    "2": ("experimental",      "ExperimentalUCT"),
    "3": ("solver",            "SolverMCTS"),
    "4": ("numba",             "NumbaMCTS"),
    "5": ("flat_numba",        "FlatNumbaMCTS"),
    "6": ("numba_solver",      "NumbaSolverMCTS"),
    "7": ("flat_numba_solver", "FlatNumbaSolverMCTS"),
    "8": ("id3",               "ID3Agent (Decision Tree)"),
}


def _ask_int(prompt: str, default: int, min_val: int = 1) -> int:
    """Pede um inteiro ao utilizador; usa default se vazio."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            val = int(raw)
            if val < min_val:
                print(f"  ⚠  Valor mínimo: {min_val}.")
                continue
            return val
        except ValueError:
            print("  ⚠  Introduz um número inteiro.")


def _ask_agent(label: str) -> tuple[str, str]:
    """Mostra lista de agentes e devolve (agent_key, display_name)."""
    print(f"\n  Escolhe o {label}:")
    for key, (_, name) in _AGENTS.items():
        print(f"    {key}. {name}")
    while True:
        choice = input(f"  Opção: ").strip()
        if choice in _AGENTS:
            key, name = _AGENTS[choice]
            return key, name
        print("  ⚠  Opção inválida.")


def run_cvc_tournament() -> None:
    """Torneio configurável via terminal entre dois agentes MCTS."""
    print("=" * 50)
    print("  TORNEIO — Configuração")
    print("=" * 50)

    key1, name1 = _ask_agent("Agente 1")
    key2, name2 = _ask_agent("Agente 2")
    n_games    = _ask_int("\n  Número de jogos", default=10)
    iters1     = _ask_int(f"  Iterações para {name1}", default=1000)
    iters2     = _ask_int(f"  Iterações para {name2}", default=1000)

    results: Counter = Counter()

    print()
    print("=" * 50)
    print(f"  TORNEIO: {name1} vs {name2}")
    print("=" * 50)
    print(f"  {n_games} jogos  |  {name1}={iters1} iter  |  {name2}={iters2} iter\n")

    for i in range(n_games):
        # Alternar cores para resultados justos
        if i % 2 == 0:
            a1, a2 = get_agent(key1, seed=i), get_agent(key2, seed=i)
            n1, n2, i1, i2 = name1, name2, iters1, iters2
        else:
            a1, a2 = get_agent(key2, seed=i), get_agent(key1, seed=i)
            n1, n2, i1, i2 = name2, name1, iters2, iters1

        result = run_cvc(
            agent1=a1, agent2=a2,
            agent1_name=n1, agent2_name=n2,
            iterations1=i1, iterations2=i2,
            verbose=False,
        )

        if result == -1:
            results["Empate"] += 1
            outcome = "Empate"
        elif result == 1:
            results[n1] += 1
            outcome = f"{n1} venceu"
        else:
            results[n2] += 1
            outcome = f"{n2} venceu"

        print(f"  Jogo {i+1:2d}: {n1} (X) vs {n2} (O) → {outcome}")

    print()
    print("─" * 50)
    print("  Resultados Finais:")
    for name, count in results.most_common():
        pct = count / n_games * 100
        print(f"    {name:<18}: {count:2d} ({pct:.0f}%)")
    print("─" * 50)


# ── Entry point ───────────────────────────────────────────────────────────────

def run_cli_game(iterations: int = 1000) -> None:
    """Menu principal da CLI — seleciona o modo de jogo.

    Args:
        iterations (int): Iterações MCTS padrão (usado em modos com IA).
    """
    print("\n╔══════════════════════════════════╗")
    print("║       POPOUT — Menu Principal    ║")
    print("╚══════════════════════════════════╝")
    print("  1. Humano vs Humano")
    print("  2. Humano vs Computador (MCTS)")
    print("  3. Computador vs Computador (torneio)")
    print()

    try:
        choice = input("Escolhe um modo [1/2/3]: ").strip()
    except EOFError:
        choice = "2"

    if choice == "1":
        run_hvh()
    elif choice == "2":
        agent_key, agent_display = _ask_agent("agente IA")
        iters = _ask_int(f"  Iterações para {agent_display}", default=iterations)
        run_hvc(iterations=iters, agent_name=agent_key)
    elif choice == "3":
        run_cvc_tournament()
    else:
        print(f"Opção '{choice}' inválida. A iniciar modo padrão (Humano vs Computador).")
        run_hvc(iterations=iterations)


if __name__ == "__main__":
    run_cli_game()
