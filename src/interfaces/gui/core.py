"""Main GUI entry point for the PopOut game."""

from __future__ import annotations

import sys
from queue import Empty, SimpleQueue
from threading import Thread

import pygame

from src.mcts.factory import get_agent
from src.mcts.protocol import MCTSEngine
from src.engine.standard.bitboard import COLS, ROWS, PopOutBoard
from src.engine.standard.rules import evaluate_after_move, board_signature, is_threefold_repetition, find_winning_cells
from src.interfaces.gui.assets import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    BOARD_WIDTH,
    CELL_SIZE,
    create_fonts,
)
from src.interfaces.gui.components import PauseMenu
from src.interfaces.gui.renderer import _draw_board, _draw_pause_menu, _draw_setup_screen
from src.interfaces.gui.state import AnimationState, Difficulty

_GAME_MODES = ("PvP", "IA", "Arena")


def _column_from_mouse(x_pos: int) -> int | None:
    """Converte coordenada X do rato para coluna 0..6, ou None fora do tabuleiro.

    Args:
        x_pos (int): Posição X do rato em pixels.

    Returns:
        int | None: Coluna (0-6) ou None se fora do board.
    """
    if x_pos < 0 or x_pos >= BOARD_WIDTH:
        return None
    return x_pos // CELL_SIZE


def _encode_move(column: int, mode_pop: bool) -> int:
    """Codifica jogada no formato do motor: drop(0..6) ou pop(7..13).

    Args:
        column (int): Coluna (0-6).
        mode_pop (bool): True para POP, False para DROP.

    Returns:
        int: Move codificado (0-13).
    """
    return column + 7 if mode_pop else column


_NUMBA_ENGINES = {"numba", "flat_numba", "numba_solver", "flat_numba_solver"}


def _make_ai_engine(difficulty: Difficulty) -> MCTSEngine:
    """Instantiate the correct AI engine for the given difficulty level.

    Calling warmup() for any Numba-based engine ensures JIT compilation
    happens at difficulty-change time rather than mid-game.
    """
    if difficulty.engine_type in _NUMBA_ENGINES:
        from src.mcts.optimized.numba_mcts import warmup
        warmup()
    return get_agent(difficulty.engine_type, seed=42)


def _cycle_game_mode(current_mode: str, direction: int) -> str:
    """Cycle through the supported GUI game modes in the requested direction."""
    try:
        idx = _GAME_MODES.index(current_mode)
    except ValueError:
        return _GAME_MODES[0]
    return _GAME_MODES[(idx + direction) % len(_GAME_MODES)]


def _next_game_mode(current_mode: str) -> str:
    """Cycle forward through the supported GUI game modes."""
    return _cycle_game_mode(current_mode, 1)


def _is_ai_turn(game_mode: str, current_player: int, human_player: int = 1) -> bool:
    """Return whether the current player is controlled by AI in the GUI mode."""
    if game_mode == "Arena":
        return True
    if game_mode == "IA":
        return current_player != human_player
    return False


def _cycle_difficulty(current: Difficulty, direction: int) -> Difficulty:
    """Cycle through AI options in the requested direction."""
    difficulties = list(Difficulty)
    idx = difficulties.index(current)
    return difficulties[(idx + direction) % len(difficulties)]


def _setup_options_for_mode(game_mode: str) -> list[str]:
    """Return setup menu fields for the chosen game mode."""
    options = ["Modo"]
    if game_mode == "IA":
        options.extend(["IA", "Lado"])
    elif game_mode == "Arena":
        options.extend(["IA P1", "IA P2"])
    options.append("Iniciar")
    return options


def launch_gui() -> None:
    """Inicia a janela pygame com interface melhorada e executa o ciclo principal."""
    try:
        pygame.init()
        pygame.display.set_caption("PopOut AI")
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        clock = pygame.time.Clock()

        fonts = create_fonts()
        font = fonts["font"]
        small_font = fonts["small_font"]
        bold_24 = fonts["bold_24"]
        bold_48 = fonts["bold_48"]
        font_18 = fonts["font_18"]
        font_18_bold = fonts["font_18_bold"]

        state: dict = {}
        pause_menu = PauseMenu()
        setup_active = True
        setup_selected_option = 0
        ai_job_token = 0
        ai_result_queue: SimpleQueue[tuple[int, int, int]] = SimpleQueue()
        ai_worker: Thread | None = None

        def reset_game(msg: str = "Novo jogo iniciado") -> None:
            nonlocal ai_job_token, ai_worker
            ai_job_token += 1
            ai_worker = None
            state["board"] = PopOutBoard()
            state["prev"] = (0, 0)
            state["mode_pop"] = False
            state["message"] = msg
            state["game_over"] = False
            state["winner"] = 0
            state["ai_thinking"] = False
            state["ai_timer"] = 0.0
            state["anim"] = AnimationState()
            state["history"] = [board_signature(state["board"])]
            state["can_declare_draw"] = False
            state["winning_cells"] = []
            state["ai_thinking"] = False
            state["ai_timer"] = 0.0

        def apply_and_animate(move: int, mover: int) -> None:
            old_p1, old_p2 = state["board"].mask_p1, state["board"].mask_p2
            state["board"].apply_move(move)
            col = move if move < 7 else move - 7
            for r in range(ROWS):
                bit = 1 << (col * 7 + r)
                if mover == 1 and (state["board"].mask_p1 & bit) and not (old_p1 & bit):
                    state["anim"].add_piece_animation(r, col)
                elif mover == 2 and (state["board"].mask_p2 & bit) and not (old_p2 & bit):
                    state["anim"].add_piece_animation(r, col)
            state["prev"] = (state["board"].mask_p1, state["board"].mask_p2)

        def check_winner(mover: int) -> None:
            w = evaluate_after_move(state["board"], mover=mover)
            if w:
                state["game_over"] = True
                state["winner"] = w
                state["message"] = f"Jogador {w} venceu!"
                score[w - 1] += 1
                winner_mask = state["board"].mask_p1 if w == 1 else state["board"].mask_p2
                state["winning_cells"] = find_winning_cells(winner_mask)

        def _update_draw_state() -> None:
            if state["game_over"]:
                state["can_declare_draw"] = False
                return
            if is_threefold_repetition(state["history"]):
                state["can_declare_draw"] = True
                state["message"] = "Repeticao tripla! Prima D para declarar empate"
                return
            if state["board"].is_full():
                pop_moves = [m for m in state["board"].legal_moves() if m >= 7]
                if not pop_moves:
                    state["game_over"] = True
                    state["winner"] = 0
                    state["message"] = "Empate - sem jogadas possiveis!"
                else:
                    state["can_declare_draw"] = True
                    state["message"] = "Tabuleiro cheio! Prima D para empate ou jogue POP"
                return
            state["can_declare_draw"] = False

        game_mode = "PvP"
        ai = _make_ai_engine(pause_menu.selected_difficulty)
        arena_ai = {
            1: _make_ai_engine(pause_menu.arena_p1_difficulty),
            2: _make_ai_engine(pause_menu.arena_p2_difficulty),
        }
        hover_col: int | None = None
        score = [0, 0]  # [p1_vitórias, p2_vitórias] — persiste entre partidas
        reset_game("Configura a partida para comecar")

        running = True
        while running:
            try:
                dt = clock.tick(60) / 1000.0
                state["anim"].update(dt)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.MOUSEMOTION:
                        if setup_active or game_mode == "Arena":
                            hover_col = None
                        else:
                            hover_col = _column_from_mouse(event.pos[0])

                    elif event.type == pygame.KEYDOWN:
                        if setup_active:
                            setup_options = _setup_options_for_mode(game_mode)
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            elif event.key == pygame.K_UP:
                                setup_selected_option = (setup_selected_option - 1) % len(setup_options)
                            elif event.key == pygame.K_DOWN:
                                setup_selected_option = (setup_selected_option + 1) % len(setup_options)
                            elif event.key in {pygame.K_LEFT, pygame.K_RIGHT, pygame.K_RETURN}:
                                step = 1 if event.key != pygame.K_LEFT else -1
                                selected = setup_options[setup_selected_option]
                                if selected == "Modo":
                                    game_mode = _cycle_game_mode(game_mode, step)
                                    setup_options = _setup_options_for_mode(game_mode)
                                    setup_selected_option = min(setup_selected_option, len(setup_options) - 1)
                                    hover_col = None
                                elif selected == "IA":
                                    pause_menu.selected_difficulty = _cycle_difficulty(pause_menu.selected_difficulty, step)
                                    ai = _make_ai_engine(pause_menu.selected_difficulty)
                                elif selected == "Lado":
                                    pause_menu.human_player = 2 if pause_menu.human_player == 1 else 1
                                elif selected == "IA P1":
                                    pause_menu.arena_p1_difficulty = _cycle_difficulty(pause_menu.arena_p1_difficulty, step)
                                    arena_ai[1] = _make_ai_engine(pause_menu.arena_p1_difficulty)
                                elif selected == "IA P2":
                                    pause_menu.arena_p2_difficulty = _cycle_difficulty(pause_menu.arena_p2_difficulty, step)
                                    arena_ai[2] = _make_ai_engine(pause_menu.arena_p2_difficulty)
                                elif selected == "Iniciar" and event.key == pygame.K_RETURN:
                                    setup_active = False
                                    _side_msg = f" — Tu: J{pause_menu.human_player}" if game_mode == "IA" else ""
                                    reset_game(f"Modo: {game_mode}{_side_msg}")
                                    if game_mode == "Arena":
                                        state["ai_timer"] = 0.3
                            continue
                        if event.key == pygame.K_ESCAPE:
                            pause_menu.toggle_pause()
                        elif pause_menu.is_paused:
                            if event.key == pygame.K_UP:
                                pause_menu.navigate(-1, game_mode)
                            elif event.key == pygame.K_DOWN:
                                pause_menu.navigate(1, game_mode)
                            elif event.key == pygame.K_RETURN:
                                selected = pause_menu.select_current(game_mode)
                                if selected == "Retomar":
                                    pause_menu.is_paused = False
                                elif selected == "Novo Jogo":
                                    reset_game()
                                    pause_menu.is_paused = False
                                elif selected == "Dificuldade":
                                    difficulties = list(Difficulty)
                                    idx = difficulties.index(pause_menu.selected_difficulty)
                                    pause_menu.selected_difficulty = difficulties[(idx + 1) % len(difficulties)]
                                    ai = _make_ai_engine(pause_menu.selected_difficulty)
                                    state["message"] = f"Dificuldade: {pause_menu.selected_difficulty.label}"
                                elif selected == "Lado":
                                    pause_menu.human_player = 2 if pause_menu.human_player == 1 else 1
                                    state["message"] = f"Tu: Jogador {pause_menu.human_player}"
                                elif selected == "Modo":
                                    game_mode = _next_game_mode(game_mode)
                                    reset_game(f"Modo: {game_mode}")
                                elif selected == "IA P1":
                                    difficulties = list(Difficulty)
                                    idx = difficulties.index(pause_menu.arena_p1_difficulty)
                                    pause_menu.arena_p1_difficulty = difficulties[(idx + 1) % len(difficulties)]
                                    arena_ai[1] = _make_ai_engine(pause_menu.arena_p1_difficulty)
                                    state["message"] = f"Arena P1: {pause_menu.arena_p1_difficulty.label}"
                                elif selected == "IA P2":
                                    difficulties = list(Difficulty)
                                    idx = difficulties.index(pause_menu.arena_p2_difficulty)
                                    pause_menu.arena_p2_difficulty = difficulties[(idx + 1) % len(difficulties)]
                                    arena_ai[2] = _make_ai_engine(pause_menu.arena_p2_difficulty)
                                    state["message"] = f"Arena P2: {pause_menu.arena_p2_difficulty.label}"
                                elif selected == "Sair":
                                    running = False
                        else:
                            if event.key == pygame.K_SPACE and not state["game_over"]:
                                state["mode_pop"] = not state["mode_pop"]
                                state["message"] = f"Modo: {'POP' if state['mode_pop'] else 'DROP'}"
                            elif event.key == pygame.K_d and state["can_declare_draw"] and not state["game_over"]:
                                state["game_over"] = True
                                state["winner"] = 0
                                state["message"] = "Empate declarado!"
                                state["can_declare_draw"] = False
                            elif event.key == pygame.K_m:
                                game_mode = _next_game_mode(game_mode)
                                reset_game(f"Modo: {game_mode}")
                                hover_col = None if game_mode == "Arena" else hover_col
                            elif event.key == pygame.K_r:
                                reset_game()

                    elif (
                        event.type == pygame.MOUSEBUTTONDOWN
                        and event.button == 1
                        and not state["game_over"]
                        and not pause_menu.is_paused
                    ):
                        if setup_active:
                            setup_options = _setup_options_for_mode(game_mode)
                            panel_x = 80
                            panel_y = 170
                            panel_w = WINDOW_WIDTH - 160
                            row_h = 68
                            start_y = panel_y + 42
                            for idx, option in enumerate(setup_options):
                                row_top = start_y + idx * row_h
                                row_rect = pygame.Rect(panel_x + 24, row_top, panel_w - 48, row_h - 10)
                                if not row_rect.collidepoint(event.pos):
                                    continue
                                setup_selected_option = idx
                                if option == "Modo":
                                    game_mode = _next_game_mode(game_mode)
                                elif option == "IA":
                                    pause_menu.selected_difficulty = _cycle_difficulty(pause_menu.selected_difficulty, 1)
                                    ai = _make_ai_engine(pause_menu.selected_difficulty)
                                elif option == "Lado":
                                    pause_menu.human_player = 2 if pause_menu.human_player == 1 else 1
                                elif option == "IA P1":
                                    pause_menu.arena_p1_difficulty = _cycle_difficulty(pause_menu.arena_p1_difficulty, 1)
                                    arena_ai[1] = _make_ai_engine(pause_menu.arena_p1_difficulty)
                                elif option == "IA P2":
                                    pause_menu.arena_p2_difficulty = _cycle_difficulty(pause_menu.arena_p2_difficulty, 1)
                                    arena_ai[2] = _make_ai_engine(pause_menu.arena_p2_difficulty)
                                elif option == "Iniciar":
                                    setup_active = False
                                    _side_msg = f" — Tu: J{pause_menu.human_player}" if game_mode == "IA" else ""
                                    reset_game(f"Modo: {game_mode}{_side_msg}")
                                    if game_mode == "Arena":
                                        state["ai_timer"] = 0.3
                                break
                            continue
                        if _is_ai_turn(game_mode, state["board"].current_player, pause_menu.human_player):
                            continue
                        col = _column_from_mouse(event.pos[0])
                        if col is None:
                            continue
                        move = _encode_move(col, state["mode_pop"])
                        if move not in state["board"].legal_moves():
                            state["message"] = "Jogada invalida!"
                            continue
                        mover = state["board"].current_player
                        state["can_declare_draw"] = False
                        apply_and_animate(move, mover)
                        check_winner(mover)
                        if not state["game_over"]:
                            state["history"].append(board_signature(state["board"]))
                            _update_draw_state()
                            if not state["can_declare_draw"] and not state["game_over"]:
                                state["message"] = ""
                            state["ai_timer"] = 0.3 if game_mode == "IA" else 0

                # --- Renderiza frame ANTES da IA pensar ---
                render_hover_col = None if game_mode == "Arena" else hover_col
                if setup_active:
                    _draw_setup_screen(
                        screen,
                        font,
                        small_font,
                        bold_24,
                        bold_48,
                        game_mode,
                        setup_selected_option,
                        pause_menu.selected_difficulty,
                        pause_menu.arena_p1_difficulty,
                        pause_menu.arena_p2_difficulty,
                        pause_menu.human_player,
                    )
                else:
                    _draw_board(
                        screen, state["board"], font, small_font,
                        bold_24, bold_48, font_18, font_18_bold,
                        state["mode_pop"], state["message"], render_hover_col,
                        state["game_over"], state["winner"], game_mode,
                        state["anim"], state["ai_thinking"],
                        winning_cells=state["winning_cells"],
                        score=score,
                        can_declare_draw=state["can_declare_draw"],
                    )
                if pause_menu.is_paused and not setup_active:
                    _draw_pause_menu(screen, font, small_font, pause_menu, game_mode)
                pygame.display.flip()

                # --- Lógica da IA (após render para não bloquear visual) ---
                if (
                    running
                    and not setup_active
                    and not state["game_over"]
                    and not pause_menu.is_paused
                    and _is_ai_turn(game_mode, state["board"].current_player, pause_menu.human_player)
                ):
                    if state["can_declare_draw"]:
                        state["game_over"] = True
                        state["winner"] = 0
                        state["message"] = "Empate declarado pela IA!"
                        state["can_declare_draw"] = False
                    elif ai_worker is None and not state["ai_thinking"]:
                        state["ai_timer"] -= dt
                        if state["ai_timer"] <= 0:
                            state["ai_thinking"] = True
                            _draw_board(
                                screen, state["board"], font, small_font,
                                bold_24, bold_48, font_18, font_18_bold,
                                state["mode_pop"], state["message"], render_hover_col,
                                state["game_over"], state["winner"], game_mode,
                                state["anim"], True,
                                winning_cells=state["winning_cells"],
                                score=score,
                                can_declare_draw=state["can_declare_draw"],
                            )
                            pygame.display.flip()
                            ai_player = state["board"].current_player
                            active_ai = arena_ai[ai_player] if game_mode == "Arena" else ai
                            active_difficulty = (
                                pause_menu.arena_p1_difficulty if ai_player == 1 else pause_menu.arena_p2_difficulty
                            ) if game_mode == "Arena" else pause_menu.selected_difficulty
                            board_snapshot = state["board"].clone()
                            current_job_token = ai_job_token

                            def _run_ai_turn(agent, board_copy, iterations, mover, job_token, result_queue) -> None:
                                move = agent.run(board_copy, iterations=iterations)
                                result_queue.put((job_token, mover, move))

                            ai_worker = Thread(
                                target=_run_ai_turn,
                                args=(
                                    active_ai,
                                    board_snapshot,
                                    active_difficulty.iterations,
                                    ai_player,
                                    current_job_token,
                                    ai_result_queue,
                                ),
                                daemon=True,
                            )
                            ai_worker.start()

                    try:
                        job_token, ai_mover, ai_move = ai_result_queue.get_nowait()
                    except Empty:
                        pass
                    else:
                        if job_token != ai_job_token:
                            continue
                        ai_worker = None
                        state["can_declare_draw"] = False
                        apply_and_animate(ai_move, ai_mover)
                        check_winner(ai_mover)
                        if not state["game_over"]:
                            state["history"].append(board_signature(state["board"]))
                            _update_draw_state()
                            if not state["can_declare_draw"] and not state["game_over"]:
                                state["message"] = f"IA J{ai_mover} jogou"
                        state["ai_thinking"] = False

            except KeyboardInterrupt:
                running = False

        pygame.quit()

    except pygame.error as e:
        print(f"Erro pygame: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            pygame.quit()
        except Exception:
            pass
