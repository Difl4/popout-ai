"""Main GUI entry point for the PopOut game."""

from __future__ import annotations

import sys

import pygame

from src.algorithms.factory import get_agent
from src.algorithms.mcts.protocol import MCTSEngine
from src.engine.standard.bitboard import COLS, ROWS, PopOutBoard
from src.engine.standard.rules import evaluate_after_move
from src.interfaces.gui.assets import (
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
    BOARD_WIDTH,
    CELL_SIZE,
    create_fonts,
)
from src.interfaces.gui.components import PauseMenu
from src.interfaces.gui.renderer import _draw_board, _draw_pause_menu
from src.interfaces.gui.state import AnimationState, Difficulty


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


def _make_ai_engine(difficulty: Difficulty) -> MCTSEngine:
    """Instantiate the correct AI engine for the given difficulty level.

    Calling warmup() when switching to a Numba-based engine ensures JIT
    compilation happens at difficulty-change time (not mid-game).
    """
    if difficulty.engine_type == "flat_numba":
        from src.algorithms.mcts.optimized.numba_mcts import warmup
        warmup()
        return get_agent("flat_numba")
    return get_agent(difficulty.engine_type, seed=42)


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

        def reset_game(msg: str = "Novo jogo iniciado") -> None:
            state["board"] = PopOutBoard()
            state["prev"] = (0, 0)
            state["mode_pop"] = False
            state["message"] = msg
            state["game_over"] = False
            state["winner"] = 0
            state["ai_thinking"] = False
            state["ai_timer"] = 0.0
            state["anim"] = AnimationState()

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
            elif state["board"].is_full():
                state["game_over"] = True
                state["winner"] = 0
                state["message"] = "Empate - tabuleiro cheio!"

        game_mode = "PvP"
        ai = _make_ai_engine(pause_menu.selected_difficulty)
        hover_col: int | None = None
        reset_game("Bem-vindo ao PopOut!")

        running = True
        while running:
            try:
                dt = clock.tick(60) / 1000.0
                state["anim"].update(dt)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.MOUSEMOTION:
                        hover_col = _column_from_mouse(event.pos[0])

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pause_menu.toggle_pause()
                        elif pause_menu.is_paused:
                            if event.key == pygame.K_UP:
                                pause_menu.navigate(-1)
                            elif event.key == pygame.K_DOWN:
                                pause_menu.navigate(1)
                            elif event.key == pygame.K_RETURN:
                                selected = pause_menu.select_current()
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
                                elif selected == "Modo":
                                    game_mode = "IA" if game_mode == "PvP" else "PvP"
                                    reset_game(f"Modo: {game_mode}")
                                elif selected == "Sair":
                                    running = False
                        else:
                            if event.key == pygame.K_SPACE and not state["game_over"]:
                                state["mode_pop"] = not state["mode_pop"]
                                state["message"] = f"Modo: {'POP' if state['mode_pop'] else 'DROP'}"
                            elif event.key == pygame.K_m:
                                game_mode = "IA" if game_mode == "PvP" else "PvP"
                                reset_game(f"Modo: {game_mode}")
                            elif event.key == pygame.K_r:
                                reset_game()

                    elif (
                        event.type == pygame.MOUSEBUTTONDOWN
                        and event.button == 1
                        and not state["game_over"]
                        and not pause_menu.is_paused
                    ):
                        if game_mode == "IA" and state["board"].current_player == 2:
                            continue
                        col = _column_from_mouse(event.pos[0])
                        if col is None:
                            continue
                        move = _encode_move(col, state["mode_pop"])
                        if move not in state["board"].legal_moves():
                            state["message"] = "Jogada invalida!"
                            continue
                        mover = state["board"].current_player
                        apply_and_animate(move, mover)
                        check_winner(mover)
                        if not state["game_over"]:
                            state["message"] = ""
                            state["ai_timer"] = 0.3 if game_mode == "IA" else 0

                # --- Renderiza frame ANTES da IA pensar ---
                _draw_board(
                    screen, state["board"], font, small_font,
                    bold_24, bold_48, font_18, font_18_bold,
                    state["mode_pop"], state["message"], hover_col,
                    state["game_over"], state["winner"], game_mode,
                    state["anim"], state["ai_thinking"],
                )
                if pause_menu.is_paused:
                    _draw_pause_menu(screen, font, small_font, pause_menu, game_mode)
                pygame.display.flip()

                # --- Lógica da IA (após render para não bloquear visual) ---
                if (
                    running
                    and not state["game_over"]
                    and not pause_menu.is_paused
                    and game_mode == "IA"
                    and state["board"].current_player == 2
                ):
                    if not state["ai_thinking"]:
                        state["ai_timer"] -= dt
                        if state["ai_timer"] <= 0:
                            state["ai_thinking"] = True
                            _draw_board(
                                screen, state["board"], font, small_font,
                                bold_24, bold_48, font_18, font_18_bold,
                                state["mode_pop"], state["message"], hover_col,
                                state["game_over"], state["winner"], game_mode,
                                state["anim"], True,
                            )
                            pygame.display.flip()
                    if state["ai_thinking"]:
                        iters = pause_menu.selected_difficulty.iterations
                        ai_move = ai.run(state["board"], iterations=iters)
                        ai_mover = state["board"].current_player
                        apply_and_animate(ai_move, ai_mover)
                        check_winner(ai_mover)
                        if not state["game_over"]:
                            state["message"] = "IA jogou"
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
