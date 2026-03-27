"""Interface gráfica em pygame para jogar PopOut localmente (2 jogadores)."""

from __future__ import annotations

import pygame

from ..algorithms.mcts.uct_standard import StandardUCT
from ..engine.bitboard import COLS, ROWS, PopOutBoard
from ..engine.rules import evaluate_after_move


CELL_SIZE = 90
BOARD_WIDTH = COLS * CELL_SIZE
BOARD_HEIGHT = ROWS * CELL_SIZE
HUD_TOP_HEIGHT = 64
HUD_BOTTOM_HEIGHT = 118
WINDOW_WIDTH = BOARD_WIDTH
WINDOW_HEIGHT = HUD_TOP_HEIGHT + BOARD_HEIGHT + HUD_BOTTOM_HEIGHT

COLOR_BG_TOP = (18, 25, 45)
COLOR_BG_BOTTOM = (10, 15, 30)
COLOR_BOARD = (38, 78, 165)
COLOR_BOARD_DARK = (22, 52, 120)
COLOR_EMPTY = (235, 237, 245)
COLOR_EMPTY_SHADOW = (185, 190, 210)
COLOR_P1 = (240, 88, 88)
COLOR_P1_HI = (255, 140, 140)
COLOR_P2 = (247, 214, 84)
COLOR_P2_HI = (255, 238, 156)
COLOR_TEXT = (245, 248, 255)
COLOR_TEXT_MUTED = (185, 198, 225)
COLOR_ACCENT = (95, 220, 255)
COLOR_ERROR = (255, 130, 130)
COLOR_PANEL = (20, 30, 58)
COLOR_PANEL_BORDER = (62, 86, 140)
COLOR_OVERLAY = (0, 0, 0, 150)
COLOR_HOVER = (140, 220, 255, 80)


def _player_color(player: int) -> tuple[int, int, int]:
    """Devolve a cor base associada ao jogador."""
    return COLOR_P1 if player == 1 else COLOR_P2


def _draw_vertical_gradient(screen: pygame.Surface, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    """Desenha um gradiente vertical no fundo para melhorar o acabamento visual."""
    for y in range(WINDOW_HEIGHT):
        ratio = y / max(1, WINDOW_HEIGHT - 1)
        r = int(top[0] * (1 - ratio) + bottom[0] * ratio)
        g = int(top[1] * (1 - ratio) + bottom[1] * ratio)
        b = int(top[2] * (1 - ratio) + bottom[2] * ratio)
        pygame.draw.line(screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))


def _draw_disc(screen: pygame.Surface, center: tuple[int, int], radius: int, color: tuple[int, int, int], highlight: tuple[int, int, int]) -> None:
    """Desenha uma peça com sombra e brilho para dar sensação de profundidade."""
    cx, cy = center
    pygame.draw.circle(screen, (0, 0, 0, 80), (cx + 2, cy + 3), radius)
    pygame.draw.circle(screen, color, center, radius)
    pygame.draw.circle(screen, highlight, (cx - radius // 3, cy - radius // 3), max(3, radius // 3))


def _draw_board(
    screen: pygame.Surface,
    board: PopOutBoard,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    mode_pop: bool,
    message: str,
    hover_col: int | None,
    game_over: bool,
    winner: int,
    game_mode: str,
) -> None:
    """Desenha tabuleiro, HUD superior/inferior, preview, hover e overlay final."""
    _draw_vertical_gradient(screen, COLOR_BG_TOP, COLOR_BG_BOTTOM)

    board_y = HUD_TOP_HEIGHT
    board_rect = pygame.Rect(0, board_y, BOARD_WIDTH, BOARD_HEIGHT)
    pygame.draw.rect(screen, COLOR_BOARD_DARK, board_rect.inflate(0, 10), border_radius=20)
    pygame.draw.rect(screen, COLOR_BOARD, board_rect, border_radius=20)

    if hover_col is not None and 0 <= hover_col < COLS and not game_over:
        hover_surface = pygame.Surface((CELL_SIZE, BOARD_HEIGHT), pygame.SRCALPHA)
        hover_surface.fill(COLOR_HOVER)
        screen.blit(hover_surface, (hover_col * CELL_SIZE, board_y))

    radius = CELL_SIZE // 2 - 10
    for r in range(ROWS):
        for c in range(COLS):
            visual_y = board_y + (ROWS - 1 - r) * CELL_SIZE
            cx = c * CELL_SIZE + CELL_SIZE // 2
            cy = visual_y + CELL_SIZE // 2

            bit = 1 << (c * 7 + r)
            if board.mask_p1 & bit:
                _draw_disc(screen, (cx, cy), radius, COLOR_P1, COLOR_P1_HI)
            elif board.mask_p2 & bit:
                _draw_disc(screen, (cx, cy), radius, COLOR_P2, COLOR_P2_HI)
            else:
                pygame.draw.circle(screen, COLOR_EMPTY_SHADOW, (cx + 1, cy + 2), radius)
                pygame.draw.circle(screen, COLOR_EMPTY, (cx, cy), radius)

    if hover_col is not None and not game_over:
        preview_color = COLOR_P2 if board.current_player == 2 else COLOR_P1
        preview_hi = COLOR_P2_HI if board.current_player == 2 else COLOR_P1_HI
        px = hover_col * CELL_SIZE + CELL_SIZE // 2
        py = HUD_TOP_HEIGHT // 2 + 6
        _draw_disc(screen, (px, py), radius - 10, preview_color, preview_hi)

    title = font.render("POPOUT", True, COLOR_TEXT)
    screen.blit(title, (14, 12))

    mode_text = "POP" if mode_pop else "DROP"
    mode_color = COLOR_ACCENT if not mode_pop else COLOR_P2
    mode_badge = pygame.Rect(WINDOW_WIDTH - 330, 12, 156, 36)
    pygame.draw.rect(screen, COLOR_PANEL, mode_badge, border_radius=10)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, mode_badge, width=2, border_radius=10)
    mode_surface = small_font.render(f"Jogada: {mode_text}", True, mode_color)
    screen.blit(mode_surface, (mode_badge.x + 16, mode_badge.y + 9))

    game_mode_badge = pygame.Rect(WINDOW_WIDTH - 170, 12, 156, 36)
    pygame.draw.rect(screen, COLOR_PANEL, game_mode_badge, border_radius=10)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, game_mode_badge, width=2, border_radius=10)
    game_mode_surface = small_font.render(f"Modo: {game_mode}", True, COLOR_ACCENT)
    screen.blit(game_mode_surface, (game_mode_badge.x + 16, game_mode_badge.y + 9))

    bottom_panel = pygame.Rect(8, HUD_TOP_HEIGHT + BOARD_HEIGHT + 8, WINDOW_WIDTH - 16, HUD_BOTTOM_HEIGHT - 16)
    pygame.draw.rect(screen, COLOR_PANEL, bottom_panel, border_radius=14)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, bottom_panel, width=2, border_radius=14)

    player_label = f"Jogador atual: {'X (1)' if board.current_player == 1 else 'O (2)'}"
    player_surface = font.render(player_label, True, _player_color(board.current_player))
    screen.blit(player_surface, (20, bottom_panel.y + 12))

    controls = "Clique: jogar | SPACE: alternar DROP/POP | M: alternar PvP/IA | R: reiniciar | ESC: sair"
    controls_surface = small_font.render(controls, True, COLOR_TEXT_MUTED)
    screen.blit(controls_surface, (20, bottom_panel.y + 48))

    if message:
        msg_color = COLOR_ERROR if "inválida" in message.lower() else COLOR_TEXT
        msg_surface = small_font.render(message, True, msg_color)
        screen.blit(msg_surface, (20, bottom_panel.y + 78))

    for c in range(COLS):
        txt = small_font.render(str(c), True, COLOR_TEXT)
        tx = c * CELL_SIZE + CELL_SIZE // 2 - txt.get_width() // 2
        ty = HUD_TOP_HEIGHT + BOARD_HEIGHT - 24
        screen.blit(txt, (tx, ty))

    if game_over:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill(COLOR_OVERLAY)
        screen.blit(overlay, (0, 0))

        box = pygame.Rect(80, WINDOW_HEIGHT // 2 - 80, WINDOW_WIDTH - 160, 160)
        pygame.draw.rect(screen, COLOR_PANEL, box, border_radius=16)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, box, width=2, border_radius=16)

        end_title = font.render(f"Vitória do jogador {winner}", True, _player_color(winner))
        end_sub = small_font.render("Pressiona R para reiniciar ou ESC para sair", True, COLOR_TEXT)

        screen.blit(end_title, (box.centerx - end_title.get_width() // 2, box.y + 44))
        screen.blit(end_sub, (box.centerx - end_sub.get_width() // 2, box.y + 88))


def _column_from_mouse(x_pos: int) -> int | None:
    """Converte coordenada X do rato para coluna 0..6, ou None fora do tabuleiro."""
    if x_pos < 0 or x_pos >= BOARD_WIDTH:
        return None
    return x_pos // CELL_SIZE


def _encode_move(column: int, mode_pop: bool) -> int:
    """Codifica jogada no formato do motor: drop(0..6) ou pop(7..13)."""
    return column + 7 if mode_pop else column


def launch_gui() -> None:
    """Inicia a janela pygame e executa o ciclo principal da partida."""
    pygame.init()
    pygame.display.set_caption("PopOut - GUI (pygame)")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 28, bold=True)
    small_font = pygame.font.SysFont("arial", 20)

    board = PopOutBoard()
    mode_pop = False
    message = ""
    game_over = False
    winner = 0
    hover_col: int | None = None
    game_mode = "PvP"
    ai = StandardUCT(seed=42)
    ai_iterations = 1200

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEMOTION:
                hover_col = _column_from_mouse(event.pos[0])

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE and not game_over:
                    mode_pop = not mode_pop
                    message = f"Modo alterado para {'POP' if mode_pop else 'DROP'}."
                elif event.key == pygame.K_m:
                    game_mode = "IA" if game_mode == "PvP" else "PvP"
                    board = PopOutBoard()
                    mode_pop = False
                    message = f"Modo alterado para {game_mode}."
                    game_over = False
                    winner = 0
                elif event.key == pygame.K_r:
                    board = PopOutBoard()
                    mode_pop = False
                    message = "Jogo reiniciado."
                    game_over = False
                    winner = 0

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game_over:
                if game_mode == "IA" and board.current_player == 2:
                    message = "Aguarda a jogada da IA."
                    continue

                col = _column_from_mouse(event.pos[0])
                if col is None:
                    message = "Clique fora da área do tabuleiro."
                    continue

                move = _encode_move(col, mode_pop)
                if move not in board.legal_moves():
                    message = "Jogada inválida para o estado atual."
                    continue

                mover = board.current_player
                board.apply_move(move)
                winner = evaluate_after_move(board, mover=mover)
                if winner:
                    game_over = True
                    message = f"Vitória do jogador {winner}."
                else:
                    message = ""

        if running and not game_over and game_mode == "IA" and board.current_player == 2:
            ai_move = ai.run(board, iterations=ai_iterations)
            ai_mover = board.current_player
            board.apply_move(ai_move)
            winner = evaluate_after_move(board, mover=ai_mover)
            if winner:
                game_over = True
                message = f"Vitória do jogador {winner}."
            else:
                message = "IA jogou."

        _draw_board(screen, board, font, small_font, mode_pop, message, hover_col, game_over, winner, game_mode)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    launch_gui()
