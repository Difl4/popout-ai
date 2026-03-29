"""Pure drawing functions for the PopOut GUI."""

from __future__ import annotations

import pygame

from src.engine.standard.bitboard import COLS, ROWS, PopOutBoard
from src.interfaces.gui.assets import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    CELL_SIZE,
    HUD_BOTTOM_HEIGHT,
    HUD_TOP_HEIGHT,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    COLOR_ACCENT,
    COLOR_ACCENT_LIGHT,
    COLOR_BG_BOTTOM,
    COLOR_BG_TOP,
    COLOR_BOARD,
    COLOR_BOARD_DARK,
    COLOR_BOARD_SHADOW,
    COLOR_EMPTY,
    COLOR_EMPTY_HOVER,
    COLOR_ERROR,
    COLOR_HOVER,
    COLOR_OVERLAY,
    COLOR_P1,
    COLOR_P1_GLOW,
    COLOR_P1_HI,
    COLOR_P2,
    COLOR_P2_GLOW,
    COLOR_P2_HI,
    COLOR_PANEL,
    COLOR_PANEL_BORDER,
    COLOR_PANEL_LIGHT,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
)
from src.interfaces.gui.state import AnimationState


def _player_color(player: int) -> tuple[int, int, int]:
    """Devolve a cor base associada ao jogador.

    Args:
        player (int): Número do jogador (1 ou 2).

    Returns:
        tuple[int, int, int]: Cor RGB do jogador.
    """
    return COLOR_P1 if player == 1 else COLOR_P2


def _player_glow(player: int) -> tuple[int, int, int]:
    """Devolve a cor de glow (brilho) do jogador.

    Args:
        player (int): Número do jogador (1 ou 2).

    Returns:
        tuple[int, int, int]: Cor RGB do glow.
    """
    return COLOR_P1_GLOW if player == 1 else COLOR_P2_GLOW


def _draw_vertical_gradient(screen: pygame.Surface, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    """Desenha um gradiente vertical no fundo com mais qualidade.

    Args:
        screen (pygame.Surface): Surface para desenhar.
        top (tuple[int, int, int]): Cor RGB do topo.
        bottom (tuple[int, int, int]): Cor RGB do fundo.
    """
    for y in range(WINDOW_HEIGHT):
        ratio = y / max(1, WINDOW_HEIGHT - 1)
        r = int(top[0] * (1 - ratio) + bottom[0] * ratio)
        g = int(top[1] * (1 - ratio) + bottom[1] * ratio)
        b = int(top[2] * (1 - ratio) + bottom[2] * ratio)
        pygame.draw.line(screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))


def _draw_glow_circle(
    screen: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    intensity: float = 0.5,
) -> None:
    """Desenha uma aura de brilho ao redor de um círculo.

    Args:
        screen (pygame.Surface): Surface para desenhar.
        center (tuple[int, int]): Centro do círculo (x, y).
        radius (int): Raio do círculo.
        color (tuple[int, int, int]): Cor RGB do glow.
        intensity (float): Intensidade do glow (0.0-1.0). Default: 0.5
    """
    cx, cy = center
    glow_surface = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
    glow_radius = int(radius * 2.5)

    for r in range(glow_radius, 0, -1):
        alpha = int(80 * intensity * (1 - r / glow_radius))
        glow_color = (*color, alpha)
        pygame.draw.circle(glow_surface, glow_color, (radius * 2, radius * 2), r)

    screen.blit(glow_surface, (cx - radius * 2, cy - radius * 2))


def _draw_disc(
    screen: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    highlight: tuple[int, int, int],
    animation_progress: float = 1.0,
    glow_intensity: float = 0.0,
) -> None:
    """Desenha uma peça com sombra, brilho e animação de chegada.

    Args:
        screen (pygame.Surface): Surface para desenhar.
        center (tuple[int, int]): Centro da peça (x, y).
        radius (int): Raio da peça.
        color (tuple[int, int, int]): Cor RGB da peça.
        highlight (tuple[int, int, int]): Cor RGB do destaque.
        animation_progress (float): Progresso da animação (0.0-1.0). Default: 1.0
        glow_intensity (float): Intensidade do glow. Default: 0.0
    """
    cx, cy = center

    if animation_progress < 1.0:
        ease = 1 - (1 - animation_progress) ** 2
        offset_y = (1 - ease) * 20
        scale = 0.7 + ease * 0.3
        actual_radius = int(radius * scale)
    else:
        offset_y = 0
        actual_radius = radius

    actual_center = (cx, cy + offset_y)

    if glow_intensity > 0:
        _draw_glow_circle(screen, actual_center, actual_radius, color, glow_intensity)

    pygame.draw.circle(screen, (0, 0, 0, 120), (actual_center[0] + 3, actual_center[1] + 4), actual_radius)
    pygame.draw.circle(screen, color, actual_center, actual_radius)

    if animation_progress > 0.5:
        highlight_offset = int(actual_radius // 3 * animation_progress)
        pygame.draw.circle(
            screen,
            highlight,
            (actual_center[0] - highlight_offset, actual_center[1] - highlight_offset),
            max(3, actual_radius // 4),
        )


def _draw_board(
    screen: pygame.Surface,
    board: PopOutBoard,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    bold_24: pygame.font.Font,
    bold_48: pygame.font.Font,
    font_18: pygame.font.Font,
    font_18_bold: pygame.font.Font,
    mode_pop: bool,
    message: str,
    hover_col: int | None,
    game_over: bool,
    winner: int,
    game_mode: str,
    anim: AnimationState,
    ai_thinking: bool = False,
) -> None:
    """Desenha tabuleiro com animações, HUD superior/inferior, preview, hover e overlay final."""
    _draw_vertical_gradient(screen, COLOR_BG_TOP, COLOR_BG_BOTTOM)

    board_y = HUD_TOP_HEIGHT
    board_rect = pygame.Rect(0, board_y, BOARD_WIDTH, BOARD_HEIGHT)

    pygame.draw.rect(screen, COLOR_BOARD_SHADOW, board_rect.inflate(40, 20), border_radius=25)
    pygame.draw.rect(screen, COLOR_BOARD_DARK, board_rect, border_radius=20, width=8)
    pygame.draw.rect(screen, COLOR_BOARD, board_rect, border_radius=20)

    if hover_col is not None and 0 <= hover_col < COLS and not game_over:
        hover_surface = pygame.Surface((CELL_SIZE, BOARD_HEIGHT), pygame.SRCALPHA)
        hover_surface.fill(COLOR_HOVER)
        screen.blit(hover_surface, (hover_col * CELL_SIZE, board_y))
        pygame.draw.rect(screen, COLOR_ACCENT_LIGHT, (hover_col * CELL_SIZE, board_y - 2, CELL_SIZE, 4))

    radius = CELL_SIZE // 2 - 12
    for r in range(ROWS):
        for c in range(COLS):
            visual_y = board_y + (ROWS - 1 - r) * CELL_SIZE
            cx = c * CELL_SIZE + CELL_SIZE // 2
            cy = visual_y + CELL_SIZE // 2

            bit = 1 << (c * 7 + r)
            anim_progress = anim.piece_animations.get((r, c), 1.0)

            if board.mask_p1 & bit:
                glow = 0.3 if (c == hover_col and not game_over) else 0.0
                _draw_disc(screen, (cx, cy), radius, COLOR_P1, COLOR_P1_HI, anim_progress, glow)
            elif board.mask_p2 & bit:
                glow = 0.3 if (c == hover_col and not game_over) else 0.0
                _draw_disc(screen, (cx, cy), radius, COLOR_P2, COLOR_P2_HI, anim_progress, glow)
            else:
                glow = 0.5 if (c == hover_col and not game_over) else 0.0
                pygame.draw.circle(screen, (0, 0, 0, 40), (cx + 1, cy + 2), radius)
                pygame.draw.circle(screen, COLOR_EMPTY, (cx, cy), radius)
                if glow > 0:
                    _draw_glow_circle(screen, (cx, cy), radius, COLOR_ACCENT, glow)

    if hover_col is not None and not game_over:
        preview_color = COLOR_P2 if board.current_player == 2 else COLOR_P1
        preview_hi = COLOR_P2_HI if board.current_player == 2 else COLOR_P1_HI
        px = hover_col * CELL_SIZE + CELL_SIZE // 2
        py = HUD_TOP_HEIGHT // 2 + 6
        if mode_pop:
            pts = [(px - 14, py - 8), (px + 14, py - 8), (px, py + 10)]
            pygame.draw.polygon(screen, preview_color, pts)
            label = small_font.render("POP", True, preview_color)
            screen.blit(label, (px - label.get_width() // 2, py - 26))
        else:
            _draw_disc(screen, (px, py), radius - 8, preview_color, preview_hi, glow_intensity=0.4)
            label = small_font.render("DROP", True, COLOR_ACCENT_LIGHT)
            screen.blit(label, (px - label.get_width() // 2, py - radius + 5))

    # === HUD SUPERIOR ===
    title = font.render("POPOUT", True, COLOR_TEXT)
    title_shadow = font.render("POPOUT", True, (0, 0, 0, 100))
    screen.blit(title_shadow, (17, 15))
    screen.blit(title, (14, 12))

    mode_text = "POP" if mode_pop else "DROP"
    mode_color = COLOR_P2 if mode_pop else COLOR_ACCENT
    mode_bg = (60, 40, 15) if mode_pop else COLOR_PANEL_LIGHT
    mode_badge = pygame.Rect(WINDOW_WIDTH - 340, 12, 156, 46)
    pygame.draw.rect(screen, mode_bg, mode_badge, border_radius=12)
    pygame.draw.rect(screen, mode_color, mode_badge, width=2, border_radius=12)

    mode_label = small_font.render("JOGADA", True, COLOR_TEXT_MUTED)
    mode_value = bold_24.render(mode_text, True, mode_color)
    screen.blit(mode_label, (mode_badge.x + 12, mode_badge.y + 4))
    screen.blit(mode_value, (mode_badge.x + 12, mode_badge.y + 24))

    if mode_pop and not game_over:
        arrow_y = board_y + BOARD_HEIGHT - 8
        for c in range(COLS):
            ax = c * CELL_SIZE + CELL_SIZE // 2
            pts = [(ax - 8, arrow_y), (ax + 8, arrow_y), (ax, arrow_y + 12)]
            pygame.draw.polygon(screen, COLOR_P2, pts)

    game_mode_badge = pygame.Rect(WINDOW_WIDTH - 170, 12, 156, 46)
    pygame.draw.rect(screen, COLOR_PANEL_LIGHT, game_mode_badge, border_radius=12)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, game_mode_badge, width=2, border_radius=12)

    game_mode_label = small_font.render("MODO", True, COLOR_TEXT_MUTED)
    game_mode_value = bold_24.render(game_mode, True, COLOR_ACCENT)
    screen.blit(game_mode_label, (game_mode_badge.x + 12, game_mode_badge.y + 4))
    screen.blit(game_mode_value, (game_mode_badge.x + 12, game_mode_badge.y + 24))

    # === PANEL INFERIOR ===
    bottom_panel = pygame.Rect(8, HUD_TOP_HEIGHT + BOARD_HEIGHT + 8, WINDOW_WIDTH - 16, HUD_BOTTOM_HEIGHT - 16)
    pygame.draw.rect(screen, COLOR_PANEL, bottom_panel, border_radius=16)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, bottom_panel, width=2, border_radius=16)

    player_color = _player_color(board.current_player)
    player_label = f"{'X (1)' if board.current_player == 1 else 'O (2)'}"
    player_title = "Jogador Atual: "

    title_surf = font.render(player_title, True, COLOR_TEXT)
    value_surf = font.render(player_label, True, player_color)

    screen.blit(title_surf, (20, bottom_panel.y + 10))
    screen.blit(value_surf, (20 + title_surf.get_width() + 10, bottom_panel.y + 6))

    if ai_thinking and game_mode == "IA" and board.current_player == 2:
        ai_indicator = "⟳ IA a pensar..."
        ai_surf = small_font.render(ai_indicator, True, COLOR_ACCENT_LIGHT)
        screen.blit(ai_surf, (WINDOW_WIDTH - 200, bottom_panel.y + 10))

    controls = "SPACE: DROP/POP | M: PvP/IA | R: Novo | ESC: Pausa | CLIQUE: Jogar"
    controls_surface = small_font.render(controls, True, COLOR_TEXT_MUTED)
    screen.blit(controls_surface, (20, bottom_panel.y + 48))

    if message:
        msg_color = COLOR_ERROR if "inválida" in message.lower() else COLOR_ACCENT_LIGHT
        msg_surface = font_18.render(message, True, msg_color)
        screen.blit(msg_surface, (20, bottom_panel.y + 78))

    for c in range(COLS):
        txt = font_18_bold.render(str(c), True, COLOR_TEXT)
        tx = c * CELL_SIZE + CELL_SIZE // 2 - txt.get_width() // 2
        ty = HUD_TOP_HEIGHT + BOARD_HEIGHT + 4
        screen.blit(txt, (tx, ty))

    # === OVERLAY DE VITÓRIA ===
    if game_over:
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
        overlay.fill(COLOR_OVERLAY)
        screen.blit(overlay, (0, 0))

        box = pygame.Rect(50, WINDOW_HEIGHT // 2 - 100, WINDOW_WIDTH - 100, 200)
        pygame.draw.rect(screen, COLOR_PANEL, box, border_radius=20)
        box_color = _player_color(winner) if winner else COLOR_ACCENT
        pygame.draw.rect(screen, box_color, box, width=4, border_radius=20)

        if winner:
            end_title = bold_48.render(f"JOGADOR {winner} VENCEU!", True, _player_color(winner))
        else:
            end_title = bold_48.render("EMPATE!", True, COLOR_ACCENT)
        end_sub = small_font.render("Pressiona R para novo jogo ou ESC para sair", True, COLOR_TEXT)

        screen.blit(end_title, (box.centerx - end_title.get_width() // 2, box.y + 30))
        screen.blit(end_sub, (box.centerx - end_sub.get_width() // 2, box.y + 120))


def _draw_pause_menu(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    pause_menu,
    game_mode: str = "PvP",
) -> None:
    """Desenha menu de pausa com opcoes navegaveis."""
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    screen.blit(overlay, (0, 0))

    menu_w, menu_h = 420, 420
    mx = (WINDOW_WIDTH - menu_w) // 2
    my = (WINDOW_HEIGHT - menu_h) // 2

    shadow = pygame.Rect(mx + 6, my + 6, menu_w, menu_h)
    pygame.draw.rect(screen, (0, 0, 0, 100), shadow, border_radius=24)

    menu_box = pygame.Rect(mx, my, menu_w, menu_h)
    pygame.draw.rect(screen, (18, 28, 58), menu_box, border_radius=24)
    pygame.draw.rect(screen, COLOR_ACCENT, menu_box, width=2, border_radius=24)

    title = font.render("PAUSA", True, COLOR_ACCENT_LIGHT)
    tx = mx + menu_w // 2 - title.get_width() // 2
    screen.blit(title, (tx, my + 20))
    line_y = my + 60
    pygame.draw.line(screen, COLOR_PANEL_BORDER, (mx + 30, line_y), (mx + menu_w - 30, line_y), 1)

    option_h = 48
    start_y = my + 75
    details = {
        "Dificuldade": pause_menu.selected_difficulty.label,
        "Modo": game_mode,
    }

    for i, option in enumerate(pause_menu.options):
        is_sel = i == pause_menu.selected_option
        oy = start_y + i * option_h

        if is_sel:
            bg = pygame.Rect(mx + 16, oy, menu_w - 32, option_h - 6)
            pygame.draw.rect(screen, (30, 50, 100), bg, border_radius=10)
            pygame.draw.rect(screen, COLOR_ACCENT, bg, width=2, border_radius=10)

        color = COLOR_TEXT if is_sel else COLOR_TEXT_MUTED
        prefix = "> " if is_sel else "  "
        label_surf = small_font.render(prefix + option, True, color)
        screen.blit(label_surf, (mx + 30, oy + 10))

        if option in details:
            detail_color = COLOR_ACCENT if is_sel else COLOR_TEXT_MUTED
            detail_surf = small_font.render(details[option], True, detail_color)
            screen.blit(detail_surf, (mx + menu_w - 30 - detail_surf.get_width(), oy + 10))

    line_y2 = start_y + len(pause_menu.options) * option_h + 5
    pygame.draw.line(screen, COLOR_PANEL_BORDER, (mx + 30, line_y2), (mx + menu_w - 30, line_y2), 1)

    hints = [
        small_font.render("Setas para navegar | ENTER selecionar", True, COLOR_TEXT_MUTED),
        small_font.render("ESC para retomar", True, COLOR_TEXT_MUTED),
    ]
    for j, h in enumerate(hints):
        screen.blit(h, (mx + menu_w // 2 - h.get_width() // 2, line_y2 + 12 + j * 24))
