"""Interface gráfica em pygame para jogar PopOut localmente (2 jogadores) - VERSÃO MELHORADA."""

from __future__ import annotations

import sys
import pickle
import pygame
from enum import Enum
from pathlib import Path

from ..algorithms.mcts.uct_standard import StandardUCT
from ..algorithms.mcts.numba_mcts import FlatNumbaMCTS, warmup
from ..algorithms.mcts.protocol import MCTSEngine
from ..engine.bitboard import COLS, ROWS, PopOutBoard
from ..engine.rules import evaluate_after_move


CELL_SIZE = 100
BOARD_WIDTH = COLS * CELL_SIZE
BOARD_HEIGHT = ROWS * CELL_SIZE
HUD_TOP_HEIGHT = 80
HUD_BOTTOM_HEIGHT = 140
WINDOW_WIDTH = BOARD_WIDTH
WINDOW_HEIGHT = HUD_TOP_HEIGHT + BOARD_HEIGHT + HUD_BOTTOM_HEIGHT

# Controlo de animações
class AnimationState:
    """Rastreia animações ativas para peças e efeitos visuais.
    
    Attributes:
        piece_animations (dict): Mapa de (row, col) -> progresso (0.0-1.0)
        particle_effects (list): Lista de efeitos de partículas ativos
        flash_effects (dict): Mapa de efeitos de flash
    """
    def __init__(self) -> None:
        """Inicializa estado de animação vazio."""
        self.piece_animations: dict[tuple[int, int], float] = {}
        self.particle_effects: list[dict] = []
        self.flash_effects: dict[str, float] = {}
    
    def add_piece_animation(self, row: int, col: int) -> None:
        """Inicia animação de entrada para uma peça.
        
        Args:
            row (int): Linha da peça (0-3).
            col (int): Coluna da peça (0-6).
        """
        self.piece_animations[(row, col)] = 0.0
    
    def update(self, dt: float) -> None:
        """Atualiza todas as animações.
        
        Args:
            dt (float): Delta time em segundos.
        """
        # Avança animações de peças
        for key in list(self.piece_animations.keys()):
            self.piece_animations[key] += dt * 8  # velocidade 8x
            if self.piece_animations[key] >= 1.0:
                del self.piece_animations[key]
        
        # Avança efeitos de partícula
        for effect in self.particle_effects[:]:
            effect['time'] += dt
            if effect['time'] >= effect['lifetime']:
                self.particle_effects.remove(effect)
        
        # Avança efeitos de flash
        for key in list(self.flash_effects.keys()):
            self.flash_effects[key] -= dt * 3
            if self.flash_effects[key] <= 0:
                del self.flash_effects[key]


# PALETA DE CORES - Moderna e elegante
COLOR_BG_TOP = (15, 20, 35)
COLOR_BG_BOTTOM = (8, 12, 25)
COLOR_BOARD = (45, 95, 190)
COLOR_BOARD_DARK = (25, 60, 140)
COLOR_BOARD_SHADOW = (12, 30, 85)
COLOR_EMPTY = (240, 242, 250)
COLOR_EMPTY_SHADOW = (200, 205, 225)
COLOR_EMPTY_HOVER = (255, 255, 255)
COLOR_P1 = (255, 100, 100)
COLOR_P1_HI = (255, 150, 150)
COLOR_P1_GLOW = (255, 80, 80)
COLOR_P2 = (255, 220, 80)
COLOR_P2_HI = (255, 240, 150)
COLOR_P2_GLOW = (255, 200, 50)
COLOR_TEXT = (250, 252, 255)
COLOR_TEXT_MUTED = (180, 195, 225)
COLOR_ACCENT = (100, 230, 255)
COLOR_ACCENT_LIGHT = (150, 245, 255)
COLOR_ERROR = (255, 120, 120)
COLOR_PANEL = (22, 35, 70)
COLOR_PANEL_LIGHT = (35, 55, 110)
COLOR_PANEL_BORDER = (70, 100, 160)
COLOR_OVERLAY = (0, 0, 0, 180)
COLOR_HOVER = (150, 230, 255, 100)


# Menu de Pausa e Dificuldade
class Difficulty(Enum):
    """Níveis de dificuldade para IA."""
    EASY         = (100,    "Fácil (100)",         "standard")
    MEDIUM       = (500,    "Médio (500)",          "standard")
    HARD         = (1000,   "Difícil (1000)",       "standard")
    EXTREME      = (2000,   "Extremo (2000)",       "standard")
    EXTREME_NUMBA= (50_000, "Extremo Numba (50k)",  "flat_numba")

    @property
    def iterations(self) -> int:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]

    @property
    def engine_type(self) -> str:
        return self.value[2]


class PauseMenu:
    """Gerencia estado do menu de pausa."""
    def __init__(self) -> None:
        self.is_paused = False
        self.selected_option = 0
        self.selected_difficulty = Difficulty.MEDIUM
        self.options = ["Retomar", "Novo Jogo", "Dificuldade", "Modo", "Sair"]

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.selected_option = 0

    def navigate(self, direction: int) -> None:
        if self.is_paused:
            self.selected_option = (self.selected_option + direction) % len(self.options)

    def select_current(self) -> str:
        return self.options[self.selected_option]


def _draw_pause_menu(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    pause_menu: PauseMenu,
    game_mode: str = "PvP",
) -> None:
    """Desenha menu de pausa com opcoes navegaveis."""
    # Overlay escuro
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 200))
    screen.blit(overlay, (0, 0))

    # Box do menu
    menu_w, menu_h = 420, 420
    mx = (WINDOW_WIDTH - menu_w) // 2
    my = (WINDOW_HEIGHT - menu_h) // 2

    # Sombra do menu
    shadow = pygame.Rect(mx + 6, my + 6, menu_w, menu_h)
    pygame.draw.rect(screen, (0, 0, 0, 100), shadow, border_radius=24)

    menu_box = pygame.Rect(mx, my, menu_w, menu_h)
    pygame.draw.rect(screen, (18, 28, 58), menu_box, border_radius=24)
    pygame.draw.rect(screen, COLOR_ACCENT, menu_box, width=2, border_radius=24)

    # Titulo com linha decorativa
    title = font.render("PAUSA", True, COLOR_ACCENT_LIGHT)
    tx = mx + menu_w // 2 - title.get_width() // 2
    screen.blit(title, (tx, my + 20))
    line_y = my + 60
    pygame.draw.line(screen, COLOR_PANEL_BORDER, (mx + 30, line_y), (mx + menu_w - 30, line_y), 1)

    # Opcoes com detalhes contextuais
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

        # Valor atual ao lado (dificuldade / modo)
        if option in details:
            detail_color = COLOR_ACCENT if is_sel else COLOR_TEXT_MUTED
            detail_surf = small_font.render(details[option], True, detail_color)
            screen.blit(detail_surf, (mx + menu_w - 30 - detail_surf.get_width(), oy + 10))

    # Linha decorativa inferior
    line_y2 = start_y + len(pause_menu.options) * option_h + 5
    pygame.draw.line(screen, COLOR_PANEL_BORDER, (mx + 30, line_y2), (mx + menu_w - 30, line_y2), 1)

    # Controlos
    hints = [
        small_font.render("Setas para navegar | ENTER selecionar", True, COLOR_TEXT_MUTED),
        small_font.render("ESC para retomar", True, COLOR_TEXT_MUTED),
    ]
    for j, h in enumerate(hints):
        screen.blit(h, (mx + menu_w // 2 - h.get_width() // 2, line_y2 + 12 + j * 24))


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


def _draw_glow_circle(screen: pygame.Surface, center: tuple[int, int], radius: int, color: tuple[int, int, int], intensity: float = 0.5) -> None:
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
    glow_intensity: float = 0.0
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
    
    # Easing function para animação suave (ease-out)
    if animation_progress < 1.0:
        ease = 1 - (1 - animation_progress) ** 2
        offset_y = (1 - ease) * 20
        scale = 0.7 + ease * 0.3
        actual_radius = int(radius * scale)
    else:
        offset_y = 0
        actual_radius = radius
    
    actual_center = (cx, cy + offset_y)
    
    # Glow se estiver com glow
    if glow_intensity > 0:
        _draw_glow_circle(screen, actual_center, actual_radius, color, glow_intensity)
    
    # Sombra (3D effect)
    pygame.draw.circle(screen, (0, 0, 0, 120), (actual_center[0] + 3, actual_center[1] + 4), actual_radius)
    
    # Peça principal
    pygame.draw.circle(screen, color, actual_center, actual_radius)
    
    # Brilho especular (só se visível)
    if animation_progress > 0.5:
        highlight_offset = int(actual_radius // 3 * animation_progress)
        pygame.draw.circle(screen, highlight, 
                          (actual_center[0] - highlight_offset, actual_center[1] - highlight_offset), 
                          max(3, actual_radius // 4))


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
    
    # Sombra profunda do tabuleiro
    pygame.draw.rect(screen, COLOR_BOARD_SHADOW, board_rect.inflate(40, 20), border_radius=25)
    pygame.draw.rect(screen, COLOR_BOARD_DARK, board_rect, border_radius=20, width=8)
    pygame.draw.rect(screen, COLOR_BOARD, board_rect, border_radius=20)

    # Hover column highlight com transição suave
    if hover_col is not None and 0 <= hover_col < COLS and not game_over:
        hover_surface = pygame.Surface((CELL_SIZE, BOARD_HEIGHT), pygame.SRCALPHA)
        hover_surface.fill(COLOR_HOVER)
        screen.blit(hover_surface, (hover_col * CELL_SIZE, board_y))
        
        # Borda de destaque no topo (drop area)
        pygame.draw.rect(screen, COLOR_ACCENT_LIGHT, 
                        (hover_col * CELL_SIZE, board_y - 2, CELL_SIZE, 4))

    # Desenha peças com animações
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
                # Célula vazia
                glow = 0.5 if (c == hover_col and not game_over) else 0.0
                pygame.draw.circle(screen, (0, 0, 0, 40), (cx + 1, cy + 2), radius)
                pygame.draw.circle(screen, COLOR_EMPTY, (cx, cy), radius)
                if glow > 0:
                    _draw_glow_circle(screen, (cx, cy), radius, COLOR_ACCENT, glow)

    # Preview na coluna hover
    if hover_col is not None and not game_over:
        preview_color = COLOR_P2 if board.current_player == 2 else COLOR_P1
        preview_hi = COLOR_P2_HI if board.current_player == 2 else COLOR_P1_HI
        px = hover_col * CELL_SIZE + CELL_SIZE // 2
        py = HUD_TOP_HEIGHT // 2 + 6
        if mode_pop:
            # Modo POP: seta para baixo em vez de peça
            pts = [(px - 14, py - 8), (px + 14, py - 8), (px, py + 10)]
            pygame.draw.polygon(screen, preview_color, pts)
            label = small_font.render("POP", True, preview_color)
            screen.blit(label, (px - label.get_width() // 2, py - 26))
        else:
            # Modo DROP: peça normal
            _draw_disc(screen, (px, py), radius - 8, preview_color, preview_hi, glow_intensity=0.4)
            label = small_font.render("DROP", True, COLOR_ACCENT_LIGHT)
            screen.blit(label, (px - label.get_width() // 2, py - radius + 5))

    # === HUD SUPERIOR ===
    title = font.render("POPOUT", True, COLOR_TEXT)
    title_shadow = font.render("POPOUT", True, (0, 0, 0, 100))
    screen.blit(title_shadow, (17, 15))
    screen.blit(title, (14, 12))

    # Badge: Mode (DROP/POP) — cor de fundo muda para distinguir visualmente
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

    # Setas na base das colunas quando em modo POP
    if mode_pop and not game_over:
        arrow_y = board_y + BOARD_HEIGHT - 8
        for c in range(COLS):
            ax = c * CELL_SIZE + CELL_SIZE // 2
            # Triangulo apontando para baixo
            pts = [(ax - 8, arrow_y), (ax + 8, arrow_y), (ax, arrow_y + 12)]
            pygame.draw.polygon(screen, COLOR_P2, pts)

    # Badge: Game Mode  (PvP/IA)
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

    # Jogador atual com indicador
    player_color = _player_color(board.current_player)
    player_label = f"{'X (1)' if board.current_player == 1 else 'O (2)'}"
    player_title = f"Jogador Atual: "
    
    title_surf = font.render(player_title, True, COLOR_TEXT)
    value_surf = font.render(player_label, True, player_color)
    
    screen.blit(title_surf, (20, bottom_panel.y + 10))
    screen.blit(value_surf, (20 + title_surf.get_width() + 10, bottom_panel.y + 6))
    
    # Indicador de IA pensando
    if ai_thinking and game_mode == "IA" and board.current_player == 2:
        ai_indicator = "⟳ IA a pensar..."
        ai_surf = small_font.render(ai_indicator, True, COLOR_ACCENT_LIGHT)
        screen.blit(ai_surf, (WINDOW_WIDTH - 200, bottom_panel.y + 10))

    # Controles
    controls = "SPACE: DROP/POP | M: PvP/IA | R: Novo | ESC: Pausa | CLIQUE: Jogar"
    controls_surface = small_font.render(controls, True, COLOR_TEXT_MUTED)
    screen.blit(controls_surface, (20, bottom_panel.y + 48))

    # Mensagem de feedback
    if message:
        msg_color = COLOR_ERROR if "inválida" in message.lower() else COLOR_ACCENT_LIGHT
        msg_surface = font_18.render(message, True, msg_color)
        screen.blit(msg_surface, (20, bottom_panel.y + 78))

    # Índices das colunas (0-6)
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
            end_title = bold_48.render(
                f"JOGADOR {winner} VENCEU!", True, _player_color(winner)
            )
        else:
            end_title = bold_48.render("EMPATE!", True, COLOR_ACCENT)
        end_sub = small_font.render("Pressiona R para novo jogo ou ESC para sair", True, COLOR_TEXT)

        screen.blit(end_title, (box.centerx - end_title.get_width() // 2, box.y + 30))
        screen.blit(end_sub, (box.centerx - end_sub.get_width() // 2, box.y + 120))


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
        warmup()
        return FlatNumbaMCTS()
    return StandardUCT(seed=42)


def launch_gui() -> None:
    """Inicia a janela pygame com interface melhorada e executa o ciclo principal."""
    try:
        pygame.init()
        pygame.display.set_caption("PopOut AI")
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        clock = pygame.time.Clock()

        # Fontes (criadas uma vez, reutilizadas)
        font = pygame.font.SysFont("arial", 32, bold=True)
        small_font = pygame.font.SysFont("arial", 20)
        bold_24 = pygame.font.SysFont("arial", 24, bold=True)
        bold_48 = pygame.font.SysFont("arial", 48, bold=True)
        font_18 = pygame.font.SysFont("arial", 18)
        font_18_bold = pygame.font.SysFont("arial", 18, bold=True)

        # --- Estado mutável do jogo ---
        state: dict = {}
        pause_menu = PauseMenu()

        def reset_game(msg: str = "Novo jogo iniciado") -> None:
            """Reinicia o estado do jogo."""
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
            """Aplica jogada, deteta peças novas e inicia animações."""
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
            """Verifica vitória ou empate após jogada."""
            w = evaluate_after_move(state["board"], mover=mover)
            if w:
                state["game_over"] = True
                state["winner"] = w
                state["message"] = f"Jogador {w} venceu!"
            elif state["board"].is_full():
                state["game_over"] = True
                state["winner"] = 0
                state["message"] = "Empate - tabuleiro cheio!"

        # Estado inicial
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

                    elif (event.type == pygame.MOUSEBUTTONDOWN
                          and event.button == 1
                          and not state["game_over"]
                          and not pause_menu.is_paused):
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
                if (running
                    and not state["game_over"]
                    and not pause_menu.is_paused
                    and game_mode == "IA"
                    and state["board"].current_player == 2):
                    if not state["ai_thinking"]:
                        state["ai_timer"] -= dt
                        if state["ai_timer"] <= 0:
                            state["ai_thinking"] = True
                            # Renderiza frame com indicador "IA a pensar..."
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


if __name__ == "__main__":
    launch_gui()
