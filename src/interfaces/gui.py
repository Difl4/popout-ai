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
    """Gerencia estado do menu de pausa.
    
    Attributes:
        is_paused (bool): Se o jogo está pausado.
        selected_option (int): Índice da opção selecionada (0-3).
        selected_difficulty (Difficulty): Dificuldade selecionada.
    """
    def __init__(self) -> None:
        """Inicializa menu de pausa."""
        self.is_paused = False
        self.selected_option = 0  # 0: Resume, 1: New Game, 2: Difficulty, 3: Quit
        self.selected_difficulty = Difficulty.MEDIUM
        self.options = ["Retomar", "Novo Jogo", "Dificuldade", "Sair"]
    
    def toggle_pause(self) -> None:
        """Alterna estado de pausa."""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.selected_option = 0  # Reset para "Retomar"
    
    def navigate(self, direction: int) -> None:
        """Navega menu com setas (up/down).
        
        Args:
            direction (int): -1 para cima, +1 para baixo.
        """
        if self.is_paused:
            self.selected_option = (self.selected_option + direction) % len(self.options)
    
    def select_current(self) -> str:
        """Retorna a opção selecionada (para processamento).
        
        Returns:
            str: Nome da opção selecionada.
        """
        return self.options[self.selected_option]


def _draw_pause_menu(
    screen: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    pause_menu: PauseMenu,
) -> None:
    """Desenha menu de pausa com opções navegáveis.
    
    Args:
        screen (pygame.Surface): Superfície pygame.
        font (pygame.font.Font): Fonte principal.
        small_font (pygame.font.Font): Fonte pequena.
        pause_menu (PauseMenu): Estado do menu.
    """
    # Overlay escuro
    overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
    overlay.fill(COLOR_OVERLAY)
    screen.blit(overlay, (0, 0))
    
    # Box do menu
    menu_width = 400
    menu_height = 350
    menu_x = (WINDOW_WIDTH - menu_width) // 2
    menu_y = (WINDOW_HEIGHT - menu_height) // 2
    
    menu_box = pygame.Rect(menu_x, menu_y, menu_width, menu_height)
    pygame.draw.rect(screen, COLOR_PANEL, menu_box, border_radius=20)
    pygame.draw.rect(screen, COLOR_ACCENT, menu_box, width=3, border_radius=20)
    
    # Título
    pause_title = font.render("PAUSADO", True, COLOR_ACCENT_LIGHT)
    screen.blit(pause_title, (menu_x + menu_width // 2 - pause_title.get_width() // 2, menu_y + 20))
    
    # Opções do menu
    option_spacing = 60
    start_y = menu_y + 80
    
    for i, option in enumerate(pause_menu.options):
        is_selected = (i == pause_menu.selected_option)
        color = COLOR_ACCENT if is_selected else COLOR_TEXT_MUTED
        
        # Fundo da opção selecionada
        if is_selected:
            option_bg = pygame.Rect(menu_x + 20, start_y + i * option_spacing - 10, menu_width - 40, 40)
            pygame.draw.rect(screen, COLOR_PANEL_LIGHT, option_bg, border_radius=8)
            pygame.draw.rect(screen, COLOR_ACCENT, option_bg, width=2, border_radius=8)
        
        # Texto da opção
        option_text = "→ " + option if is_selected else "  " + option
        option_surf = small_font.render(option_text, True, color)
        screen.blit(option_surf, (menu_x + 30, start_y + i * option_spacing))
    
    # Instrções no fundo
    hint = small_font.render("↑↓ Navegar | ENTER Selecionar | ESC Retomar", True, COLOR_TEXT_MUTED)
    screen.blit(hint, (menu_x + menu_width // 2 - hint.get_width() // 2, menu_y + menu_height - 40))


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

    # Preview de peça na coluna hover
    if hover_col is not None and not game_over:
        preview_color = COLOR_P2 if board.current_player == 2 else COLOR_P1
        preview_hi = COLOR_P2_HI if board.current_player == 2 else COLOR_P1_HI
        px = hover_col * CELL_SIZE + CELL_SIZE // 2
        py = HUD_TOP_HEIGHT // 2 + 6
        _draw_disc(screen, (px, py), radius - 8, preview_color, preview_hi, glow_intensity=0.4)
        
        # Rótulo "PRÓXIMA JOGADA"
        preview_label = small_font.render("PRÓXIMA", True, COLOR_ACCENT_LIGHT)
        screen.blit(preview_label, (px - preview_label.get_width() // 2, py - radius + 5))

    # === HUD SUPERIOR ===
    title = font.render("POPOUT", True, COLOR_TEXT)
    title_shadow = font.render("POPOUT", True, (0, 0, 0, 100))
    screen.blit(title_shadow, (17, 15))
    screen.blit(title, (14, 12))

    # Badge: Mode (DROP/POP)
    mode_text = "POP" if mode_pop else "DROP"
    mode_color = COLOR_P2 if mode_pop else COLOR_ACCENT
    mode_badge = pygame.Rect(WINDOW_WIDTH - 340, 12, 156, 46)
    pygame.draw.rect(screen, COLOR_PANEL_LIGHT, mode_badge, border_radius=12)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, mode_badge, width=2, border_radius=12)
    
    mode_label = small_font.render("JOGADA", True, COLOR_TEXT_MUTED)
    mode_value = pygame.font.SysFont("arial", 24, bold=True).render(mode_text, True, mode_color)
    screen.blit(mode_label, (mode_badge.x + 12, mode_badge.y + 4))
    screen.blit(mode_value, (mode_badge.x + 12, mode_badge.y + 24))

    # Badge: Game Mode  (PvP/IA)
    game_mode_badge = pygame.Rect(WINDOW_WIDTH - 170, 12, 156, 46)
    pygame.draw.rect(screen, COLOR_PANEL_LIGHT, game_mode_badge, border_radius=12)
    pygame.draw.rect(screen, COLOR_PANEL_BORDER, game_mode_badge, width=2, border_radius=12)
    
    game_mode_label = small_font.render("MODO", True, COLOR_TEXT_MUTED)
    game_mode_value = pygame.font.SysFont("arial", 24, bold=True).render(game_mode, True, COLOR_ACCENT)
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
    value_surf = pygame.font.SysFont("arial", 32, bold=True).render(player_label, True, player_color)
    
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
        msg_surface = pygame.font.SysFont("arial", 18).render(message, True, msg_color)
        screen.blit(msg_surface, (20, bottom_panel.y + 78))

    # Índices das colunas (0-6)
    for c in range(COLS):
        txt = pygame.font.SysFont("arial", 18, bold=True).render(str(c), True, COLOR_TEXT)
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
        pygame.draw.rect(screen, _player_color(winner), box, width=4, border_radius=20)

        end_title = pygame.font.SysFont("arial", 48, bold=True).render(
            f"JOGADOR {winner} VENCEU!", True, _player_color(winner)
        )
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
    """Inicia a janela pygame com interface melhorada e executa o ciclo principal.
    
    Raises:
        pygame.error: Se houver erro ao inicializar pygame.
        SystemExit: Se houver erro irrecuperável.
    """
    try:
        pygame.init()
        pygame.display.set_caption("PopOut - GUI Melhorada (pygame)")
        screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        clock = pygame.time.Clock()

        # Fontes modernizadas
        font = pygame.font.SysFont("arial", 32, bold=True)
        small_font = pygame.font.SysFont("arial", 20)

        # Estado do jogo
        board = PopOutBoard()
        mode_pop = False
        message = ""
        game_over = False
        winner = 0
        hover_col: int | None = None
        game_mode = "PvP"
        ai = StandardUCT(seed=42)
        ai_iterations = 10000
        ai_thinking = False
        ai_delay = 0.3  # segundos antes de IA começar a pensar
        ai_timer = 0.0

        # Sistema de animações
        anim = AnimationState()
        
        # Menu de pausa
        pause_menu = PauseMenu()

        # Matriz anterior para detetar mudanças
        prev_board_state = (board.mask_p1, board.mask_p2)

        running = True
        while running:
            try:
                dt = clock.tick(60) / 1000.0  # Delta time em segundos

                # Atualiza animações
                anim.update(dt)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                    elif event.type == pygame.MOUSEMOTION:
                        hover_col = _column_from_mouse(event.pos[0])

                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pause_menu.toggle_pause()
                        elif event.key == pygame.K_UP and pause_menu.is_paused:
                            pause_menu.navigate(-1)
                        elif event.key == pygame.K_DOWN and pause_menu.is_paused:
                            pause_menu.navigate(1)
                        elif event.key == pygame.K_RETURN and pause_menu.is_paused:
                            # Processar seleção do menu
                            selected = pause_menu.select_current()
                            if selected == "Retomar":
                                pause_menu.is_paused = False
                            elif selected == "Novo Jogo":
                                board = PopOutBoard()
                                prev_board_state = (board.mask_p1, board.mask_p2)
                                mode_pop = False
                                message = "Novo jogo iniciado"
                                game_over = False
                                winner = 0
                                ai_thinking = False
                                anim = AnimationState()
                                pause_menu.is_paused = False
                            elif selected == "Dificuldade":
                                # Cicla através dos níveis de dificuldade
                                difficulties = list(Difficulty)
                                current_idx = difficulties.index(pause_menu.selected_difficulty)
                                pause_menu.selected_difficulty = difficulties[(current_idx + 1) % len(difficulties)]
                                ai_iterations = pause_menu.selected_difficulty.iterations
                                ai = _make_ai_engine(pause_menu.selected_difficulty)
                                message = f"Dificuldade: {pause_menu.selected_difficulty.label}"
                            elif selected == "Sair":
                                running = False
                        elif event.key == pygame.K_SPACE and not game_over and not pause_menu.is_paused:
                            mode_pop = not mode_pop
                            message = f"Modo: {'POP' if mode_pop else 'DROP'}"
                        elif event.key == pygame.K_m and not pause_menu.is_paused:
                            game_mode = "IA" if game_mode == "PvP" else "PvP"
                            board = PopOutBoard()
                            prev_board_state = (board.mask_p1, board.mask_p2)
                            mode_pop = False
                            message = f"Modo: {game_mode}"
                            game_over = False
                            winner = 0
                            ai_thinking = False
                            anim = AnimationState()
                        elif event.key == pygame.K_r and not pause_menu.is_paused:
                            board = PopOutBoard()
                            prev_board_state = (board.mask_p1, board.mask_p2)
                            mode_pop = False
                            message = "Novo jogo iniciado"
                            game_over = False
                            winner = 0
                            ai_thinking = False
                            anim = AnimationState()

                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game_over and not pause_menu.is_paused:
                        if game_mode == "IA" and board.current_player == 2:
                            continue

                        col = _column_from_mouse(event.pos[0])
                        if col is None:
                            continue

                        move = _encode_move(col, mode_pop)
                        if move not in board.legal_moves():
                            message = "Jogada inválida!"
                            continue

                        mover = board.current_player
                        board.apply_move(move)
                        
                        # Inicia animação para a peça colocada
                        # Encontra a peça que foi colocada
                        for r in range(ROWS):
                            bit = 1 << (col * 7 + r)
                            if mover == 1 and board.mask_p1 & bit and not (prev_board_state[0] & bit):
                                anim.add_piece_animation(r, col)
                            elif mover == 2 and board.mask_p2 & bit and not (prev_board_state[1] & bit):
                                anim.add_piece_animation(r, col)
                        
                        prev_board_state = (board.mask_p1, board.mask_p2)

                        winner = evaluate_after_move(board, mover=mover)
                        if winner:
                            game_over = True
                            message = f"Jogador {winner} venceu!"
                        else:
                            message = ""
                            ai_thinking = False
                            ai_timer = ai_delay if game_mode == "IA" else 0

                # Lógica da IA (só executa se não pausado)
                if running and not game_over and not pause_menu.is_paused and game_mode == "IA" and board.current_player == 2:
                    if not ai_thinking:
                        ai_timer -= dt
                        if ai_timer <= 0:
                            ai_thinking = True
                    
                    if ai_thinking:
                        ai_move = ai.run(board, iterations=ai_iterations)
                        ai_mover = board.current_player
                        board.apply_move(ai_move)
                        
                        # Inicia animação para a peça da IA
                        for r in range(ROWS):
                            ai_col = ai_move if ai_move < 7 else ai_move - 7
                            bit = 1 << (ai_col * 7 + r)
                            if ai_mover == 2 and board.mask_p2 & bit and not (prev_board_state[1] & bit):
                                anim.add_piece_animation(r, ai_col)
                        
                        prev_board_state = (board.mask_p1, board.mask_p2)
                        
                        winner = evaluate_after_move(board, mover=ai_mover)
                        if winner:
                            game_over = True
                            message = f"Jogador {winner} venceu!"
                        else:
                            message = "IA jogou"
                        ai_thinking = False

                # Renderiza
                _draw_board(screen, board, font, small_font, mode_pop, message, hover_col, game_over, winner, game_mode, anim, ai_thinking)
                
                # Renderiza menu de pausa se ativo
                if pause_menu.is_paused:
                    _draw_pause_menu(screen, font, small_font, pause_menu)
                
                pygame.display.flip()
            
            except KeyboardInterrupt:
                print("\n⏹️  Jogo interrompido pelo utilizador")
                running = False
            except Exception as loop_error:
                print(f"❌ Erro no loop de jogo: {type(loop_error).__name__}: {loop_error}")
                running = False

        pygame.quit()
    
    except pygame.error as e:
        print(f"❌ Erro pygame: {e}", file=sys.stderr)
        sys.exit(1)
    except ImportError as e:
        print(f"❌ Módulo não encontrado: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro inesperado: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Cleanup seguro
        try:
            pygame.quit()
        except:
            pass


if __name__ == "__main__":
    launch_gui()
