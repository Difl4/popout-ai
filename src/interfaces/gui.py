"""Interface gráfica em pygame para jogar PopOut localmente (2 jogadores) - VERSÃO MELHORADA."""

from __future__ import annotations

import pygame

from ..algorithms.mcts.uct_standard import StandardUCT
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
    """Tracks active animations para peças e efeitos visuais."""
    def __init__(self):
        self.piece_animations: dict[tuple[int, int], float] = {}
        self.particle_effects: list[dict] = []
        self.flash_effects: dict[str, float] = {}
    
    def add_piece_animation(self, row: int, col: int):
        """Inicia animação de entrada para uma peça."""
        self.piece_animations[(row, col)] = 0.0
    
    def update(self, dt: float):
        """Atualiza todas as animações (dt em segundos)."""
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


def _player_color(player: int) -> tuple[int, int, int]:
    """Devolve a cor base associada ao jogador."""
    return COLOR_P1 if player == 1 else COLOR_P2


def _player_glow(player: int) -> tuple[int, int, int]:
    """Devolve a cor de glow (brilho) do jogador."""
    return COLOR_P1_GLOW if player == 1 else COLOR_P2_GLOW


def _draw_vertical_gradient(screen: pygame.Surface, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> None:
    """Desenha um gradiente vertical no fundo com mais qualidade."""
    for y in range(WINDOW_HEIGHT):
        ratio = y / max(1, WINDOW_HEIGHT - 1)
        r = int(top[0] * (1 - ratio) + bottom[0] * ratio)
        g = int(top[1] * (1 - ratio) + bottom[1] * ratio)
        b = int(top[2] * (1 - ratio) + bottom[2] * ratio)
        pygame.draw.line(screen, (r, g, b), (0, y), (WINDOW_WIDTH, y))


def _draw_glow_circle(screen: pygame.Surface, center: tuple[int, int], radius: int, color: tuple[int, int, int], intensity: float = 0.5) -> None:
    """Desenha uma aura de brilho ao redor de um círculo."""
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
    """Desenha uma peça com sombra, brilho e animação de chegada."""
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
    controls = "SPACE: DROP/POP | M: PvP/IA | R: Novo | ESC: Sair | CLIQUE: Jogar"
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
    """Converte coordenada X do rato para coluna 0..6, ou None fora do tabuleiro."""
    if x_pos < 0 or x_pos >= BOARD_WIDTH:
        return None
    return x_pos // CELL_SIZE


def _encode_move(column: int, mode_pop: bool) -> int:
    """Codifica jogada no formato do motor: drop(0..6) ou pop(7..13)."""
    return column + 7 if mode_pop else column


def launch_gui() -> None:
    """Inicia a janela pygame com interface melhorada e executa o ciclo principal."""
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

    # Matriz anterior para detetar mudanças
    prev_board_state = (board.mask_p1, board.mask_p2)

    running = True
    while running:
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
                    running = False
                elif event.key == pygame.K_SPACE and not game_over:
                    mode_pop = not mode_pop
                    message = f"Modo: {'POP' if mode_pop else 'DROP'}"
                elif event.key == pygame.K_m:
                    game_mode = "IA" if game_mode == "PvP" else "PvP"
                    board = PopOutBoard()
                    prev_board_state = (board.mask_p1, board.mask_p2)
                    mode_pop = False
                    message = f"Modo: {game_mode}"
                    game_over = False
                    winner = 0
                    ai_thinking = False
                    anim = AnimationState()
                elif event.key == pygame.K_r:
                    board = PopOutBoard()
                    prev_board_state = (board.mask_p1, board.mask_p2)
                    mode_pop = False
                    message = "Novo jogo iniciado"
                    game_over = False
                    winner = 0
                    ai_thinking = False
                    anim = AnimationState()

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not game_over:
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

        # Lógica da IA
        if running and not game_over and game_mode == "IA" and board.current_player == 2:
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
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    launch_gui()
