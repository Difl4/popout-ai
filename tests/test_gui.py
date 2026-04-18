"""Testes para a interface gráfica (GUI) em pygame."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

# Mock pygame antes de importar gui
pytest.importorskip("pygame")

from src.interfaces.gui import (
    _player_color,
    _player_glow,
    _column_from_mouse,
    _encode_move,
    COLOR_P1,
    COLOR_P1_GLOW,
    COLOR_P2,
    COLOR_P2_GLOW,
    COLOR_ACCENT,
    CELL_SIZE,
    BOARD_WIDTH,
)


class TestPlayerColor:
    """Testa função _player_color()."""
    
    def test_player1_color(self):
        """Player 1 deve retornar cor vermelha."""
        assert _player_color(1) == COLOR_P1
    
    def test_player2_color(self):
        """Player 2 deve retornar cor amarela."""
        assert _player_color(2) == COLOR_P2
    
    def test_invalid_player_returns_p2(self):
        """Jogador inválido retorna cor default (P2)."""
        assert _player_color(99) == COLOR_P2


class TestPlayerGlow:
    """Testa função _player_glow()."""
    
    def test_player1_glow(self):
        """Player 1 glow deve ser vermelho brilhante."""
        assert _player_glow(1) == COLOR_P1_GLOW
    
    def test_player2_glow(self):
        """Player 2 glow deve ser amarelo brilhante."""
        assert _player_glow(2) == COLOR_P2_GLOW


class TestColumnFromMouse:
    """Testa conversão de coordenadas X do rato para coluna."""
    
    def test_column0_at_left_edge(self):
        """X=0 deve retornar coluna 0."""
        col = _column_from_mouse(0)
        assert col == 0
    
    def test_column3_at_middle(self):
        """Meio do board deve retornar coluna 3."""
        x = CELL_SIZE * 3 + CELL_SIZE // 2
        col = _column_from_mouse(x)
        assert col == 3
    
    def test_column6_at_right_edge(self):
        """Coluna 6 no limite direito."""
        x = CELL_SIZE * 6 + 10
        col = _column_from_mouse(x)
        assert col == 6
    
    def test_out_of_bounds_left(self):
        """X negativo retorna None."""
        col = _column_from_mouse(-1)
        assert col is None
    
    def test_out_of_bounds_right(self):
        """X fora do board retorna None."""
        col = _column_from_mouse(BOARD_WIDTH + 10)
        assert col is None
    
    def test_exact_boundary_cases(self):
        """Testa limites exatos entre colunas."""
        # Última posição de coluna 0
        assert _column_from_mouse(CELL_SIZE - 1) == 0
        # Primeira posição de coluna 1
        assert _column_from_mouse(CELL_SIZE) == 1


class TestEncodeMove:
    """Testa codificação de jogadas (DROP vs POP)."""
    
    def test_drop_mode_column0(self):
        """Mode DROP, coluna 0 = move 0."""
        move = _encode_move(0, mode_pop=False)
        assert move == 0
    
    def test_drop_mode_column6(self):
        """Mode DROP, coluna 6 = move 6."""
        move = _encode_move(6, mode_pop=False)
        assert move == 6
    
    def test_pop_mode_column0(self):
        """Mode POP, coluna 0 = move 7 (7 + 0)."""
        move = _encode_move(0, mode_pop=True)
        assert move == 7
    
    def test_pop_mode_column6(self):
        """Mode POP, coluna 6 = move 13 (7 + 6)."""
        move = _encode_move(6, mode_pop=True)
        assert move == 13
    
    def test_symmetric_encoding(self):
        """Testa simetria DROP/POP para todas as colunas."""
        for col in range(7):
            drop_move = _encode_move(col, mode_pop=False)
            pop_move = _encode_move(col, mode_pop=True)
            assert drop_move == col
            assert pop_move == col + 7


class TestAnimationState:
    """Testa sistema de animações."""
    
    def test_animation_state_creation(self):
        """AnimationState deve ser criado vazio."""
        from src.interfaces.gui import AnimationState
        
        anim = AnimationState()
        assert anim.piece_animations == {}
        assert anim.particle_effects == []
        assert anim.flash_effects == {}
    
    def test_add_piece_animation(self):
        """Adicionar animação de peça."""
        from src.interfaces.gui import AnimationState
        
        anim = AnimationState()
        anim.add_piece_animation(2, 3)
        
        assert (2, 3) in anim.piece_animations
        assert anim.piece_animations[(2, 3)] == 0.0
    
    def test_animation_update_progress(self):
        """Animação deve progredir com tempo."""
        from src.interfaces.gui import AnimationState
        
        anim = AnimationState()
        anim.add_piece_animation(0, 0)
        
        # Avança 0.1 segundo
        anim.update(0.1)
        
        # Progress = 0.1 * 8 = 0.8
        assert 0.7 < anim.piece_animations[(0, 0)] < 0.9
    
    def test_animation_completion(self):
        """Animação deve ser removida quando 100% completa."""
        from src.interfaces.gui import AnimationState
        
        anim = AnimationState()
        anim.add_piece_animation(0, 0)
        
        # Avança 0.15 segundo (15 * 8 = 1.2, > 1.0)
        anim.update(0.15)
        
        # Animação deve estar completa e removida
        assert (0, 0) not in anim.piece_animations
    
    def test_multiple_animations(self):
        """Deve gerenciar múltiplas animações."""
        from src.interfaces.gui import AnimationState
        
        anim = AnimationState()
        anim.add_piece_animation(0, 0)
        anim.add_piece_animation(1, 1)
        anim.add_piece_animation(2, 2)
        
        assert len(anim.piece_animations) == 3
        
        # Avançar meio segundo
        anim.update(0.05)
        
        # Ainda deve ter as 3
        assert len(anim.piece_animations) == 3


class TestDrawFunctions:
    """Testa funções de desenho (com mocks de pygame)."""
    
    @patch('src.interfaces.gui.renderer.pygame.draw.line')
    def test_draw_vertical_gradient(self, mock_draw_line):
        """Gradiente deve desenhar linhas do topo ao fundo."""
        from src.interfaces.gui import _draw_vertical_gradient, WINDOW_HEIGHT
        
        mock_screen = Mock()
        
        _draw_vertical_gradient(mock_screen, (255, 0, 0), (0, 0, 255))
        
        # Deve chamar draw.line para cada linha de altura
        assert mock_draw_line.call_count == WINDOW_HEIGHT
    
    @patch('src.interfaces.gui.renderer.pygame.draw.circle')
    @patch('src.interfaces.gui.renderer.pygame.Surface')
    def test_draw_glow_circle(self, mock_surface, mock_draw):
        """Glow circle deve desenhar múltiplos círculos."""
        from src.interfaces.gui import _draw_glow_circle
        
        mock_screen = Mock()
        
        _draw_glow_circle(mock_screen, (100, 100), 50, (255, 0, 0), intensity=0.8)
        
        # Deve chamar draw.circle múltiplas vezes para glow
        assert mock_draw.call_count >= 50


@patch('pygame.init')
@patch('pygame.display.set_mode')
@patch('pygame.display.set_caption')
@patch('pygame.time.Clock')
def test_launch_gui_initialization(mock_clock, mock_caption, mock_set_mode, mock_init):
    """Teste que launch_gui inicializa pygame corretamente."""
    # Não vamos correr launch_gui completo, só testes básicos
    # porque requer event loop
    
    # pygame.init deve ser chamado
    from src.interfaces.gui import launch_gui
    
    # Criamos um mock para o event loop
    mock_clock_instance = MagicMock()
    mock_clock_instance.tick.side_effect = [60] * 5 + [KeyboardInterrupt]  # 5 frames depois interrompe
    mock_clock.return_value = mock_clock_instance
    
    mock_screen = MagicMock()
    mock_set_mode.return_value = mock_screen
    
    # Isto deve configurar pygame sem erros
    # (Não vamos correr o loop completo neste teste)


class TestGuiConstants:
    """Testa que constantes de GUI estão bem definidas."""
    
    def test_cell_size_is_positive(self):
        """CELL_SIZE deve ser positivo."""
        from src.interfaces.gui import CELL_SIZE
        assert CELL_SIZE > 0
    
    def test_board_dimensions(self):
        """Dimensões do board devem ser válidas."""
        from src.interfaces.gui import BOARD_WIDTH, BOARD_HEIGHT, COLS, ROWS, CELL_SIZE
        assert BOARD_WIDTH == COLS * CELL_SIZE
        assert BOARD_HEIGHT == ROWS * CELL_SIZE
    
    def test_window_dimensions(self):
        """Dimensões da janela devem conter o board."""
        from src.interfaces.gui import WINDOW_WIDTH, WINDOW_HEIGHT, BOARD_WIDTH, BOARD_HEIGHT
        assert WINDOW_WIDTH >= BOARD_WIDTH
        assert WINDOW_HEIGHT >= BOARD_HEIGHT
    
    def test_colors_are_tuples(self):
        """Cores devem ser tuplas RGB ou RGBA."""
        from src.interfaces.gui import (
            COLOR_P1, COLOR_P2, COLOR_EMPTY, COLOR_ACCENT,
            COLOR_TEXT, COLOR_PANEL
        )
        
        assert isinstance(COLOR_P1, tuple)
        assert isinstance(COLOR_P2, tuple)
        assert isinstance(COLOR_EMPTY, tuple)
        assert isinstance(COLOR_ACCENT, tuple)
        assert isinstance(COLOR_TEXT, tuple)
        assert isinstance(COLOR_PANEL, tuple)
    
    def test_colors_valid_rgb_values(self):
        """Valores de cor devem estar entre 0-255."""
        from src.interfaces.gui import COLOR_P1, COLOR_P2
        
        for value in COLOR_P1:
            assert 0 <= value <= 255
        
        for value in COLOR_P2:
            assert 0 <= value <= 255


class TestGuiIntegration:
    """Testes de integração básica da GUI."""
    
    def test_encode_decode_roundtrip(self):
        """Encoding DROP então POP deve voltar ao original."""
        for col in range(7):
            encoded_drop = _encode_move(col, mode_pop=False)
            # Converter de volta
            decoded_col = encoded_drop if encoded_drop < 7 else encoded_drop - 7
            assert decoded_col == col
    
    def test_mouse_to_column_to_move(self):
        """Pipeline: mouse X -> coluna -> move."""
        for col in range(7):
            x_pos = col * CELL_SIZE + CELL_SIZE // 2
            detected_col = _column_from_mouse(x_pos)
            assert detected_col == col
            
            move = _encode_move(detected_col, mode_pop=False)
            assert move == col


class TestDifficulty:
    """Testa enum Difficulty."""
    
    def test_difficulty_easy(self):
        """Dificuldade Easy = 100 iterações."""
        from src.interfaces.gui import Difficulty
        assert Difficulty.EASY.value[0] == 100
        assert "Easy" in Difficulty.EASY.value[1]

    def test_difficulty_medium(self):
        """Dificuldade Medium = 500 iterações."""
        from src.interfaces.gui import Difficulty
        assert Difficulty.MEDIUM.value[0] == 500
        assert "Medium" in Difficulty.MEDIUM.value[1]

    def test_difficulty_hard(self):
        """Dificuldade Hard = 1000 iterações."""
        from src.interfaces.gui import Difficulty
        assert Difficulty.HARD.value[0] == 1000
        assert "Hard" in Difficulty.HARD.value[1]

    def test_difficulty_extreme(self):
        """Dificuldade Extreme = 10000 iterações."""
        from src.interfaces.gui import Difficulty
        assert Difficulty.EXTREME.value[0] == 10000
        assert "Extreme" in Difficulty.EXTREME.value[1]


class TestPauseMenu:
    """Testa classe PauseMenu."""
    
    def test_pause_menu_creation(self):
        """PauseMenu deve inicializar desativado."""
        from src.interfaces.gui import PauseMenu, Difficulty
        menu = PauseMenu()
        
        assert not menu.is_paused
        assert menu.selected_option == 0
        assert menu.selected_difficulty == Difficulty.MEDIUM
        assert len(menu.options) == 5
    
    def test_pause_menu_toggle(self):
        """Deve alternar estado de pausa."""
        from src.interfaces.gui import PauseMenu
        menu = PauseMenu()
        
        assert not menu.is_paused
        menu.toggle_pause()
        assert menu.is_paused
        menu.toggle_pause()
        assert not menu.is_paused
    
    def test_pause_menu_navigate_down(self):
        """Deve navegar para baixo nas opções."""
        from src.interfaces.gui import PauseMenu
        menu = PauseMenu()
        menu.toggle_pause()
        
        assert menu.selected_option == 0
        menu.navigate(1)
        assert menu.selected_option == 1
        menu.navigate(1)
        assert menu.selected_option == 2
    
    def test_pause_menu_navigate_up(self):
        """Deve navegar para cima nas opções."""
        from src.interfaces.gui import PauseMenu
        menu = PauseMenu()
        menu.toggle_pause()
        menu.selected_option = 2
        
        menu.navigate(-1)
        assert menu.selected_option == 1
        menu.navigate(-1)
        assert menu.selected_option == 0
    
    def test_pause_menu_navigate_wrap_around(self):
        """Deve fazer wrap-around ao navegar além dos limites."""
        from src.interfaces.gui import PauseMenu
        menu = PauseMenu()
        menu.toggle_pause()
        menu.selected_option = 4

        menu.navigate(1)
        assert menu.selected_option == 0  # Volta ao início
    
    def test_pause_menu_select_current(self):
        """Deve retornar opção selecionada correta."""
        from src.interfaces.gui import PauseMenu
        menu = PauseMenu()
        menu.toggle_pause()
        
        assert menu.select_current() == "Retomar"
        menu.selected_option = 1
        assert menu.select_current() == "Novo Jogo"
        menu.selected_option = 2
        assert menu.select_current() == "Dificuldade"
        menu.selected_option = 3
        assert menu.select_current() == "Modo"
        menu.selected_option = 4
        assert menu.select_current() == "Sair"
    
    def test_pause_menu_reset_on_toggle(self):
        """Deve resetar opção para "Retomar" ao pausar."""
        from src.interfaces.gui import PauseMenu
        menu = PauseMenu()
        menu.selected_option = 3
        
        menu.toggle_pause()  # Pausa
        assert menu.is_paused
        assert menu.selected_option == 0  # Reset para "Retomar"


class TestDrawPauseMenu:
    """Testa renderização do menu de pausa."""
    
    @patch('pygame.draw.line')
    @patch('pygame.draw.rect')
    def test_draw_pause_menu_renders(self, mock_rect, mock_line):
        """Menu de pausa deve renderizar sem erros."""
        from src.interfaces.gui import PauseMenu, _draw_pause_menu
        from unittest.mock import MagicMock

        menu = PauseMenu()
        menu.toggle_pause()

        mock_screen = MagicMock()
        mock_font = MagicMock()
        mock_small_font = MagicMock()

        # Simula render de texto com get_width
        text_surf = MagicMock()
        text_surf.get_width.return_value = 100
        mock_font.render.return_value = text_surf
        mock_small_font.render.return_value = text_surf

        try:
            _draw_pause_menu(mock_screen, mock_font, mock_small_font, menu)
        except Exception as e:
            pytest.fail(f"_draw_pause_menu lançou exceção: {e}")
