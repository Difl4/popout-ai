"""GUI components (PauseMenu) for the PopOut GUI."""

from __future__ import annotations

import pygame

from src.interfaces.gui.state import Difficulty


class PauseMenu:
    """Gerencia estado do menu de pausa."""

    def __init__(self) -> None:
        self.is_paused = False
        self.selected_option = 0
        self.selected_difficulty = Difficulty.MEDIUM
        self.arena_p1_difficulty = Difficulty.HARD
        self.arena_p2_difficulty = Difficulty.SOLVER
        self.human_player = 1
        self.options = ["Retomar", "Novo Jogo", "Dificuldade", "Lado", "Modo", "IA P1", "IA P2", "Sair"]

    def get_options(self, game_mode: str) -> list[str]:
        """Return only the pause-menu options that apply to the current mode."""
        mode_options = ["Retomar", "Novo Jogo", "Modo"]
        if game_mode == "IA":
            mode_options.extend(["Dificuldade", "Lado"])
        elif game_mode == "Arena":
            mode_options.extend(["IA P1", "IA P2"])
        mode_options.append("Sair")
        return mode_options

    def toggle_pause(self) -> None:
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.selected_option = 0

    def navigate(self, direction: int, game_mode: str = "PvP") -> None:
        if self.is_paused:
            self.selected_option = (self.selected_option + direction) % len(self.get_options(game_mode))

    def select_current(self, game_mode: str = "PvP") -> str:
        return self.get_options(game_mode)[self.selected_option]

    def handle_event(self, event: pygame.event.Event) -> str | None:
        """Process a pygame event and return the selected option name if confirmed.

        Args:
            event: A pygame event.

        Returns:
            str | None: Selected option name on RETURN, otherwise None.
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.navigate(-1)
            elif event.key == pygame.K_DOWN:
                self.navigate(1)
            elif event.key == pygame.K_RETURN:
                return self.select_current()
        return None

    def draw(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font, game_mode: str = "PvP") -> None:
        """Draw the pause menu overlay.

        Args:
            screen: Pygame surface to draw on.
            font: Primary font.
            small_font: Secondary (smaller) font.
            game_mode: Current game mode string (e.g. "PvP" or "IA").
        """
        from src.interfaces.gui.renderer import _draw_pause_menu
        _draw_pause_menu(screen, font, small_font, self, game_mode)
