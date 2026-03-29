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
