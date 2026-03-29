"""Animation state, difficulty, and coordinate utilities for the PopOut GUI."""

from __future__ import annotations

from enum import Enum

from src.interfaces.gui.assets import CELL_SIZE, BOARD_WIDTH


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
        for key in list(self.piece_animations.keys()):
            self.piece_animations[key] += dt * 8
            if self.piece_animations[key] >= 1.0:
                del self.piece_animations[key]

        for effect in self.particle_effects[:]:
            effect['time'] += dt
            if effect['time'] >= effect['lifetime']:
                self.particle_effects.remove(effect)

        for key in list(self.flash_effects.keys()):
            self.flash_effects[key] -= dt * 3
            if self.flash_effects[key] <= 0:
                del self.flash_effects[key]


class Difficulty(Enum):
    """Níveis de dificuldade para IA."""
    EASY          = (100,     "Fácil (100)",          "standard")
    MEDIUM        = (500,     "Médio (500)",           "standard")
    HARD          = (1000,    "Difícil (1000)",        "standard")
    EXTREME       = (10000,   "Extremo (2000)",        "standard")
    EXTREME_NUMBA = (100_000, "Extremo Numba (100k)",  "flat_numba")

    @property
    def iterations(self) -> int:
        return self.value[0]

    @property
    def label(self) -> str:
        return self.value[1]

    @property
    def engine_type(self) -> str:
        return self.value[2]


class CoordinateMapper:
    """Converts pixel coordinates to board coordinates."""

    @staticmethod
    def col_from_x(x_pos: int) -> int | None:
        """Convert mouse x position to board column.

        Args:
            x_pos (int): X pixel position.

        Returns:
            int | None: Column (0-6) or None if outside board.
        """
        if x_pos < 0 or x_pos >= BOARD_WIDTH:
            return None
        return x_pos // CELL_SIZE
