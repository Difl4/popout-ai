"""Visual constants and font creation for the PopOut GUI."""

from __future__ import annotations

import pygame

from src.engine.standard.bitboard import COLS, ROWS

CELL_SIZE = 100
BOARD_WIDTH = COLS * CELL_SIZE
BOARD_HEIGHT = ROWS * CELL_SIZE
HUD_TOP_HEIGHT = 80
HUD_BOTTOM_HEIGHT = 140
WINDOW_WIDTH = BOARD_WIDTH
WINDOW_HEIGHT = HUD_TOP_HEIGHT + BOARD_HEIGHT + HUD_BOTTOM_HEIGHT

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


def create_fonts() -> dict:
    """Create and return game fonts. Must be called after pygame.init()."""
    return {
        "font": pygame.font.SysFont("arial", 32, bold=True),
        "small_font": pygame.font.SysFont("arial", 20),
        "bold_24": pygame.font.SysFont("arial", 24, bold=True),
        "bold_48": pygame.font.SysFont("arial", 48, bold=True),
        "font_18": pygame.font.SysFont("arial", 18),
        "font_18_bold": pygame.font.SysFont("arial", 18, bold=True),
    }
