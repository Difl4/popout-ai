"""Engine package — exposes all public symbols from standard/ and optimized/ subpackages."""

from src.engine.standard.bitboard import PopOutBoard, ROWS, COLS, COL_SIZE
from src.engine.standard.rules import (
    has_won,
    check_winner_for_player,
    evaluate_after_move,
    board_signature,
    is_threefold_repetition,
    is_draw,
)

__all__ = [
    "PopOutBoard", "ROWS", "COLS", "COL_SIZE",
    "has_won", "check_winner_for_player", "evaluate_after_move",
    "board_signature", "is_threefold_repetition", "is_draw",
]
