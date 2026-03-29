"""Central configuration — single source of truth for game parameters.

Both Python and Numba engines import constants from here.
"""

BOARD_WIDTH: int = 7    # number of columns
BOARD_HEIGHT: int = 6   # number of rows
WIN_CONDITION: int = 4  # pieces in a row required to win
COL_SIZE: int = 7       # bits allocated per column in bitboard encoding
