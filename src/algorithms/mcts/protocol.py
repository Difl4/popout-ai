"""MCTSEngine protocol — the single interface all MCTS engines must satisfy.

Depending on this abstraction (instead of concrete classes) follows the
Dependency Inversion Principle: callers declare what they need (run()), not
what engine they get.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ...engine.bitboard import PopOutBoard


@runtime_checkable
class MCTSEngine(Protocol):
    """Structural interface for any MCTS engine.

    Both BaseMCTS subclasses and FlatNumbaMCTS satisfy this protocol
    without explicit inheritance — checked via isinstance() at runtime.
    """

    def run(self, board: "PopOutBoard", iterations: int = 10_000) -> int:
        """Return the best move integer (0-13) for the given board state."""
        ...
