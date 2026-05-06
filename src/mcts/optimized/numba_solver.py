"""Numba-accelerated MCTS-Solver engines for PopOut.

Two acceleration tiers, mirroring the standard/optimized split for plain MCTS:

NumbaSolverMCTS
    Hybrid: Python tree + proof propagation, Numba simulation and expansion.
    Inherits SolverMCTS — proof logic is identical to the pure Python version,
    making it the correctness reference for FlatNumbaSolverMCTS.
    Expected throughput: ~8 000–15 000 iterations/s.

FlatNumbaSolverMCTS
    Full JIT: entire loop (select, expand, simulate, backprop, proof propagation)
    compiled by Numba.  Nodes live in pre-allocated numpy flat arrays.
    Expected throughput: ~30 000–60 000 iterations/s.
    Exits early once the root position is proven (status != UNKNOWN).

Call warmup_solver() once at startup to pay the JIT compilation cost upfront.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.engine.standard.bitboard import PopOutBoard
from src.mcts.standard.uct_solver import (
    SolverMCTS,
    SolverNode,
    STATUS_WIN,
    STATUS_LOSS,
    STATUS_DRAW,
    STATUS_UNKNOWN,
)
from src.mcts.optimized.numba_search import (
    nb_expand_step,
    nb_simulate,
    nb_solver_mcts_run,
)


# ── NumbaSolverMCTS ───────────────────────────────────────────────────────────

class _NumbaSolverNode(SolverNode):
    """SolverNode whose __post_init__ defers legal_moves() to nb_expand_step.

    NumbaSolverMCTS.expand() populates untried_moves via the Numba kernel,
    avoiding a redundant Python-side legal_moves() round-trip.
    Terminal and draw status are still set here so proof propagation works
    correctly before any children are added.
    """

    def __post_init__(self) -> None:
        self.untried_moves = []  # filled by NumbaSolverMCTS.expand()

        if self.terminal_winner != 0:
            if self.terminal_winner == self.state.current_player:
                self.status = STATUS_WIN
            else:
                self.status = STATUS_LOSS
            self.distance = 0


class NumbaSolverMCTS(SolverMCTS):
    """MCTS-Solver with Numba-accelerated expansion and simulation.

    Only expand() and simulate() are replaced with Numba kernels.  All proof
    logic (_propagate_status, _update_node_status, best_child, get_best_move)
    is inherited from SolverMCTS unchanged, so outputs can be compared
    directly against the pure Python solver to verify correctness.

    Call warmup_solver() once before benchmarking to pay JIT cost upfront.
    """

    def __init__(
        self,
        exploration_c: float = 1.414,
        rollout_depth: int = 150,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(exploration_c=exploration_c, rollout_depth=rollout_depth, seed=seed)
        if seed is not None:
            np.random.seed(seed)

    def expand(self, node: SolverNode) -> SolverNode:  # type: ignore[override]
        """Add one child using nb_expand_step (apply + evaluate + legal_moves in one JIT call)."""
        if node.is_terminal or not node.untried_moves:
            return node

        untried = node.untried_moves
        idx  = self.random.randrange(len(untried))
        move = untried[idx]
        untried[idx] = untried[-1]
        untried.pop()

        st = node.state
        m1, m2, cp, winner, legal_arr, n_legal = nb_expand_step(
            np.int64(st.mask_p1),
            np.int64(st.mask_p2),
            np.int32(st.current_player),
            np.int32(move),
        )

        next_state = PopOutBoard(int(m1), int(m2), int(cp))
        mover      = st.current_player

        child = _NumbaSolverNode(
            state=next_state,
            parent=node,
            move_from_parent=move,
            mover=mover,
            terminal_winner=int(winner),
        )
        child.untried_moves = [int(legal_arr[i]) for i in range(n_legal)]

        # No winner but no legal moves either → stalemate / draw.
        if int(winner) == 0 and n_legal == 0:
            child.status   = STATUS_DRAW
            child.distance = 0

        node.children[move] = child
        return child

    def simulate(self, node: SolverNode) -> float:  # type: ignore[override]
        """Run a Numba-compiled random rollout from *node*."""
        state         = node.state
        initial_mover = node.mover if node.mover is not None else (3 - state.current_player)
        return float(
            nb_simulate(
                np.int64(state.mask_p1),
                np.int64(state.mask_p2),
                np.int32(state.current_player),
                np.int32(initial_mover),
                np.int32(node.terminal_winner),
                np.int32(self.rollout_depth),
            )
        )


# ── FlatNumbaSolverMCTS ───────────────────────────────────────────────────────

class FlatNumbaSolverMCTS:
    """MCTS-Solver with the complete search loop JIT-compiled by Numba.

    Architecture
    ------------
    Extends FlatNumbaMCTS with two extra flat arrays:
      status[]   — proof status per node (UNKNOWN / WIN / LOSS / DRAW)
      distance[] — minimax distance to the nearest proven terminal

    A single call to nb_solver_mcts_run() runs all iterations in compiled code,
    paying the Python<->JIT crossing cost only once per search.  The loop exits
    early as soon as the root node is proven.

    Interface
    ---------
    Mirrors FlatNumbaMCTS.run()::

        move = FlatNumbaSolverMCTS().run(board, iterations=100_000)

    Note: not a BaseMCTS subclass — satisfies MCTSEngine protocol instead.
    """

    def __init__(
        self,
        max_nodes: int = 200_000,
        exploration_c: float = 1.414,
        rollout_depth: int = 150,
        seed: Optional[int] = None,
    ) -> None:
        self.exploration_c = float(exploration_c)
        self.rollout_depth = rollout_depth
        if seed is not None:
            np.random.seed(seed)

        N = max_nodes
        self._visits     = np.zeros(N,          dtype=np.int32)
        self._value      = np.zeros(N,          dtype=np.float64)
        self._parent     = np.full(N, -1,       dtype=np.int32)
        self._move_fp    = np.empty(N,          dtype=np.int32)
        self._mover_arr  = np.zeros(N,          dtype=np.int32)
        self._mp1        = np.empty(N,          dtype=np.int64)
        self._mp2        = np.empty(N,          dtype=np.int64)
        self._player_arr = np.empty(N,          dtype=np.int32)
        self._terminal   = np.zeros(N,          dtype=np.int32)
        self._children   = np.full((N, 14), -1, dtype=np.int32)
        self._n_children = np.zeros(N,          dtype=np.int32)
        self._untried    = np.empty((N, 14),    dtype=np.int32)
        self._n_untried  = np.zeros(N,          dtype=np.int32)
        self._status     = np.zeros(N,          dtype=np.int32)
        self._distance   = np.zeros(N,          dtype=np.int32)

    def run(self, board: PopOutBoard, iterations: int = 10_000) -> int:
        """Return the best move integer (0-13) for the given board state."""
        return int(nb_solver_mcts_run(
            np.int64(board.mask_p1),
            np.int64(board.mask_p2),
            np.int32(board.current_player),
            np.int32(iterations),
            self.exploration_c,
            np.int32(self.rollout_depth),
            self._visits, self._value, self._parent, self._move_fp,
            self._mover_arr, self._mp1, self._mp2, self._player_arr,
            self._terminal, self._children, self._n_children,
            self._untried, self._n_untried,
            self._status, self._distance,
        ))


# ── warmup ────────────────────────────────────────────────────────────────────

def warmup_solver() -> None:
    """Trigger JIT compilation of all solver @njit functions once.

    Call this once at startup (~1-3 s); subsequent calls return immediately
    because Numba caches compiled functions to disk.

    Usage::

        from src.mcts.optimized.numba_solver import warmup_solver
        warmup_solver()
        move = FlatNumbaSolverMCTS().run(board, iterations=100_000)
    """
    dummy = PopOutBoard()
    NumbaSolverMCTS(rollout_depth=10).run(dummy, iterations=1)
    FlatNumbaSolverMCTS(rollout_depth=10).run(dummy, iterations=1)
