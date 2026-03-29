"""Numba-accelerated MCTS engines for PopOut.

Single Responsibility: this module owns the two accelerated MCTS classes
(NumbaMCTS and FlatNumbaMCTS) and the warmup helper. All JIT kernel
functions live in src/engine/optimized/numba_rules.py and
src/algorithms/mcts/optimized/numba_search.py.

Classes
-------
NumbaMCTS
    Python tree traversal + Numba simulation rollout.  Inherits BaseMCTS.
FlatNumbaMCTS
    Entire MCTS loop inside a single @njit call (100k+ iter/s).
    Satisfies MCTSEngine protocol without BaseMCTS inheritance.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.engine.standard.bitboard import PopOutBoard
from src.algorithms.mcts.standard.base import BaseMCTS, MCTSNode
from src.algorithms.mcts.optimized.numba_search import (
    nb_expand_step,
    nb_mcts_run,
    nb_simulate,
)


# ── NumbaMCTS ─────────────────────────────────────────────────────────────────

class _NumbaNode(MCTSNode):
    """MCTSNode whose __post_init__ skips the Python legal_moves() call.

    NumbaMCTS.expand() populates untried_moves via nb_expand_step instead,
    avoiding a redundant Python<->bitboard round-trip.
    """

    def __post_init__(self) -> None:
        self.untried_moves = []   # filled by NumbaMCTS.expand


class NumbaMCTS(BaseMCTS):
    """MCTS with Numba-accelerated rollouts and Python-side micro-optimisations.

    Improvements over BaseMCTS
    --------------------------
    simulate  -- replaced by nb_simulate (@njit, pure int64 arithmetic).
    expand    -- single nb_expand_step call (apply + evaluate + legal_moves)
                + O(1) swap-pop removal from untried_moves list.
    best_child -- log(node.visits) cached once per call (BaseMCTS recomputes
                 it per child); q() inlined; exploration_c bound as local.

    Call warmup() once before benchmarking to pay the JIT cost upfront.
    """

    def __init__(
        self,
        exploration_c: float = 1.414,
        rollout_depth: int = 150,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(exploration_c=exploration_c, rollout_depth=rollout_depth, seed=seed)
        if seed is not None:
            np.random.seed(seed)   # seed Numba's RNG (separate from Python's)

    def best_child(self, node: MCTSNode) -> MCTSNode:
        c = self.exploration_c
        log_n = math.log(max(1, node.visits))
        best_score = -math.inf
        best = None
        for child in node.children.values():
            v = child.visits
            if v == 0:
                score = math.inf
            else:
                score = child.value_sum / v + c * math.sqrt(log_n / v)
            if score > best_score:
                best_score = score
                best = child
        return best

    def expand(self, node: MCTSNode) -> MCTSNode:
        if node.is_terminal or not node.untried_moves:
            return node

        untried = node.untried_moves
        idx = self.random.randrange(len(untried))
        move = untried[idx]
        untried[idx] = untried[-1]   # O(1) swap-pop
        untried.pop()

        st = node.state
        m1, m2, cp, winner, legal_arr, n_legal = nb_expand_step(
            np.int64(st.mask_p1),
            np.int64(st.mask_p2),
            np.int32(st.current_player),
            np.int32(move),
        )

        next_state = PopOutBoard(int(m1), int(m2), int(cp))
        mover = st.current_player
        child = _NumbaNode(
            state=next_state,
            parent=node,
            move_from_parent=move,
            mover=mover,
            terminal_winner=int(winner),
        )
        child.untried_moves = [int(legal_arr[i]) for i in range(n_legal)]
        node.children[move] = child
        return child

    def simulate(self, node: MCTSNode) -> float:
        state = node.state
        initial_mover = (
            node.mover if node.mover is not None else (3 - state.current_player)
        )
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


# ── FlatNumbaMCTS ─────────────────────────────────────────────────────────────

class FlatNumbaMCTS:
    """MCTS with the complete search loop JIT-compiled by Numba (100k+ iter/s).

    Architecture
    ------------
    Tree nodes live in pre-allocated numpy arrays indexed by node ID.
    A single call to nb_mcts_run() runs all iterations in compiled code,
    paying the Python<->JIT crossing cost only once per search.

    The arrays are allocated once in __init__ and reused across calls.
    Root slot (index 0) is re-initialised at the start of each run().

    Interface
    ---------
    Mirrors BaseMCTS.run()::

        move = FlatNumbaMCTS().run(board, iterations=100_000)

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
        self._visits     = np.zeros(N,        dtype=np.int32)
        self._value      = np.zeros(N,        dtype=np.float64)
        self._parent     = np.full(N, -1,     dtype=np.int32)
        self._move_fp    = np.empty(N,        dtype=np.int32)
        self._mover_arr  = np.zeros(N,        dtype=np.int32)
        self._mp1        = np.empty(N,        dtype=np.int64)
        self._mp2        = np.empty(N,        dtype=np.int64)
        self._player_arr = np.empty(N,        dtype=np.int32)
        self._terminal   = np.zeros(N,        dtype=np.int32)
        self._children   = np.full((N, 14), -1, dtype=np.int32)
        self._n_children = np.zeros(N,        dtype=np.int32)
        self._untried    = np.empty((N, 14),  dtype=np.int32)
        self._n_untried  = np.zeros(N,        dtype=np.int32)

    def run(self, board: PopOutBoard, iterations: int = 10_000) -> int:
        """Return the best move integer (0-13) for the given board state."""
        return int(nb_mcts_run(
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
        ))


# ── warmup ────────────────────────────────────────────────────────────────────

def warmup() -> None:
    """Trigger JIT compilation of all @njit functions once.

    Call this once at startup (~1-3 s); subsequent calls return immediately
    because Numba caches compiled functions to disk.

    Usage::

        from src.algorithms.mcts.optimized.numba_mcts import warmup, FlatNumbaMCTS
        warmup()
        move = FlatNumbaMCTS().run(board, iterations=100_000)
    """
    dummy = PopOutBoard()
    NumbaMCTS(rollout_depth=10).run(dummy, iterations=1)
    FlatNumbaMCTS(rollout_depth=10).run(dummy, iterations=1)
