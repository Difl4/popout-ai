"""Numba-accelerated MCTS-Solver engines for PopOut, with optional tree reuse.

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

from collections import deque
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
            np.int32(1),   # n_initial: fresh search
        ))


# ── ReuseFlatNumbaSolverMCTS ──────────────────────────────────────────────────

class ReuseFlatNumbaSolverMCTS:
    """FlatNumbaSolverMCTS with inter-turn tree reuse.

    Architecture
    ------------
    The flat numpy arrays that store the search tree are preserved between
    calls to run().  After each search the Python wrapper locates the
    chosen child's subtree inside the flat arrays and compacts it back to
    index 0 via a BFS traversal.  The next search call passes the compacted
    size as ``n_initial`` to ``nb_solver_mcts_run``, which skips root
    initialisation and continues building on top of the inherited nodes.

    Why this is especially valuable for the Solver
    -----------------------------------------------
    Plain tree reuse inherits visit counts and Q-values.  Solver reuse
    also inherits **proven WIN / LOSS / DRAW nodes**: positions that were
    formally solved in a previous turn do not need to be re-proved.  For
    endgame positions where many moves lead to forced outcomes, this can
    save the majority of computation.

    Memory
    ------
    The flat arrays (max_nodes entries) are allocated once in __init__ and
    reused across calls and across games.  The compaction step is O(k)
    where k is the subtree size (~N/196 of the previous iteration budget).

    Usage
    -----
    ::

        from src.mcts.optimized.numba_mcts import warmup
        from src.mcts.optimized.numba_solver import ReuseFlatNumbaSolverMCTS

        warmup()                                      # compile Numba kernels
        agent = ReuseFlatNumbaSolverMCTS(seed=42)
        agent.reset()                                 # new game

        move = agent.run(board, 10_000)
        board.apply_move(move)

        opp = opponent.run(board, 10_000)
        board.apply_move(opp)

        move = agent.run(board, 10_000, opponent_move=opp)  # reuses tree
        ...
        agent.reset()                                 # next game
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

        # Index of the best-move child after the last run() call.
        # Used to advance into the opponent's subtree on the next call.
        self._best_child_idx: int = -1

    # ── public helpers ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Discard the cached tree.  Call at the start of every new game."""
        self._best_child_idx = -1

    @property
    def cached_visits(self) -> int:
        """Visits in the cached best-child node (0 when cache is empty)."""
        if self._best_child_idx < 0:
            return 0
        return int(self._visits[self._best_child_idx])

    # ── main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        board: PopOutBoard,
        iterations: int = 10_000,
        *,
        opponent_move: Optional[int] = None,
    ) -> int:
        """Return the best move, optionally reusing the tree from the previous turn.

        Parameters
        ----------
        board : PopOutBoard
            Current board state after the opponent has played.
        iterations : int
            Number of new MCTS-Solver iterations to run.
        opponent_move : int | None, keyword-only
            Move the opponent just played (0-13).  When given, the cached
            subtree is compacted to index 0 and the kernel continues from
            there.  When omitted, a fresh search is performed.
        """
        n_initial = np.int32(1)   # default: fresh search

        # ── Step 1: find the new root in the flat arrays ──────────────────
        if self._best_child_idx >= 0 and opponent_move is not None:
            opp_idx = self._find_child_idx(self._best_child_idx, opponent_move)
            if opp_idx >= 0:
                # Compact the surviving subtree into slots 0..k-1.
                n_initial = np.int32(self._compact_subtree_to_root(opp_idx))
            # else: opponent played an unexpected move → fresh start (n_initial=1)

        # ── Step 2: run the Numba kernel ──────────────────────────────────
        best_move = int(nb_solver_mcts_run(
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
            n_initial,
        ))

        # ── Step 3: cache best-move child index for the next turn ─────────
        self._best_child_idx = self._find_child_idx(0, best_move)

        return best_move

    # ── private helpers ───────────────────────────────────────────────────────

    def _find_child_idx(self, parent_idx: int, move: int) -> int:
        """Return the flat-array index of the child reached by *move* from
        *parent_idx*, or -1 if that child does not exist in the tree."""
        nc = int(self._n_children[parent_idx])
        for i in range(nc):
            cid = int(self._children[parent_idx, i])
            if int(self._move_fp[cid]) == move:
                return cid
        return -1

    def _compact_subtree_to_root(self, old_root_idx: int) -> int:
        """BFS-compact the subtree at *old_root_idx* into array slots 0..k-1.

        All parent / children references are remapped to the new indices.
        Returns k, the number of nodes in the compacted subtree.

        The root of the compacted tree is at index 0 with parent = -1.
        """
        # ── BFS traversal: collect nodes in breadth-first order ───────────
        old_to_new: dict[int, int] = {}
        order: list[int] = []
        queue: deque[int] = deque([old_root_idx])

        while queue:
            old = queue.popleft()
            if old in old_to_new:
                continue
            old_to_new[old] = len(order)
            order.append(old)
            nc = int(self._n_children[old])
            for i in range(nc):
                child = int(self._children[old, i])
                if child >= 0 and child not in old_to_new:
                    queue.append(child)

        k = len(order)

        # ── copy fields into temporary arrays (avoids in-place aliasing) ──
        tmp_visits    = np.empty(k,       dtype=np.int32)
        tmp_value     = np.empty(k,       dtype=np.float64)
        tmp_parent    = np.full(k, -1,    dtype=np.int32)
        tmp_move_fp   = np.empty(k,       dtype=np.int32)
        tmp_mover     = np.empty(k,       dtype=np.int32)
        tmp_mp1       = np.empty(k,       dtype=np.int64)
        tmp_mp2       = np.empty(k,       dtype=np.int64)
        tmp_player    = np.empty(k,       dtype=np.int32)
        tmp_terminal  = np.empty(k,       dtype=np.int32)
        tmp_nc        = np.zeros(k,       dtype=np.int32)
        tmp_nu        = np.empty(k,       dtype=np.int32)
        tmp_status    = np.empty(k,       dtype=np.int32)
        tmp_distance  = np.empty(k,       dtype=np.int32)
        tmp_children  = np.full((k, 14), -1, dtype=np.int32)
        tmp_untried   = np.empty((k, 14),    dtype=np.int32)

        for old_idx, new_idx in old_to_new.items():
            tmp_visits[new_idx]   = self._visits[old_idx]
            tmp_value[new_idx]    = self._value[old_idx]
            tmp_move_fp[new_idx]  = self._move_fp[old_idx]
            tmp_mover[new_idx]    = self._mover_arr[old_idx]
            tmp_mp1[new_idx]      = self._mp1[old_idx]
            tmp_mp2[new_idx]      = self._mp2[old_idx]
            tmp_player[new_idx]   = self._player_arr[old_idx]
            tmp_terminal[new_idx] = self._terminal[old_idx]
            tmp_status[new_idx]   = self._status[old_idx]
            tmp_distance[new_idx] = self._distance[old_idx]

            # Remap parent (-1 for the new root)
            old_p = int(self._parent[old_idx])
            tmp_parent[new_idx] = old_to_new[old_p] if old_p in old_to_new else -1

            # Remap children
            nc = int(self._n_children[old_idx])
            tmp_nc[new_idx] = nc
            for i in range(nc):
                old_c = int(self._children[old_idx, i])
                tmp_children[new_idx, i] = old_to_new.get(old_c, -1)

            # Copy untried moves verbatim (move integers, not indices)
            nu = int(self._n_untried[old_idx])
            tmp_nu[new_idx] = nu
            for i in range(nu):
                tmp_untried[new_idx, i] = self._untried[old_idx, i]

        # ── write compacted subtree back into slots 0..k-1 ────────────────
        self._visits[:k]       = tmp_visits
        self._value[:k]        = tmp_value
        self._parent[:k]       = tmp_parent
        self._move_fp[:k]      = tmp_move_fp
        self._mover_arr[:k]    = tmp_mover
        self._mp1[:k]          = tmp_mp1
        self._mp2[:k]          = tmp_mp2
        self._player_arr[:k]   = tmp_player
        self._terminal[:k]     = tmp_terminal
        self._n_children[:k]   = tmp_nc
        self._n_untried[:k]    = tmp_nu
        self._status[:k]       = tmp_status
        self._distance[:k]     = tmp_distance
        self._children[:k, :]  = tmp_children
        self._untried[:k, :]   = tmp_untried

        return k


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
    agent = ReuseFlatNumbaSolverMCTS(rollout_depth=10)
    agent.run(dummy, iterations=1)                          # fresh path (n_initial=1)
    # reuse path: compact a 1-node tree and call with n_initial=1 (same code path)
    agent.reset()
