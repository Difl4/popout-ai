"""MCTS with inter-turn tree reuse.

Core idea
---------
A vanilla MCTS call discards its entire search tree after every move:

    run() → build N-node tree → choose move M → throw everything away

This wastes the pre-computed knowledge about positions the opponent is
likely to play into.  Tree reuse preserves the surviving subtree:

    Turn 1 │ run(board)
           │   → builds N-node tree, chooses move M
           │   → _root advanced to children[M]          (≈0.35·N visits)
    Turn 2 │ run(board, opponent_move=O)
           │   → _root advanced to children[O]          (≈0.12·N visits)
           │   → N new iterations on top of inherited tree
           │   → chooses move M2, _root advanced to children[M2]
    ...

Why inherited statistics remain valid
--------------------------------------
Each node stores ``value_sum`` from its *mover's* perspective.
``backpropagate`` alternates rewards (r, 1-r, r, …) as it climbs the
tree — the same alternation that a fresh search would apply.  Adding new
simulations to a pre-seeded subtree is therefore self-consistent: no
re-normalisation is needed.

Classes
-------
_TreeReuseMixin
    Pure state-management mixin.  Must appear *before* the concrete MCTS
    class in the MRO so its run() shadows BaseMCTS.run().
ReuseUCT
    Tree reuse on top of StandardUCT (~9 k iter/s).
"""

from __future__ import annotations

from typing import Optional

from src.engine.standard.bitboard import PopOutBoard
from src.mcts.standard.base import BaseMCTS, MCTSNode
from src.mcts.standard.uct_standard import StandardUCT


class _TreeReuseMixin:
    """Mixin that adds inter-turn tree reuse to any BaseMCTS subclass.

    State
    -----
    _root : MCTSNode | None
        Root of the currently cached subtree, or None when no tree is
        available (start of game, unexpected opponent move, after reset).

    MRO requirement
    ---------------
    This mixin must be listed *before* the concrete engine class::

        class ReuseUCT(_TreeReuseMixin, StandardUCT): ...

    Doing so makes Python's MRO resolve ``run`` to this mixin first,
    while ``expand``, ``simulate``, and ``backpropagate`` are still
    provided by the concrete subclass.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)   # cooperative __init__ chain
        self._root: Optional[MCTSNode] = None

    # ── public helpers ────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Discard the cached tree.

        Call at the start of every new game so no state from the
        previous game bleeds into the next.
        """
        self._root = None

    @property
    def cached_visits(self) -> int:
        """Visit count in the cached root node (0 when cache is empty)."""
        return 0 if self._root is None else self._root.visits

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
            Current board state *after* the opponent has played.
        iterations : int
            Number of *new* MCTS iterations to run on top of any
            inherited tree.
        opponent_move : int | None, keyword-only
            Move the opponent just played (integer 0-13).  When given,
            the cached subtree is advanced past that move before
            searching.  When omitted (first turn, or opponent's move is
            unknown), a fresh tree is created.

        Returns
        -------
        int
            Best move integer in [0, 13].
        """
        # ── Step 1: advance root past the opponent's reply ────────────────
        #
        # Before starting the new search we need to "fast-forward" the
        # cached tree by two plies: (a) our previous move (already done
        # at the end of the previous run() call) and (b) the opponent's
        # reply received now.
        #
        # Three cases:
        #   A. opponent_move given AND it was already explored in our tree
        #      → inherit that subtree (the happy path).
        #   B. opponent_move given BUT the move was never explored
        #      → opponent surprised us; start fresh.
        #   C. opponent_move not given (first turn or unknown)
        #      → we don't know where to advance; start fresh.

        if self._root is not None:
            if opponent_move is not None:
                candidate = self._root.children.get(opponent_move)
                if candidate is not None:
                    # Case A — inherit the subtree.
                    self._root = candidate
                    self._root.parent = None  # detach: old tree → GC-eligible
                else:
                    # Case B — unexpected opponent move.
                    self._root = None
            else:
                # Case C — no opponent move supplied.
                self._root = None

        # ── Step 2: create a fresh root when nothing was inherited ────────
        if self._root is None:
            self._root = MCTSNode(state=board.clone())
            if not self._root.untried_moves:
                raise ValueError("Estado sem jogadas legais.")
        else:
            # Sync the board reference.  The game is deterministic so the
            # states are already identical; this is a defensive measure.
            self._root.state = board.clone()

        # ── Step 3: run MCTS iterations ───────────────────────────────────
        #
        # Bind methods to locals once — avoids repeated attribute lookups
        # inside the hot loop (measurable at high iteration counts).
        root      = self._root
        _select   = self.select
        _expand   = self.expand
        _simulate = self.simulate
        _backprop = self.backpropagate

        for _ in range(iterations):
            leaf   = _select(root)
            child  = _expand(leaf)
            reward = _simulate(child)
            _backprop(child, reward)

        # ── Step 4: choose the move with the most visits ──────────────────
        #
        # "Most visits" (not highest Q) is the standard MCTS decision rule:
        # it is more robust because visit count is less noisy than Q when
        # the budget is limited.
        if not root.children:
            # Defensive: no expansion happened (e.g. isolated terminal node).
            fallback = root.untried_moves[0] if root.untried_moves else -1
            self._root = None
            return fallback

        best_move = max(root.children, key=lambda m: root.children[m].visits)

        # ── Step 5: advance root to our chosen move ───────────────────────
        #
        # After the caller applies best_move and the opponent replies,
        # the next run() call will advance _root past that reply.
        chosen = root.children.get(best_move)
        self._root = chosen
        if self._root is not None:
            self._root.parent = None  # detach: GC the branches we won't visit

        return best_move


# ── Concrete reuse engine ─────────────────────────────────────────────────────

class ReuseUCT(_TreeReuseMixin, StandardUCT):
    """StandardUCT with inter-turn tree reuse (~9 k iter/s).

    Identical to StandardUCT except the search tree is preserved between
    consecutive calls to run().  Pass ``opponent_move`` to activate
    reuse; omit it for a stateless call equivalent to StandardUCT.

    Game-loop usage::

        agent = ReuseUCT(seed=42)
        agent.reset()                                      # new game

        # Turn 1 — no cached tree yet
        move = agent.run(board, 1_000)
        board.apply_move(move)

        # Opponent replies
        opp = opponent.run(board, 500)
        board.apply_move(opp)

        # Turn 2 — inherit the subtree after opponent's move
        move = agent.run(board, 1_000, opponent_move=opp)
        board.apply_move(move)
        ...

        agent.reset()                                      # next game
    """

    def __init__(
        self,
        exploration_c: float = 1.414,
        rollout_depth: int = 30,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            exploration_c=exploration_c,
            rollout_depth=rollout_depth,
            seed=seed,
        )
