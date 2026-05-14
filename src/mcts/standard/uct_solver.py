"""MCTS-Solver: MCTS variant with game-theoretic proof propagation and minimax distance.

══════════════════════════════════════════════════════════════════════════════
OVERVIEW
══════════════════════════════════════════════════════════════════════════════

Standard MCTS estimates position quality through random rollouts and visits.
MCTS-Solver extends this by *proving* whether positions are forced wins, losses,
or draws — and propagating those proofs up the tree.  Once the root is proven,
no more iterations are needed.

The algorithm still runs the 4 classic MCTS phases (Select → Expand →
Simulate → Backpropagate), but adds a 5th logical step after backpropagation:
proof propagation (_propagate_status), which travels up the tree marking nodes
whose outcomes are now mathematically certain.

══════════════════════════════════════════════════════════════════════════════
NODE STATUS  (always from the perspective of state.current_player)
══════════════════════════════════════════════════════════════════════════════

  STATUS_UNKNOWN (0) — outcome not yet proved; MCTS stats drive exploration
  STATUS_WIN     (1) — the player to move has a forced win from this position
  STATUS_LOSS    (2) — the player to move is in a forced loss
  STATUS_DRAW    (3) — forced draw (neither player can do better)

Important: status is relative to whoever is about to move in that node's
state.  A child with STATUS_LOSS is *good* for the parent — the parent's move
led to a position where the *opponent* is in a forced loss.

══════════════════════════════════════════════════════════════════════════════
DISTANCE  (minimax distance to terminal)
══════════════════════════════════════════════════════════════════════════════

  Terminal node (leaf)     → distance = 0
  Proven WIN  (OR  node)   → distance = min(distance of LOSS children) + 1
                             (win as fast as possible)
  Proven LOSS (AND node)   → distance = max(distance of WIN  children) + 1
                             (lose as slowly as possible)

In an OR node the current player can *choose* the best child, so we pick the
shortest path to a win.  In an AND node the opponent controls the choice, so
we assume they will pick the path that makes us lose fastest; we can only
delay by choosing moves that maximise the depth before that happens.

══════════════════════════════════════════════════════════════════════════════
PROOF LOGIC (how a node becomes proven)
══════════════════════════════════════════════════════════════════════════════

A node N is proven WIN  if at least one child has STATUS_LOSS.
  → N's player can *choose* that move, so the win is forced.

A node N is proven LOSS if every child (and there are no untried moves left)
  has STATUS_WIN.
  → No matter what N's player does, the opponent is in a winning position.

A node N is proven DRAW if every child is proven and none is LOSS or WIN
  (all DRAW), or if at least one child is DRAW and none is LOSS.
  → N's player can force at least a draw.

These rules mirror the AND/OR tree structure of a two-player game:
  • Current player node = OR  node (one good child suffices for WIN)
  • After the move, opponent node = AND node (all must be bad for a LOSS)

══════════════════════════════════════════════════════════════════════════════
SELECTION PRIORITY (what best_child returns during tree traversal)
══════════════════════════════════════════════════════════════════════════════

  1. If any child has STATUS_LOSS → go there (fastest proven win)
  2. If all children are proven and none is LOSS → delay loss (max distance)
  3. Otherwise → UCB1 on STATUS_UNKNOWN children (standard MCTS exploration)

══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.mcts.standard.base import BaseMCTS, Move
from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import board_signature, evaluate_after_move

# ── Status constants ──────────────────────────────────────────────────────────
STATUS_UNKNOWN: int = 0
STATUS_WIN: int = 1
STATUS_LOSS: int = 2
STATUS_DRAW: int = 3


# ── SolverNode ────────────────────────────────────────────────────────────────

@dataclass
class SolverNode:
    """MCTS node extended with a proof status and minimax distance.

    Attributes
    ----------
    state            : Board position stored at this node.
    parent           : Parent node (None for the root).
    move_from_parent : The move that was played to reach this node.
    mover            : The player ID who *made* the move arriving here
                       (i.e. state.current_player *before* the move was applied).
                       Used to compute rollout rewards from the perspective of
                       the player who expanded this child.
    visits           : Number of times this node has been visited.
    value_sum        : Accumulated reward from all visits (used to compute Q).
    children         : Dict mapping Move → child SolverNode.
    untried_moves    : Moves that have not yet been expanded into children.
                       Populated in __post_init__ from state.legal_moves().
    terminal_winner  : Player ID of the winner if this is a terminal game state,
                       or 0 if the game is still going.
    status           : Proof status (STATUS_* constant).
    distance         : Minimax distance to the proven terminal (0 if unknown).
    """

    state: PopOutBoard
    parent: Optional["SolverNode"] = None
    move_from_parent: Optional[Move] = None
    mover: Optional[int] = None
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[Move, "SolverNode"] = field(default_factory=dict)
    untried_moves: List[Move] = field(default_factory=list)
    terminal_winner: int = 0
    status: int = STATUS_UNKNOWN
    distance: int = 0

    def __post_init__(self) -> None:
        """Initialise untried_moves and immediately set status for terminal nodes."""
        self.untried_moves = self.state.legal_moves()

        if self.terminal_winner != 0:
            # The game is already over.  Determine WIN/LOSS from the perspective
            # of state.current_player (the player who would move *next*, i.e. the
            # one who did NOT make the winning/losing move).
            #
            # Example: player 1 pops a piece and that causes player 2 to get
            # 4-in-a-row (a "popout win for player 2").  After the move is
            # applied, state.current_player is player 1 again.  terminal_winner
            # is player 2.  Since terminal_winner != current_player → LOSS for
            # the player about to move (player 1).
            if self.terminal_winner == self.state.current_player:
                self.status = STATUS_WIN
            else:
                self.status = STATUS_LOSS
            self.distance = 0

        elif not self.untried_moves:
            # No legal moves and no winner → stalemate / draw.
            self.status = STATUS_DRAW
            self.distance = 0

    @property
    def is_terminal(self) -> bool:
        """True if this node represents a finished game (no more moves possible)."""
        return self.terminal_winner != 0 or (
            self.status == STATUS_DRAW and not self.untried_moves and not self.children
        )

    def q(self) -> float:
        """Mean reward (exploitation value) for UCB1 computation."""
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


# ── SolverMCTS ────────────────────────────────────────────────────────────────

class SolverMCTS(BaseMCTS):
    """MCTS with game-theoretic proof propagation (MCTS-Solver).

    Extends BaseMCTS by overriding the four MCTS phases plus adding a proof
    propagation step.  The key difference from vanilla MCTS:

    • Nodes carry a proof STATUS (WIN/LOSS/DRAW/UNKNOWN).
    • After every backpropagation, proofs are pushed upward through ancestors.
    • Selection skips subtrees that are already proven.
    • Final move selection uses proofs first, visit counts only as fallback.
    • run() exits early once the root itself is proven.
    """

    # ── Phase 1 — Expansion ───────────────────────────────────────────────────

    def expand(self, node: SolverNode) -> SolverNode:  # type: ignore[override]
        """Add one new child to *node* by trying a previously untried move.

        Uses an O(1) swap-pop instead of list.remove() to pick and remove the
        chosen move from untried_moves without shifting the rest of the list.

        Returns the new child node, or *node* itself if it is terminal or fully
        expanded (no untried moves remain).
        """
        if node.is_terminal or not node.untried_moves:
            return node

        # O(1) swap-pop: swap chosen element with last, then pop last.
        # Avoids the O(n) cost of list.remove() or list.pop(i).
        idx = self.random.randrange(len(node.untried_moves))
        move = node.untried_moves[idx]
        node.untried_moves[idx] = node.untried_moves[-1]
        node.untried_moves.pop()

        next_state = node.state.clone()
        mover = next_state.current_player        # player who is about to move
        next_state.apply_move(move)
        winner = evaluate_after_move(next_state, mover=mover)  # 0 or player id

        child = SolverNode(
            state=next_state,
            parent=node,
            move_from_parent=move,
            mover=mover,
            terminal_winner=winner,
        )
        node.children[move] = child
        return child

    # ── Phase 1b — Selection ──────────────────────────────────────────────────

    def select(self, node: SolverNode) -> SolverNode:  # type: ignore[override]
        """Walk the tree from *node* downward, returning a node to expand.

        Descends as long as:
          • The node is not terminal (game not over).
          • The node's status is UNKNOWN (not yet proven — proven subtrees are
            already handled and we should not waste iterations there).
          • All moves have been tried (untried_moves is empty).
          • There is at least one child to recurse into.

        When any condition fails, the current node is the selection target and
        expand() will be called on it next.
        """
        current = node
        while (
            not current.is_terminal
            and current.status == STATUS_UNKNOWN
            and not current.untried_moves
            and current.children
        ):
            current = self.best_child(current)
        return current

    def best_child(self, node: SolverNode) -> SolverNode:  # type: ignore[override]
        """Choose the best child of *node* according to solver priority rules.

        Priority (from highest to lowest):
          1. A child with STATUS_LOSS  → this is a forced win for us; pick the
             one with *smallest* distance to finish the game fastest.
          2. All children proven, none LOSS  → we cannot improve our outcome;
             pick the child with *largest* distance to delay the inevitable.
          3. Normal MCTS  → apply UCB1 on STATUS_UNKNOWN children only.
             (Proven-bad children are excluded from exploration to avoid
             wasting iterations in already-resolved subtrees.)
        """
        children = list(node.children.values())

        # Priority 1: any child where the opponent is forced to lose → take it
        # immediately, choosing the shortest path (smallest distance).
        loss_children = [c for c in children if c.status == STATUS_LOSS]
        if loss_children:
            return min(loss_children, key=lambda c: c.distance)

        # Priority 2: fully explored with no winning option → delay loss.
        # This is reached only when no child is LOSS and all children are proven
        # (no UNKNOWN children and no untried moves).
        unknown_children = [c for c in children if c.status == STATUS_UNKNOWN]
        if not unknown_children and not node.untried_moves:
            return max(children, key=lambda c: c.distance)

        # Priority 3: standard UCB1 on unknown children.
        # If there are no unknown children at all (but there *are* untried moves),
        # fall back to all children — this is an unusual edge case.
        candidates = unknown_children if unknown_children else children
        log_n = math.log(max(1, node.visits))
        best_score = -float("inf")
        best = candidates[0]
        for child in candidates:
            if child.visits == 0:
                score = float("inf")   # unvisited nodes have infinite priority
            else:
                score = child.q() + self.exploration_c * math.sqrt(log_n / child.visits)
            if score > best_score:
                best_score = score
                best = child
        return best

    # ── Phase 2 — Simulation (rollout) ────────────────────────────────────────

    def simulate(self, node: SolverNode) -> float:  # type: ignore[override]
        """Run a pure random rollout from *node* and return a reward in [0, 1].

        Returns
        -------
        1.0  if the player who made the move arriving at *node* (node.mover) wins.
        0.0  if that player loses.
        0.5  for a draw or if the rollout depth limit is reached without a result.

        Why pure random (no tactical look-ahead)?
        ─────────────────────────────────────────
        The rollout's job is to estimate *positional quality* — how often does
        random play from this position lead to a win?  Injecting tactical moves
        (e.g. always playing a 1-move win when available) would bias that estimate:
        positions with immediate threats look artificially strong regardless of
        their deeper structure.

        For MCTS-Solver this matters doubly: the tree already *proves* wins and
        losses through STATUS propagation.  The rollout only guides exploration
        for UNKNOWN nodes.  Corrupting those statistics with tactical shortcuts
        can interfere with the solver's proof convergence.

        Rollout termination conditions (in order checked each step):
          1. Same position repeated ≥ 3 times      → draw (0.5)  [Rule 3: AI declares]
          2. No legal moves                        → draw (0.5)
          3. A player wins after a move            → 1.0 or 0.0
          4. rollout_depth steps exhausted         → draw (0.5)
          Note: full board (Rule 2) is NOT a draw — pop moves may still be available.

        Note on sig_counter:
          board_signature() produces a compact hash of the board state.
          We use a Counter (dict) for O(1) insertion and lookup; checking
          all values via any(count >= 3 ...) is O(distinct_positions) which
          is small in practice.
        """
        state = node.state.clone()
        # initial_mover: the player we track the outcome for.
        # node.mover is the player who moved to reach this node.
        # If mover is None (e.g. the root), we infer it as the other player.
        initial_mover = node.mover if node.mover is not None else (3 - state.current_player)

        # If this node is already terminal, return immediately without rolling out.
        if node.terminal_winner != 0:
            return 1.0 if node.terminal_winner == initial_mover else 0.0

        sig_counter: Counter = Counter([board_signature(state)])
        choice = self.random.choice

        for _ in range(self.rollout_depth):
            # Rule 2: full board is NOT an automatic draw — player can still pop.
            if any(count >= 3 for count in sig_counter.values()):
                return 0.5   # Rule 3: AI always declares draw on threefold

            legal = state.legal_moves()
            if not legal:
                return 0.5

            curr_p = state.current_player
            state.apply_move(choice(legal))   # random move

            winner = evaluate_after_move(state, mover=curr_p)
            if winner != 0:
                return 1.0 if winner == initial_mover else 0.0

            sig_counter[board_signature(state)] += 1

        return 0.5   # depth exhausted — treat as draw

    # ── Phase 3 — Backpropagation + Proof propagation ─────────────────────────

    def backpropagate(self, node: SolverNode, reward: float) -> None:  # type: ignore[override]
        """Update visit counts / value sums AND propagate proofs upward.

        Step 1 (inherited from BaseMCTS.backpropagate):
          Walk from *node* to the root, incrementing visits and adding reward.
          The reward flips at each level (1 - r) because the two players have
          opposite objectives: what is good for one is bad for the other.

        Step 2 (_propagate_status):
          Walk from *node*'s parent upward, recomputing STATUS and distance for
          each ancestor until no change occurs.  Stops early because status
          transitions are monotone (UNKNOWN → WIN/LOSS/DRAW, never back), so
          once an ancestor is unchanged, all further ancestors are also unchanged.
        """
        super().backpropagate(node, reward)
        self._propagate_status(node)

    def _propagate_status(self, node: SolverNode) -> None:
        """Walk from *node*'s parent to the root, updating proof status.

        Stops as soon as _update_node_status returns False (no change occurred).
        This early-exit is valid because:
          • STATUS transitions are one-way (UNKNOWN → proven).
          • Distance only changes if the underlying children change.
          • If neither changed for a node, no ancestor can be affected.
        """
        current = node.parent
        while current is not None:
            if not self._update_node_status(current):
                break
            current = current.parent

    def _update_node_status(self, node: SolverNode) -> bool:
        """Recompute *node*'s STATUS and distance from its current children.

        This implements the AND/OR proof rules:

          OR  (current player chooses):
            WIN  ← ∃ child with STATUS_LOSS   (we can choose that move)
            LOSS ← ∀ children STATUS_WIN  AND no untried moves
            DRAW ← ∀ children proven AND ∃ child STATUS_DRAW AND none STATUS_LOSS

          Distance:
            WIN  → min distance among LOSS children + 1  (win fast)
            LOSS → max distance among WIN  children + 1  (lose slow)
            DRAW → 0 (distance not meaningful for draws here)

        Returns True if STATUS or distance changed (caller should continue
        propagating upward), False otherwise.
        """
        if node.is_terminal:
            return False   # terminal nodes are already at their final status

        children = list(node.children.values())
        if not children:
            return False   # no children yet — nothing to infer

        # ── WIN: at least one child where the opponent is forced to lose ──────
        # We only need ONE such child (OR logic: we choose the best move).
        loss_children = [c for c in children if c.status == STATUS_LOSS]
        if loss_children:
            new_status = STATUS_WIN
            # Pick the shortest path to our win (minimax: minimize distance).
            new_distance = min(c.distance for c in loss_children) + 1
            if node.status == new_status and node.distance == new_distance:
                return False
            node.status = new_status
            node.distance = new_distance
            return True

        # ── LOSS or DRAW: only conclusive once ALL moves have been explored ───
        # If there are still untried moves or unknown children, we cannot yet
        # conclude that the position is losing or drawn — a future child might
        # turn out to be a LOSS (good for us).
        fully_explored = (
            not node.untried_moves
            and all(c.status != STATUS_UNKNOWN for c in children)
        )
        if not fully_explored:
            return False

        win_children  = [c for c in children if c.status == STATUS_WIN]
        draw_children = [c for c in children if c.status == STATUS_DRAW]

        if draw_children:
            # At least one child leads to a draw and we can choose it.
            new_status   = STATUS_DRAW
            new_distance = 0
        elif win_children:
            # Every child is a WIN for the opponent.
            # Under PopOut Rule 2: if the board is currently full, the player
            # to move can declare draw instead of accepting a forced loss.
            if node.state.is_full():
                new_status   = STATUS_DRAW
                new_distance = 0
            else:
                new_status   = STATUS_LOSS
                new_distance = max(c.distance for c in win_children) + 1
        else:
            return False   # edge case: no win/draw/loss children (shouldn't happen)

        if node.status == new_status and node.distance == new_distance:
            return False
        node.status   = new_status
        node.distance = new_distance
        return True

    # ── Final move selection ───────────────────────────────────────────────────

    def get_best_move(self, root: SolverNode) -> Move:
        """Choose the best move from *root* after all iterations are done.

        Priority (from highest to lowest):

          1. Proven WIN  → pick the child with STATUS_LOSS and smallest distance
                           (win as quickly as possible).
          2. Proven DRAW → pick any child with STATUS_DRAW
                           (secure the draw rather than risk a loss).
          3. Proven LOSS → pick the child with STATUS_WIN and largest distance
                           (delay the inevitable as long as possible; opponent
                           might make a mistake and the game ends in draw by
                           repetition before the proof line is reached).
          4. Unknown     → standard MCTS fallback: most-visited child
                           (visit count is the most reliable proxy for value
                           when no proof exists).
        """
        # Case 1: proven win — finish fast.
        loss_children = [
            (m, c) for m, c in root.children.items() if c.status == STATUS_LOSS
        ]
        if loss_children:
            return min(loss_children, key=lambda mc: mc[1].distance)[0]

        # Case 2: proven draw — take it.
        if root.status == STATUS_DRAW:
            draw_children = [
                (m, c) for m, c in root.children.items() if c.status == STATUS_DRAW
            ]
            if draw_children:
                return draw_children[0][0]

        # Case 3: proven loss — delay.
        if root.status == STATUS_LOSS:
            return max(root.children.items(), key=lambda mc: mc[1].distance)[0]

        # Case 4: no proof available — most visited child (standard MCTS).
        best_move, _ = max(root.children.items(), key=lambda mc: mc[1].visits)
        return best_move

    # ── Main entry point ──────────────────────────────────────────────────────

    def _run_iterations(self, root: SolverNode, iterations: int) -> None:
        """Run up to *iterations* MCTS-Solver iterations from *root* in-place."""
        for _ in range(iterations):
            if root.status != STATUS_UNKNOWN and not root.untried_moves:
                break
            leaf   = self.select(root)
            child  = self.expand(leaf)
            reward = self.simulate(child)
            self.backpropagate(child, reward)

    def run(self, root_state: PopOutBoard, iterations: int = 10_000) -> Move:
        """Run MCTS-Solver for up to *iterations* iterations and return the best move.

        Each iteration is the classic four-phase loop:
          select → expand → simulate → backpropagate (+proof propagation)

        Exits only once the root is proven AND all immediate root moves have been
        tried at least once.  This guarantees every possible first move is
        evaluated before stopping, so get_best_move compares all winning lines
        and reliably returns the one with the shortest minimax distance.
        """
        root = SolverNode(state=root_state.clone())
        if not root.untried_moves:
            raise ValueError("No legal moves in the given state.")

        self._run_iterations(root, iterations)
        return self.get_best_move(root)


# ── ReuseSolverMCTS ───────────────────────────────────────────────────────────

class ReuseSolverMCTS(SolverMCTS):
    """SolverMCTS with inter-turn tree reuse (pure Python reference implementation).

    Preserves the SolverNode tree between calls to run().  After each search
    the root is advanced to children[best_move]; on the next call it is
    advanced to children[opponent_move] if that child was already explored,
    inheriting its visit counts and proof status.

    Usage
    -----
    ::

        agent = ReuseSolverMCTS(seed=42)
        agent.reset()

        move = agent.run(board, 1_000)
        board.apply_move(move)

        opp = opponent.run(board, 1_000)
        board.apply_move(opp)

        move = agent.run(board, 1_000, opponent_move=opp)
        ...
        agent.reset()  # next game
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._root: Optional[SolverNode] = None

    def reset(self) -> None:
        """Discard the cached tree. Call at the start of every new game."""
        self._root = None

    @property
    def cached_visits(self) -> int:
        """Visits in the cached root node (0 when cache is empty)."""
        return self._root.visits if self._root is not None else 0

    def run(  # type: ignore[override]
        self,
        root_state: PopOutBoard,
        iterations: int = 10_000,
        *,
        opponent_move: Optional[Move] = None,
    ) -> Move:
        """Return the best move, optionally reusing the tree from the previous turn.

        Parameters
        ----------
        root_state : PopOutBoard
            Current board state (after the opponent has played).
        iterations : int
            Number of new MCTS-Solver iterations to run on top of the cached tree.
        opponent_move : Move | None, keyword-only
            Move the opponent just played.  When given and the child was explored,
            the cached subtree is reused.  When omitted, a fresh search is done.
        """
        # Advance root through the opponent's move (if we have a cached tree).
        if self._root is not None and opponent_move is not None:
            child = self._root.children.get(opponent_move)
            if child is not None:
                self._root = child
                self._root.parent = None   # detach to let GC reclaim old tree
            else:
                self._root = None          # opponent played an unexplored move
        else:
            self._root = None              # no cache or no opponent_move → fresh

        if self._root is None:
            self._root = SolverNode(state=root_state.clone())

        self._run_iterations(self._root, iterations)
        best_move = self.get_best_move(self._root)

        # Advance root to the chosen child for the next turn.
        next_root = self._root.children.get(best_move)
        if next_root is not None:
            next_root.parent = None
        self._root = next_root

        return best_move
