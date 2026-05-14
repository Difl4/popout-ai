"""MCTS-specific @njit search kernels — no Python objects cross the JIT boundary.

Single Responsibility: this module owns only the JIT-compiled MCTS search
logic.  Pure game mechanics live in src/engine/optimized/numba_rules.py.

Functions
---------
nb_expand_step          — apply move + evaluate + compute legal moves in one JIT call.
nb_simulate             — full random rollout in pure int64 arithmetic.
_nb_best_child_id       — UCT child selection on flat arrays.
nb_mcts_run             — complete MCTS loop (select, expand, simulate, backprop).

Solver extensions
-----------------
_nb_update_node_status  — recompute WIN/LOSS/DRAW status for one node from its children.
_nb_propagate_status    — walk from a node to the root propagating proof status.
_nb_best_child_solver_id — proof-aware child selection (priority: LOSS > delay > UCB1).
nb_solver_mcts_run      — complete MCTS-Solver loop with proof propagation and early exit.

Solver status constants (mirror uct_solver.py)
----------------------------------------------
_S_UNKNOWN = 0   STATUS_UNKNOWN
_S_WIN     = 1   STATUS_WIN
_S_LOSS    = 2   STATUS_LOSS
_S_DRAW    = 3   STATUS_DRAW
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from src.engine.optimized.numba_bitboard import nb_apply_move, nb_is_full, nb_legal_moves
from src.engine.optimized.numba_rules import nb_evaluate_after_move, nb_is_threefold_repetition

# Solver proof-status constants — must match uct_solver.STATUS_* values exactly.
_S_UNKNOWN = 0
_S_WIN     = 1
_S_LOSS    = 2
_S_DRAW    = 3


@njit(cache=True)
def nb_expand_step(
    mask_p1: np.int64,
    mask_p2: np.int64,
    current_player: np.int32,
    move: np.int32,
):
    """Apply a move, evaluate for a winner, and compute the resulting legal moves.

    Bundles three kernels into one JIT call to minimise Python<->Numba crossings.
    Returns: (new_mask_p1, new_mask_p2, new_player, winner, moves_array, n_moves)
    """
    mover = current_player
    new_m1, new_m2, new_cp = nb_apply_move(mask_p1, mask_p2, current_player, move)
    winner = nb_evaluate_after_move(new_m1, new_m2, mover)
    moves, n = nb_legal_moves(new_m1, new_m2, new_cp)
    return new_m1, new_m2, new_cp, winner, moves, n


@njit(cache=True)
def nb_simulate(
    mask_p1: np.int64,
    mask_p2: np.int64,
    current_player: np.int32,
    initial_mover: np.int32,
    terminal_winner: np.int32,
    rollout_depth: np.int32,
) -> np.float64:
    """Full random rollout in pure int64 arithmetic.

    Threefold repetition is detected via a linear scan over the history buffer
    (O(depth) per step — negligible for depth <= 50).

    Returns reward from initial_mover's perspective: 1.0 win / 0.5 draw / 0.0 loss.
    """
    if terminal_winner != 0:
        return np.float64(1.0) if terminal_winner == initial_mover else np.float64(0.0)

    hist_p1 = np.empty(rollout_depth + 1, dtype=np.int64)
    hist_p2 = np.empty(rollout_depth + 1, dtype=np.int64)
    hist_p1[0] = mask_p1
    hist_p2[0] = mask_p2
    hist_size = np.int32(1)

    for _ in range(rollout_depth):
        # Rule 2: full board is NOT an automatic draw — player can still pop.
        # nb_legal_moves returns only pop moves when full; the n_moves == 0
        # check below handles the truly stuck case.
        if nb_is_threefold_repetition(hist_p1, hist_p2, hist_size):
            return np.float64(0.5)   # Rule 3: AI always declares draw on threefold

        moves, n_moves = nb_legal_moves(mask_p1, mask_p2, current_player)
        if n_moves == 0:
            return np.float64(0.5)

        move = moves[np.random.randint(np.int32(0), n_moves)]
        curr_p = current_player
        mask_p1, mask_p2, current_player = nb_apply_move(mask_p1, mask_p2, current_player, move)

        winner = nb_evaluate_after_move(mask_p1, mask_p2, curr_p)
        if winner != 0:
            return np.float64(1.0) if winner == initial_mover else np.float64(0.0)

        hist_p1[hist_size] = mask_p1
        hist_p2[hist_size] = mask_p2
        hist_size += np.int32(1)

    return np.float64(0.5)


@njit(cache=True)
def _nb_best_child_id(
    node_id: np.int32,
    visits, value, children, n_children,
    exploration_c: np.float64,
) -> np.int32:
    """UCT child selection on flat arrays. Returns child node id."""
    nc = n_children[node_id]
    pv = visits[node_id]
    log_n = math.log(max(1.0, float(pv)))
    c = exploration_c
    best_score = -1e300
    best = np.int32(-1)
    for i in range(nc):
        cid = children[node_id, i]
        cv = visits[cid]
        if cv == 0:
            return cid
        score = value[cid] / float(cv) + c * math.sqrt(log_n / float(cv))
        if score > best_score:
            best_score = score
            best = cid
    return best


@njit(cache=True)
def nb_mcts_run(
    root_m1: np.int64,
    root_m2: np.int64,
    root_player: np.int32,
    iterations: np.int32,
    exploration_c: np.float64,
    rollout_depth: np.int32,
    visits, value, parent, move_fp, mover_arr,
    mp1, mp2, player_arr, terminal,
    children, n_children, untried, n_untried,
) -> np.int32:
    """Complete MCTS loop in Numba — select, expand, simulate, backpropagate.

    Arrays are pre-allocated by FlatNumbaMCTS and reused across calls.
    Only the root slot (index 0) is re-initialised on each call.
    """
    max_nodes = visits.shape[0]

    # Initialise root (node 0)
    mp1[0]        = root_m1
    mp2[0]        = root_m2
    player_arr[0] = root_player
    mover_arr[0]  = np.int32(3 - root_player)
    terminal[0]   = np.int32(0)
    visits[0]     = np.int32(0)
    value[0]      = np.float64(0.0)
    n_children[0] = np.int32(0)
    parent[0]     = np.int32(-1)
    legal, n_legal = nb_legal_moves(root_m1, root_m2, root_player)
    n_untried[0]  = n_legal
    for i in range(n_legal):
        untried[0, i] = legal[i]

    n_nodes = np.int32(1)

    for _ in range(iterations):
        if n_nodes >= max_nodes - 1:
            break

        # SELECT
        cur = np.int32(0)
        while (terminal[cur] == 0
               and n_untried[cur] == 0
               and n_children[cur] > 0):
            cur = _nb_best_child_id(cur, visits, value, children, n_children, exploration_c)

        # EXPAND
        if terminal[cur] == 0 and n_untried[cur] > 0:
            nu   = n_untried[cur]
            idx  = np.random.randint(np.int32(0), nu)
            move = untried[cur, idx]
            untried[cur, idx] = untried[cur, nu - 1]
            n_untried[cur] -= np.int32(1)

            pm1 = mp1[cur]
            pm2 = mp2[cur]
            ppl = player_arr[cur]
            mv  = ppl

            new_m1, new_m2, new_cp = nb_apply_move(pm1, pm2, ppl, move)
            win = nb_evaluate_after_move(new_m1, new_m2, np.int32(mv))

            cid = n_nodes
            n_nodes += np.int32(1)

            mp1[cid]        = new_m1
            mp2[cid]        = new_m2
            player_arr[cid] = new_cp
            parent[cid]     = cur
            move_fp[cid]    = move
            mover_arr[cid]  = np.int32(mv)
            terminal[cid]   = win
            visits[cid]     = np.int32(0)
            value[cid]      = np.float64(0.0)
            n_children[cid] = np.int32(0)

            if win == 0:
                leg, nl = nb_legal_moves(new_m1, new_m2, new_cp)
                n_untried[cid] = nl
                for i in range(nl):
                    untried[cid, i] = leg[i]
            else:
                n_untried[cid] = np.int32(0)

            children[cur, n_children[cur]] = cid
            n_children[cur] += np.int32(1)
            cur = cid

        # SIMULATE
        ini_mov = mover_arr[cur]
        if ini_mov == 0:
            ini_mov = np.int32(3 - player_arr[cur])
        reward = nb_simulate(
            mp1[cur], mp2[cur], player_arr[cur],
            ini_mov, terminal[cur], rollout_depth,
        )

        # BACKPROPAGATE
        r    = reward
        node = cur
        while node >= 0:
            visits[node] += np.int32(1)
            value[node]  += r
            r    = np.float64(1.0) - r
            node = parent[node]

    # Best move = child of root with most visits
    best_move = np.int32(-1)
    best_v    = np.int32(-1)
    for i in range(n_children[0]):
        cid = children[0, i]
        v   = visits[cid]
        if v > best_v:
            best_v    = v
            best_move = move_fp[cid]

    return best_move


# ── Solver JIT kernels ────────────────────────────────────────────────────────

@njit(cache=True)
def _nb_update_node_status(
    node_id: np.int32,
    terminal, status, distance, children, n_children, n_untried,
    mp1, mp2,
) -> bool:
    """Recompute proof status and minimax distance for *node_id* from its children.

    Implements AND/OR proof rules (node is an OR-node — current player chooses):

      WIN  ← any child has STATUS_LOSS  (we pick the fastest)
      LOSS ← all children STATUS_WIN and no untried moves  (opponent forces it)
      DRAW ← fully explored, at least one DRAW child, no LOSS child

    Returns True if status or distance changed (caller should continue
    propagating upward), False otherwise.
    """
    if terminal[node_id] != 0:
        return False  # game-ending nodes have fixed status

    nc = n_children[node_id]
    if nc == 0:
        return False  # no children yet — nothing to infer

    # WIN: at least one child where the opponent is in a forced loss.
    has_loss    = False
    min_loss_d  = np.int32(999999)
    for i in range(nc):
        cid = children[node_id, i]
        if status[cid] == _S_LOSS:
            has_loss = True
            if distance[cid] < min_loss_d:
                min_loss_d = distance[cid]

    if has_loss:
        new_status   = np.int32(_S_WIN)
        new_distance = min_loss_d + np.int32(1)
        if status[node_id] == new_status and distance[node_id] == new_distance:
            return False
        status[node_id]   = new_status
        distance[node_id] = new_distance
        return True

    # LOSS / DRAW: only conclusive once every move has been tried.
    if n_untried[node_id] > 0:
        return False
    for i in range(nc):
        if status[children[node_id, i]] == _S_UNKNOWN:
            return False

    has_draw    = False
    has_win_c   = False
    max_win_d   = np.int32(0)
    for i in range(nc):
        cid = children[node_id, i]
        s   = status[cid]
        if s == _S_DRAW:
            has_draw = True
        elif s == _S_WIN:
            has_win_c = True
            if distance[cid] > max_win_d:
                max_win_d = distance[cid]

    if has_draw:
        new_status   = np.int32(_S_DRAW)
        new_distance = np.int32(0)
    elif has_win_c:
        # Under PopOut Rule 2: if the board is full the current player can
        # declare draw rather than accept a forced loss.
        if nb_is_full(mp1[node_id], mp2[node_id]):
            new_status   = np.int32(_S_DRAW)
            new_distance = np.int32(0)
        else:
            new_status   = np.int32(_S_LOSS)
            new_distance = max_win_d + np.int32(1)
    else:
        return False

    if status[node_id] == new_status and distance[node_id] == new_distance:
        return False
    status[node_id]   = new_status
    distance[node_id] = new_distance
    return True


@njit(cache=True)
def _nb_propagate_status(
    node_id: np.int32,
    parent, terminal, status, distance, children, n_children, n_untried,
    mp1, mp2,
) -> None:
    """Walk from *node_id*'s parent to the root, updating proof status.

    Stops as soon as _nb_update_node_status reports no change — valid because
    status transitions are monotone (UNKNOWN → proven, never reversed).
    """
    current = parent[node_id]
    while current >= 0:
        if not _nb_update_node_status(
            current, terminal, status, distance, children, n_children, n_untried,
            mp1, mp2,
        ):
            break
        current = parent[current]


@njit(cache=True)
def _nb_best_child_solver_id(
    node_id: np.int32,
    visits, value, children, n_children, n_untried,
    status, distance,
    exploration_c: np.float64,
) -> np.int32:
    """Proof-aware child selection for the solver.

    Priority (highest to lowest):
      1. STATUS_LOSS child  → forced win; pick shortest distance.
      2. All children proven, none LOSS  → delay loss; pick largest distance.
      3. UCB1 on STATUS_UNKNOWN children only.
    """
    nc = n_children[node_id]

    # Priority 1: any child where opponent is in forced loss.
    has_loss  = False
    best_loss = np.int32(-1)
    min_dist  = np.int32(999999)
    for i in range(nc):
        cid = children[node_id, i]
        if status[cid] == _S_LOSS:
            has_loss = True
            if distance[cid] < min_dist:
                min_dist  = distance[cid]
                best_loss = cid
    if has_loss:
        return best_loss

    # Priority 2: fully proven subtree — delay the inevitable.
    all_proven = n_untried[node_id] == 0
    if all_proven:
        for i in range(nc):
            if status[children[node_id, i]] == _S_UNKNOWN:
                all_proven = False
                break
    if all_proven:
        best  = np.int32(-1)
        max_d = np.int32(-1)
        for i in range(nc):
            cid = children[node_id, i]
            if distance[cid] > max_d:
                max_d = distance[cid]
                best  = cid
        return best

    # Priority 3: UCB1 restricted to UNKNOWN children.
    log_n      = math.log(max(1.0, float(visits[node_id])))
    c          = exploration_c
    best_score = -1e300
    best       = np.int32(-1)
    for i in range(nc):
        cid = children[node_id, i]
        if status[cid] != _S_UNKNOWN:
            continue
        cv = visits[cid]
        if cv == 0:
            return cid
        score = value[cid] / float(cv) + c * math.sqrt(log_n / float(cv))
        if score > best_score:
            best_score = score
            best       = cid

    if best < 0 and nc > 0:
        best = children[node_id, 0]  # edge case: no unknown child but untried remain
    return best


@njit(cache=True)
def nb_solver_mcts_run(
    root_m1: np.int64,
    root_m2: np.int64,
    root_player: np.int32,
    iterations: np.int32,
    exploration_c: np.float64,
    rollout_depth: np.int32,
    visits, value, parent, move_fp, mover_arr,
    mp1, mp2, player_arr, terminal,
    children, n_children, untried, n_untried,
    status, distance,
    n_initial: np.int32,
) -> np.int32:
    """Complete MCTS-Solver loop in Numba — select, expand, simulate, backprop, proof-propagate.

    Extends nb_mcts_run with two additional flat arrays (status, distance) and
    proof propagation after every backpropagation step.  Runs all *iterations*
    even after the root is proven: continuing to expand the root's untried moves
    discovers additional winning lines and refines minimax distances, so the
    final move selection reliably returns the fastest win rather than the first
    one proved.

    Arrays are pre-allocated by FlatNumbaSolverMCTS and reused across calls.

    n_initial
        Pass ``np.int32(1)`` for a fresh search: node 0 is (re-)initialised
        from *root_m1/m2/player* before any iterations run.
        Pass ``np.int32(k)`` where k > 1 for tree reuse: node 0 has already
        been populated by a Python-side subtree compaction; the kernel skips
        initialisation and starts with *k* pre-existing nodes.
    """
    max_nodes = visits.shape[0]

    if n_initial == np.int32(1):
        # Fresh search — initialise root slot (node 0) from the board state.
        mp1[0]        = root_m1
        mp2[0]        = root_m2
        player_arr[0] = root_player
        mover_arr[0]  = np.int32(3 - root_player)
        terminal[0]   = np.int32(0)
        visits[0]     = np.int32(0)
        value[0]      = np.float64(0.0)
        n_children[0] = np.int32(0)
        parent[0]     = np.int32(-1)
        status[0]     = np.int32(_S_UNKNOWN)
        distance[0]   = np.int32(0)
        legal, n_legal = nb_legal_moves(root_m1, root_m2, root_player)
        n_untried[0]  = n_legal
        for i in range(n_legal):
            untried[0, i] = legal[i]
        if n_legal == 0:
            status[0] = np.int32(_S_DRAW)

    # n_initial == 1  → fresh tree with 1 node (the root).
    # n_initial  > 1  → reuse: k nodes already compacted into slots 0..k-1.
    n_nodes = n_initial

    for _ in range(iterations):
        if n_nodes >= max_nodes - 1:
            break
        if status[0] != _S_UNKNOWN and n_untried[0] == 0:
            break   # proven & every first move tried — distance is optimal

        # SELECT: descend while node is fully expanded, unproven, and non-terminal.
        cur = np.int32(0)
        while (terminal[cur] == 0
               and status[cur] == _S_UNKNOWN
               and n_untried[cur] == 0
               and n_children[cur] > 0):
            cur = _nb_best_child_solver_id(
                cur, visits, value, children, n_children, n_untried,
                status, distance, exploration_c,
            )

        # EXPAND: add one child for an untried move.
        # Allow expanding WIN nodes too — their untried moves may lead to LOSS
        # children with shorter distances than the winning lines already proven.
        if terminal[cur] == 0 and n_untried[cur] > 0 and status[cur] != _S_LOSS and status[cur] != _S_DRAW:
            nu   = n_untried[cur]
            idx  = np.random.randint(np.int32(0), nu)
            move = untried[cur, idx]
            untried[cur, idx] = untried[cur, nu - 1]
            n_untried[cur]   -= np.int32(1)

            pm1 = mp1[cur]
            pm2 = mp2[cur]
            ppl = player_arr[cur]
            mv  = ppl

            new_m1, new_m2, new_cp = nb_apply_move(pm1, pm2, ppl, move)
            win = nb_evaluate_after_move(new_m1, new_m2, np.int32(mv))

            cid = n_nodes
            n_nodes += np.int32(1)

            mp1[cid]        = new_m1
            mp2[cid]        = new_m2
            player_arr[cid] = new_cp
            parent[cid]     = cur
            move_fp[cid]    = move
            mover_arr[cid]  = np.int32(mv)
            terminal[cid]   = win
            visits[cid]     = np.int32(0)
            value[cid]      = np.float64(0.0)
            n_children[cid] = np.int32(0)
            distance[cid]   = np.int32(0)

            if win != 0:
                n_untried[cid] = np.int32(0)
                # terminal_winner vs next-to-move determines status
                if win == new_cp:
                    status[cid] = np.int32(_S_WIN)
                else:
                    status[cid] = np.int32(_S_LOSS)
            else:
                leg, nl = nb_legal_moves(new_m1, new_m2, new_cp)
                n_untried[cid] = nl
                for i in range(nl):
                    untried[cid, i] = leg[i]
                status[cid] = np.int32(_S_DRAW) if nl == 0 else np.int32(_S_UNKNOWN)

            children[cur, n_children[cur]] = cid
            n_children[cur] += np.int32(1)
            cur = cid

        # SIMULATE: random rollout from the selected/expanded node.
        ini_mov = mover_arr[cur]
        if ini_mov == 0:
            ini_mov = np.int32(3 - player_arr[cur])
        reward = nb_simulate(
            mp1[cur], mp2[cur], player_arr[cur],
            ini_mov, terminal[cur], rollout_depth,
        )

        # BACKPROPAGATE
        r    = reward
        node = cur
        while node >= 0:
            visits[node] += np.int32(1)
            value[node]  += r
            r    = np.float64(1.0) - r
            node = parent[node]

        # PROOF PROPAGATION: update status from cur upward to root.
        _nb_propagate_status(
            cur, parent, terminal, status, distance, children, n_children, n_untried,
            mp1, mp2,
        )

    # BEST MOVE — proof priority, fall back to most-visited.
    best_move = np.int32(-1)

    # Priority 1: child with STATUS_LOSS (forced win — pick fastest).
    has_loss = False
    min_dist = np.int32(999999)
    for i in range(n_children[0]):
        cid = children[0, i]
        if status[cid] == _S_LOSS:
            has_loss = True
            if distance[cid] < min_dist:
                min_dist  = distance[cid]
                best_move = move_fp[cid]
    if has_loss:
        return best_move

    # Priority 2: root proven DRAW — take any DRAW child.
    if status[0] == _S_DRAW:
        for i in range(n_children[0]):
            cid = children[0, i]
            if status[cid] == _S_DRAW:
                return move_fp[cid]

    # Priority 3: root proven LOSS — delay as long as possible.
    if status[0] == _S_LOSS:
        max_dist = np.int32(-1)
        for i in range(n_children[0]):
            cid = children[0, i]
            if distance[cid] > max_dist:
                max_dist  = distance[cid]
                best_move = move_fp[cid]
        return best_move

    # Priority 4: no proof — most-visited child.
    best_v = np.int32(-1)
    for i in range(n_children[0]):
        cid = children[0, i]
        v   = visits[cid]
        if v > best_v:
            best_v    = v
            best_move = move_fp[cid]

    return best_move
