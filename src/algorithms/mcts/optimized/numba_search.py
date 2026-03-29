"""MCTS-specific @njit search kernels — no Python objects cross the JIT boundary.

Single Responsibility: this module owns only the JIT-compiled MCTS search
logic.  Pure game mechanics live in src/engine/optimized/numba_rules.py.

Functions
---------
nb_expand_step  — apply move + evaluate + compute legal moves in one JIT call.
nb_simulate     — full random rollout in pure int64 arithmetic.
_nb_best_child_id — UCT child selection on flat arrays.
nb_mcts_run     — complete MCTS loop (select, expand, simulate, backprop).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from src.engine.optimized.numba_rules import (
    nb_apply_move,
    nb_evaluate_after_move,
    nb_has_won,
    nb_is_full,
    nb_legal_moves,
)


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
        if nb_is_full(mask_p1, mask_p2):
            return np.float64(0.5)

        rep_count = np.int32(0)
        for i in range(hist_size):
            if hist_p1[i] == mask_p1 and hist_p2[i] == mask_p2:
                rep_count += np.int32(1)
        if rep_count >= 3:
            return np.float64(0.5)

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
