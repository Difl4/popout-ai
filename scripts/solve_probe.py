"""Probe how close the MCTS-Solver is to proving PopOut is a first-player win.

Three experiments:

  1. Root proof (scaling nodes)
       Run FlatNumbaSolverMCTS from the empty board, increasing max_nodes until
       the root is proven or we hit a practical memory limit.
       The previous attempt hit the 400k node cap — this test scales the budget.

  2. Post-opening proof
       Fix P1's first move (col 3, the solver's favourite opening), then try to
       prove the resulting position is a forced LOSS for P2.
       Proving one subtree is far cheaper than proving all 7 first moves at once.
       If this succeeds, "drop col 3" is a proven first-player winning move.

  3. Game depth probe
       Both players use the solver at 100k iterations.
       Reports proof status at every move to show when positions become provable.

Usage
-----
    conda run -n popout-ai python3 scripts/solve_probe.py
"""

from __future__ import annotations

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mcts.optimized.numba_mcts import warmup
from src.mcts.optimized.numba_solver import FlatNumbaSolverMCTS
from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import evaluate_after_move

_STATUS    = {0: "UNKNOWN", 1: "WIN", 2: "LOSS", 3: "DRAW"}
_S_UNKNOWN = 0
_S_WIN     = 1
_S_LOSS    = 2
_S_DRAW    = 3

_BYTES_PER_NODE = 176   # measured: 10×int32 + 2×int64 + 1×float64 + 2×(14,)int32


def _move_label(m: int) -> str:
    return f"{'DROP' if m < 7 else 'POP'} col {m % 7}"


# ── Experiment 1: root proof with increasing node budget ─────────────────────

def root_proof_test() -> None:
    print("=" * 64)
    print("Experiment 1 — Root proof (scaling max_nodes)")
    print("  Bottleneck is nodes, not iterations.")
    print("  Previous run saturated the 400k node cap immediately.")
    print("=" * 64)
    print(f"  {'max_nodes':>12}  {'RAM':>7}  {'root':>9}  {'dist':>5}  {'nodes':>10}  {'time':>6}  {'best'}")
    print("  " + "-" * 66)

    board = PopOutBoard()

    for max_nodes in [400_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]:
        mb = max_nodes * _BYTES_PER_NODE // (1024 * 1024)
        solver = FlatNumbaSolverMCTS(max_nodes=max_nodes)

        # Use 2× max_nodes as the iteration cap so the node budget is always
        # the binding constraint, not the iteration count.
        iters = max_nodes * 2

        t0   = time.perf_counter()
        move = solver.run(board, iterations=iters)
        dt   = time.perf_counter() - t0

        status   = int(solver._status[0])
        distance = int(solver._distance[0])
        nodes    = int(solver._visits[0])   # ≈ iterations completed ≈ nodes created

        tag = _STATUS[status]
        print(
            f"  {max_nodes:>12,}  {mb:>5}MB  {tag:>9}  {distance:>5}  "
            f"{nodes:>10,}  {dt:>5.1f}s  {_move_label(move)}"
        )

        if status == _S_WIN:
            print(f"\n  *** ROOT PROVEN: P1 has a forced win in {distance} moves! ***")
            return

    print("\n  Root not proven. The empty-board search space is still too large.")


# ── Experiment 2: post-opening proof (fix P1's first move) ───────────────────

def post_opening_proof(opening_col: int = 3) -> None:
    print()
    print("=" * 64)
    print(f"Experiment 2 — Post-opening proof  (P1 opens: DROP col {opening_col})")
    print("  P1 commits to one opening move; we try to prove the resulting")
    print("  position is LOSS for P2 — which would make that opening a")
    print("  proven first-player win.")
    print("=" * 64)
    print(f"  {'max_nodes':>12}  {'RAM':>7}  {'P2 status':>10}  {'dist':>5}  {'nodes':>10}  {'time':>6}")
    print("  " + "-" * 60)

    board_after_opening = PopOutBoard()
    board_after_opening.apply_move(opening_col)   # P1 drops opening_col

    for max_nodes in [400_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]:
        mb     = max_nodes * _BYTES_PER_NODE // (1024 * 1024)
        solver = FlatNumbaSolverMCTS(max_nodes=max_nodes)
        iters  = max_nodes * 2

        t0   = time.perf_counter()
        _    = solver.run(board_after_opening.clone(), iterations=iters)
        dt   = time.perf_counter() - t0

        status   = int(solver._status[0])
        distance = int(solver._distance[0])
        nodes    = int(solver._visits[0])

        tag = _STATUS[status]

        proof_note = ""
        if status == _S_LOSS:
            proof_note = f"  ← PROVEN: DROP col {opening_col} is a forced win for P1 in {distance + 1} moves!"
        elif status == _S_DRAW:
            proof_note = f"  ← PROVEN DRAW from this opening"

        print(
            f"  {max_nodes:>12,}  {mb:>5}MB  {tag:>10}  {distance:>5}  "
            f"{nodes:>10,}  {dt:>5.1f}s{proof_note}"
        )

        if status in (_S_LOSS, _S_WIN):
            return

    print(f"\n  Position after DROP col {opening_col} not proven within tested budgets.")


# ── Experiment 3: game depth probe ───────────────────────────────────────────

def game_depth_probe(iters_per_move: int = 100_000) -> None:
    print()
    print("=" * 64)
    print(f"Experiment 3 — Game depth probe  ({iters_per_move:,} iter/move)")
    print("  Both players use FlatNumbaSolverMCTS.")
    print("  Shows the move at which positions first become provable.")
    print("=" * 64)

    solver = FlatNumbaSolverMCTS(max_nodes=400_000)
    board  = PopOutBoard()

    first_proof: int | None = None

    print(
        f"\n  {'#':>3}  {'P':>2}  {'status':>9}  {'dist':>5}  {'action':>14}  {'time':>5}"
    )
    print("  " + "-" * 50)

    for move_num in range(1, 60):
        player = board.current_player
        t0     = time.perf_counter()
        move   = solver.run(board, iterations=iters_per_move)
        dt     = time.perf_counter() - t0

        status   = int(solver._status[0])
        distance = int(solver._distance[0])
        tag      = _STATUS[status]

        marker = ""
        if status != _S_UNKNOWN and first_proof is None:
            first_proof = move_num
            marker = "  ← first proof"

        print(f"  {move_num:>3}  P{player}  {tag:>9}  {distance:>5}  {_move_label(move):>14}  {dt:.1f}s{marker}")

        board.apply_move(move)
        winner = evaluate_after_move(board, mover=player)
        if winner:
            print(f"\n  Game over — Player {winner} wins on move {move_num}.")
            break
        if board.is_full():
            print(f"\n  Draw — board full after {move_num} moves.")
            break

    if first_proof:
        print(f"\n  First proof at move {first_proof}.")


# ── Experiment 3b: replay walk-back ─────────────────────────────────────────

def replay_walkback() -> None:
    """Replay the known proven game line and walk backwards to find the
    earliest position that can be proven within a practical node budget.

    The depth-probe game proved at move 9 (after 8 moves played) with 400k nodes.
    We walk back one move at a time to find how early in that specific line
    the proof becomes achievable as we increase the node budget.
    """
    # Exact move sequence from the depth-probe game that led to the first proof.
    # This is P1's opening line: both players following the solver's best moves.
    GAME_SEQ = [3, 3, 3, 3, 2, 4, 4, 2]  # 8 moves; position after these is proven at 400k

    print()
    print("=" * 64)
    print("Experiment 3b — Walk-back along the proven game line")
    print("  Starts from the proven position (after 8 moves) and walks")
    print("  backwards to find the earliest provable prefix.")
    print("  Stops when 10M nodes can't prove the position.")
    print("=" * 64)
    print(f"\n  {'depth':>6}  {'player':>7}  {'max_nodes':>10}  {'status':>9}  {'dist':>5}  {'time':>6}")
    print("  " + "-" * 54)

    for depth in range(len(GAME_SEQ), -1, -1):
        board = PopOutBoard()
        for m in GAME_SEQ[:depth]:
            board.apply_move(m)

        player = board.current_player
        proven = False

        for max_nodes in [400_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]:
            mb     = max_nodes * _BYTES_PER_NODE // (1024 * 1024)
            solver = FlatNumbaSolverMCTS(max_nodes=max_nodes)
            t0     = time.perf_counter()
            solver.run(board.clone(), iterations=max_nodes * 2)
            dt     = time.perf_counter() - t0

            status   = int(solver._status[0])
            distance = int(solver._distance[0])

            if status != _S_UNKNOWN:
                print(
                    f"  {depth:>6}  P{player}      {max_nodes:>10,}  {_STATUS[status]:>9}"
                    f"  {distance:>5}  {dt:>5.1f}s  (proven!)"
                )
                proven = True
                break

        if not proven:
            print(f"  {depth:>6}  P{player}      {'10,000,000':>10}  {'UNKNOWN':>9}  {'—':>5}  {'—':>6}  (not proven)")
            print(f"\n  Earliest provable prefix: depth {depth + 1} (after {depth + 1} moves).")
            return

    print(f"\n  All positions in the line provable — root (depth 0) is proven!")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Warming up Numba JIT...")
    t0 = time.perf_counter()
    warmup()
    print(f"Warmup done in {time.perf_counter() - t0:.1f}s\n")

    root_proof_test()
    post_opening_proof(opening_col=3)
    game_depth_probe(iters_per_move=100_000)
    replay_walkback()
