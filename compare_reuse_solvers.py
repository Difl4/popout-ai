"""Correctness comparison: ReuseSolverMCTS (pure Python) vs ReuseFlatNumbaSolverMCTS.

Both implement the same MCTS-Solver algorithm with tree reuse.
If the flat variant is correct, win rates should be close to 50-50.

Usage
-----
    python compare_reuse_solvers.py                  # 40 games, 500 iters
    python compare_reuse_solvers.py --games 80 --iters 1000
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import evaluate_after_move, board_signature


def play_game(agent_a, agent_b, iters: int, seed: int, a_is_p1: bool) -> int:
    """Play one game. Returns winner player id (0 = draw)."""
    rng = random.Random(seed)
    board = PopOutBoard()
    sig_counter: Counter = Counter([board_signature(board)])

    last_a_mv: int | None = None
    last_b_mv: int | None = None

    while True:
        legal = board.legal_moves()
        if not legal:
            return 0

        is_a_turn = (board.current_player == 1) == a_is_p1

        if is_a_turn:
            move = agent_a.run(board, iters, opponent_move=last_b_mv)
            last_a_mv = move
        else:
            move = agent_b.run(board, iters, opponent_move=last_a_mv)
            last_b_mv = move

        mover = board.current_player
        board.apply_move(move)

        winner = evaluate_after_move(board, mover=mover)
        if winner != 0:
            return winner

        sig = board_signature(board)
        sig_counter[sig] += 1
        if sig_counter[sig] >= 3:
            return 0


def run_comparison(n_games: int, iters: int, seed: int) -> None:
    from src.mcts.optimized.numba_mcts import warmup
    from src.mcts.optimized.numba_solver import warmup_solver, ReuseFlatNumbaSolverMCTS
    from src.mcts.standard.uct_solver import ReuseSolverMCTS

    print("Warming up Numba kernels...", flush=True)
    warmup()
    warmup_solver()
    print("Done.\n")

    wins_flat   = 0  # ReuseFlatNumbaSolverMCTS wins
    wins_python = 0  # ReuseSolverMCTS wins
    draws       = 0

    rng = random.Random(seed)

    print(f"Playing {n_games} games ({iters} iters/move each)...")
    print(f"{'Game':>4}  {'P1':>16}  {'P2':>16}  {'Winner':>20}  {'Result':>8}")
    print("-" * 72)

    t_start = time.perf_counter()

    for g in range(n_games):
        flat_is_p1 = (g % 2 == 0)
        gseed = rng.randint(0, 10**9)

        flat   = ReuseFlatNumbaSolverMCTS(seed=gseed)
        python = ReuseSolverMCTS(seed=gseed)
        flat.reset()
        python.reset()

        if flat_is_p1:
            winner = play_game(flat, python, iters, gseed, a_is_p1=True)
            p1_label, p2_label = "Flat(P1)", "Python(P2)"
        else:
            winner = play_game(python, flat, iters, gseed, a_is_p1=False)
            p1_label, p2_label = "Python(P1)", "Flat(P2)"

        if winner == 0:
            draws += 1
            result = "Draw"
        elif (winner == 1) == flat_is_p1:
            wins_flat += 1
            result = "Flat wins"
        else:
            wins_python += 1
            result = "Python wins"

        print(f"{g+1:>4}  {p1_label:>16}  {p2_label:>16}  {result:>20}", flush=True)

    elapsed = time.perf_counter() - t_start
    total = n_games - draws

    print("\n" + "=" * 72)
    print(f"Results after {n_games} games ({elapsed:.1f}s)")
    print(f"  ReuseFlatNumbaSolverMCTS wins : {wins_flat:>3}  ({100*wins_flat/n_games:.1f}%)")
    print(f"  ReuseSolverMCTS wins          : {wins_python:>3}  ({100*wins_python/n_games:.1f}%)")
    print(f"  Draws                         : {draws:>3}  ({100*draws/n_games:.1f}%)")
    if total > 0:
        print(f"\n  Win rate flat (excl. draws)   : {100*wins_flat/total:.1f}%")
        print(f"  Win rate python (excl. draws) : {100*wins_python/total:.1f}%")
    print()
    if total > 0:
        ratio = wins_flat / total
        if 0.35 <= ratio <= 0.65:
            print("VERDICT: win rates are close to 50-50 — flat variant looks correct.")
        else:
            print("VERDICT: significant skew detected — investigate the flat variant.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()
    run_comparison(args.games, args.iters, args.seed)
