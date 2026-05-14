"""Benchmark: Standard MCTS vs Tree-Reuse MCTS.

Compara três pares de agentes:
  1. ReuseUCT          vs StandardUCT          (~9 k iter/s)
  2. ReuseNumbaMCTS    vs NumbaMCTS             (~50 k iter/s)
  3. ReuseFlatNumbaSolverMCTS vs FlatNumbaSolverMCTS  (~30-60 k iter/s)

Métricas
--------
  - Taxa de vitória do agente com reuso
  - Visitas herdadas por turno (nós que o kernel recebe já preenchidos)
  - Overhead de tempo por jogada (reuse vs. fresh)

Uso
---
    python benchmark_reuse.py                    # todos os pares, defaults
    python benchmark_reuse.py --pair uct         # só par 1
    python benchmark_reuse.py --pair solver      # só par 3
    python benchmark_reuse.py --games 40 --iters 500
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import evaluate_after_move, board_signature


# ── tipos ─────────────────────────────────────────────────────────────────────

@dataclass
class GameStats:
    winner: int
    n_moves: int
    times_reuse: list[float] = field(default_factory=list)
    times_fresh: list[float] = field(default_factory=list)
    inherited:   list[int]   = field(default_factory=list)


# ── motor de jogo ─────────────────────────────────────────────────────────────

def _call(agent, board: PopOutBoard, iters: int, opp_mv) -> int:
    """Despacha run() correctamente para agentes com e sem reuso."""
    if hasattr(agent, '_root') or hasattr(agent, '_best_child_idx'):
        return agent.run(board, iters, opponent_move=opp_mv)
    return agent.run(board, iters)


def _peek_inherited(agent, opp_mv) -> int:
    """Espia as visitas herdadas ANTES de run() avançar o root."""
    # ReuseUCT / ReuseNumbaMCTS: _root aponta para o children[our_move];
    # as visitas de children[opp_mv] são as que vão ser herdadas.
    if hasattr(agent, '_root'):
        root = agent._root
        if root is not None and opp_mv is not None:
            candidate = root.children.get(opp_mv)
            return candidate.visits if candidate else 0
        return 0

    # ReuseFlatNumbaSolverMCTS: _best_child_idx aponta para children[our_move].
    if hasattr(agent, '_best_child_idx'):
        bc = agent._best_child_idx
        if bc >= 0 and opp_mv is not None:
            cid = agent._find_child_idx(bc, opp_mv)
            return int(agent._visits[cid]) if cid >= 0 else 0
        return 0

    return 0


def play_game(reuse_agent, fresh_agent, iters: int, seed: int,
              reuse_is_p1: bool) -> GameStats:
    rng = random.Random(seed)
    board = PopOutBoard()
    sig_counter: Counter = Counter([board_signature(board)])
    stats = GameStats(winner=0, n_moves=0)

    last_reuse_mv: int | None = None
    last_fresh_mv: int | None = None

    while True:
        legal = board.legal_moves()
        if not legal:
            break

        is_reuse_turn = (board.current_player == 1) == reuse_is_p1

        if is_reuse_turn:
            # espia visitas herdadas antes da jogada
            stats.inherited.append(_peek_inherited(reuse_agent, last_fresh_mv))
            t0 = time.perf_counter()
            move = _call(reuse_agent, board, iters, last_fresh_mv)
            stats.times_reuse.append(time.perf_counter() - t0)
            last_reuse_mv = move
        else:
            t0 = time.perf_counter()
            move = _call(fresh_agent, board, iters, last_reuse_mv)
            stats.times_fresh.append(time.perf_counter() - t0)
            last_fresh_mv = move

        mover = board.current_player
        board.apply_move(move)
        stats.n_moves += 1

        winner = evaluate_after_move(board, mover=mover)
        if winner != 0:
            stats.winner = winner
            return stats

        sig = board_signature(board)
        sig_counter[sig] += 1
        if sig_counter[sig] >= 3:
            return stats

    return stats


# ── torneio ───────────────────────────────────────────────────────────────────

def run_pair(
    label_reuse: str,
    label_fresh: str,
    make_reuse,      # callable() → agent
    make_fresh,      # callable() → agent
    n_games: int,
    iters: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    results = Counter()
    all_inherited:    list[int]   = []
    all_times_reuse:  list[float] = []
    all_times_fresh:  list[float] = []
    move_counts:      list[int]   = []

    half = n_games // 2

    print(f"\n{'═'*62}")
    print(f"  {label_reuse}  vs  {label_fresh}")
    print(f"  {n_games} jogos  |  {iters} iter/turno  |  seed={seed}")
    print(f"{'═'*62}")

    for block, (reuse_is_p1, n) in enumerate([(True, half), (False, n_games - half)]):
        tag = f"Reuse=P{'1' if reuse_is_p1 else '2'}"
        print(f"\n  [{tag}] {n} jogos", end="  ")
        for i in range(n):
            g_seed = rng.randint(0, 10**6)
            ra = make_reuse(); ra.reset() if hasattr(ra, 'reset') else None
            fa = make_fresh()
            gs = play_game(ra, fa, iters, g_seed, reuse_is_p1)

            # Normalizar winner para perspectiva do reuse
            reuse_player = 1 if reuse_is_p1 else 2
            if gs.winner == reuse_player:
                results['win'] += 1
            elif gs.winner == 0:
                results['draw'] += 1
            else:
                results['loss'] += 1

            all_inherited.extend(gs.inherited)
            all_times_reuse.extend(gs.times_reuse)
            all_times_fresh.extend(gs.times_fresh)
            move_counts.append(gs.n_moves)
            _bar(i + 1, n)

    total = results['win'] + results['draw'] + results['loss']
    score = (results['win'] + 0.5 * results['draw']) / total

    print(f"\n\n  {'─'*60}")
    print(f"  RESULTADOS  ({total} jogos, perspectiva do {label_reuse})")
    print(f"  {'─'*60}")
    print(f"  Vitórias  : {results['win']:>4d}  ({100*results['win']/total:5.1f}%)")
    print(f"  Empates   : {results['draw']:>4d}  ({100*results['draw']/total:5.1f}%)")
    print(f"  Derrotas  : {results['loss']:>4d}  ({100*results['loss']/total:5.1f}%)")
    print(f"  Score     : {score:.3f}  (0.500 = par)")

    print(f"\n  {'─'*60}")
    print(f"  VISITAS HERDADAS  (root no início de cada turno do agente reuse)")
    print(f"  {'─'*60}")
    if all_inherited:
        nonzero  = [v for v in all_inherited if v > 0]
        hit_rate = 100 * len(nonzero) / len(all_inherited)
        avg_all  = sum(all_inherited) / len(all_inherited)
        avg_nz   = sum(nonzero) / len(nonzero) if nonzero else 0
        print(f"  Amostras          : {len(all_inherited):>6d}")
        print(f"  Taxa de acerto    : {hit_rate:>5.1f}%  (oponente jogou move explorado)")
        print(f"  Média (todos)     : {avg_all:>6.1f} visitas  ({100*avg_all/iters:4.1f}% do orçamento)")
        print(f"  Média (acertos)   : {avg_nz:>6.1f} visitas")
        print(f"  Máximo herdado    : {max(all_inherited):>6d} visitas")

    print(f"\n  {'─'*60}")
    print(f"  TEMPO POR JOGADA")
    print(f"  {'─'*60}")
    if all_times_reuse and all_times_fresh:
        avg_r = 1000 * sum(all_times_reuse) / len(all_times_reuse)
        avg_f = 1000 * sum(all_times_fresh) / len(all_times_fresh)
        print(f"  {label_reuse:<30s}: {avg_r:>7.1f} ms/jogada")
        print(f"  {label_fresh:<30s}: {avg_f:>7.1f} ms/jogada")
        diff  = avg_r - avg_f
        print(f"  Overhead reuse              : {diff:>+7.2f} ms  ({100*diff/avg_f:+.1f}%)")

    print(f"\n  {'─'*60}")
    print(f"  DURAÇÃO DAS PARTIDAS")
    print(f"  {'─'*60}")
    if move_counts:
        print(f"  Média  : {sum(move_counts)/len(move_counts):.1f} jogadas")
        print(f"  Mín/Máx: {min(move_counts)} / {max(move_counts)}")
    print()


# ── utilitários ───────────────────────────────────────────────────────────────

def _bar(done: int, total: int) -> None:
    w = 25
    f = int(w * done / total)
    print(f"\r  [{'█'*f}{'░'*(w-f)}] {done}/{total}", end='', flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

PAIRS = {
    'uct':    ('UCT',    'ReuseUCT',     'StandardUCT'),
    'numba':  ('Numba',  'ReuseNumbaMCTS', 'NumbaMCTS'),
    'solver': ('Solver', 'ReuseFlatNumbaSolverMCTS', 'FlatNumbaSolverMCTS'),
    'all':    None,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark: Fresh vs Reuse MCTS")
    parser.add_argument("--games", type=int, default=30,
                        help="número de jogos por par (default: 30)")
    parser.add_argument("--iters", type=int, default=400,
                        help="iterações por turno (default: 400)")
    parser.add_argument("--seed",  type=int, default=0,
                        help="semente global (default: 0)")
    parser.add_argument("--pair",  choices=list(PAIRS), default="all",
                        help="par a testar (default: all)")
    args = parser.parse_args()

    # ── warmup Numba ──────────────────────────────────────────────────────────
    if args.pair in ('numba', 'solver', 'all'):
        print("Compilando kernels Numba (warmup)...", end=" ", flush=True)
        from src.mcts.optimized.numba_mcts import warmup
        from src.mcts.optimized.numba_solver import warmup_solver
        warmup(); warmup_solver()
        print("pronto.\n")

    # ── par 1: UCT ────────────────────────────────────────────────────────────
    if args.pair in ('uct', 'all'):
        from src.mcts.standard.uct_standard import StandardUCT
        from src.mcts.standard.uct_reuse import ReuseUCT
        run_pair(
            label_reuse="ReuseUCT",
            label_fresh="StandardUCT",
            make_reuse=lambda: ReuseUCT(seed=42),
            make_fresh=lambda: StandardUCT(seed=43),
            n_games=args.games, iters=args.iters, seed=args.seed,
        )

    # ── par 2: NumbaMCTS ──────────────────────────────────────────────────────
    if args.pair in ('numba', 'all'):
        from src.mcts.optimized.numba_mcts import NumbaMCTS
        from src.mcts.optimized.numba_reuse import ReuseNumbaMCTS
        run_pair(
            label_reuse="ReuseNumbaMCTS",
            label_fresh="NumbaMCTS",
            make_reuse=lambda: ReuseNumbaMCTS(seed=42),
            make_fresh=lambda: NumbaMCTS(seed=43),
            n_games=args.games, iters=args.iters, seed=args.seed,
        )

    # ── par 3: FlatNumbaSolverMCTS ────────────────────────────────────────────
    if args.pair in ('solver', 'all'):
        from src.mcts.optimized.numba_solver import FlatNumbaSolverMCTS, ReuseFlatNumbaSolverMCTS
        run_pair(
            label_reuse="ReuseFlatNumbaSolver",
            label_fresh="FlatNumbaSolver",
            make_reuse=lambda: ReuseFlatNumbaSolverMCTS(seed=42),
            make_fresh=lambda: FlatNumbaSolverMCTS(seed=43),
            n_games=args.games, iters=args.iters, seed=args.seed,
        )
