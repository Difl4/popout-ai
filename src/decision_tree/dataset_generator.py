"""Geração em lote de dataset (estado -> melhor jogada) via MCTS."""

from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from src.mcts.factory import get_agent
from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import extended_features

# Iteration levels the opponent can drop to when blundering.
# Sampled uniformly so every severity of mistake is equally represented.
BLUNDER_GRID = [100, 300, 500, 1_000, 3_000, 5_000, 10_000, 30_000]


def _mirror_record(record: dict) -> dict:
    """Return the horizontally mirrored version of a board-state record.

    PopOut is left-right symmetric: column c mirrors to column 6-c.
    Applying this to every recorded position doubles the dataset for free.
    """
    out: dict = {}
    for key, val in record.items():
        if key.startswith("cell_"):
            _, row, col = key.split("_")
            out[f"cell_{row}_{6 - int(col)}"] = val
        elif key == "best_move":
            move_type, col = val.split("_")
            out["best_move"] = f"{move_type}_{6 - int(col)}"
        else:
            out[key] = val
    return out


def _worker_generate_sample(args: tuple) -> list[dict]:
    """Play one full game and return a record for every position.

    Top-level function so multiprocessing can pickle it for spawn workers.

    Design:
    - The oracle (FlatNumbaSolverMCTS at oracle_iterations) plays one fixed
      side for the entire game and labels EVERY position — regardless of whose
      turn it is — with its best move.  This ensures all training labels are
      oracle-quality.
    - Which side the oracle plays alternates per game (even seed → player 1,
      odd seed → player 2) so the dataset covers both player perspectives.
    - opponent_type controls how the opponent moves:
        "oracle_blunder": oracle strength most of the time; with probability
          blunder_rate drops to a random iteration count from blunder_grid.
        "blocking_random": picks a random legal move, but (1) never plays a
          move that accidentally wins the game for itself, and (2) always
          blocks the oracle's immediate wins when a blocking move exists.
    - forced_first_move (int 0-6 or None): when set, the opponent's very first
      move is forced to drop_<forced_first_move> for systematic opening
      coverage.  After that, normal opponent logic resumes.

    Args:
        args: (seed, oracle_iterations, blunder_rate, blunder_grid,
               opponent_type, forced_first_move)

    Returns:
        List of feature dicts, one per position (typically 40–80 after mirroring).
        Returns an empty list on failure.
    """
    seed, oracle_iterations, blunder_rate, blunder_grid, opponent_type, forced_first_move = args

    import sys
    _root = str(Path(__file__).resolve().parent.parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src.engine.standard.bitboard import PopOutBoard as _Board
    from src.engine.standard.rules import (
        evaluate_after_move,
        extended_features as _ext_feat,
        _can_win_in_one,
    )
    from src.mcts.optimized.numba_solver import FlatNumbaSolverMCTS

    rng           = random.Random(seed)
    agent         = FlatNumbaSolverMCTS(max_nodes=oracle_iterations + 256, seed=seed)
    oracle_player = 1 if seed % 2 == 0 else 2   # alternate which side oracle plays
    board         = _Board()
    records: list[dict] = []
    opponent_move_idx   = 0   # counts how many moves the opponent has made

    try:
        for _ in range(300):  # safety cap — real games end well before this
            legal = board.legal_moves()
            if not legal:
                break

            mover = board.current_player

            # Always label with the oracle's best move from this position.
            oracle_move = agent.run(board, iterations=oracle_iterations)
            # _status[0] is the root node's proof status after run():
            #   0 = UNKNOWN (heuristic best-guess move)
            #   1 = WIN  (forced win — fastest path, provably optimal)
            #   2 = LOSS (forced loss — slowest path, provably optimal)
            #   3 = DRAW (proven draw)
            # All three non-UNKNOWN statuses produce a mathematically determined
            # label, so all are worth oversampling during training.
            is_proven   = int(agent._status[0] != 0)
            label       = f"drop_{oracle_move}" if oracle_move < 7 else f"pop_{oracle_move - 7}"
            features    = _ext_feat(board)
            features["best_move"] = label
            features["is_proven"] = is_proven
            records.append(features)

            # Decide which move actually advances the game.
            if mover == oracle_player:
                move = oracle_move
            elif opponent_type == "blocking_random":
                # First opponent move: force a specific column for opening coverage.
                if opponent_move_idx == 0 and forced_first_move is not None:
                    forced_drop = forced_first_move  # 0–6, always legal at game start
                    move = forced_drop if forced_drop in legal else rng.choice(legal)
                else:
                    # One pass over legal moves: collect non-self-winning moves,
                    # and the subset of those that also block oracle's immediate win.
                    non_winning: list[int] = []
                    blocking_non_winning: list[int] = []
                    for m in legal:
                        b_tmp = board.clone()
                        b_tmp.apply_move(m)
                        if evaluate_after_move(b_tmp, mover=mover) == mover:
                            continue  # skip: random pop/drop accidentally wins
                        non_winning.append(m)
                        if not _can_win_in_one(b_tmp, oracle_player):
                            blocking_non_winning.append(m)

                    if blocking_non_winning:
                        move = rng.choice(blocking_non_winning)
                    elif non_winning:
                        move = rng.choice(non_winning)
                    else:
                        move = rng.choice(legal)  # all moves win or lose — unavoidable
                opponent_move_idx += 1
            else:
                # oracle_blunder: play at oracle strength, occasionally drop to weaker.
                if rng.random() < blunder_rate:
                    opp_iters = rng.choice(blunder_grid)
                    move      = agent.run(board, iterations=opp_iters)
                else:
                    move = oracle_move   # reuse the oracle result — no extra call needed
                opponent_move_idx += 1

            board.apply_move(move)
            if evaluate_after_move(board, mover=mover) != 0:
                break
    except Exception:
        pass

    return records + [_mirror_record(r) for r in records]


def generate_dataset_parallel(
    n_games: int = 5_000,
    oracle_iterations: int = 100_000,
    blunder_rate: float = 0.10,
    blunder_grid: list[int] | None = None,
    seed: int = 42,
    n_workers: int | None = None,
    oracle_blunder_fraction: float = 0.40,
) -> pd.DataFrame:
    """Generate a dataset by playing full games with FlatNumbaSolverMCTS.

    Games are split into two opponent strategies:

    oracle_blunder (oracle_blunder_fraction of n_games):
        Opponent plays at oracle strength most of the time but drops to a
        weaker iteration count sampled from blunder_grid with probability
        blunder_rate.  Covers optimal-line positions.

    blocking_random (remaining games):
        Opponent picks a random legal move each turn, but never plays a move
        that accidentally wins the game for itself, and always prefers moves
        that block the oracle's immediate wins.  The opponent's very first
        move cycles through columns 0–6 (game_index % 7) for systematic
        opening coverage.  Produces diverse, full-length games across all
        opening variations.

    Which side the oracle plays alternates per game (even seed → P1,
    odd seed → P2) so both player perspectives are equally represented.

    Total rows ≈ n_games × average_game_length × 2 (mirroring).

    Args:
        n_games:                  Total number of games to play.
        oracle_iterations:        Iterations for the oracle label and the
                                  oracle_blunder opponent's non-blunder moves.
        blunder_rate:             Probability of a blunder on any opponent move
                                  in oracle_blunder games (default 0.10).
        blunder_grid:             Iteration counts for blunders. Defaults to
                                  BLUNDER_GRID.
        seed:                     Base random seed; game i uses seed + i.
        n_workers:                Parallel processes. Defaults to os.cpu_count().
        oracle_blunder_fraction:  Fraction of games using oracle_blunder
                                  opponent (default 0.40). The rest use
                                  blocking_random.

    Returns:
        DataFrame with one row per position: cell features, tactical features,
        current_player, is_proven, and best_move label ("drop_c" or "pop_c").

    Raises:
        RuntimeError: If no records were generated.
    """
    if blunder_grid is None:
        blunder_grid = BLUNDER_GRID

    if n_workers is None:
        n_workers = os.cpu_count() or 1

    n_oracle_blunder   = round(n_games * oracle_blunder_fraction)
    n_blocking_random  = n_games - n_oracle_blunder

    args_list: list[tuple] = []
    for i in range(n_oracle_blunder):
        args_list.append(
            (seed + i, oracle_iterations, blunder_rate, blunder_grid, "oracle_blunder", None)
        )
    for i in range(n_blocking_random):
        forced_col = i % 7  # cycles 0–6 for systematic opening coverage
        args_list.append(
            (seed + n_oracle_blunder + i, oracle_iterations, blunder_rate, blunder_grid,
             "blocking_random", forced_col)
        )

    chunksize = max(1, n_games // (n_workers * 4))

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        for game_records in executor.map(_worker_generate_sample, args_list, chunksize=chunksize):
            rows.extend(game_records)

    if not rows:
        raise RuntimeError("Dataset generation failed — all games produced no records.")

    return pd.DataFrame(rows)


def make_agent(variant: str, seed: int | None = None):
    """
    Cria agente MCTS conforme variante pedida.

    Args:
        variant (str): Nome da variante.
            'uct_standard'     — Python MCTS padrão.
            'uct_experimental' — Python MCTS experimental.
        seed (int | None): Semente opcional.

    Returns:
        MCTSEngine: Instância de agente MCTS.

    Raises:
        ValueError: Se variante for desconhecida.
        TypeError: Se os tipos dos argumentos forem inválidos.
    """
    if not isinstance(variant, str):
        raise TypeError(f"variant deve ser string, recebeu {type(variant).__name__}")
    if seed is not None and not isinstance(seed, int):
        raise TypeError(f"seed deve ser int ou None, recebeu {type(seed).__name__}")
    _variant_map: dict[str, str] = {
        "uct_standard": "standard",
        "uct_experimental": "experimental",
    }
    if variant not in _variant_map:
        raise ValueError(f"Variante desconhecida: {variant}")
    return get_agent(_variant_map[variant], seed=seed)


def randomize_state(steps: int, rng: random.Random) -> PopOutBoard:
    """
    Gera estado plausível aplicando jogadas aleatórias.

    Args:
        steps (int): Número de jogadas aleatórias.
        rng (random.Random): Gerador aleatório.

    Returns:
        PopOutBoard: Estado resultante.

    Raises:
        TypeError: If steps is not int or rng is not random.Random.
        ValueError: If steps is negative.
    """
    # Type validation
    if not isinstance(steps, int) or isinstance(steps, bool):
        raise TypeError(f"steps must be int, got {type(steps).__name__}")
    if not isinstance(rng, random.Random):
        raise TypeError(f"rng must be random.Random, got {type(rng).__name__}")

    # Value validation
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")

    board = PopOutBoard()
    for _ in range(steps):
        legal = board.legal_moves()
        if not legal:
            break
        move = rng.choice(legal)
        board.apply_move(move)
    return board


def generate_dataset(
    variant: str,
    n_samples: int = 200,
    iterations: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Gera dataset de treino para ID3 baseado em decisões de MCTS.

    Args:
        variant (str): Variante MCTS.
        n_samples (int): Número de amostras.
        iterations (int): Iterações MCTS por estado.
        seed (int): Semente global.

    Returns:
        pd.DataFrame: Dataset com features do estado e rótulo da jogada.

    Raises:
        TypeError: If parameters have wrong types.
        ValueError: If parameters are out of valid ranges.
    """
    # Type validation
    if not isinstance(variant, str):
        raise TypeError(f"variant must be string, got {type(variant).__name__}")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool):
        raise TypeError(f"n_samples must be int, got {type(n_samples).__name__}")
    if not isinstance(iterations, int) or isinstance(iterations, bool):
        raise TypeError(f"iterations must be int, got {type(iterations).__name__}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    # Value validation
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    rng = random.Random(seed)
    agent = make_agent(variant, seed=seed)

    rows = []
    skipped = 0
    for i in range(n_samples):
        steps = rng.randint(0, 20)
        board = randomize_state(steps=steps, rng=rng)

        legal = board.legal_moves()
        if not legal:
            continue

        try:
            best_move_int = agent.run(board, iterations=iterations)
        except (TimeoutError, RuntimeError, Exception) as e:
            print(f"  Amostra {i}: Erro ao correr MCTS: {type(e).__name__}: {e}")
            skipped += 1
            continue

        # Converter inteiro (0-13) para tuplo (tipo, coluna)
        if best_move_int < 7:
            best_move = ("drop", best_move_int)
        else:
            best_move = ("pop", best_move_int - 7)

        features = extended_features(board)
        features["best_move"] = f"{best_move[0]}_{best_move[1]}"
        rows.append(features)

    if skipped > 0:
        print(f"  {skipped} amostras puladas por erro.")

    return pd.DataFrame(rows)


def main() -> None:
    """
    Entry point de linha de comandos para geração em lote.
    """
    parser = argparse.ArgumentParser(description="Geração de datasets PopOut com variantes de MCTS.")
    parser.add_argument("--variant", type=str, default="uct_standard", help="Variante MCTS.")
    parser.add_argument("--samples", type=int, default=200, help="Número de estados a gerar.")
    parser.add_argument("--iterations", type=int, default=150, help="Iterações MCTS por estado.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    parser.add_argument("--output", type=str, default="", help="CSV de saída (opcional).")
    args = parser.parse_args()

    df = generate_dataset(
        variant=args.variant,
        n_samples=args.samples,
        iterations=args.iterations,
        seed=args.seed,
    )

    output = args.output or f"data/generated/{args.variant}.csv"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset gerado: {output_path} | linhas={len(df)}")


if __name__ == "__main__":
    main()
