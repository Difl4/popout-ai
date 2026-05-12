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
    Both sides are played by the same FlatNumbaSolverMCTS instance — each
    run() call rebuilds the search tree from scratch, so there is no state
    bleed between moves.  Starting from the empty board produces realistic
    opening, mid-game, and end-game positions and naturally surfaces the
    tactical situations where pop moves are optimal.

    Args:
        args: (seed, iterations)

    Returns:
        List of feature dicts, one per move played (typically 20–40).
        Returns an empty list on failure.
    """
    seed, iterations = args

    import sys
    _root = str(Path(__file__).resolve().parent.parent.parent)
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src.engine.standard.bitboard import PopOutBoard as _Board
    from src.engine.standard.rules import evaluate_after_move, extended_features as _ext_feat
    from src.mcts.optimized.numba_solver import FlatNumbaSolverMCTS

    agent   = FlatNumbaSolverMCTS(max_nodes=iterations + 256, seed=seed)
    board   = _Board()
    records: list[dict] = []

    try:
        for _ in range(300):  # safety cap — real games end well before this
            legal = board.legal_moves()
            if not legal:
                break

            mover         = board.current_player
            best_move_int = agent.run(board, iterations=iterations)

            label    = f"drop_{best_move_int}" if best_move_int < 7 else f"pop_{best_move_int - 7}"
            features = _ext_feat(board)
            features["best_move"] = label
            records.append(features)

            board.apply_move(best_move_int)
            if evaluate_after_move(board, mover=mover) != 0:
                break
    except Exception:
        pass

    return records + [_mirror_record(r) for r in records]


def generate_dataset_parallel(
    n_games: int = 200,
    iterations: int = 100_000,
    seed: int = 42,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Generate a dataset by playing full games with FlatNumbaSolverMCTS.

    Each worker plays one complete game (both sides), recording every
    (board-state, best-move) pair from the opening through to the terminal
    position.  This produces realistic positions and naturally includes
    tactical pop moves that random scrambling misses.

    Total rows ≈ n_games × average game length (~25–35 moves).

    Args:
        n_games:    Number of full games to play.
        iterations: Maximum MCTS iterations per move.  The solver exits early
                    once the position is proven.  100 000 gives near-optimal
                    oracle quality.
        seed:       Base random seed; game i uses seed + i for reproducibility.
        n_workers:  Parallel processes.  Defaults to os.cpu_count().

    Returns:
        DataFrame with one row per position: 42 cell features, current_player,
        and best_move label ("drop_c" or "pop_c").

    Raises:
        RuntimeError: If no records were generated.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    args_list = [(seed + i, iterations) for i in range(n_games)]
    chunksize  = max(1, n_games // (n_workers * 4))

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
