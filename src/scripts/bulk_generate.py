"""Geração em lote de dataset (estado -> melhor jogada) via MCTS."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Type

import pandas as pd

from ..algorithms.mcts.uct_experimental import ExperimentalUCT
from ..algorithms.mcts.uct_standard import StandardUCT
from ..engine.bitboard import PopOutBoard


def make_agent(variant: str, seed: int | None = None):
    """
    Cria agente MCTS conforme variante pedida.

    Args:
        variant (str): Nome da variante ('uct_standard' ou 'uct_experimental').
        seed (int | None): Semente opcional.

    Returns:
        BaseMCTS: Instância de agente MCTS.

    Raises:
        ValueError: Se variante for desconhecida.
    """
    mapping: dict[str, Type] = {
        "uct_standard": StandardUCT,
        "uct_experimental": ExperimentalUCT,
    }
    if variant not in mapping:
        raise ValueError(f"Variante desconhecida: {variant}")
    return mapping[variant](seed=seed)


def randomize_state(steps: int, rng: random.Random) -> PopOutBoard:
    """
    Gera estado plausível aplicando jogadas aleatórias.

    Args:
        steps (int): Número de jogadas aleatórias.
        rng (random.Random): Gerador aleatório.

    Returns:
        PopOutBoard: Estado resultante.
    """
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
    """
    rng = random.Random(seed)
    agent = make_agent(variant, seed=seed)

    rows = []
    for _ in range(n_samples):
        steps = rng.randint(0, 20)
        board = randomize_state(steps=steps, rng=rng)

        legal = board.legal_moves()
        if not legal:
            continue

        best_move_int = agent.run(board, iterations=iterations)
        
        # Converter inteiro (0-13) para tuplo (tipo, coluna)
        if best_move_int < 7:
            best_move = ("drop", best_move_int)
        else:
            best_move = ("pop", best_move_int - 7)
        
        features = board.to_feature_dict()
        features["best_move"] = f"{best_move[0]}_{best_move[1]}"
        rows.append(features)

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
