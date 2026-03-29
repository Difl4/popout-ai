"""Implementação UCT padrão baseada em BaseMCTS."""

from __future__ import annotations

from src.algorithms.mcts.standard.base import BaseMCTS


class StandardUCT(BaseMCTS):
    """
    Variante padrão de UCT para PopOut.

    Mantém a lógica da classe base sem alterações estruturais.
    """

    def __init__(self, exploration_c: float = 1.414, rollout_depth: int = 30, seed: int | None = None) -> None:
        """
        Inicializa o UCT padrão.

        Args:
            exploration_c (float): Peso da exploração.
            rollout_depth (int): Profundidade de simulação.
            seed (int | None): Semente opcional.
        """
        super().__init__(exploration_c=exploration_c, rollout_depth=rollout_depth, seed=seed)
