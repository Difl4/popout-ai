"""Variações experimentais de UCT para testes de desempenho."""

from __future__ import annotations

from .base import BaseMCTS, MCTSNode


class ExperimentalUCT(BaseMCTS):
    """
    Variante experimental de UCT com viés para exploração inicial.

    Estratégia:
    - Se existir filho não visitado, escolhe-o primeiro.
    - Caso contrário, aplica UCT normal da classe base.
    """

    def best_child(self, node: MCTSNode) -> MCTSNode:
        """
        Escolhe melhor filho com política experimental.

        Args:
            node (MCTSNode): Nó pai.

        Returns:
            MCTSNode: Filho selecionado.
        """
        for child in node.children.values():
            if child.visits == 0:
                return child
        return super().best_child(node)
