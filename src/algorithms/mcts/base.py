"""Base MCTS com os 4 passos: seleção, expansão, simulação e retropropagação."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ...engine.bitboard import PopOutBoard
from ...engine.rules import board_signature, evaluate_after_move, is_draw


Move = Tuple[str, int]


@dataclass
class MCTSNode:
    """
    Nó da árvore MCTS.

    Attributes:
        state (PopOutBoard): Estado associado ao nó.
        parent (Optional[MCTSNode]): Nó pai.
        move_from_parent (Optional[Move]): Jogada que gerou este nó.
        mover (Optional[int]): Jogador que fez a jogada de entrada neste nó.
    """

    state: PopOutBoard
    parent: Optional["MCTSNode"] = None
    move_from_parent: Optional[Move] = None
    mover: Optional[int] = None
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[Move, "MCTSNode"] = field(default_factory=dict)
    untried_moves: List[Move] = field(default_factory=list)
    terminal_winner: int = 0

    def __post_init__(self) -> None:
        """
        Inicializa lista de jogadas não testadas para o estado atual.
        """
        self.untried_moves = self.state.legal_moves(self.state.current_player)

    @property
    def is_terminal(self) -> bool:
        """
        Indica se o nó representa um estado terminal conhecido.

        Returns:
            bool: True se existir vencedor.
        """
        return self.terminal_winner != 0

    def q(self) -> float:
        """
        Valor médio do nó.

        Returns:
            float: Média value_sum / visits.
        """
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


class BaseMCTS:
    """
    Implementação base de MCTS para jogos adversariais.

    Pode ser extendida para variações de UCT.
    """

    def __init__(self, exploration_c: float = 1.414, rollout_depth: int = 30, seed: Optional[int] = None) -> None:
        """
        Inicializa hiperparâmetros do MCTS.

        Args:
            exploration_c (float): Constante de exploração UCT.
            rollout_depth (int): Profundidade máxima de simulação.
            seed (Optional[int]): Semente de aleatoriedade.
        """
        self.exploration_c = exploration_c
        self.rollout_depth = rollout_depth
        self.random = random.Random(seed)

    def best_child(self, node: MCTSNode) -> MCTSNode:
        """
        Seleciona filho pelo critério UCT.

        Args:
            node (MCTSNode): Nó pai.

        Returns:
            MCTSNode: Filho com melhor score UCT.
        """
        best_score = -float("inf")
        best = None

        for child in node.children.values():
            if child.visits == 0:
                score = float("inf")
            else:
                exploit = child.q()
                explore = self.exploration_c * math.sqrt(math.log(max(1, node.visits)) / child.visits)
                score = exploit + explore
            if score > best_score:
                best_score = score
                best = child

        return best  # type: ignore[return-value]

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Fase de seleção: desce pela árvore até nó expansível/terminal.

        Args:
            node (MCTSNode): Nó raiz atual.

        Returns:
            MCTSNode: Nó selecionado.
        """
        current = node
        while not current.is_terminal and not current.untried_moves and current.children:
            current = self.best_child(current)
        return current

    def expand(self, node: MCTSNode) -> MCTSNode:
        """
        Fase de expansão: cria um novo filho a partir de jogada não explorada.

        Args:
            node (MCTSNode): Nó a expandir.

        Returns:
            MCTSNode: Filho criado (ou próprio nó se não expansível).
        """
        if node.is_terminal or not node.untried_moves:
            return node

        move = self.random.choice(node.untried_moves)
        node.untried_moves.remove(move)

        next_state = node.state.clone()
        mover = next_state.current_player
        next_state.apply_move(move, switch_player=True)

        winner = evaluate_after_move(next_state, mover=mover)
        child = MCTSNode(state=next_state, parent=node, move_from_parent=move, mover=mover, terminal_winner=winner)
        node.children[move] = child
        return child

    def simulate(self, node: MCTSNode) -> float:
        """
        Fase de simulação (rollout) a partir do nó.

        Retorna recompensa do ponto de vista do jogador que vai jogar no nó pai,
        usando:
        - 1.0 vitória do jogador raiz
        - 0.0 derrota
        - 0.5 empate

        Args:
            node (MCTSNode): Nó de arranque do rollout.

        Returns:
            float: Resultado numérico da simulação.
        """
        state = node.state.clone()
        history = [board_signature(state)]

        if node.parent is None:
            root_player = state.current_player
        else:
            root_player = node.parent.state.current_player

        if node.terminal_winner != 0:
            return 1.0 if node.terminal_winner == root_player else 0.0

        for _ in range(self.rollout_depth):
            if is_draw(state, history):
                return 0.5

            legal = state.legal_moves(state.current_player)
            if not legal:
                return 0.5

            move = self.random.choice(legal)
            mover = state.current_player
            state.apply_move(move, switch_player=True)
            history.append(board_signature(state))

            winner = evaluate_after_move(state, mover=mover)
            if winner != 0:
                return 1.0 if winner == root_player else 0.0

        return 0.5

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        """
        Fase de retropropagação: atualiza estatísticas até à raiz.

        Args:
            node (MCTSNode): Nó final da iteração.
            reward (float): Recompensa da simulação.
        """
        current = node
        r = reward
        while current is not None:
            current.visits += 1
            current.value_sum += r
            current = current.parent
            r = 1.0 - r

    def run(self, root_state: PopOutBoard, iterations: int = 300) -> Move:
        """
        Executa iterações MCTS e devolve a melhor jogada por visitas.

        Args:
            root_state (PopOutBoard): Estado inicial.
            iterations (int): Número de iterações.

        Returns:
            Move: Jogada escolhida.
        """
        root = MCTSNode(state=root_state.clone())

        if not root.untried_moves:
            raise ValueError("Estado sem jogadas legais.")

        for _ in range(iterations):
            leaf = self.select(root)
            child = self.expand(leaf)
            reward = self.simulate(child)
            self.backpropagate(child, reward)

        best_move, _ = max(root.children.items(), key=lambda kv: kv[1].visits)
        return best_move
