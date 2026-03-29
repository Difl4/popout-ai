"""Base MCTS com os 4 passos: seleção, expansão, simulação e retropropagação."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import board_signature, evaluate_after_move

# JOGADA AGORA É APENAS UM INTEIRO (0-13)
Move = int

@dataclass
class MCTSNode:
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
        self.untried_moves = self.state.legal_moves()

    @property
    def is_terminal(self) -> bool:
        return self.terminal_winner != 0

    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

class BaseMCTS:
    def __init__(self, exploration_c: float = 1.414, rollout_depth: int = 50, seed: Optional[int] = None) -> None:
        self.exploration_c = exploration_c
        self.rollout_depth = rollout_depth
        self.random = random.Random(seed)

    def best_child(self, node: MCTSNode) -> MCTSNode:
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
        return best

    def select(self, node: MCTSNode) -> MCTSNode:
        current = node
        while not current.is_terminal and not current.untried_moves and current.children:
            current = self.best_child(current)
        return current

    def expand(self, node: MCTSNode) -> MCTSNode:
        if node.is_terminal or not node.untried_moves:
            return node

        move = self.random.choice(node.untried_moves)
        node.untried_moves.remove(move)

        next_state = node.state.clone()
        mover = next_state.current_player
        next_state.apply_move(move)

        winner = evaluate_after_move(next_state, mover=mover)
        child = MCTSNode(state=next_state, parent=node, move_from_parent=move, mover=mover, terminal_winner=winner)
        node.children[move] = child
        return child

    def simulate(self, node: MCTSNode) -> float:
        state = node.state.clone()
        initial_mover = node.mover if node.mover is not None else (3 - state.current_player)

        if node.terminal_winner != 0:
            return 1.0 if node.terminal_winner == initial_mover else 0.0

        # OTIMIZAÇÃO: Usar Counter para O(1) lookups em vez de lista
        sig_counter: Counter = Counter([board_signature(state)])
        choice = self.random.choice

        for _ in range(self.rollout_depth):
            if state.is_full():
                return 0.5

            # OTIMIZAÇÃO: Verificar threefold directamente (O(1) lookup)
            if any(count >= 3 for count in sig_counter.values()):
                return 0.5

            legal = state.legal_moves()
            if not legal:
                return 0.5

            move = choice(legal)
            curr_p = state.current_player
            state.apply_move(move)

            winner = evaluate_after_move(state, mover=curr_p)
            if winner != 0:
                return 1.0 if winner == initial_mover else 0.0

            # OTIMIZAÇÃO: Incrementar counter (O(1)) em vez de append a lista
            sig = board_signature(state)
            sig_counter[sig] += 1

        return 0.5

    def backpropagate(self, node: MCTSNode, reward: float) -> None:
        current = node
        r = reward
        while current is not None:
            current.visits += 1
            current.value_sum += r
            current = current.parent
            r = 1.0 - r

    def run(self, root_state: PopOutBoard, iterations: int = 10000) -> Move:
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
