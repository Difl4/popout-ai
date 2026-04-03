"""MCTS-Solver: variante com prova de vitória/derrota e distância minimax.

Status de cada nó — da perspectiva de state.current_player:
  STATUS_UNKNOWN (0) — ainda não provado
  STATUS_WIN     (1) — o jogador a mover tem vitória forçada
  STATUS_LOSS    (2) — o jogador a mover está em derrota forçada
  STATUS_DRAW    (3) — empate forçado

Distância (d): número de plies até ao estado terminal.
  - Nó terminal           → d = 0
  - Vitória provada (OR)  → d = min(d_filhos_LOSS) + 1
  - Derrota provada (AND) → d = max(d_filhos_WIN)  + 1  (atrasa a derrota)

Selecção:
  1. Se existirem filhos LOSS → escolher o de menor d  (matar mais depressa)
  2. Se todos os filhos forem WIN → escolher o de maior d (aguentar mais)
  3. Caso contrário → UCB1 nos filhos UNKNOWN


"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.algorithms.mcts.standard.base import BaseMCTS, Move
from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import board_signature, evaluate_after_move

# ── Constantes de status ──────────────────────────────────────────────────────
STATUS_UNKNOWN: int = 0
STATUS_WIN: int = 1
STATUS_LOSS: int = 2
STATUS_DRAW: int = 3


# ── Nó do solver ─────────────────────────────────────────────────────────────

@dataclass
class SolverNode:
    """Nó MCTS estendido com status provado e distância minimax."""

    state: PopOutBoard
    parent: Optional["SolverNode"] = None
    move_from_parent: Optional[Move] = None
    mover: Optional[int] = None
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[Move, "SolverNode"] = field(default_factory=dict)
    untried_moves: List[Move] = field(default_factory=list)
    terminal_winner: int = 0
    status: int = STATUS_UNKNOWN
    distance: int = 0

    def __post_init__(self) -> None:
        self.untried_moves = self.state.legal_moves()
        if self.terminal_winner != 0:
            # Verificar quem realmente ganhou em relação ao current_player
            if self.terminal_winner == self.state.current_player:
                # O current_player ganhou (ex: pop do adversário deu-nos 4 em linha)
                self.status = STATUS_WIN
            else:
                # O current_player perdeu (o jogador anterior ganhou)
                self.status = STATUS_LOSS
            self.distance = 0
        elif not self.untried_moves:
            # Sem vencedor e sem jogadas legais → empate
            self.status = STATUS_DRAW
            self.distance = 0

    @property
    def is_terminal(self) -> bool:
        return self.terminal_winner != 0 or (
            self.status == STATUS_DRAW and not self.untried_moves and not self.children
        )

    def q(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


# ── SolverMCTS ────────────────────────────────────────────────────────────────

class SolverMCTS(BaseMCTS):
    """MCTS com prova de vitória: selecção por status/distância + retroprop de prova.

    Herda BaseMCTS e sobrepõe expand, best_child, simulate, backpropagate e run.
    Todos os nós criados são SolverNode (compatíveis com a interface de MCTSNode).
    """

    # ── Expansão ───────────────────────────────────────────────────────────────

    def expand(self, node: SolverNode) -> SolverNode:  # type: ignore[override]
        if node.is_terminal or not node.untried_moves:
            return node

        # O(1) swap-pop em vez de list.remove()
        idx = self.random.randrange(len(node.untried_moves))
        move = node.untried_moves[idx]
        node.untried_moves[idx] = node.untried_moves[-1]
        node.untried_moves.pop()

        next_state = node.state.clone()
        mover = next_state.current_player
        next_state.apply_move(move)
        winner = evaluate_after_move(next_state, mover=mover)

        child = SolverNode(
            state=next_state,
            parent=node,
            move_from_parent=move,
            mover=mover,
            terminal_winner=winner,
        )
        node.children[move] = child
        return child

    # ── Selecção ───────────────────────────────────────────────────────────────

    def select(self, node: SolverNode) -> SolverNode:  # type: ignore[override]
        """Desce a árvore evitando subárvores já provadas."""
        current = node
        while (
            not current.is_terminal
            and current.status == STATUS_UNKNOWN
            and not current.untried_moves
            and current.children
        ):
            current = self.best_child(current)
        return current

    def best_child(self, node: SolverNode) -> SolverNode:  # type: ignore[override]
        """Prioridade: vitória provada (min d) > UCB1 > atrasar derrota (max d)."""
        children = list(node.children.values())

        # 1. Filhos LOSS → jogamos para lá (são vitórias para nós)
        loss_children = [c for c in children if c.status == STATUS_LOSS]
        if loss_children:
            return min(loss_children, key=lambda c: c.distance)

        # 2. Sem filhos desconhecidos nem movimentos por explorar → tudo provado
        unknown_children = [c for c in children if c.status == STATUS_UNKNOWN]
        if not unknown_children and not node.untried_moves:
            # Melhor opção disponível: atrasar a derrota o máximo possível
            return max(children, key=lambda c: c.distance)

        # 3. UCB1 nos filhos ainda desconhecidos
        candidates = unknown_children if unknown_children else children
        log_n = math.log(max(1, node.visits))
        best_score = -float("inf")
        best = candidates[0]
        for child in candidates:
            if child.visits == 0:
                score = float("inf")
            else:
                score = child.q() + self.exploration_c * math.sqrt(log_n / child.visits)
            if score > best_score:
                best_score = score
                best = child
        return best

    # ── Simulação com detecção imediata de vitória ─────────────────────────────

    def simulate(self, node: SolverNode) -> float:  # type: ignore[override]
        """Rollout com verificação de vitória a 1 movimento em cada passo."""
        state = node.state.clone()
        initial_mover = node.mover if node.mover is not None else (3 - state.current_player)

        if node.terminal_winner != 0:
            return 1.0 if node.terminal_winner == initial_mover else 0.0

        sig_counter: Counter = Counter([board_signature(state)])
        choice = self.random.choice

        for _ in range(self.rollout_depth):
            if state.is_full():
                return 0.5
            if any(count >= 3 for count in sig_counter.values()):
                return 0.5

            legal = state.legal_moves()
            if not legal:
                return 0.5

            curr_p = state.current_player

            # Jogar imediatamente se curr_p tiver vitória a 1 movimento
            for move in legal:
                test = state.clone()
                test.apply_move(move)
                if evaluate_after_move(test, mover=curr_p) == curr_p:
                    return 1.0 if curr_p == initial_mover else 0.0

            # Sem vitória imediata — movimento aleatório
            state.apply_move(choice(legal))

            # Verificar se o movimento aleatório resultou em vitória (ex: pop infeliz)
            winner = evaluate_after_move(state, mover=curr_p)
            if winner != 0:
                return 1.0 if winner == initial_mover else 0.0

            sig_counter[board_signature(state)] += 1

        return 0.5

    # ── Retropropagação com prova ──────────────────────────────────────────────

    def backpropagate(self, node: SolverNode, reward: float) -> None:  # type: ignore[override]
        super().backpropagate(node, reward)
        self._propagate_status(node)

    def _propagate_status(self, node: SolverNode) -> None:
        """Propaga status/distância provados de baixo para cima até à raiz."""
        current = node.parent
        while current is not None:
            if not self._update_node_status(current):
                break
            current = current.parent

    def _update_node_status(self, node: SolverNode) -> bool:
        """Recalcula status e distância do nó com base nos filhos conhecidos.

        Returns:
            True se houve alteração (status ou distância mudou).
        """
        if node.is_terminal:
            return False

        children = list(node.children.values())
        if not children:
            return False

        # ── Vitória: existe pelo menos um filho LOSS ──────────────────────────
        loss_children = [c for c in children if c.status == STATUS_LOSS]
        if loss_children:
            new_status = STATUS_WIN
            new_distance = min(c.distance for c in loss_children) + 1
            if node.status == new_status and node.distance == new_distance:
                return False
            node.status = new_status
            node.distance = new_distance
            return True

        # ── Só conclui LOSS/DRAW quando não há movimentos inexplorados ────────
        fully_explored = (
            not node.untried_moves
            and all(c.status != STATUS_UNKNOWN for c in children)
        )
        if not fully_explored:
            return False

        win_children = [c for c in children if c.status == STATUS_WIN]
        draw_children = [c for c in children if c.status == STATUS_DRAW]

        if draw_children:
            # Podemos forçar um empate — melhor do que perder
            new_status = STATUS_DRAW
            new_distance = 0
        elif win_children:
            # Derrota forçada — atrasar o máximo possível
            new_status = STATUS_LOSS
            new_distance = max(c.distance for c in win_children) + 1
        else:
            return False

        if node.status == new_status and node.distance == new_distance:
            return False
        node.status = new_status
        node.distance = new_distance
        return True

    # ── Escolha final da jogada ────────────────────────────────────────────────

    def get_best_move(self, root: SolverNode) -> Move:
        """Vitória provada (menor d) > empate > atrasar derrota > mais visitado."""
        # Proven win → kill fastest
        loss_children = [
            (m, c) for m, c in root.children.items() if c.status == STATUS_LOSS
        ]
        if loss_children:
            return min(loss_children, key=lambda mc: mc[1].distance)[0]

        # Proven draw → take it (better than any loss)
        if root.status == STATUS_DRAW:
            draw_children = [
                (m, c) for m, c in root.children.items() if c.status == STATUS_DRAW
            ]
            if draw_children:
                return draw_children[0][0]

        # Proven loss → delay as long as possible instead of picking randomly
        if root.status == STATUS_LOSS:
            return max(root.children.items(), key=lambda mc: mc[1].distance)[0]

        # No proof available → standard MCTS (most visited)
        best_move, _ = max(root.children.items(), key=lambda mc: mc[1].visits)
        return best_move

    def run(self, root_state: PopOutBoard, iterations: int = 10_000) -> Move:
        root = SolverNode(state=root_state.clone())
        if not root.untried_moves:
            raise ValueError("Estado sem jogadas legais.")

        for _ in range(iterations):
            if root.status != STATUS_UNKNOWN:
                break  # Solução provada — terminar cedo
            leaf = self.select(root)
            child = self.expand(leaf)
            reward = self.simulate(child)
            self.backpropagate(child, reward)

        return self.get_best_move(root)
