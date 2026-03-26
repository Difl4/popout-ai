"""Módulo de representação do tabuleiro PopOut usando estrutura eficiente."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

ROWS = 6
COLS = 7
CONNECT_N = 4


@dataclass
class PopOutBoard:
    """
    Representa o estado do jogo PopOut.

    O tabuleiro é armazenado como matriz [linha][coluna], onde:
    - 0: vazio
    - 1: jogador 1
    - 2: jogador 2
    """

    rows: int = ROWS
    cols: int = COLS
    board: List[List[int]] = field(default_factory=lambda: [[0] * COLS for _ in range(ROWS)])
    current_player: int = 1

    def clone(self) -> "PopOutBoard":
        """
        Cria uma cópia profunda do estado atual.

        Returns:
            PopOutBoard: Novo objeto com o mesmo estado.
        """
        new_board = PopOutBoard(rows=self.rows, cols=self.cols)
        new_board.board = [row[:] for row in self.board]
        new_board.current_player = self.current_player
        return new_board

    def legal_drop_moves(self) -> List[int]:
        """
        Lista colunas válidas para jogada de drop.

        Returns:
            List[int]: Colunas onde o topo ainda está livre.
        """
        return [c for c in range(self.cols) if self.board[0][c] == 0]

    def legal_pop_moves(self, player: Optional[int] = None) -> List[int]:
        """
        Lista colunas válidas para jogada pop do jogador.

        Args:
            player (Optional[int]): Jogador alvo (usa current_player se None).

        Returns:
            List[int]: Colunas onde a célula da base pertence ao jogador.
        """
        p = player if player is not None else self.current_player
        return [c for c in range(self.cols) if self.board[self.rows - 1][c] == p]

    def legal_moves(self, player: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Lista todas as jogadas legais (drop e pop).

        Args:
            player (Optional[int]): Jogador alvo (usa current_player se None).

        Returns:
            List[Tuple[str, int]]: Tuplos no formato ('drop'|'pop', coluna).
        """
        p = player if player is not None else self.current_player
        moves = [("drop", c) for c in self.legal_drop_moves()]
        moves.extend(("pop", c) for c in self.legal_pop_moves(p))
        return moves

    def apply_drop(self, col: int, player: Optional[int] = None) -> bool:
        """
        Executa uma jogada drop numa coluna.

        Args:
            col (int): Coluna da jogada.
            player (Optional[int]): Jogador que joga (usa current_player se None).

        Returns:
            bool: True se a jogada foi aplicada, False se inválida.
        """
        p = player if player is not None else self.current_player
        if col < 0 or col >= self.cols or self.board[0][col] != 0:
            return False

        for r in range(self.rows - 1, -1, -1):
            if self.board[r][col] == 0:
                self.board[r][col] = p
                return True
        return False

    def apply_pop(self, col: int, player: Optional[int] = None) -> bool:
        """
        Executa uma jogada pop removendo a peça da base da coluna.

        Após a remoção, todas as peças acima descem uma posição.

        Args:
            col (int): Coluna da jogada.
            player (Optional[int]): Jogador que joga (usa current_player se None).

        Returns:
            bool: True se a jogada foi aplicada, False se inválida.
        """
        p = player if player is not None else self.current_player
        if col < 0 or col >= self.cols:
            return False
        if self.board[self.rows - 1][col] != p:
            return False

        for r in range(self.rows - 1, 0, -1):
            self.board[r][col] = self.board[r - 1][col]
        self.board[0][col] = 0
        return True

    def apply_move(self, move: Tuple[str, int], switch_player: bool = True) -> bool:
        """
        Aplica uma jogada genérica ('drop' ou 'pop').

        Args:
            move (Tuple[str, int]): Jogada no formato (tipo, coluna).
            switch_player (bool): Alterna jogador após jogada válida.

        Returns:
            bool: True se aplicada com sucesso.
        """
        move_type, col = move
        ok = False
        if move_type == "drop":
            ok = self.apply_drop(col, self.current_player)
        elif move_type == "pop":
            ok = self.apply_pop(col, self.current_player)

        if ok and switch_player:
            self.current_player = 2 if self.current_player == 1 else 1
        return ok

    def is_full(self) -> bool:
        """
        Verifica se o tabuleiro está cheio (sem drops possíveis).

        Returns:
            bool: True se não houver colunas livres no topo.
        """
        return all(self.board[0][c] != 0 for c in range(self.cols))

    def to_feature_dict(self) -> dict:
        """
        Converte o estado em dicionário de features para ID3.

        Returns:
            dict: Mapeamento célula->valor e jogador atual.
        """
        features = {"current_player": self.current_player}
        for r in range(self.rows):
            for c in range(self.cols):
                features[f"cell_{r}_{c}"] = self.board[r][c]
        return features

    def __str__(self) -> str:
        """
        Representação textual do tabuleiro.

        Returns:
            str: Tabuleiro formatado para debug.
        """
        symbols = {0: ".", 1: "X", 2: "O"}
        lines = []
        for row in self.board:
            lines.append(" ".join(symbols[v] for v in row))
        lines.append("0 1 2 3 4 5 6")
        return "\n".join(lines)
