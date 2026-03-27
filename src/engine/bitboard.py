"""Implementação do tabuleiro usando Bitboards e Move Encoding (0-13)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List

ROWS = 6
COLS = 7
COL_SIZE = 7 

@dataclass
class PopOutBoard:
    mask_p1: int = 0
    mask_p2: int = 0
    current_player: int = 1

    def clone(self) -> "PopOutBoard":
        return PopOutBoard(self.mask_p1, self.mask_p2, self.current_player)

    @property
    def merged_mask(self) -> int:
        return self.mask_p1 | self.mask_p2

    def legal_moves(self) -> List[int]:
        """
        Gera jogadas como inteiros:
        0-6: Drop na coluna 0-6
        7-13: Pop na coluna 0-6
        """
        moves = []
        full_mask = self.merged_mask
        target_mask = self.mask_p1 if self.current_player == 1 else self.mask_p2

        for c in range(COLS):
            # Check Drop: Verifica se o bit do topo da coluna está vazio
            if not (full_mask & (1 << (c * COL_SIZE + 5))):
                moves.append(c) # Move 0 a 6
            
            # Check Pop: Verifica se o bit da base da coluna pertence ao jogador
            if target_mask & (1 << (c * COL_SIZE)):
                moves.append(c + 7) # Move 7 a 13
                
        return moves

    def apply_move(self, move: int) -> None:
        """Aplica a jogada (0-13) diretamente."""
        p = self.current_player
        
        if move < 7:
            # DROP LOGIC
            col = move
            col_mask = (self.merged_mask >> (col * COL_SIZE)) & 0b111111
            new_piece = ((col_mask + 1) & ~col_mask) << (col * COL_SIZE)
            if p == 1: self.mask_p1 |= new_piece
            else: self.mask_p2 |= new_piece
        else:
            # POP LOGIC
            col = move - 7
            shift = col * COL_SIZE
            col_bit_mask = 0b111111 << shift
            
            self.mask_p1 = (self.mask_p1 & ~col_bit_mask) | ((self.mask_p1 & col_bit_mask) >> 1) & col_bit_mask
            self.mask_p2 = (self.mask_p2 & ~col_bit_mask) | ((self.mask_p2 & col_bit_mask) >> 1) & col_bit_mask
            
            # Limpa o "lixo" que possa ter subido para o 7º bit
            CLEAN_MASK = ~0x4081020408102040 
            self.mask_p1 &= CLEAN_MASK
            self.mask_p2 &= CLEAN_MASK

        # Troca de jogador com bitwise trick (3 - 1 = 2; 3 - 2 = 1)
        self.current_player = 3 - self.current_player

    def is_full(self) -> bool:
        # TOP_ROW_MASK: bit 5, 12, 19, 26, 33, 40, 47 representam a linha de topo
        # de cada coluna (linhas 0-6, posição 5 em cada coluna)
        TOP_ROW_MASK = 0x810204081020
        return (self.merged_mask & TOP_ROW_MASK) == TOP_ROW_MASK

    def to_feature_dict(self) -> dict:
        """
        Converte o estado do tabuleiro em dicionário de features para ML.
        
        Returns:
            dict: Dicionário com features categóricas por célula e jogador actual.
        """
        features = {}
        
        # Cell features: cada célula como 0 (vazia), 1 (jogador 1) ou 2 (jogador 2)
        for r in range(ROWS):
            for c in range(COLS):
                bit = 1 << (c * COL_SIZE + r)
                if self.mask_p1 & bit:
                    features[f"cell_{r}_{c}"] = 1
                elif self.mask_p2 & bit:
                    features[f"cell_{r}_{c}"] = 2
                else:
                    features[f"cell_{r}_{c}"] = 0
        
        # Jogador actual
        features["current_player"] = self.current_player
        
        return features

    def __str__(self) -> str:
        res = []
        for r in range(ROWS - 1, -1, -1):
            line = []
            for c in range(COLS):
                bit = 1 << (c * COL_SIZE + r)
                if self.mask_p1 & bit: line.append("X")
                elif self.mask_p2 & bit: line.append("O")
                else: line.append(".")
            res.append(" ".join(line))
        res.append("0 1 2 3 4 5 6")
        return "\n".join(res)