"""Gerenciamento persistente de estado de jogo (save/load)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import NamedTuple

from src.engine.standard.bitboard import PopOutBoard
from src.interfaces.gui.state import Difficulty


class GameState(NamedTuple):
    """Encapsula estado completo de um jogo.
    
    Attributes:
        board (PopOutBoard): Estado do tabuleiro.
        mode_pop (bool): Se está em modo POP (True) ou DROP (False).
        game_mode (str): "PvP" ou "IA".
        difficulty (Difficulty): Nível de dificuldade.
        ai_iterations (int): Número de iterações MCTS.
        message (str): Mensagem de feedback do último movimento.
        game_over (bool): Se o jogo terminou.
        winner (int): Jogador vencedor (0 se não há).
        move_count (int): Número de movimentos realizados.
    """
    board: PopOutBoard
    mode_pop: bool
    game_mode: str
    difficulty: Difficulty
    ai_iterations: int
    message: str = ""
    game_over: bool = False
    winner: int = 0
    move_count: int = 0


class GameSaveManager:
    """Gerencia save/load de estado de jogo com pickle.
    
    Attributes:
        save_dir (Path): Diretório onde guardar arquivos de save.
        filename (str): Nome do arquivo de save padrão.
    """
    
    def __init__(
        self,
        save_dir: str | Path = "./data/saves",
        filename: str = "autosave.pkl"
    ) -> None:
        """Inicializa manager de save.
        
        Args:
            save_dir (str | Path): Diretório para guardar saves.
            filename (str): Nome do arquivo de save padrão.
        """
        self.save_dir = Path(save_dir)
        self.filename = filename
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def save_path(self) -> Path:
        """Retorna caminho completo do arquivo de save."""
        return self.save_dir / self.filename
    
    @property
    def has_save(self) -> bool:
        """Verifica se existe um save guardado."""
        return self.save_path.exists()
    
    def save_game(self, state: GameState) -> None:
        """Guarda estado de jogo em arquivo.
        
        Args:
            state (GameState): Estado do jogo a guardar.
            
        Raises:
            IOError: Se não conseguir escrever o arquivo.
        """
        try:
            with open(self.save_path, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise IOError(f"Erro ao guardar jogo: {e}")
    
    def load_game(self) -> GameState | None:
        """Carrega estado de jogo de arquivo.
        
        Returns:
            GameState | None: Estado do jogo ou None se não existir.
            
        Raises:
            IOError: Se houver erro ao ler o arquivo.
        """
        if not self.has_save:
            return None
        
        try:
            with open(self.save_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise IOError(f"Erro ao carregar jogo: {e}")
    
    def delete_save(self) -> None:
        """Apaga arquivo de save.
        
        Raises:
            IOError: Se não conseguir apagar o arquivo.
        """
        if self.has_save:
            try:
                self.save_path.unlink()
            except Exception as e:
                raise IOError(f"Erro ao apagar save: {e}")
    
    def list_saves(self) -> list[Path]:
        """Lista todos os arquivos de save disponíveis.
        
        Returns:
            list[Path]: Lista de caminhos de save encontrados.
        """
        return sorted(self.save_dir.glob("*.pkl"))
    
    def get_save_info(self) -> dict | None:
        """Obtém informações sobre o save (sem carregar completo).
        
        Returns:
            dict | None: Dicionário com info ou None.
        """
        state = self.load_game()
        if state is None:
            return None
        
        return {
            "game_mode": state.game_mode,
            "difficulty": state.difficulty.value[1],
            "move_count": state.move_count,
            "game_over": state.game_over,
            "winner": state.winner if state.game_over else None,
        }


if __name__ == "__main__":
    # Exemplo de uso
    from src.engine.standard.bitboard import PopOutBoard
    
    # Cria estado de exemplo
    board = PopOutBoard()
    state = GameState(
        board=board,
        mode_pop=False,
        game_mode="PvP",
        difficulty=Difficulty.MEDIUM,
        ai_iterations=500,
        message="Jogo iniciado",
        game_over=False,
        winner=0,
        move_count=0
    )
    
    # Guarda e carrega
    manager = GameSaveManager()
    print(f"💾 Guardando jogo em: {manager.save_path}")
    manager.save_game(state)
    
    loaded = manager.load_game()
    if loaded:
        print(f"✅ Jogo carregado: Modo={loaded.game_mode}, Dificuldade={loaded.difficulty.value[1]}")
        print(f"   Info: {manager.get_save_info()}")
