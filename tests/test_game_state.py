"""Testes para gerenciamento de estado de jogo (save/load)."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest

from src.engine.standard.bitboard import PopOutBoard
from src.game_state import GameState, GameSaveManager, Difficulty


class TestGameState:
    """Testa classe GameState."""
    
    def test_game_state_creation(self) -> None:
        """Deve criar estado de jogo válido."""
        board = PopOutBoard()
        state = GameState(
            board=board,
            mode_pop=False,
            game_mode="PvP",
            difficulty=Difficulty.MEDIUM,
            ai_iterations=500,
            message="Teste",
            game_over=False,
            winner=0,
            move_count=0
        )
        
        assert state.board == board
        assert state.mode_pop is False
        assert state.game_mode == "PvP"
        assert state.difficulty == Difficulty.MEDIUM
        assert state.ai_iterations == 500
        assert state.message == "Teste"
        assert state.game_over is False
        assert state.winner == 0
        assert state.move_count == 0
    
    def test_game_state_defaults(self) -> None:
        """Deve usar valores padrão para campos opcionais."""
        board = PopOutBoard()
        state = GameState(
            board=board,
            mode_pop=True,
            game_mode="IA",
            difficulty=Difficulty.HARD,
            ai_iterations=1000
        )
        
        assert state.message == ""
        assert state.game_over is False
        assert state.winner == 0
        assert state.move_count == 0


class TestGameSaveManager:
    """Testa classe GameSaveManager."""
    
    @pytest.fixture
    def temp_save_dir(self) -> Path:
        """Cria diretório temporário para saves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_manager_creation(self, temp_save_dir: Path) -> None:
        """Deve criar manager com diretório de save."""
        manager = GameSaveManager(save_dir=temp_save_dir)
        
        assert manager.save_dir == temp_save_dir
        assert manager.filename == "autosave.pkl"
        assert manager.save_dir.exists()
    
    def test_save_path_property(self, temp_save_dir: Path) -> None:
        """Deve retornar caminho completo de save."""
        manager = GameSaveManager(save_dir=temp_save_dir)
        expected = temp_save_dir / "autosave.pkl"
        
        assert manager.save_path == expected
    
    def test_has_save_initially_false(self, temp_save_dir: Path) -> None:
        """Não deve ter save inicialmente."""
        manager = GameSaveManager(save_dir=temp_save_dir)
        
        assert not manager.has_save
    
    def test_save_and_load_game(self, temp_save_dir: Path) -> None:
        """Deve guardar e carregar jogo com sucesso."""
        board = PopOutBoard()
        state = GameState(
            board=board,
            mode_pop=True,
            game_mode="IA",
            difficulty=Difficulty.HARD,
            ai_iterations=1000,
            message="Teste save",
            game_over=False,
            winner=0,
            move_count=5
        )
        
        manager = GameSaveManager(save_dir=temp_save_dir)
        
        # Guarda
        manager.save_game(state)
        assert manager.has_save
        
        # Carrega
        loaded = manager.load_game()
        assert loaded is not None
        assert loaded.mode_pop == True
        assert loaded.game_mode == "IA"
        assert loaded.difficulty == Difficulty.HARD
        assert loaded.message == "Teste save"
        assert loaded.move_count == 5
    
    def test_load_game_none_when_no_save(self, temp_save_dir: Path) -> None:
        """Deve retornar None se não houver save."""
        manager = GameSaveManager(save_dir=temp_save_dir)
        
        loaded = manager.load_game()
        assert loaded is None
    
    def test_delete_save(self, temp_save_dir: Path) -> None:
        """Deve apagar arquivo de save."""
        board = PopOutBoard()
        state = GameState(
            board=board,
            mode_pop=False,
            game_mode="PvP",
            difficulty=Difficulty.MEDIUM,
            ai_iterations=500
        )
        
        manager = GameSaveManager(save_dir=temp_save_dir)
        manager.save_game(state)
        assert manager.has_save
        
        manager.delete_save()
        assert not manager.has_save
    
    def test_list_saves(self, temp_save_dir: Path) -> None:
        """Deve listar todos os saves disponíveis."""
        board = PopOutBoard()
        state1 = GameState(
            board=board,
            mode_pop=False,
            game_mode="PvP",
            difficulty=Difficulty.EASY,
            ai_iterations=100
        )
        state2 = GameState(
            board=board,
            mode_pop=True,
            game_mode="IA",
            difficulty=Difficulty.HARD,
            ai_iterations=1000
        )
        
        manager = GameSaveManager(save_dir=temp_save_dir)
        
        # Guarda dois saves com nomes diferentes
        manager.save_game(state1)
        manager2 = GameSaveManager(save_dir=temp_save_dir, filename="save2.pkl")
        manager2.save_game(state2)
        
        saves = manager.list_saves()
        assert len(saves) == 2
        assert all(s.suffix == ".pkl" for s in saves)
    
    def test_get_save_info(self, temp_save_dir: Path) -> None:
        """Deve retornar informações sobre save sem carregar completo."""
        board = PopOutBoard()
        state = GameState(
            board=board,
            mode_pop=True,
            game_mode="IA",
            difficulty=Difficulty.HARD,
            ai_iterations=1000,
            game_over=True,
            winner=1,
            move_count=42
        )
        
        manager = GameSaveManager(save_dir=temp_save_dir)
        manager.save_game(state)
        
        info = manager.get_save_info()
        assert info is not None
        assert info["game_mode"] == "IA"
        assert "Difícil" in info["difficulty"]
        assert info["move_count"] == 42
        assert info["game_over"] is True
        assert info["winner"] == 1
    
    def test_get_save_info_none_when_no_save(self, temp_save_dir: Path) -> None:
        """Deve retornar None quando não há save."""
        manager = GameSaveManager(save_dir=temp_save_dir)
        
        info = manager.get_save_info()
        assert info is None
    
    def test_board_serialization(self, temp_save_dir: Path) -> None:
        """PopOutBoard deve ser serializável com pickle."""
        board = PopOutBoard()
        # Aplica alguns movimentos
        board.apply_move(0)  # DROP na coluna 0
        board.apply_move(1)  # DROP na coluna 1
        
        state = GameState(
            board=board,
            mode_pop=False,
            game_mode="PvP",
            difficulty=Difficulty.MEDIUM,
            ai_iterations=500
        )
        
        manager = GameSaveManager(save_dir=temp_save_dir)
        manager.save_game(state)
        loaded = manager.load_game()
        
        assert loaded is not None
        assert loaded.board.mask_p1 == board.mask_p1
        assert loaded.board.mask_p2 == board.mask_p2
        assert loaded.board.current_player == board.current_player
