"""Testes de integração para fluxos completos do projeto PopOut."""

from __future__ import annotations

import pytest
import tempfile
from pathlib import Path

from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import evaluate_after_move
from src.mcts.standard.uct_standard import StandardUCT
from src.decision_tree.id3.learner import ID3Classifier
from src.decision_tree.dataset_generator import generate_dataset


class TestFullMCTSGameFlow:
    """Testa um fluxo completo de jogo com MCTS."""
    
    def test_mcts_plays_complete_game(self):
        """MCTS deve conseguir jogar um jogo completo sem erros."""
        board = PopOutBoard()
        ai = StandardUCT(seed=42)
        moves_played = 0
        max_moves = 100
        
        while moves_played < max_moves:
            move = ai.run(board, iterations=50)
            player = board.current_player
            
            # Move deve ser legal
            assert move in board.legal_moves()
            
            board.apply_move(move)
            moves_played += 1
            
            winner = evaluate_after_move(board, mover=player)
            if winner:
                assert winner in [1, 2]
                break
    
    def test_mcts_deterministic_with_seed(self):
        """MCTS com mesmo seed deve produzir mesmo resultado."""
        board = PopOutBoard()
        
        ai1 = StandardUCT(seed=42)
        move1 = ai1.run(board, iterations=100)
        
        ai2 = StandardUCT(seed=42)
        move2 = ai2.run(board, iterations=100)
        
        assert move1 == move2


class TestID3TrainingPipeline:
    """Testa fluxo completo de treinamento ID3."""
    
    def test_dataset_generation_and_training(self):
        """Deve gerar dataset e treinar ID3 com sucesso."""
        # Gerar dataset
        df = generate_dataset(
            variant='uct_standard',
            n_samples=50,
            iterations=100,
            seed=42
        )
        
        assert len(df) > 0
        assert 'best_move' in df.columns
        assert df['best_move'].nunique() > 0
        
        # Treinar classifier
        classifier = ID3Classifier()
        classifier.fit(df, target='best_move')
        
        # Testar score
        score = classifier.score(df, target='best_move')
        assert 0 <= score <= 1
        assert score > 0.5  # Deve ter accuracy razoável
    
    def test_id3_predictions_are_valid(self):
        """Predições ID3 devem ser labels válidas do dataset."""
        df = generate_dataset(
            variant='uct_standard',
            n_samples=50,
            iterations=100,
            seed=42
        )
        
        classifier = ID3Classifier()
        classifier.fit(df, target='best_move')
        
        # Fazer predições
        predictions = classifier.predict(df)
        
        # Todas predições devem estar no conjunto de labels
        valid_moves = set(df['best_move'].unique())
        for pred in predictions:
            assert pred in valid_moves


class TestBoardStateEquality:
    """Testa igualdade e hashing de boards."""
    
    def test_board_equality(self):
        """Boards com mesmo estado devem ser iguais."""
        board1 = PopOutBoard()
        board2 = PopOutBoard()
        
        assert board1 == board2
    
    def test_board_inequality_after_move(self):
        """Boards divergentes devem ser diferentes."""
        board1 = PopOutBoard()
        board2 = PopOutBoard()
        
        board1.apply_move(0)
        
        assert board1 != board2
    
    def test_board_hashable(self):
        """Board deve ser hashable e usável em sets/dicts."""
        board1 = PopOutBoard()
        board2 = PopOutBoard()
        board2.apply_move(0)
        
        # Deve conseguir criar um set com boards
        board_set = {board1, board2}
        assert len(board_set) == 2
        
        # Deve conseguir usar como chave de dict
        board_dict = {board1: "inicial", board2: "depois de move"}
        assert board_dict[board1] == "inicial"
        assert board_dict[board2] == "depois de move"
    
    def test_board_hash_consistency(self):
        """Hash do mesmo board deve ser sempre igual."""
        board = PopOutBoard()
        board.apply_move(3)
        
        hash1 = hash(board)
        hash2 = hash(board)
        
        assert hash1 == hash2


class TestGameFlowWithDifferentModes:
    """Testa diferentes modos de jogo."""
    
    def test_pvp_game_flow(self):
        """Simula um jogo Player vs Player."""
        board = PopOutBoard()
        moves_played = 0
        
        # Move alternadamente
        player_moves = [3, 3, 2, 2, 1, 1, 0]
        
        for move in player_moves:
            if move in board.legal_moves():
                player = board.current_player
                board.apply_move(move)
                moves_played += 1
                
                winner = evaluate_after_move(board, mover=player)
                if winner:
                    break
        
        assert moves_played > 0
    
    def test_pvai_game_flow(self):
        """Simula Player vs AI com clima de gameplay real."""
        board = PopOutBoard()
        ai = StandardUCT(seed=42)
        moves_played = 0
        max_moves = 50
        
        human_moves = [3, 2, 1, 0]  # Predefinido
        move_idx = 0
        
        while moves_played < max_moves:
            player = board.current_player
            
            if player == 1 and move_idx < len(human_moves):
                # Human move
                move = human_moves[move_idx]
                move_idx += 1
            else:
                # AI move
                move = ai.run(board, iterations=50)
            
            if move not in board.legal_moves():
                break
            
            board.apply_move(move)
            moves_played += 1
            
            winner = evaluate_after_move(board, mover=player)
            if winner:
                break
        
        assert moves_played > 0


class TestBulkDatasetGeneration:
    """Testa geração de datasets em lote."""
    
    def test_generate_multiple_samples_different_seeds(self):
        """Datasets com seeds diferentes devem conter amostras diferentes."""
        df1 = generate_dataset(variant='uct_standard', n_samples=20, seed=1)
        df2 = generate_dataset(variant='uct_standard', n_samples=20, seed=2)
        
        # Devem ter dados gerados
        assert len(df1) > 0
        assert len(df2) > 0
        
        # Não devem ser exatamente iguais
        assert not df1.equals(df2)
    
    def test_dataset_structure_consistency(self):
        """Dataset deve ter estrutura consistente."""
        df = generate_dataset(
            variant='uct_standard',
            n_samples=30,
            iterations=100
        )
        
        # Verificar colunas esperadas
        assert 'best_move' in df.columns
        
        # Verificar tipos de dados
        assert all(isinstance(move, str) for move in df['best_move'])
        
        # Verificar nomenclatura de moves (drop_N ou pop_N)
        for move in df['best_move'].unique():
            assert move.startswith('drop_') or move.startswith('pop_')


class TestEndToEndPipeline:
    """Testa pipeline completo: Gerar dados → Treinar → Prever."""
    
    def test_complete_pipeline(self):
        """Executa pipeline completo com sucesso."""
        # 1. Gerar dados
        dataset = generate_dataset(
            variant='uct_standard',
            n_samples=40,
            iterations=100,
            seed=42
        )
        assert len(dataset) > 0
        
        # 2. Treinar modelo
        model = ID3Classifier()
        model.fit(dataset, target='best_move')
        
        # 3. Avaliar
        train_score = model.score(dataset, target='best_move')
        assert train_score > 0.4
        
        # 4. Fazer predições
        sample = dataset.iloc[:5]
        predictions = model.predict(sample)
        actuals = sample['best_move'].tolist()
        
        # Verificar que predições são válidas
        assert len(predictions) == len(actuals)
        assert all(isinstance(p, str) for p in predictions)
    
    def test_pipeline_with_different_mcts_variants(self):
        """Validar que pipeline funciona com diferentes variantes de MCTS."""
        for variant in ['uct_standard', 'uct_experimental']:
            dataset = generate_dataset(
                variant=variant,
                n_samples=20,
                iterations=50,
                seed=42
            )
            assert len(dataset) > 0
            
            model = ID3Classifier()
            model.fit(dataset, target='best_move')
            
            score = model.score(dataset, target='best_move')
            assert 0 <= score <= 1
