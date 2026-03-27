"""Testes para validação robusta de entrada (input validation)."""

import pytest
import pandas as pd
import numpy as np
import random

from src.algorithms.id3.learner import ID3Classifier
from src.engine.bitboard import PopOutBoard
from src.scripts.bulk_generate import make_agent, randomize_state, generate_dataset


class TestID3Validation:
    """Testes de validação para ID3Classifier.fit()."""

    def test_fit_none_dataframe(self):
        """fit() rejeita None como DataFrame."""
        clf = ID3Classifier()
        with pytest.raises(TypeError, match="Expected DataFrame"):
            clf.fit(None, "target")

    def test_fit_wrong_type_dataframe(self):
        """fit() rejeita tipos não-DataFrame."""
        clf = ID3Classifier()
        with pytest.raises(TypeError, match="Expected DataFrame"):
            clf.fit([1, 2, 3], "target")

    def test_fit_wrong_type_target(self):
        """fit() rejeita target não-string."""
        clf = ID3Classifier()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(TypeError, match="string"):
            clf.fit(df, 123)

    def test_fit_empty_dataframe(self):
        """fit() rejeita DataFrame vazio."""
        clf = ID3Classifier()
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            clf.fit(df, "target")

    def test_fit_missing_target_column(self):
        """fit() rejeita DataFrames sem coluna target."""
        clf = ID3Classifier()
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with pytest.raises(ValueError, match="not found"):
            clf.fit(df, "missing_column")

    def test_fit_with_nan_values(self):
        """fit() rejeita DataFrames com NaN."""
        clf = ID3Classifier()
        df = pd.DataFrame({
            "a": [1, np.nan],
            "b": [3, 4],
            "target": ["x", "y"]
        })
        with pytest.raises(ValueError, match="NaN"):
            clf.fit(df, "target")

    def test_fit_valid_dataframe(self):
        """fit() aceita DataFrame válido."""
        clf = ID3Classifier()
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "target": ["A", "B", "A"]
        })
        # Não deve lançar exceção
        clf.fit(df, "target")
        assert clf.root is not None
        assert clf.target_name == "target"

    def test_fit_multiple_columns(self):
        """fit() funciona com múltiplas features."""
        clf = ID3Classifier()
        df = pd.DataFrame({
            "col1": [1, 1, 2, 2],
            "col2": ["a", "b", "a", "b"],
            "col3": [10, 20, 30, 40],
            "label": ["yes", "no", "yes", "no"]
        })
        clf.fit(df, "label")
        assert clf.root is not None


class TestBitboardValidation:
    """Testes de validação para PopOutBoard.apply_move()."""

    def test_apply_move_wrong_type(self):
        """apply_move() rejeita tipos não-inteiros."""
        board = PopOutBoard()
        with pytest.raises(TypeError, match="integer"):
            board.apply_move("0")

    def test_apply_move_float(self):
        """apply_move() rejeita floats."""
        board = PopOutBoard()
        with pytest.raises(TypeError, match="integer"):
            board.apply_move(5.5)

    def test_apply_move_boolean(self):
        """apply_move() rejeita booleans (mesmo sendo int em Python)."""
        board = PopOutBoard()
        with pytest.raises(TypeError, match="integer"):
            board.apply_move(True)

    def test_apply_move_negative(self):
        """apply_move() rejeita moves negativos."""
        board = PopOutBoard()
        with pytest.raises(ValueError, match="range"):
            board.apply_move(-1)

    def test_apply_move_too_large(self):
        """apply_move() rejeita moves > 13."""
        board = PopOutBoard()
        with pytest.raises(ValueError, match="range"):
            board.apply_move(14)

    def test_apply_move_valid_drops(self):
        """apply_move() aceita drops válidos (0-6)."""
        board = PopOutBoard()
        for move in range(7):
            board_copy = PopOutBoard()
            board_copy.mask_p1 = board.mask_p1
            board_copy.mask_p2 = board.mask_p2
            board_copy.current_player = board.current_player
            # Não deve lançar exceção
            board_copy.apply_move(move)

    def test_apply_move_valid_pops(self):
        """apply_move() aceita pops válidos (7-13)."""
        board = PopOutBoard()
        # Primeiro fazer alguns drops
        board.apply_move(0)  # P1 drops
        board.apply_move(1)  # P2 drops
        board.apply_move(2)  # P1 drops
        # Agora fazer pops
        for move in range(7, 14):
            board_copy = PopOutBoard()
            board_copy.mask_p1 = board.mask_p1
            board_copy.mask_p2 = board.mask_p2
            board_copy.current_player = board.current_player
            # Não deve lançar exceção
            board_copy.apply_move(move)

    def test_apply_move_boundary_0(self):
        """apply_move() aceita 0 (limite inferior)."""
        board = PopOutBoard()
        board.apply_move(0)  # Sem erro
        assert board.current_player == 2

    def test_apply_move_boundary_13(self):
        """apply_move() aceita 13 (limite superior)."""
        board = PopOutBoard()
        # Setup com algumas peças
        board.apply_move(0)
        board.apply_move(1)
        board.apply_move(2)
        board.apply_move(3)
        # Apply pop
        board.apply_move(13)  # Sem erro


class TestBulkGenerateValidation:
    """Testes de validação para bulk_generate.py."""

    # Tests for make_agent
    def test_make_agent_wrong_variant_type(self):
        """make_agent() rejeita variant não-string."""
        with pytest.raises(TypeError, match="string"):
            make_agent(123)

    def test_make_agent_wrong_seed_type(self):
        """make_agent() rejeita seed não-int/None."""
        with pytest.raises(TypeError, match="seed"):
            make_agent("uct_standard", seed="42")

    def test_make_agent_invalid_variant(self):
        """make_agent() rejeita variante desconhecida."""
        with pytest.raises(ValueError, match="desconhecida"):
            make_agent("invalid_variant")

    def test_make_agent_valid_standard(self):
        """make_agent() cria StandardUCT corretamente."""
        agent = make_agent("uct_standard", seed=42)
        assert agent is not None

    def test_make_agent_valid_experimental(self):
        """make_agent() cria ExperimentalUCT corretamente."""
        agent = make_agent("uct_experimental", seed=42)
        assert agent is not None

    def test_make_agent_none_seed(self):
        """make_agent() aceita seed=None."""
        agent = make_agent("uct_standard", seed=None)
        assert agent is not None

    # Tests for randomize_state
    def test_randomize_state_wrong_steps_type(self):
        """randomize_state() rejeita steps não-inteiro."""
        rng = random.Random(42)
        with pytest.raises(TypeError, match="int"):
            randomize_state("5", rng)

    def test_randomize_state_wrong_rng_type(self):
        """randomize_state() rejeita rng não-Random."""
        with pytest.raises(TypeError, match="random.Random"):
            randomize_state(5, 42)

    def test_randomize_state_negative_steps(self):
        """randomize_state() rejeita steps negativos."""
        rng = random.Random(42)
        with pytest.raises(ValueError, match="non-negative"):
            randomize_state(-1, rng)

    def test_randomize_state_zero_steps(self):
        """randomize_state() aceita steps=0."""
        rng = random.Random(42)
        board = randomize_state(0, rng)
        assert isinstance(board, PopOutBoard)

    def test_randomize_state_valid(self):
        """randomize_state() cria board válido."""
        rng = random.Random(42)
        board = randomize_state(5, rng)
        assert isinstance(board, PopOutBoard)
        assert board.current_player in [1, 2]

    # Tests for generate_dataset
    def test_generate_dataset_wrong_variant_type(self):
        """generate_dataset() rejeita variant não-string."""
        with pytest.raises(TypeError, match="variant"):
            generate_dataset(123)

    def test_generate_dataset_wrong_n_samples_type(self):
        """generate_dataset() rejeita n_samples não-int."""
        with pytest.raises(TypeError, match="n_samples"):
            generate_dataset("uct_standard", n_samples="10")

    def test_generate_dataset_wrong_iterations_type(self):
        """generate_dataset() rejeita iterations não-int."""
        with pytest.raises(TypeError, match="iterations"):
            generate_dataset("uct_standard", iterations="50")

    def test_generate_dataset_wrong_seed_type(self):
        """generate_dataset() rejeita seed não-int."""
        with pytest.raises(TypeError, match="seed"):
            generate_dataset("uct_standard", seed="42")

    def test_generate_dataset_negative_n_samples(self):
        """generate_dataset() rejeita n_samples <= 0."""
        with pytest.raises(ValueError, match="positive"):
            generate_dataset("uct_standard", n_samples=0)

    def test_generate_dataset_negative_iterations(self):
        """generate_dataset() rejeita iterations <= 0."""
        with pytest.raises(ValueError, match="positive"):
            generate_dataset("uct_standard", iterations=0)

    def test_generate_dataset_negative_seed(self):
        """generate_dataset() rejeita seed < 0."""
        with pytest.raises(ValueError, match="non-negative"):
            generate_dataset("uct_standard", seed=-1)

    def test_generate_dataset_invalid_variant(self):
        """generate_dataset() rejeita variante inválida."""
        with pytest.raises(ValueError, match="desconhecida"):
            generate_dataset("invalid", n_samples=1, iterations=1)

    def test_generate_dataset_valid_small(self):
        """generate_dataset() cria dataset válido."""
        df = generate_dataset("uct_standard", n_samples=2, iterations=10, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "best_move" in df.columns


class TestValidationIntegration:
    """Testes de integração para validação entre componentes."""

    def test_validation_chain_id3_to_bitboard(self):
        """ID3 validation interage corretamente com bitboard."""
        clf = ID3Classifier()
        
        # Criar dataset válido com features do bitboard
        boards = [PopOutBoard() for _ in range(3)]
        features = [b.to_feature_dict() for b in boards]
        df = pd.DataFrame(features)
        df["label"] = ["A", "B", "A"]
        
        # Fit deve funcionar
        clf.fit(df, "label")
        assert clf.root is not None

    def test_validation_chain_bulk_to_id3(self):
        """Geração de dataset e ID3 validation integram."""
        # Gerar dataset pequeno
        df = generate_dataset("uct_standard", n_samples=3, iterations=10, seed=42)
        
        # ID3 fit deve aceitar
        clf = ID3Classifier()
        if not df.empty:
            clf.fit(df, "best_move")
            assert clf.root is not None

    def test_multiple_validation_errors_caught(self):
        """Múltiplos erros de validação são detectados independentemente."""
        errors = []
        
        # Teste 1: ID3 com None
        try:
            clf = ID3Classifier()
            clf.fit(None, "target")
        except TypeError:
            errors.append("id3_none")
        
        # Teste 2: Bitboard com move inválido
        try:
            board = PopOutBoard()
            board.apply_move(-1)
        except ValueError:
            errors.append("bitboard_negative")
        
        # Teste 3: bulk_generate com variante inválida
        try:
            make_agent("invalid")
        except ValueError:
            errors.append("bulk_invalid_variant")
        
        # Todos os erros foram detectados
        assert len(errors) == 3
