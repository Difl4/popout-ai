"""Testes para src/scripts/bulk_generate.py — geração de dataset."""

import pytest
import pandas as pd

from src.ml.dataset_generator import generate_dataset, make_agent, randomize_state
from src.algorithms.mcts.standard.uct_standard import StandardUCT
from src.algorithms.mcts.standard.uct_experimental import ExperimentalUCT
from src.engine.standard.bitboard import COLS, ROWS, PopOutBoard

import random


# ── make_agent ───────────────────────────────────────────────────────────────

class TestMakeAgent:
    def test_standard_variant(self):
        agent = make_agent("uct_standard", seed=42)
        assert isinstance(agent, StandardUCT)

    def test_experimental_variant(self):
        agent = make_agent("uct_experimental", seed=42)
        assert isinstance(agent, ExperimentalUCT)

    def test_unknown_variant_raises(self):
        with pytest.raises(ValueError, match="desconhecida"):
            make_agent("nonexistent_variant")


# ── randomize_state ──────────────────────────────────────────────────────────

class TestRandomizeState:
    def test_returns_popout_board(self):
        rng = random.Random(42)
        board = randomize_state(steps=5, rng=rng)
        assert isinstance(board, PopOutBoard)

    def test_zero_steps_returns_empty(self):
        rng = random.Random(42)
        board = randomize_state(steps=0, rng=rng)
        # Tabuleiro deve estar vazio (sem peças)
        assert board.mask_p1 == 0
        assert board.mask_p2 == 0

    def test_many_steps_does_not_crash(self):
        rng = random.Random(42)
        board = randomize_state(steps=50, rng=rng)
        assert isinstance(board, PopOutBoard)

    def test_state_has_pieces_after_steps(self):
        rng = random.Random(42)
        board = randomize_state(steps=10, rng=rng)
        # Contar peças nos bitboards
        p1_pieces = bin(board.mask_p1).count('1')
        p2_pieces = bin(board.mask_p2).count('1')
        total_pieces = p1_pieces + p2_pieces
        assert total_pieces > 0


# ── generate_dataset ─────────────────────────────────────────────────────────

class TestGenerateDataset:
    def test_returns_dataframe(self):
        df = generate_dataset(variant="uct_standard", n_samples=5, iterations=10, seed=42)
        assert isinstance(df, pd.DataFrame)

    def test_has_best_move_column(self):
        df = generate_dataset(variant="uct_standard", n_samples=5, iterations=10, seed=42)
        assert "best_move" in df.columns

    def test_has_current_player_column(self):
        df = generate_dataset(variant="uct_standard", n_samples=5, iterations=10, seed=42)
        assert "current_player" in df.columns

    def test_has_cell_columns(self):
        df = generate_dataset(variant="uct_standard", n_samples=3, iterations=10, seed=42)
        assert "cell_0_0" in df.columns
        assert f"cell_{ROWS-1}_{COLS-1}" in df.columns

    def test_best_move_format(self):
        """best_move deve ter formato 'tipo_coluna', ex: 'drop_3'."""
        df = generate_dataset(variant="uct_standard", n_samples=5, iterations=10, seed=42)
        for move in df["best_move"]:
            parts = move.split("_")
            assert len(parts) == 2
            assert parts[0] in ("drop", "pop")
            assert 0 <= int(parts[1]) < COLS

    def test_row_count(self):
        """Número de linhas deve ser <= n_samples (pode ser menor se estados sem jogadas)."""
        df = generate_dataset(variant="uct_standard", n_samples=10, iterations=10, seed=42)
        assert 0 < len(df) <= 10

    def test_experimental_variant_works(self):
        df = generate_dataset(variant="uct_experimental", n_samples=3, iterations=10, seed=42)
        assert len(df) > 0
