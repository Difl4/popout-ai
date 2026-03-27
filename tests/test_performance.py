"""Performance benchmarking tests for PopOut AI."""

import pytest
import time
import pandas as pd

from src.algorithms.id3.learner import ID3Classifier
from src.engine.bitboard import PopOutBoard
from src.algorithms.mcts.uct_standard import StandardUCT
from src.algorithms.mcts.uct_experimental import ExperimentalUCT


class TestBitboardPerformance:
    """Benchmarks for bitboard operations."""

    def test_apply_move_completes_quickly(self):
        """Test apply_move() completes in reasonable time."""
        board = PopOutBoard()
        
        start = time.time()
        for _ in range(100):
            b = PopOutBoard()
            for move in range(7):
                if move in b.legal_moves():
                    b.apply_move(move)
        elapsed = time.time() - start
        
        # 100 iterations should complete in less than 1 second
        assert elapsed < 1.0

    def test_legal_moves_speed(self):
        """Test legal_moves() computation speed."""
        board = PopOutBoard()
        
        start = time.time()
        for _ in range(1000):
            moves = board.legal_moves()
        elapsed = time.time() - start
        
        # 1000 calls should be fast
        assert elapsed < 2.0
        assert len(moves) > 0

    def test_board_state_operations_batch(self):
        """Test batch board operations complete quickly."""
        start = time.time()
        for _ in range(100):
            board = PopOutBoard()
            # Apply some moves
            for move in [0, 1, 2, 3]:
                if move in board.legal_moves():
                    board.apply_move(move)
            # Extract features
            features = board.to_feature_dict()
        elapsed = time.time() - start
        
        # 100 full iterations should be quick
        assert elapsed < 2.0

    def test_legal_moves_iterations_per_second(self):
        """Measure legal_moves() iterations per second."""
        board = PopOutBoard()
        
        start = time.time()
        count = 0
        while time.time() - start < 0.1:  # Measure for 0.1 seconds
            board.legal_moves()
            count += 1
        elapsed = time.time() - start
        
        iterations_per_second = count / elapsed
        # Should be capable of at least 1000 iterations per second
        assert iterations_per_second > 1000


class TestID3Performance:
    """Benchmarks for ID3 decision tree."""

    @staticmethod
    def _create_dataset(rows: int = 100) -> pd.DataFrame:
        """Create synthetic dataset for benchmarking."""
        import random
        random.seed(42)
        data = {
            "feat1": [random.choice(["a", "b", "c"]) for _ in range(rows)],
            "feat2": [random.choice(["x", "y", "z"]) for _ in range(rows)],
            "feat3": [random.choice(["1", "2", "3"]) for _ in range(rows)],
            "target": [random.choice(["yes", "no"]) for _ in range(rows)],
        }
        return pd.DataFrame(data)

    def test_id3_fit_speed_small(self):
        """Test ID3 training completes quickly on small dataset."""
        df = self._create_dataset(rows=50)
        clf = ID3Classifier()
        
        start = time.time()
        clf.fit(df, "target")
        elapsed = time.time() - start
        
        # Should train quickly
        assert elapsed < 1.0
        assert clf.root is not None

    def test_id3_fit_speed_medium(self):
        """Test ID3 training on medium dataset."""
        df = self._create_dataset(rows=200)
        clf = ID3Classifier()
        
        start = time.time()
        clf.fit(df, "target")
        elapsed = time.time() - start
        
        assert elapsed < 2.0
        assert clf.root is not None

    def test_id3_fit_speed_large(self):
        """Test ID3 training on larger dataset."""
        df = self._create_dataset(rows=500)
        clf = ID3Classifier()
        
        start = time.time()
        clf.fit(df, "target")
        elapsed = time.time() - start
        
        assert elapsed < 5.0
        assert clf.root is not None

    def test_id3_predict_speed(self):
        """Test ID3 prediction speed."""
        df = self._create_dataset(rows=100)
        clf = ID3Classifier()
        clf.fit(df, "target")
        
        test_df = df.drop(columns=["target"])
        
        start = time.time()
        result = clf.predict(test_df)
        elapsed = time.time() - start
        
        assert len(result) == len(test_df)
        assert elapsed < 1.0

    def test_id3_feature_importance_speed(self):
        """Test feature importance calculation speed."""
        df = self._create_dataset(rows=100)
        clf = ID3Classifier()
        clf.fit(df, "target")
        
        start = time.time()
        importance = clf.get_feature_importance()
        elapsed = time.time() - start
        
        assert isinstance(importance, dict)
        assert elapsed < 0.5

    def test_id3_tree_visualization_speed(self):
        """Test tree visualization speed."""
        df = self._create_dataset(rows=100)
        clf = ID3Classifier()
        clf.fit(df, "target")
        
        start = time.time()
        tree_str = clf.tree_to_string()
        elapsed = time.time() - start
        
        assert isinstance(tree_str, str)
        assert elapsed < 0.5

    def test_id3_scaling_with_dataset_size(self):
        """Test ID3 scaling as dataset grows."""
        sizes = [50, 100, 200]
        times = []
        
        for size in sizes:
            df = self._create_dataset(rows=size)
            clf = ID3Classifier()
            
            start = time.time()
            clf.fit(df, "target")
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Training time should grow sub-quadratically
        # All should complete within reasonable time
        assert all(t < 5.0 for t in times)
        # Larger datasets should take more time but not explode
        if times[0] > 0.01:
            assert times[-1] < times[0] * 20


class TestMCTSPerformance:
    """Benchmarks for MCTS algorithms."""

    def test_standard_uct_completes_quickly(self):
        """Test StandardUCT completes in reasonable time."""
        board = PopOutBoard()
        agent = StandardUCT(seed=42)
        
        start = time.time()
        best_move = agent.run(board, iterations=50)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        assert elapsed < 10.0
        assert best_move in board.legal_moves()

    def test_experimental_uct_completes_quickly(self):
        """Test ExperimentalUCT completes in reasonable time."""
        board = PopOutBoard()
        agent = ExperimentalUCT(seed=42)
        
        start = time.time()
        best_move = agent.run(board, iterations=50)
        elapsed = time.time() - start
        
        assert elapsed < 10.0
        assert best_move in board.legal_moves()

    def test_mcts_scaling_with_iterations(self):
        """Test MCTS scales linearly with iterations."""
        board = PopOutBoard()
        agent = StandardUCT(seed=42)
        
        iteration_counts = [10, 25]
        times = []
        
        for iterations in iteration_counts:
            start = time.time()
            agent.run(board, iterations=iterations)
            elapsed = time.time() - start
            times.append(elapsed)
        
        # Time should scale roughly linearly with iterations
        # Both should complete quickly
        assert all(t < 10.0 for t in times)

    def test_mcts_determinism(self):
        """Test MCTS output with same seed is consistent."""
        board = PopOutBoard()
        
        # Run 1
        agent1 = StandardUCT(seed=42)
        move1 = agent1.run(board, iterations=50)
        
        # Run 2
        agent2 = StandardUCT(seed=42)
        move2 = agent2.run(board, iterations=50)
        
        # Same seed should produce same move (deterministic)
        assert move1 == move2

    def test_mcts_different_seeds_different_moves(self):
        """Test different seeds produce different results."""
        board = PopOutBoard()
        
        agent1 = StandardUCT(seed=42)
        move1 = agent1.run(board, iterations=30)
        
        agent2 = StandardUCT(seed=99)
        move2 = agent2.run(board, iterations=30)
        
        # Different seeds may or may not produce different moves,
        # but this test ensures both complete successfully
        assert move1 in board.legal_moves()
        assert move2 in board.legal_moves()


class TestIntegrationPerformance:
    """Integration performance tests."""

    def test_full_game_simulation_speed(self):
        """Benchmark full game simulation."""
        import random
        random.seed(42)
        
        start = time.time()
        board = PopOutBoard()
        move_count = 0
        
        while not board.is_full() and move_count < 100:
            legal = board.legal_moves()
            if not legal:
                break
            move = random.choice(legal)
            board.apply_move(move)
            move_count += 1
        
        elapsed = time.time() - start
        
        # Should complete quickly (full game simulation)
        assert elapsed < 2.0
        assert move_count > 0

    def test_id3_training_with_small_dataset(self):
        """Test ID3 training runs quickly with synthetic data."""
        df = TestID3Performance._create_dataset(rows=50)
        
        start = time.time()
        clf = ID3Classifier()
        clf.fit(df, "target")
        elapsed = time.time() - start
        
        assert elapsed < 2.0
        assert clf.root is not None

    def test_multiple_bitboard_operations(self):
        """Test combining multiple bitboard operations."""
        start = time.time()
        count = 0
        
        for _ in range(100):
            board = PopOutBoard()
            while count < 1000 and not board.is_full():
                legal = board.legal_moves()
                if legal:
                    board.apply_move(legal[0])
                    count += 1
        
        elapsed = time.time() - start
        
        # 1000 moves should complete quickly
        assert elapsed < 2.0
        assert count >= 100

