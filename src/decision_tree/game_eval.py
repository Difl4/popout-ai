"""Parallel game evaluation — worker functions for ProcessPoolExecutor.

Each worker process is initialised once via _init_eval_worker (section 5) or
_init_cross_worker (section 6), which loads classifiers and warms up Numba JIT.
Individual tasks are then cheap: just play games against a fresh MCTS instance.
"""
from __future__ import annotations

# Per-process globals populated by the initialiser
_g_clf_feat    = None
_g_clf_raw     = None
_g_Flat_MCTS   = None
_g_Solver_MCTS = None


# ── Initialisers (called once per worker process) ─────────────────────────────

def _init_eval_worker(pkl_feat: str, pkl_raw: str) -> None:
    """Load both classifiers and warm up Numba — used by section 5 workers."""
    global _g_clf_feat, _g_clf_raw, _g_Flat_MCTS, _g_Solver_MCTS
    import pickle
    from src.mcts.optimized.numba_mcts import warmup, FlatNumbaMCTS
    from src.mcts.optimized.numba_solver import FlatNumbaSolverMCTS
    with open(pkl_feat, 'rb') as f:
        _g_clf_feat = pickle.load(f)
    with open(pkl_raw, 'rb') as f:
        _g_clf_raw = pickle.load(f)
    warmup()
    _g_Flat_MCTS   = FlatNumbaMCTS
    _g_Solver_MCTS = FlatNumbaSolverMCTS


def _init_cross_worker() -> None:
    """Warm up Numba only — models are loaded per-task in section 6."""
    global _g_Solver_MCTS
    from src.mcts.optimized.numba_mcts import warmup
    from src.mcts.optimized.numba_solver import FlatNumbaSolverMCTS
    warmup()
    _g_Solver_MCTS = FlatNumbaSolverMCTS


# ── Shared helpers ────────────────────────────────────────────────────────────

def _make_agent(is_raw: bool, is_pure: bool, clf):
    import types
    import pandas as pd
    from src.engine.standard.rules import extended_features
    from src.decision_tree.id3_agent import ID3Agent
    from src.decision_tree.id3_agent_raw import ID3AgentRaw

    agent = ID3AgentRaw() if is_raw else ID3Agent()
    agent._classifier = clf

    if is_pure:
        if is_raw:
            def _run(self, board, iterations=0):
                row = pd.Series({k: str(v) for k, v in board.to_feature_dict().items()})
                return self._parse_move(self._classifier.predict_one(row), board.legal_moves())
        else:
            def _run(self, board, iterations=0):
                features = extended_features(board)
                row = pd.Series({k: str(v) for k, v in features.items() if k != 'is_proven'})
                return self._parse_move(self._classifier.predict_one(row), board.legal_moves())
        agent.run = types.MethodType(_run, agent)

    return agent


def _play_games(agent, mcts_cls, iters: int, n_games: int, seed: int):
    import random
    from src.engine.standard.bitboard import PopOutBoard
    from src.engine.standard.rules import evaluate_after_move
    random.seed(seed)
    mcts = mcts_cls(max_nodes=iters + 256)
    wins = losses = draws = 0
    for i in range(n_games):
        tree_p = 1 if i % 2 == 0 else 2
        board = PopOutBoard()
        result = 0
        for _ in range(120):
            legal = board.legal_moves()
            if not legal:
                break
            mover = board.current_player
            move = agent.run(board) if mover == tree_p else mcts.run(board, iterations=iters)
            board.apply_move(move)
            winner = evaluate_after_move(board, mover=mover)
            if winner:
                result = 1 if winner == tree_p else -1
                break
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws


# ── Task functions ────────────────────────────────────────────────────────────

def eval_config(is_raw: bool, is_pure: bool,
                engine_name: str, iters: int, n_games: int, seed: int):
    """Section 5 task: play games using classifiers loaded by _init_eval_worker."""
    clf = _g_clf_raw if is_raw else _g_clf_feat
    agent = _make_agent(is_raw, is_pure, clf)
    mcts_cls = _g_Flat_MCTS if engine_name == 'FlatNumbaMCTS' else _g_Solver_MCTS
    return _play_games(agent, mcts_cls, iters, n_games, seed)


def eval_cross_config(pkl_path: str, is_raw: bool,
                      iters: int, n_games: int, seed: int):
    """Section 6 task: load model from pkl_path, play games vs FlatNumbaSolverMCTS."""
    import pickle
    with open(pkl_path, 'rb') as f:
        clf = pickle.load(f)
    agent = _make_agent(is_raw, False, clf)
    return _play_games(agent, _g_Solver_MCTS, iters, n_games, seed)
