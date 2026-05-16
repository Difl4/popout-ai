"""Parallel game evaluation — worker functions for ProcessPoolExecutor.

Each worker process is initialised once via _init_eval_worker (section 5),
which loads classifiers and warms up Numba JIT.
Individual tasks are then cheap: just play games against a fresh MCTS instance.

Results are returned as (w_p1, l_p1, d_p1, w_p2, l_p2, d_p2) so callers can
report first-player and second-player win rates separately.
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
    """Play n_games alternating sides; return (w_p1, l_p1, d_p1, w_p2, l_p2, d_p2)."""
    import random
    from src.engine.standard.bitboard import PopOutBoard
    from src.engine.standard.rules import evaluate_after_move
    random.seed(seed)
    mcts = mcts_cls(max_nodes=iters + 256)
    w_p1 = l_p1 = d_p1 = 0
    w_p2 = l_p2 = d_p2 = 0
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
        if tree_p == 1:
            if result == 1:    w_p1 += 1
            elif result == -1: l_p1 += 1
            else:              d_p1 += 1
        else:
            if result == 1:    w_p2 += 1
            elif result == -1: l_p2 += 1
            else:              d_p2 += 1
    return w_p1, l_p1, d_p1, w_p2, l_p2, d_p2


# ── Task function ─────────────────────────────────────────────────────────────

def eval_config(is_raw: bool, is_pure: bool,
                engine_name: str, iters: int, n_games: int, seed: int):
    """Section 5 task: play games using classifiers loaded by _init_eval_worker."""
    clf = _g_clf_raw if is_raw else _g_clf_feat
    agent = _make_agent(is_raw, is_pure, clf)
    mcts_cls = _g_Flat_MCTS if engine_name == 'FlatNumbaMCTS' else _g_Solver_MCTS
    return _play_games(agent, mcts_cls, iters, n_games, seed)
