"""Parallel tournament worker — one game per task call.

Must be a top-level module (not nested) so multiprocessing can pickle it.
Place in the project root; workers self-add the root to sys.path.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ID3 agents are stateless after loading — cache them once per worker process
# so each pickle is loaded exactly once rather than once per game.
_ID3_CACHE: dict[str, object] = {}


def init_worker() -> None:
    """Called once per worker process at startup — warms up Numba JIT."""
    try:
        from src.mcts.optimized.numba_mcts import warmup
        warmup()
    except Exception:
        pass  # Numba unavailable; ID3 agents still work


def run_game_task(task: tuple) -> int:
    """Play one game and return result from agent_a's perspective.

    task = (agent_a_key, iters_a, agent_b_key, iters_b, a_is_p1, seed)

    Returns
    -------
    1  agent_a wins
    2  agent_b wins
    0  draw
    """
    agent_a_key, iters_a, agent_b_key, iters_b, a_is_p1, seed = task

    import random
    random.seed(seed)

    from src.engine.standard.bitboard import PopOutBoard
    from src.engine.standard.rules import evaluate_after_move
    from src.mcts.factory import get_agent

    if a_is_p1:
        p1_key, p1_iters = agent_a_key, iters_a
        p2_key, p2_iters = agent_b_key, iters_b
    else:
        p1_key, p1_iters = agent_b_key, iters_b
        p2_key, p2_iters = agent_a_key, iters_a

    def _get(key: str):
        if key.startswith("id3"):
            if key not in _ID3_CACHE:
                _ID3_CACHE[key] = get_agent(key)
            return _ID3_CACHE[key]
        return get_agent(key)

    p1 = _get(p1_key)
    p2 = _get(p2_key)

    board = PopOutBoard()
    sig_count: Counter = Counter()
    game_result = 0

    for _ in range(300):
        sig = (board.mask_p1, board.mask_p2)
        sig_count[sig] += 1
        if sig_count[sig] >= 3:
            break
        if not board.legal_moves():
            break
        pl = board.current_player - 1
        agent = p1 if pl == 0 else p2
        iters = p1_iters if pl == 0 else p2_iters
        move = agent.run(board, iterations=iters)
        mover = board.current_player
        board.apply_move(move)
        r = evaluate_after_move(board, mover=mover)
        if r != 0:
            game_result = r
            break

    # game_result: 1=P1 won, 2=P2 won, 0=draw
    # convert to agent_a perspective
    if a_is_p1:
        return game_result          # agent_a is P1
    else:
        if game_result == 1: return 2   # P1 (agent_b) won
        if game_result == 2: return 1   # P2 (agent_a) won
        return 0
