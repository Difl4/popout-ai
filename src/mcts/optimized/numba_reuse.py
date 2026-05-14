"""NumbaMCTS with inter-turn tree reuse.

Combines the Numba-accelerated expand/simulate kernels from NumbaMCTS
with the tree-reuse state machine from _TreeReuseMixin.

At ~50 k iter/s NumbaMCTS is 5× faster than StandardUCT; tree reuse
compounds this by seeding each search with pre-computed visit statistics
from the previous turn.

Class
-----
ReuseNumbaMCTS
    Tree reuse on top of NumbaMCTS (~50 k iter/s).
    See uct_reuse.ReuseUCT for a full usage example.
"""

from __future__ import annotations

from typing import Optional

from src.mcts.standard.uct_reuse import _TreeReuseMixin
from src.mcts.optimized.numba_mcts import NumbaMCTS


class ReuseNumbaMCTS(_TreeReuseMixin, NumbaMCTS):
    """NumbaMCTS with inter-turn tree reuse (~50 k iter/s).

    The Python-side search tree (MCTSNode objects) is preserved between
    calls to run().  The Numba-compiled rollout kernel (nb_simulate) is
    still invoked for every simulation — no Python overhead on the hot
    path.

    Call warmup() once at startup to pay the JIT compilation cost, and
    reset() at the start of every new game.

    Usage::

        from src.mcts.optimized.numba_mcts import warmup
        from src.mcts.optimized.numba_reuse import ReuseNumbaMCTS

        warmup()                                    # compile Numba kernels
        agent = ReuseNumbaMCTS(seed=42)
        agent.reset()

        move = agent.run(board, 5_000)
        board.apply_move(move)

        opp = opponent.run(board, 5_000)
        board.apply_move(opp)

        move = agent.run(board, 5_000, opponent_move=opp)
    """

    def __init__(
        self,
        exploration_c: float = 1.414,
        rollout_depth: int = 150,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            exploration_c=exploration_c,
            rollout_depth=rollout_depth,
            seed=seed,
        )
