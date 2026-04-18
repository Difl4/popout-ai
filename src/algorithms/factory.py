"""Agent factory — centralises MCTSEngine instantiation.

Use get_agent() instead of importing engine classes directly.  This keeps
callers (GUI, CLI) decoupled from concrete implementations.
"""

from __future__ import annotations

from typing import Any


def get_agent(name: str, **kwargs: Any):
    """Instantiate and return an MCTS engine by name.

    Parameters
    ----------
    name : str
        One of ``"standard"``, ``"experimental"``, ``"numba"``, ``"flat_numba"``.
    **kwargs :
        Forwarded to the engine constructor (e.g. ``seed``, ``rollout_depth``).

    Returns
    -------
    MCTSEngine
        A concrete engine satisfying the MCTSEngine protocol.

    Raises
    ------
    ValueError
        If *name* is not a known agent identifier.
    """
    if name == "standard":
        from src.algorithms.mcts.standard.uct_standard import StandardUCT
        return StandardUCT(**kwargs)
    elif name == "experimental":
        from src.algorithms.mcts.standard.uct_experimental import ExperimentalUCT
        return ExperimentalUCT(**kwargs)
    elif name == "solver":
        from src.algorithms.mcts.standard.uct_solver import SolverMCTS
        return SolverMCTS(**kwargs)
    elif name == "numba":
        from src.algorithms.mcts.optimized.numba_mcts import NumbaMCTS
        return NumbaMCTS(**kwargs)
    elif name == "flat_numba":
        from src.algorithms.mcts.optimized.numba_mcts import FlatNumbaMCTS
        return FlatNumbaMCTS(**kwargs)
    elif name == "numba_solver":
        from src.algorithms.mcts.optimized.numba_solver import NumbaSolverMCTS
        return NumbaSolverMCTS(**kwargs)
    elif name == "flat_numba_solver":
        from src.algorithms.mcts.optimized.numba_solver import FlatNumbaSolverMCTS
        return FlatNumbaSolverMCTS(**kwargs)
    else:
        raise ValueError(
            f"Unknown agent '{name}'. "
            "Valid names: standard, experimental, solver, numba, flat_numba, "
            "numba_solver, flat_numba_solver."
        )
