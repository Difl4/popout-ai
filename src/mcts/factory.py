"""Agent factory — centralises MCTSEngine instantiation.

Use get_agent() instead of importing engine classes directly.  This keeps
callers (GUI, CLI) decoupled from concrete implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
        from src.mcts.standard.uct_standard import StandardUCT
        return StandardUCT(**kwargs)
    elif name == "experimental":
        from src.mcts.standard.uct_experimental import ExperimentalUCT
        return ExperimentalUCT(**kwargs)
    elif name == "solver":
        from src.mcts.standard.uct_solver import SolverMCTS
        return SolverMCTS(**kwargs)
    elif name == "numba":
        from src.mcts.optimized.numba_mcts import NumbaMCTS
        return NumbaMCTS(**kwargs)
    elif name == "flat_numba":
        from src.mcts.optimized.numba_mcts import FlatNumbaMCTS
        return FlatNumbaMCTS(**kwargs)
    elif name == "numba_solver":
        from src.mcts.optimized.numba_solver import NumbaSolverMCTS
        return NumbaSolverMCTS(**kwargs)
    elif name == "flat_numba_solver":
        from src.mcts.optimized.numba_solver import FlatNumbaSolverMCTS
        return FlatNumbaSolverMCTS(**kwargs)
    elif name == "reuse":
        from src.mcts.standard.uct_reuse import ReuseUCT
        return ReuseUCT(**kwargs)
    elif name == "reuse_numba":
        from src.mcts.optimized.numba_reuse import ReuseNumbaMCTS
        return ReuseNumbaMCTS(**kwargs)
    elif name == "reuse_flat_numba_solver":
        from src.mcts.optimized.numba_solver import ReuseFlatNumbaSolverMCTS
        return ReuseFlatNumbaSolverMCTS(**kwargs)
    elif name == "id3":
        from src.decision_tree.id3_agent import ID3Agent
        return ID3Agent(**kwargs)
    elif name == "id3_raw":
        from src.decision_tree.id3_agent_raw import ID3AgentRaw
        return ID3AgentRaw(**kwargs)
    elif name == "id3_v1":
        from src.decision_tree.id3_agent import ID3Agent
        _v1 = _PROJECT_ROOT / "data/generated/v1_3000games_100k"
        kwargs.setdefault("pickle_path", str(_v1 / "id3_model.pkl"))
        kwargs.setdefault("dataset_path", str(_v1 / "popout_dt_dataset.csv"))
        return ID3Agent(**kwargs)
    elif name == "id3_raw_v1":
        from src.decision_tree.id3_agent_raw import ID3AgentRaw
        _v1 = _PROJECT_ROOT / "data/generated/v1_3000games_100k"
        kwargs.setdefault("pickle_path", str(_v1 / "id3_model_raw.pkl"))
        kwargs.setdefault("dataset_path", str(_v1 / "popout_dt_dataset.csv"))
        return ID3AgentRaw(**kwargs)
    else:
        raise ValueError(
            f"Unknown agent '{name}'. "
            "Valid names: standard, experimental, solver, numba, flat_numba, "
<<<<<<< Updated upstream
            "numba_solver, flat_numba_solver, id3, id3_raw, id3_v1, id3_raw_v1."
=======
            "numba_solver, flat_numba_solver, reuse, reuse_numba, "
            "reuse_flat_numba_solver, id3, id3_raw."
>>>>>>> Stashed changes
        )
