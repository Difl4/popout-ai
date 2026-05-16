"""MCTS agents and supporting infrastructure for PopOut.

Exposes:

- ``BaseMCTS``, ``MCTSNode``: object-oriented base implementation.
- ``MCTSEngine``: runtime-checkable protocol every playable agent satisfies.
- ``StandardUCT``, ``ExperimentalUCT``, ``SolverMCTS``, ``ReuseSolverMCTS``:
  standard-Python agents.
- ``NumbaMCTS``, ``FlatNumbaMCTS``, ``NumbaSolverMCTS``,
  ``FlatNumbaSolverMCTS``, ``ReuseFlatNumbaSolverMCTS``: Numba-JIT-accelerated
  variants (require ``numba``; symbols are ``None`` if numba is unavailable).
- ``warmup``, ``warmup_solver``: pre-compile the JIT kernels.

Use :mod:`src.mcts.factory` to instantiate any agent by string name.
"""

from src.mcts.standard.base import BaseMCTS, MCTSNode
from src.mcts.protocol import MCTSEngine
from src.mcts.standard.uct_standard import StandardUCT
from src.mcts.standard.uct_experimental import ExperimentalUCT
from src.mcts.standard.uct_solver import SolverMCTS, ReuseSolverMCTS

try:
    from src.mcts.optimized.numba_mcts import NumbaMCTS, FlatNumbaMCTS, warmup
    from src.mcts.optimized.numba_solver import (
        NumbaSolverMCTS,
        FlatNumbaSolverMCTS,
        ReuseFlatNumbaSolverMCTS,
        warmup_solver,
    )
except ImportError:  # numba not installed
    NumbaMCTS = None  # type: ignore[assignment,misc]
    FlatNumbaMCTS = None  # type: ignore[assignment,misc]
    NumbaSolverMCTS = None  # type: ignore[assignment,misc]
    FlatNumbaSolverMCTS = None  # type: ignore[assignment,misc]
    ReuseFlatNumbaSolverMCTS = None  # type: ignore[assignment,misc]
    warmup = None  # type: ignore[assignment]
    warmup_solver = None  # type: ignore[assignment]

__all__ = [
    "BaseMCTS",
    "MCTSNode",
    "MCTSEngine",
    "StandardUCT",
    "ExperimentalUCT",
    "SolverMCTS",
    "ReuseSolverMCTS",
    "NumbaMCTS",
    "FlatNumbaMCTS",
    "NumbaSolverMCTS",
    "FlatNumbaSolverMCTS",
    "ReuseFlatNumbaSolverMCTS",
    "warmup",
    "warmup_solver",
]
