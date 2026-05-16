# PopOut AI - Full Project Context for AI Assistants

Last updated: 2026-05-16

This document is a handoff brief for any AI assistant working on this repository.
Read it before changing code, debugging tests, editing notebooks, or explaining
the project. It captures the project objective, architecture, core invariants,
data/model files, commands, current local verification state, and the main risks.

## 1. Project Identity

`popout-ai` is a Python project for an Artificial Intelligence university
assignment, 2025/2026, about **PopOut**, a Connect Four variant.

The assignment requires:

- A playable PopOut implementation.
- Human vs Human, Human vs Computer, and Computer vs Computer scenarios.
- Monte Carlo Tree Search using UCT.
- At least one meaningful MCTS variant beyond standard UCT.
- A decision tree learned with ID3, implemented from scratch.
- Iris dataset work as a decision-tree warmup.
- A PopOut dataset generated from MCTS decisions.
- A trained ID3 tree that predicts PopOut moves.
- Documentation, analysis notebooks, tests, and performance comparison.

The project goes beyond the baseline with bitboards, MCTS-Solver proof
propagation, Numba-accelerated flat-array search, tree reuse experiments, a
Pygame GUI, a CLI, generated datasets, saved ID3 models, and a broad pytest
suite.

## 2. Game Rules and Encoding

Board:

- 7 columns x 6 rows.
- Player 1 is usually `X`; player 2 is usually `O`.
- Win condition is 4 in a row: horizontal, vertical, or diagonal.

Moves are encoded as integers:

- `0..6`: drop in column `0..6`.
- `7..13`: pop from column `0..6`.
- Human CLI syntax uses `d3` for drop column 3 and `p0` for pop column 0.

PopOut-specific behavior:

- A player may pop only if they own the bottom piece in that column.
- When a piece is popped, all pieces above it shift down.
- If a pop creates winning lines for both players, the mover wins.
- Full-board and threefold-repetition draw behavior is implemented, but AI
  handling is simplified/conservative: AI declares a threefold repetition draw;
  on full boards it continues with pop moves when possible.

Core invariant:

- Row `0` is the bottom row.
- Each column uses 7 bits in the bitboard: 6 playable cells plus 1 guard bit.
- Do not change move encoding or row/column conventions without updating the
  engine, optimized kernels, GUI, CLI, dataset features, and tests.

## 3. Repository Layout

Important top-level files:

- `README.md`: install/run overview.
- `environment.yml`: intended Conda environment, Python 3.10.
- `setup.py`: editable package setup, currently minimal dependencies only.
- `play.py`: direct GUI launcher.
- `AI_CONTEXT.md`: this AI handoff document.
- `docs/IA_2526_Trab.pdf`: assignment statement.

Main source tree:

```text
src/
  __main__.py                    # python -m src entry point
  config.py                      # BOARD_WIDTH, BOARD_HEIGHT, WIN_CONDITION, COL_SIZE
  game_state.py                  # GUI save/load state via pickle
  engine/
    standard/
      bitboard.py                # PopOutBoard, legal moves, apply_move, features
      rules.py                   # wins, draw/repetition helpers, ID3 features
    optimized/
      numba_bitboard.py          # Numba versions of move/rule primitives
      numba_rules.py             # Numba win/draw/evaluation kernels
  mcts/
    protocol.py                  # MCTSEngine protocol: run(board, iterations) -> move
    factory.py                   # get_agent(name, **kwargs)
    standard/
      base.py                    # classic MCTS node and four phases
      uct_standard.py            # StandardUCT
      uct_experimental.py        # ExperimentalUCT
      uct_solver.py              # SolverMCTS and ReuseSolverMCTS
    optimized/
      numba_mcts.py              # NumbaMCTS, FlatNumbaMCTS, warmup()
      numba_solver.py            # Numba solver and reuse flat solver variants
      numba_search.py            # JIT search loops and rollout kernels
  decision_tree/
    discretizer.py               # quantile binning for numeric features
    dataset_generator.py         # MCTS-labeled dataset generation
    id3/learner.py               # custom ID3Classifier
    id3_agent.py                 # playable ID3 agent with tactical safeguards
    id3_agent_raw.py             # raw-feature ID3 agent, also has safeguards
  interfaces/
    cli.py                       # terminal game and tournaments
    gui.py                       # compatibility import for GUI launch
    gui/
      core.py                    # Pygame loop, setup, AI thread management
      renderer.py                # drawing functions
      components.py              # PauseMenu
      state.py                   # Difficulty enum and animation state
      assets.py                  # GUI constants/colors/fonts
  utils/
    numba_tools.py               # Numba cache/tooling helpers
```

Other important folders:

- `tests/`: pytest suite covering rules, board behavior, MCTS, solver, Numba,
  ID3, dataset generation, CLI, GUI helpers, validation, integration, and
  performance.
- `notebooks/`: assignment/report notebooks, including the main solution and
  technical documentation.
- `data/generated/`: generated CSV datasets and pickled ID3 models.
- `data/figures/`: generated plots for reports.
- `scripts/`: training and analysis utilities.

## 4. Environment and Commands

Intended setup:

```bash
conda env create -f environment.yml
conda activate popout-ai
```

Alternative editable install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

The Conda environment is the safer official path because `setup.py` does not
list all runtime/report dependencies. `environment.yml` includes Python 3.10,
numpy, pandas, matplotlib, notebook, pytest, pygame, numba, seaborn,
scikit-learn, and editable install.

Run GUI:

```bash
python -m src
```

Run CLI:

```bash
python -m src --cli
```

Generate a small dataset:

```bash
python -m src.decision_tree.dataset_generator --variant uct_standard --samples 200 --iterations 150 --seed 42
```

Open main notebook:

```bash
jupyter notebook notebooks/PopOut_Solution.ipynb
```

Run tests in the intended environment:

```bash
pytest -q
```

Warm up optimized Numba engines before benchmarking:

```python
from src.mcts.optimized.numba_mcts import warmup
warmup()
```

## 5. Engine Details

Primary board class:

- `src/engine/standard/bitboard.py::PopOutBoard`

Fields:

- `mask_p1: int`
- `mask_p2: int`
- `current_player: int`

Important methods:

- `clone()`
- `merged_mask`
- `legal_moves()`
- `apply_move(move)`
- `is_full()`
- `to_feature_dict()`
- `__str__`, `__eq__`, `__hash__`

Bitboard details:

- `COL_SIZE = 7`.
- Top playable row bit is row index `5`.
- `TOP_ROW_MASK = 0x810204081020`.
- Clean mask for playable bits is equivalent to
  `sum(0b111111 << (i * 7) for i in range(7))`.

Rules:

- `has_won(bitmask)` uses bit shifts for vertical, horizontal, and diagonal
  4-in-a-row checks.
- `evaluate_after_move(board, mover)` implements the simultaneous-win PopOut
  tiebreak.
- `board_signature(board)` returns `(mask_p1, mask_p2)` for repetition checks.
- `is_threefold_repetition(history_signatures)` uses counts over signatures.
- `extended_features(board)` adds tactical features for ID3:
  `threats_me`, `threats_opp`, `center_me`, `center_opp`, `phase`,
  `can_win`, and `opp_wins_next`.

Important caution:

- `board_signature()` ignores `current_player`. This is intentional in the
  current repetition logic. Do not change it casually.

## 6. MCTS Architecture

All playable/search agents should satisfy:

```python
run(board: PopOutBoard, iterations: int = 10_000) -> int
```

Protocol:

- `src/mcts/protocol.py::MCTSEngine`

Factory:

- `src/mcts/factory.py::get_agent(name, **kwargs)`

Known factory names:

- `standard`
- `experimental`
- `solver`
- `numba`
- `flat_numba`
- `numba_solver`
- `flat_numba_solver`
- `reuse_solver`
- `reuse_flat_numba_solver`
- `id3`
- `id3_raw`
- `id3_v1`, `id3_raw_v1`
- `id3_v3`, `id3_raw_v3`
- `id3_v4`, `id3_raw_v4`

Standard MCTS:

- Implemented in `src/mcts/standard/base.py`.
- Uses `MCTSNode` with state, parent, move, mover, visits, value sum, children,
  untried moves, and terminal winner.
- Phases: selection, expansion, simulation, backpropagation.
- UCT score is exploitation plus exploration:
  `Q + C * sqrt(log(parent_visits) / child_visits)`.
- Rollouts are random and stop on win, no legal moves, threefold repetition, or
  rollout depth.

Standard variants:

- `StandardUCT`: baseline UCT. Defaults to `rollout_depth=30`.
- `ExperimentalUCT`: prioritizes unvisited children before normal UCT.
- `SolverMCTS`: proof-aware MCTS-Solver.
- `ReuseSolverMCTS`: pure Python solver that preserves tree state across turns.

MCTS-Solver:

- Implemented in `src/mcts/standard/uct_solver.py`.
- Node status constants:
  - `STATUS_UNKNOWN = 0`
  - `STATUS_WIN = 1`
  - `STATUS_LOSS = 2`
  - `STATUS_DRAW = 3`
- Status is from the perspective of `state.current_player`.
- A node is a WIN if any child is a LOSS.
- A node is a LOSS if all fully explored children are WIN.
- A node is a DRAW if it can force at least a draw and no winning child exists.
- Distance tracks minimax distance so the solver prefers faster wins and slower
  losses.

Optimized MCTS:

- `NumbaMCTS`: Python tree with Numba expansion/simulation.
- `FlatNumbaMCTS`: full MCTS loop in JIT-compiled flat numpy arrays.
- `NumbaSolverMCTS`: Python proof logic with Numba expansion/simulation.
- `FlatNumbaSolverMCTS`: full solver loop in JIT-compiled flat arrays.
- `ReuseFlatNumbaSolverMCTS`: flat solver with inter-turn tree reuse and subtree
  compaction.

Numba constraints:

- Hot kernels must stay Numba-compatible: no Python objects, no normal dicts,
  no arbitrary lists in `@njit` code, stable numeric dtypes.
- Optimized code mirrors standard engine/rule behavior. If changing game logic,
  update both standard and optimized implementations plus tests.

## 7. ID3 and Dataset Architecture

Custom ID3 implementation:

- `src/decision_tree/id3/learner.py::ID3Classifier`
- Does not use scikit-learn to train a decision tree.
- Handles categorical features.
- Computes entropy and information gain.
- Has a pandas/DataFrame interface but internally encodes categories into numpy
  arrays for faster training.
- Handles pure nodes, no-feature fallback, max depth, unknown branch fallback to
  node majority label, scoring, feature importance, and tree visualization.

Discretization:

- `src/decision_tree/discretizer.py`
- `fit_quantile_bins(df, columns, q=3)` learns quantile edges.
- `apply_bins(df, bins)` returns a transformed copy with categorical bin labels.

PopOut dataset generation:

- `src/decision_tree/dataset_generator.py`
- Simple path: randomizes states and labels with standard MCTS.
- Parallel/full-game path: uses `FlatNumbaSolverMCTS`/reuse solver, blunder
  tiers, forced opening coverage, and horizontal mirroring.
- Target column is `best_move`, with labels like `drop_3` and `pop_1`.
- `is_proven` marks positions where the solver root was proven.

Playable ID3 agents:

- `ID3Agent`: uses raw cell features plus tactical features.
- `ID3AgentRaw`: uses raw cells plus current player for model input.
- Both currently include tactical immediate-win/immediate-block safeguards
  before consulting the tree. This is important for reporting: distinguish
  pure classifier accuracy from playable hybrid-agent behavior.

Training behavior:

- Agents lazy-load pickled models when available.
- If a model pickle is missing, agents train from CSV and save the pickle.
- Proven positions are oversampled during training.

## 8. Data and Models

Measured local generated CSV/model files:

```text
data/generated/popout_dt_dataset.csv
  rows=193,058, cols=52, proven=75,830
  top labels: drop_3, drop_4, drop_2, drop_1, drop_5

data/generated/v1_3000games_100k/popout_dt_dataset.csv
  rows=119,676, cols=52, proven=57,272

data/generated/v2_5000games_100k/popout_dt_dataset.csv
  rows=155,494, cols=52, proven=67,306

data/generated/uct_standard.csv
  rows=200, cols=44, small fallback dataset without is_proven
```

Pickled model files exist for:

- `data/generated/v1_3000games_100k/id3_model.pkl`
- `data/generated/v1_3000games_100k/id3_model_raw.pkl`
- `data/generated/v2_5000games_100k/id3_model.pkl`
- `data/generated/v2_5000games_100k/id3_model_raw.pkl`
- `data/generated/v3_5000games_tiered/id3_model.pkl`
- `data/generated/v3_5000games_tiered/id3_model_raw.pkl`
- `data/generated/v3_5000games_tiered/id3_model_d15.pkl`
- `data/generated/v3_5000games_tiered/id3_model_raw_d15.pkl`
- `data/generated/v4_75k_100k/id3_model.pkl`
- `data/generated/v4_75k_100k/id3_model_raw.pkl`

Caution:

- The `v4_75k_100k` factory entries point to a dataset path under that folder,
  but only pickles are currently present locally. Loading works if the pickle
  exists; retraining v4 from CSV would need the missing CSV.
- Generated CSVs and pickle files can be large. Do not regenerate or overwrite
  them unless explicitly asked.

## 9. Interfaces

CLI:

- `src/interfaces/cli.py`
- Menu supports:
  - Human vs Human.
  - Human vs Computer.
  - Computer vs Computer tournament.
- Agent selection includes standard, experimental, solver, Numba variants, and
  ID3.
- Tournaments alternate colors for fairness.

GUI:

- `src/interfaces/gui/core.py`
- Uses Pygame.
- Modes:
  - `PvP`
  - `IA`
  - `Arena`
- AI turns run on a background `Thread` with a job token to discard stale
  results after resets/mode changes.
- Numba engines are warmed at difficulty selection time.
- GUI difficulties are defined in `src/interfaces/gui/state.py::Difficulty`.
- `ReuseFlatNumbaSolverMCTS` is available as a GUI difficulty.

Game state persistence:

- `src/game_state.py`
- Uses pickle to save/load `GameState` under `./data/saves` by default.

## 10. Tests and Current Local Verification

The full test command was run in the current local interpreter:

```bash
pytest -q
```

Current local interpreter:

```text
Python 3.13.11
numba: not installed
sklearn: not installed
seaborn: not installed
```

Full test result in this environment:

- Collection fails with 4 errors because `numba` is missing:
  - `tests/test_numba_mcts.py`
  - `tests/test_numba_rules.py`
  - `tests/test_numba_search.py`
  - `tests/test_numba_solver.py`

Non-Numba-focused run:

```bash
pytest -q \
  --ignore=tests/test_numba_mcts.py \
  --ignore=tests/test_numba_rules.py \
  --ignore=tests/test_numba_search.py \
  --ignore=tests/test_numba_solver.py \
  --ignore=tests/test_reuse_mcts.py
```

Result:

- `289 passed`
- `3 failed`
- `1 warning`

The 3 failures were:

- `tests/test_integration.py::TestID3TrainingPipeline::test_dataset_generation_and_training`
  - generated a tiny 50-row dataset and got train score `0.2`, below expected
    `> 0.5`.
- `tests/test_integration.py::TestEndToEndPipeline::test_complete_pipeline`
  - generated a tiny 40-row dataset and got train score `0.2`, below expected
    `> 0.4`.
- `tests/test_mcts_solver.py::TestSolverVsFlatNumba::test_wins_at_least_one_game`
  - still imports Numba and fails due missing `numba`.

Interpretation:

- The Numba failures are environment issues in this local interpreter.
- The two ID3 integration failures are not caused by missing Numba; they involve
  very small generated datasets and threshold assumptions. Re-run in the
  intended Conda environment before deciding whether they are real regressions.

## 11. Current Worktree State

Before editing this context file, local `git status --short` showed existing
modified files:

```text
M notebooks/PopOut_Solution.ipynb
M notebooks/Technical_Documentation.ipynb
M notebooks/guide.ipynb
M scripts/solve_probe.py
```

Those changes were already present and should be treated as user work. Do not
revert them unless explicitly asked.

This file is currently untracked:

```text
?? AI_CONTEXT.md
```

## 12. Known Risks and Caveats

Environment:

- Use the Conda env from `environment.yml` for authoritative testing.
- The active local Python 3.13 environment is missing important dependencies.

Dependencies:

- `setup.py` lists only numpy, pandas, matplotlib, and notebook.
- Runtime/report features also need pygame, numba, pytest, seaborn, and
  scikit-learn. Treat `environment.yml` as the more complete dependency source.

Academic compliance:

- Do not replace the custom ID3 training with `sklearn.tree.DecisionTreeClassifier`.
- If scikit-learn appears in notebooks, keep it limited to utilities such as
  metrics, train/test splitting, or dataset loading, and explain that the tree
  itself is custom.
- Clearly report that playable ID3 agents use tactical safeguards, while the
  classifier itself is still a custom ID3 tree.

Notebook readiness:

- Because notebooks are modified locally, inspect outputs before submission.
- Remove stale errors from saved notebook outputs.
- Ensure any `sklearn` references cannot be misunderstood as training the
  required ID3 tree.

Testing:

- Numba test modules import optimized code at collection time, so missing Numba
  stops collection before most tests run.
- For partial testing without Numba, ignore Numba-specific test files, but do
  not treat that as a full validation.

Performance:

- Flat Numba engines rely on preallocated arrays and numeric kernels. Avoid
  introducing Python object abstractions in optimized hot paths.
- Call warmup before benchmarking; otherwise JIT compile time contaminates
  timing results.

Generated artifacts:

- Pickle models are convenient for demos but are binary and version-sensitive.
- If changing feature schemas or model code, old pickle files may no longer
  match. Regenerate intentionally and document the command used.

## 13. Common Tasks and Where to Work

Change game rules:

- Standard: `src/engine/standard/bitboard.py`, `src/engine/standard/rules.py`
- Optimized mirror: `src/engine/optimized/numba_bitboard.py`,
  `src/engine/optimized/numba_rules.py`
- Tests: `tests/test_bitboard.py`, `tests/test_rules.py`,
  `tests/test_numba_rules.py`

Change MCTS behavior:

- Standard search: `src/mcts/standard/base.py`
- Solver proof logic: `src/mcts/standard/uct_solver.py`
- Optimized kernels: `src/mcts/optimized/numba_search.py`
- Factory names: `src/mcts/factory.py`
- Tests: `tests/test_mcts.py`, `tests/test_mcts_solver.py`,
  `tests/test_numba_mcts.py`, `tests/test_numba_solver.py`

Change ID3:

- Learner: `src/decision_tree/id3/learner.py`
- Discretization: `src/decision_tree/discretizer.py`
- Playable agents: `src/decision_tree/id3_agent.py`,
  `src/decision_tree/id3_agent_raw.py`
- Dataset generation: `src/decision_tree/dataset_generator.py`
- Tests: `tests/test_id3.py`, `tests/test_discretizer.py`,
  `tests/test_bulk_generate.py`, `tests/test_integration.py`

Change GUI:

- Main loop/input/threading: `src/interfaces/gui/core.py`
- Drawing: `src/interfaces/gui/renderer.py`
- Menu components: `src/interfaces/gui/components.py`
- Difficulty list: `src/interfaces/gui/state.py`
- Tests: `tests/test_gui.py`

Change CLI:

- `src/interfaces/cli.py`
- Tests: `tests/test_cli.py`

Update project documentation:

- `README.md`
- `PROJECT_SUMMARY_AND_REVIEW.md`
- `AI_CONTEXT.md`
- assignment notebooks under `notebooks/`

## 14. Guidance for Future AI Assistants

Default priorities:

1. Preserve correctness of the standard Python engine first.
2. Keep optimized Numba behavior equivalent to standard behavior.
3. Preserve assignment compliance, especially custom ID3.
4. Avoid modifying generated data/models unless requested.
5. Run focused tests after changes; use the Conda environment for final results.

When debugging:

- Start with standard Python versions before optimized Numba versions.
- Use `get_agent()` rather than directly importing concrete agent classes in new
  app/interface code.
- Check move legality before applying moves in user-facing flows.
- Remember that model predictions are strings like `drop_3`, while engine moves
  are integers.
- Treat notebook and script changes already in the worktree as user changes.

When explaining the project:

- Emphasize bitboard efficiency, UCT MCTS, solver proof propagation, Numba
  acceleration, generated MCTS-labeled datasets, and custom ID3.
- Be transparent about hybrid ID3 agents using immediate tactical safeguards.
- Mention that proper testing requires the Python 3.10 Conda environment with
  Numba installed.
