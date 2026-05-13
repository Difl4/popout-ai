# PopOut AI Project Summary and Review

This document explains the project for someone who needs to understand the assignment, the code organization, the algorithms used, the logic behind the implementation, the current evidence/results, and the main improvement opportunities.

## 1. Assignment Context

The assignment `IA_2526_Trab.pdf` asks for an Artificial Intelligence project about **PopOut**, a Connect-4 variant where players can either:

- **Drop** a piece into a column.
- **Pop** one of their own pieces from the bottom row, making the pieces above fall down.

The required work is:

- Implement a playable PopOut program.
- Support three scenarios:
  - Human vs Human.
  - Human vs Computer.
  - Computer vs Computer with two different algorithms.
- Implement **Monte Carlo Tree Search (MCTS)** using **UCT**.
- Explore variants beyond standard MCTS.
- Implement a **Decision Tree learned with ID3**, without using scikit-learn or similar libraries to train the tree.
- Use the Iris dataset as a warm-up decision-tree task.
- Generate a PopOut dataset using MCTS: `(state, best_move)`.
- Train an ID3 tree that predicts the next move for PopOut.
- Document the solution and results in notebooks/slides.

The assignment evaluation weights are:

- 30% adversarial strategy.
- 30% decision trees.
- 30% technical quality and performance evaluation.
- 10% communication.

## 2. Project Structure

The project is organized as a Python package under `src/`, with notebooks, tests, datasets, and scripts around it.

Main folders:

- `src/engine/`: PopOut board representation and rules.
- `src/mcts/`: MCTS agents and optimized variants.
- `src/decision_tree/`: ID3 implementation, dataset generation, and playable ID3 agents.
- `src/interfaces/`: CLI and Pygame GUI.
- `tests/`: Pytest test suite.
- `notebooks/`: assignment-facing notebooks and analysis.
- `data/generated/`: generated PopOut datasets and trained ID3 pickle files.
- `scripts/`: helper scripts for training/probing.

Important entry points:

- GUI: `python -m src`
- CLI: `python -m src --cli`
- Alternative direct launcher: `python play.py`

## 3. Game Engine Logic

The core board class is `PopOutBoard` in `src/engine/standard/bitboard.py`.

The board is represented with two integer bitmasks:

- `mask_p1`: occupied cells for player 1.
- `mask_p2`: occupied cells for player 2.

The game uses a 7-column by 6-row board. Each column is stored with 7 bits: 6 playable bits plus one guard bit. This makes win detection fast and avoids false connections across columns.

Moves are encoded as integers:

- `0..6`: drop in column `0..6`.
- `7..13`: pop in column `0..6`.

Main board operations:

- `legal_moves()` returns all legal drops and pops for the current player.
- `apply_move(move)` mutates the board and switches the current player.
- `is_full()` checks whether the top row is full.
- `to_feature_dict()` converts the board to machine-learning features.

Rules are implemented in `src/engine/standard/rules.py`.

Important rule logic:

- `has_won(bitmask)` detects four-in-a-row using bit shifts.
- `evaluate_after_move(board, mover)` applies the PopOut simultaneous-win rule: if a pop creates winning lines for both players, the player who made the move wins.
- `is_threefold_repetition(history)` detects repeated states.
- `extended_features(board)` adds tactical features for ID3.

The rules are solid and efficient. The bitboard design is one of the strongest technical parts of the project.

## 4. MCTS Algorithms

The standard MCTS implementation is in `src/mcts/standard/base.py`.

It follows the four classic phases:

1. **Selection**: traverse the tree using UCT.
2. **Expansion**: add one untried move as a child node.
3. **Simulation**: play random rollout moves until win, draw, repetition, or depth limit.
4. **Backpropagation**: update visits and value estimates up the tree.

UCT formula:

```text
score = exploitation + exploration
score = Q + C * sqrt(log(parent_visits) / child_visits)
```

Implemented variants:

- `StandardUCT`: baseline UCT MCTS.
- `ExperimentalUCT`: small variation that prioritizes unvisited children first.
- `SolverMCTS`: MCTS-Solver with proof propagation.
- `NumbaMCTS`: optimized MCTS using Numba kernels.
- `FlatNumbaMCTS`: faster flat-array Numba implementation.
- `NumbaSolverMCTS`: optimized solver-style MCTS.
- `FlatNumbaSolverMCTS`: fastest solver variant.

The agent factory in `src/mcts/factory.py` centralizes agent creation.

### MCTS-Solver

`src/mcts/standard/uct_solver.py` adds game-theoretic proof labels to nodes:

- `UNKNOWN`
- `WIN`
- `LOSS`
- `DRAW`

The status is from the perspective of the player to move at that node.

The solver can prove forced wins/losses/draws in parts of the game tree. It also stores minimax distance, choosing faster wins and slower losses. This goes beyond the minimum assignment requirement and is a strong technical contribution.

### Numba Optimization

The optimized code under `src/mcts/optimized/` moves hot MCTS loops into JIT-compiled functions.

The technical documentation reports approximate throughput:

- `StandardUCT`: about 9k iterations/s.
- `SolverMCTS`: about 8k iterations/s.
- `NumbaMCTS`: about 50k iterations/s.
- `NumbaSolverMCTS`: about 37k iterations/s.
- `FlatNumbaMCTS`: about 179k iterations/s.
- `FlatNumbaSolverMCTS`: about 168k iterations/s.

This is very good evidence of performance engineering.

## 5. Decision Tree / ID3 Logic

The ID3 implementation is in `src/decision_tree/id3/learner.py`.

It is implemented from scratch and does not use scikit-learn to train the tree.

Main parts:

- `entropy(labels)`: computes Shannon entropy.
- `information_gain(df, feature, target)`: computes gain for one feature.
- `build_tree(...)`: recursively chooses the feature with highest information gain.
- `predict_one(row)`: follows the learned tree for one example.
- `score(df, target)`: computes accuracy.
- `get_feature_importance()`: counts how often each feature appears in splits.

The implementation handles:

- Pure nodes.
- No-feature fallback to majority class.
- Maximum depth.
- Unknown prediction branch fallback to the node majority class.
- Input validation.

### Iris Dataset

The Iris workflow is in `notebooks/ID3_Decision_Tree.ipynb`.

Because ID3 is categorical, numerical Iris attributes are discretized using quantile bins in `src/decision_tree/discretizer.py`.

Notebook results:

- 50 fold evaluations: 10 folds x 5 repetitions.
- Mean accuracy: **95.07%**.
- Standard deviation: **5.63%**.
- F1 scores:
  - Iris-setosa: 0.986.
  - Iris-versicolor: 0.925.
  - Iris-virginica: 0.941.

This satisfies the warm-up dataset requirement well.

### PopOut Dataset

The PopOut dataset generator is in `src/decision_tree/dataset_generator.py`.

There are two dataset-generation paths:

- Simple generation: randomize states, then label each state with a standard MCTS move.
- Parallel generation: play full games and label every position using `FlatNumbaSolverMCTS`.

The main generated dataset is:

- File: `data/generated/popout_dt_dataset.csv`
- Rows: **119,676**
- Columns: **52**
- Features:
  - 42 raw cell features.
  - `current_player`.
  - Tactical features: threats, center control, game phase, immediate win flags.
  - Target: `best_move`.
  - Metadata: `is_proven`.

Dataset label distribution:

- Drop labels: **109,896**
- Pop labels: **9,780**
- Proven positions: **57,272** / 119,676 = **47.9%**

The dataset uses horizontal mirroring to double useful samples because PopOut is symmetric left-to-right.

### Playable ID3 Agents

Two playable ID3 agents exist:

- `ID3Agent`: uses raw board cells plus tactical features.
- `ID3AgentRaw`: uses only raw cell features and current player.

`ID3Agent` also adds tactical safeguards before asking the tree:

1. Play immediate winning move if available.
2. Block opponent immediate win if possible.
3. Otherwise use ID3 prediction.

This makes the ID3 agent more robust in actual games, although it means the playable agent is not purely tree-only.

PopOut ID3 notebook results:

- Train accuracy: **0.854**
- Test accuracy: **0.843**
- Tactical accuracy:
  - `can_win=1`: 0.791
  - `opp_wins_next=1`: 0.750
  - pop moves: 0.894

Head-to-head results from the notebook show the ID3 agents can compete with optimized MCTS variants, with win rates around 40-53% in several configurations. Because PopOut appears to have a strong first-player advantage at high search budgets, alternating colors is important.

## 6. Interfaces

The CLI in `src/interfaces/cli.py` supports:

- Human vs Human.
- Human vs Computer.
- Computer vs Computer tournament.

CLI move syntax:

- `d3`: drop in column 3.
- `p0`: pop from column 0.

The GUI is implemented with Pygame under `src/interfaces/gui/`.

The assignment requirement for three game scenarios is covered.

## 7. Tests and Verification

The test suite is large for a student project.

Test files cover:

- Board rules and bitboard behavior.
- Game state persistence.
- MCTS behavior.
- MCTS-Solver behavior.
- Numba rules/search/MCTS.
- ID3 learning.
- Discretization.
- Dataset generation.
- CLI parsing.
- GUI imports/logic.
- Integration and validation.
- Performance checks.

I ran the tests in the current terminal environment.

Full `pytest -q` result:

- Failed during collection because `numba` is not installed in the active Python 3.13 environment.
- The project environment file expects Python 3.10 and includes `numba`.

Non-Numba-focused test run:

- **283 passed**
- **2 failed**

The two failures were:

1. `tests/test_mcts_solver.py::TestSolverVsFlatNumba::test_wins_at_least_one_game`
   - Cause: imports `FlatNumbaMCTS`, but `numba` is missing in the active environment.
   - This is an environment issue, not necessarily a code issue.

2. `tests/test_discretizer.py::TestApplyBins::test_applies_labels`
   - Cause: expected pandas `object` dtype, but `apply_bins()` returns pandas `StringDtype`.
   - This is a real mismatch between implementation and test expectation. Either the test should accept string dtype, or the implementation should force `astype(object)`.

## 8. Found Issues and Risks

These are the main issues I found.

### Environment mismatch

The active terminal uses Python 3.13 and does not have `numba`, while `environment.yml` specifies Python 3.10 with `numba`.

This blocks the optimized test suite unless the correct Conda environment is used.

### README command errors

The README says:

```bash
python -m src.ml.dataset_generator --variant uct_standard --samples 200 --iterations 150 --seed 42
```

But there is no `src.ml.dataset_generator`. The correct module is:

```bash
python -m src.decision_tree.dataset_generator --variant uct_standard --samples 200 --iterations 150 --seed 42
```

The README also says:

```bash
jupyter notebook PopOut_Solution.ipynb
```

But the file is under `notebooks/`, so the clearer command is:

```bash
jupyter notebook notebooks/PopOut_Solution.ipynb
```

### Main notebook has saved sklearn import error

`notebooks/PopOut_Solution.ipynb` contains saved output showing:

```text
ModuleNotFoundError: No module named 'sklearn'
```

The environment file includes `scikit-learn`, but the saved notebook state may look broken to a professor if submitted as-is.

Also, the assignment forbids using scikit-learn to automatically define/train decision trees. The project's real ID3 implementation is custom, but any use of `sklearn` in notebooks should be clearly limited to loading data or splitting/evaluation, not training a tree.

### Technical notebook has a DecisionTreeClassifier reference

`notebooks/Technical_Documentation.ipynb` contains a `DecisionTreeClassifier()` reference. Even if it is only illustrative or unrelated to the submitted ID3 implementation, it is a presentation risk because the assignment explicitly says not to use libraries to train decision trees.

Recommendation: remove that reference or add a clear note that the submitted decision tree is `ID3Classifier` from scratch.

### Discretizer dtype test mismatch

`apply_bins()` returns pandas string dtype after `.astype(str)`, but the test expects `object`.

This is not a serious algorithmic problem, but it should be fixed before submission.

### Setup dependencies are incomplete

`setup.py` lists only:

- numpy
- pandas
- matplotlib
- notebook

But the project also needs, depending on features:

- pygame
- numba
- pytest
- seaborn

`environment.yml` is more complete, so the Conda environment should be the official installation path.

### PopOut draw rules are simplified for AI

The rules for full board and threefold repetition are implemented, but AI behavior is simplified:

- AI declares repetition draws automatically.
- On a full board, AI continues by popping if possible.

This is acceptable as a strategy choice, but should be explained in the notebook/presentation.

### ID3 playable agent includes hard-coded tactical safeguards

`ID3Agent` uses immediate-win and immediate-block logic before using the tree.

This improves gameplay, but for academic honesty the report should distinguish:

- Pure ID3 classifier accuracy.
- Hybrid playable ID3 agent behavior.

### Generated model files are binary

The trained pickle files are useful for fast demos, but the notebook should also show how to reproduce them from data. Otherwise, someone reviewing the project may not know whether the model was generated correctly.

## 9. Improvement Ideas

Highest-priority improvements before submission:

1. Re-run the project in the intended Conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate popout-ai
   pytest -q
   ```

2. Fix or update the discretizer dtype test.

3. Clean notebook outputs so no failed `sklearn` import appears in the final submitted notebook.

4. Fix README commands.

5. Add a short “Compliance with assignment” section in the main notebook:
   - MCTS implemented.
   - UCT implemented.
   - MCTS variants explored.
   - ID3 implemented from scratch.
   - Iris discretization done.
   - PopOut dataset generated from MCTS.
   - Three game modes supported.

Medium-priority improvements:

1. Add a transposition table to MCTS/Solver using Zobrist hashing.
2. Add deterministic tactical rollout policy experiments and compare against random rollouts.
3. Add better handling/documentation for PopOut draw declarations.
4. Add a confusion matrix for PopOut move prediction.
5. Add a clearer comparison between pure ID3 and hybrid ID3 agent.
6. Add a small reproducible dataset-generation command that finishes quickly for demos.
7. Add command-line flags for GUI/CLI agent selection and iteration budget.

Advanced improvement:

1. Implement proof-number search or alpha-beta with transposition tables to move closer to a full PopOut solver.

## 10. Overall Assessment

This is a strong project technically. It goes beyond the minimum assignment in several ways:

- Efficient bitboard engine.
- Standard MCTS with UCT.
- Experimental MCTS variant.
- MCTS-Solver with proof propagation.
- Numba-optimized search engines.
- Generated PopOut dataset with a large number of samples.
- Custom ID3 implementation.
- Iris and PopOut decision-tree workflows.
- CLI and GUI.
- Broad test suite.
- Performance notebooks and benchmark evidence.

The main weaknesses are not conceptual; they are submission/readiness issues:

- Environment mismatch in the current terminal.
- Some stale or incorrect documentation commands.
- Saved notebook errors.
- One real test mismatch in discretization.
- Need to clearly state where scikit-learn is not used for tree training.

## 11. Suggested Grade Estimate

If submitted exactly as currently stored, I would estimate:

```text
17 / 20
```

Reasoning:

- Adversarial strategy: very strong, likely 9/10 or better.
- Decision trees: strong custom ID3 and datasets, but needs clearer pure-vs-hybrid explanation.
- Technical quality: strong, but affected by environment/test/readme/notebook polish issues.
- Communication: notebooks and docs are good, but should be cleaned before final submission.

If the environment is fixed, notebooks are cleaned, README commands are corrected, and all tests pass, this could reasonably move toward:

```text
18.5 / 20
```

