# 🤖 Project: PopOut AI (MCTS & ID3)

> **Goal:** Build an agent to play PopOut (Connect-4 variant) using MCTS and an ID3-learned Decision Tree.
> **Key Tech:** Python, Bitboards, Custom ID3 (no scikit-learn). 

---

## 🎯 Game Engine Core (PopOut Rules)
* **Standard Moves:** Players alternate dropping discs into 7 columns.
* **Pop Moves:** A player can remove their own color disc from the bottom row.
* **Gravity:** Popping a disc causes all discs above it to drop one space.
* **Win Condition:** First to 4-in-a-row (horizontally, vertically, or diagonally).
* **Conflict Rule:** If a pop move creates 4-in-a-row for both, the player who moved wins.
* **Draw Rules:** Board is full (player choice) or state repeats 3 times.

## 🏗 System Architecture
* **Bitboard Logic:** Represents board as two 64-bit integers for $O(1)$ win-checks and fast gravity shifts.
* **MCTS (UCT):** Adversarial search using the Upper Confidence Bound for Trees.
* **ID3 Algorithm:** Custom training procedure for classification.
    * **Task 1:** Classify Iris dataset (requires numerical discretization).
    * **Task 2:** Classify PopOut moves based on MCTS-generated (State -> Move) data.

## 📂 Project Structure (Future-Proof)
```text
/popout-ai
├── data/               
│   ├── raw/            # Iris.csv
│   └── generated/      # Datasets for different MCTS variants (e.g., uct_v1.csv, uct_v2.csv)
├── src/                
│   ├── engine/         
│   │   ├── bitboard.py # 64-bit board logic (Drop/Pop)
│   │   └── rules.py    # Win/Draw/Repetition logic
│   ├── algorithms/     
│   │   ├── mcts/       # 📂 MCTS Variation Folder
│   │   │   ├── base.py # Abstract class with the 4 MCTS steps
│   │   │   ├── uct_standard.py # Standard UCT implementation
│   │   │   └── uct_experimental.py # Variations in child selection/breadth 
│   │   └── id3/        
│   │       ├── learner.py    # Core ID3 procedure
│   │       └── discretizer.py # Numerical-to-categorical logic
│   ├── scripts/        
│   │   └── bulk_generate.py  # Script that accepts a variant name as an argument
│   └── interfaces/     
│       ├── cli.py      # Fast headless mode
│       └── gui.py      # Visualization for presentation
└── PopOut_Solution.ipynb # Integrated documentation & results