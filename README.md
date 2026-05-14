# PopOut AI

Projeto de Inteligencia Artificial (2025/2026) — jogo **PopOut** (variante de Connect Four) com agentes baseados em **MCTS** e arvores de decisao **ID3**.

---

## Estrutura do projeto

```text
popout-ai/
├── src/
│   ├── __main__.py            # Entry point (python -m src)
│   ├── config.py              # Configuracao global (dimensoes do tabuleiro, etc.)
│   ├── game_state.py          # Persistencia de estado
│   ├── engine/
│   │   ├── standard/
│   │   │   ├── bitboard.py    # Representacao do tabuleiro (bitboard)
│   │   │   └── rules.py       # Regras do jogo e detecao de vitoria
│   │   └── optimized/
│   │       ├── numba_bitboard.py  # Bitboard JIT (@njit)
│   │       └── numba_rules.py    # Kernels JIT das regras (@njit)
│   ├── mcts/
│   │   ├── factory.py         # Instanciacao de agentes por nome
│   │   ├── protocol.py        # Interface MCTSEngine
│   │   ├── standard/
│   │   │   ├── base.py              # MCTS base (select, expand, simulate, backprop)
│   │   │   ├── uct_standard.py      # Agente StandardUCT
│   │   │   ├── uct_experimental.py  # Agente ExperimentalUCT
│   │   │   └── uct_solver.py        # MCTS-Solver com provas AND/OR + distancia minimax
│   │   └── optimized/
│   │       ├── numba_mcts.py    # NumbaMCTS e FlatNumbaMCTS (JIT)
│   │       ├── numba_solver.py  # NumbaSolverMCTS e FlatNumbaSolverMCTS (JIT)
│   │       └── numba_search.py  # Kernels JIT de pesquisa MCTS (@njit)
│   ├── decision_tree/
│   │   ├── discretizer.py       # Discretizacao de atributos numericos (quantis)
│   │   ├── dataset_generator.py # Geracao de datasets (estado -> jogada) via MCTS
│   │   ├── id3_agent.py         # Agente ID3 jogavel (implementa MCTSEngine)
│   │   └── id3/
│   │       └── learner.py       # Classificador ID3 (sem scikit-learn)
│   ├── interfaces/
│   │   ├── cli.py             # Interface de linha de comandos (HvH, HvC, CvC)
│   │   └── gui/               # Interface grafica (pygame)
│   │       ├── core.py        # Loop principal e logica de entrada
│   │       ├── renderer.py    # Funcoes de desenho
│   │       ├── assets.py      # Constantes visuais e fontes
│   │       ├── components.py  # Componentes UI (menus, etc.)
│   │       └── state.py       # Estado de animacao e dificuldade
│   └── utils/
│       └── numba_tools.py     # Limpeza de cache e recompilacao Numba
├── data/
│   └── generated/             # Datasets CSV gerados pelo MCTS
├── notebooks/
│   ├── PopOut_Solution.ipynb  # Notebook principal da entrega
│   └── ID3_Decision_Tree.ipynb
├── tests/                     # Suite de testes (pytest)
├── environment.yml            # Ambiente Conda
└── setup.py                   # Configuracao do pacote
```

---

## Pre-requisitos

- **Python 3.10**
- Recomendado: **Conda** (`environment.yml` incluido)
- Alternativa: `venv` + `pip`

---

## Instalacao

### Opcao A — Conda (recomendado)

```bash
conda env create -f environment.yml
conda activate popout-ai
```

### Opcao B — venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Como executar

### GUI (interface grafica)

```bash
python -m src
```

### CLI (terminal)

```bash
python -m src --cli
```

No jogo: `d3` = drop na coluna 3, `p0` = pop na coluna 0.

### Gerar dataset

```bash
python -m src.decision_tree.dataset_generator --variant uct_standard --samples 200 --iterations 150 --seed 42
```

### Notebook

```bash
jupyter notebook notebooks/PopOut_Solution.ipynb
```

---

## Testes

```bash
pytest -v
```

---

## Cenarios de jogo suportados

| Cenario | Disponivel |
|---------|-----------|
| Humano vs Humano | GUI (modo PvP) e CLI |
| Humano vs Computador | GUI e CLI (MCTS ou ID3) |
| Computador vs Computador | CLI — torneio configuravel (MCTS vs MCTS, MCTS vs ID3, etc.) |
