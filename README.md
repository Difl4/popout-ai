# PopOut AI

Projeto de Inteligencia Artificial (2025/2026) — jogo **PopOut** (variante de Connect Four) com agentes baseados em **MCTS** e arvores de decisao **ID3**.

---

## Estrutura do projeto

```text
popout-ai/
├── src/
│   ├── __main__.py            # Entry point (python -m src)
│   ├── game_state.py          # Persistencia de estado
│   ├── engine/
│   │   ├── bitboard.py        # Representacao do tabuleiro (bitboard)
│   │   └── rules.py           # Regras do jogo e detecao de vitoria
│   ├── algorithms/
│   │   ├── mcts/
│   │   │   ├── base.py        # MCTS base (select, expand, simulate, backprop)
│   │   │   ├── protocol.py    # Interface MCTSEngine
│   │   │   ├── uct_standard.py
│   │   │   ├── uct_experimental.py
│   │   │   ├── numba_mcts.py  # Variantes aceleradas com Numba
│   │   │   └── kernels.py     # Kernels JIT para bitboard
│   │   └── id3/
│   │       ├── learner.py     # Classificador ID3
│   │       └── discretizer.py # Discretizacao de atributos numericos
│   ├── interfaces/
│   │   ├── cli.py             # Interface de linha de comandos
│   │   └── gui.py             # Interface grafica (pygame)
│   └── scripts/
│       └── bulk_generate.py   # Geracao de datasets via MCTS
├── data/
│   └── generated/             # Datasets gerados pelo MCTS
├── tests/                     # Suite de testes (pytest)
├── PopOut_Solution.ipynb       # Notebook principal da entrega
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
python -m src.scripts.bulk_generate --variant uct_standard --samples 200 --iterations 150 --seed 42
```

### Notebook

```bash
jupyter notebook PopOut_Solution.ipynb
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
| Humano vs Humano | GUI (modo PvP) |
| Humano vs Computador | GUI e CLI (MCTS) |
| Computador vs Computador | Em desenvolvimento (MCTS vs ID3) |
