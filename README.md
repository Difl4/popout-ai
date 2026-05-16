# PopOut AI

**Inteligência Artificial 2025/2026 — Trabalho Prático**

Implementação do jogo **PopOut** (variante do Connect-4 com jogadas *pop*) e dois
agentes adversariais: um baseado em **Monte Carlo Tree Search** com várias
variantes (UCT, MCTS-Solver, Numba JIT, *tree reuse*) e outro baseado em
**árvores de decisão ID3** treinadas a partir das decisões do MCTS.

**Grupo**

- [Duarte Gomes: Duarte Meneses dos Santos Sousa Gomes, 202409386]
- [José Sousa: José Paulo Pacheco de Sousa, 202405046]
- [ALUNO 3: Tiago Braga da Cruz Frada de Sousa, 202405406]

---

## Estrutura do projecto

```text
popout-ai/
├── docs/
│   ├── IA_2526_Trab.pdf            # Enunciado oficial
│   └── autoavaliacao.md            # Ficheiro de auto-avaliação
├── src/
│   ├── __main__.py                 # Entry point: python -m src
│   ├── config.py                   # Dimensões do tabuleiro
│   ├── engine/
│   │   ├── standard/               # Bitboard + regras em Python puro
│   │   └── optimized/              # Mirror em Numba JIT (@njit)
│   ├── mcts/
│   │   ├── factory.py              # get_agent("nome") → instância
│   │   ├── protocol.py             # MCTSEngine (runtime-checkable Protocol)
│   │   ├── standard/               # BaseMCTS, StandardUCT, ExperimentalUCT, SolverMCTS, ReuseSolverMCTS
│   │   └── optimized/              # NumbaMCTS, FlatNumbaMCTS, NumbaSolverMCTS,
│   │                               # FlatNumbaSolverMCTS, ReuseFlatNumbaSolverMCTS
│   ├── decision_tree/
│   │   ├── id3/learner.py          # Classificador ID3 (sem scikit-learn)
│   │   ├── discretizer.py          # Discretização quantile (Iris)
│   │   ├── dataset_generator.py    # Geração paralela (state, best_move) via MCTS
│   │   ├── id3_agent.py            # Agente híbrido: _forced_move + ID3+features
│   │   └── id3_agent_raw.py        # Agente híbrido: _forced_move + ID3 raw cells
│   ├── interfaces/
│   │   ├── cli.py                  # CLI: HvH, HvC, CvC, torneio
│   │   └── gui/                    # GUI Pygame (PvP, HvC, Arena CvC)
│   └── utils/numba_tools.py        # Gestão de cache Numba
├── data/
│   ├── figures/                    # PNGs de benchmarks e visualizações
│   └── generated/                  # Datasets CSV + modelos ID3 pickle (v1..v4)
├── notebooks/
│   ├── PopOut_Solution.ipynb              # Notebook principal da entrega
│   ├── ID3_Decision_Tree.ipynb            # Iris warm-up
│   ├── PopOut_Decision_Tree_Pipeline.ipynb  # Pipeline ID3 detalhado
│   └── iris.csv                            # Dataset Iris
├── scripts/
│   ├── apply_leakage_fix.py        # Reproduz análise 3-split de leakage
│   ├── oversample_experiment.py    # Compara OVERSAMPLE_FACTOR ∈ {4,8,16}
│   ├── solve_probe.py              # Sonda o MCTS-Solver em PopOut
│   ├── train_id3.py                # Treina ID3+features e grava .pkl
│   └── train_id3_raw.py            # Treina ID3 raw e grava .pkl
├── tests/                          # Suite pytest (~17 ficheiros)
├── environment.yml                 # Ambiente Conda (recomendado)
├── setup.py                        # pip install -e .
├── README.md
└── AI_CONTEXT.md                   # Briefing técnico (uso interno)
```

---

## Pré-requisitos

- **Python 3.10**
- Recomendado: **Conda** (ficheiro `environment.yml` incluído)

---

## Instalação

### Opção A — Conda (recomendado)

```bash
conda env create -f environment.yml
conda activate popout-ai
```

### Opção B — venv + pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

> O `setup.py` foi alinhado com o `environment.yml`, pelo que esta opção instala
> todas as dependências necessárias (numpy, pandas, matplotlib, seaborn, notebook,
> pygame, numba, scikit-learn). O extra `[dev]` adiciona `pytest` + `pytest-cov`.

---

## Execução

### GUI (Pygame)

```bash
python -m src
```

Modos disponíveis no menu inicial:

- **PvP** — Humano vs Humano.
- **Humano vs IA** — selecciona dificuldade (100 a 100 000 iterações MCTS, ou um dos 6 modelos ID3 pré-treinados).
- **Arena** — Computador vs Computador, dois agentes seleccionáveis independentemente.

### CLI

```bash
python -m src --cli
```

Sintaxe de jogadas: `d3` = drop na coluna 3, `p0` = pop da coluna 0.

Modos: 1) HvH, 2) HvC, 3) CvC (torneio com configuração de agentes e número de jogos).

### Notebooks

```bash
jupyter notebook notebooks/PopOut_Solution.ipynb
```

O notebook principal é `PopOut_Solution.ipynb` — entry point da entrega.

---

## Testes

```bash
pytest -q
```

A suite cobre regras, bitboard, MCTS standard, MCTS-Solver, kernels Numba,
ID3, discretização, dataset generator e integração.

---

## Cenários de jogo suportados

| Cenário | GUI | CLI |
|---------|-----|-----|
| Humano vs Humano | ✓ PvP | ✓ |
| Humano vs Computador | ✓ (13 dificuldades) | ✓ (escolha de agente) |
| Computador vs Computador | ✓ Arena | ✓ Torneio configurável |

---

## Documentos relacionados

- [`docs/IA_2526_Trab.pdf`](docs/IA_2526_Trab.pdf) — enunciado oficial.
- [`docs/autoavaliacao.md`](docs/autoavaliacao.md) — ficheiro de auto-avaliação.
- `AI_CONTEXT.md` — briefing técnico estruturado para handoff e revisão.
