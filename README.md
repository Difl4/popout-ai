# PopOut AI

**Inteligência Artificial 2025/2026 — Trabalho Prático**

Implementação do jogo **PopOut** (variante do Connect-4 com jogadas *pop*) e dois
agentes adversariais: um baseado em **Monte Carlo Tree Search** com várias
variantes (UCT, MCTS-Solver, Numba JIT, *tree reuse*) e outro baseado em
**árvores de decisão ID3** treinadas a partir das decisões do MCTS.

**Grupo**

- [Duarte Meneses dos Santos Sousa Gomes, 202409386]
- [José Paulo Pacheco de Sousa, 202405046]
- [Tiago Braga da Cruz Frada de Sousa, 202405406]

---

## Estrutura do projecto

```text
popout-ai/
├── docs/
│   └── IA_2526_Trab.pdf                     # Enunciado oficial
├── src/
│   ├── __main__.py                          # Entry point: python -m src
│   ├── config.py                            # Dimensões do tabuleiro (7×6, WIN=4)
│   ├── engine/
│   │   ├── standard/                        # Bitboard + regras em Python puro
│   │   └── optimized/                       # Mirror em Numba JIT (@njit)
│   ├── mcts/
│   │   ├── factory.py                       # get_agent("nome") → instância
│   │   ├── protocol.py                      # MCTSEngine (runtime-checkable Protocol)
│   │   ├── standard/                        # StandardUCT, ExperimentalUCT, SolverMCTS, ReuseSolverMCTS
│   │   └── optimized/                       # NumbaMCTS, FlatNumbaMCTS, NumbaSolverMCTS,
│   │                                        # FlatNumbaSolverMCTS, ReuseFlatNumbaSolverMCTS
│   ├── decision_tree/
│   │   ├── id3/learner.py                   # Classificador ID3 (sem scikit-learn)
│   │   ├── discretizer.py                   # Discretização quantile
│   │   ├── dataset_generator.py             # Geração paralela de (estado, best_move) via MCTS
│   │   ├── game_eval.py                     # Avaliação paralela de agentes (usado pelo pipeline notebook)
│   │   ├── id3_agent.py                     # Agente híbrido: _forced_move + ID3 + features
│   │   └── id3_agent_raw.py                 # Agente híbrido: _forced_move + ID3 células brutas
│   ├── interfaces/
│   │   ├── cli.py                           # CLI: HvH, HvC, CvC, torneio
│   │   └── gui/                             # GUI Pygame (PvP, HvC, Arena CvC)
│   └── utils/numba_tools.py                 # Gestão de cache Numba
├── data/
│   ├── iris.csv                             # Dataset Iris (exercício ID3 de aquecimento)
│   ├── figures/                             # PNGs de benchmarks e visualizações
│   └── generated/                           # Datasets CSV + modelos pickle treinados
│       ├── tournament/                      # Resultados do torneio round-robin completo
│       └── v1..v4/                          # popout_dt_dataset.csv, id3_model.pkl,
│                                            # id3_model_raw.pkl, results/
├── notebooks/
│   ├── PopOut_Full_Report.ipynb             # Relatório técnico completo (entrega principal)
│   ├── PopOut_Decision_Tree_Pipeline.ipynb  # Pipeline ID3 detalhado
│   └── ID3_Decision_Tree.ipynb             # Iris warm-up
├── scripts/                                 # Scripts experimentais e utilitários auxiliares
│   ├── oversample_experiment.py             # Compara OVERSAMPLE_FACTOR ∈ {4, 8, 16}
│   ├── solve_probe.py                       # Sonda de prova do MCTS-Solver em PopOut
│   └── tournament_worker.py                 # Worker de torneios paralelos (ProcessPoolExecutor)
├── tests/                                   # Suite pytest (18 ficheiros)
├── play.py                                  # Atalho para lançar a GUI (equivalente a python -m src)
├── environment.yml                          # Ambiente Conda (recomendado)
├── setup.py                                 # pip install -e .
└── README.md
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
# ou equivalentemente:
python play.py
```

Modos disponíveis no menu inicial:

- **PvP** — Humano vs Humano.
- **Humano vs IA** — selecciona dificuldade (100 a 100 000 iterações MCTS, ou um dos modelos ID3 pré-treinados).
- **Arena** — Computador vs Computador, dois agentes seleccionáveis independentemente.

### CLI

```bash
python -m src --cli
```

Sintaxe de jogadas: `d3` = drop na coluna 3, `p0` = pop da coluna 0.

Modos: 1) HvH, 2) HvC, 3) CvC (torneio com configuração de agentes e número de jogos).

### Notebooks

```bash
jupyter notebook notebooks/PopOut_Full_Report.ipynb
```

- **`PopOut_Full_Report.ipynb`** — relatório técnico completo; ponto de entrada da entrega.
- **`PopOut_Decision_Tree_Pipeline.ipynb`** — pipeline detalhado do ID3: geração de dataset, treino, avaliação paralela.
- **`ID3_Decision_Tree.ipynb`** — exercício de aquecimento com o dataset Iris.

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
| Humano vs Computador | ✓ (múltiplas dificuldades) | ✓ (escolha de agente) |
| Computador vs Computador | ✓ Arena | ✓ Torneio configurável |

---

## Documentos relacionados

- [`docs/IA_2526_Trab.pdf`](docs/IA_2526_Trab.pdf) — enunciado oficial.
- [github.com/Difl4/popout-ai](https://github.com/Difl4/popout-ai) — repositório GitHub.
