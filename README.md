# PopOut AI

Projeto em Python para o jogo **PopOut (variante de Connect Four)** com foco em:
- motor de jogo (regras + bitboard),
- agentes de decisão com **MCTS** (UCT standard e experimental),
- geração de datasets para treino,
- componentes de aprendizagem (ID3),
- interface de linha de comandos (CLI).

---

## Estrutura do projeto

```text
popout-ai/
├── environment.yml
├── setup.py
├── PopOut_Solution.ipynb
├── src/
│   ├── algorithms/
│   │   ├── mcts/
│   │   │   ├── base.py
│   │   │   ├── uct_standard.py
│   │   │   └── uct_experimental.py
│   │   └── id3/
│   │       ├── discretizer.py
│   │       └── learner.py
│   ├── engine/
│   │   ├── bitboard.py
│   │   └── rules.py
│   ├── interfaces/
│   │   ├── cli.py
│   │   └── gui.py
│   └── scripts/
│       └── bulk_generate.py
└── tests/
```

---

## Pré-requisitos

- **Python 3.10**
- Recomendado: **Conda** (há ficheiro `environment.yml` pronto)
- Alternativa: `venv` + `pip`

Dependências principais:
- `numpy`
- `pandas`
- `matplotlib`
- `notebook`

---

## Instalação

### Opção A — Conda (recomendado)

No diretório `popout-ai/`:

```bash
conda env create -f environment.yml
conda activate popout-ai
```

> O `environment.yml` já instala o projeto em modo editável (`-e .`).

---

### Opção B — venv + pip

No diretório `popout-ai/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

---

## Como executar

> Todos os comandos abaixo devem ser corridos dentro da pasta `popout-ai/`.

### 1) Jogar no terminal (CLI)

```bash
python -m src.interfaces.cli
```

No jogo:
- `d3` = drop na coluna 3
- `p0` = pop na coluna 0

A CLI coloca:
- jogador humano como **X (Jogador 1)**
- IA como **O (Jogador 2)**

---

### 2) Gerar dataset em lote (MCTS)

Script: `src/scripts/bulk_generate.py`

Exemplo:

```bash
python -m src.scripts.bulk_generate --variant uct_standard --samples 200 --iterations 150 --seed 42
```

Parâmetros disponíveis:
- `--variant` : `uct_standard` ou `uct_experimental`
- `--samples` : número de estados a gerar
- `--iterations` : iterações MCTS por estado
- `--seed` : semente aleatória
- `--output` : caminho CSV de saída (opcional)

Se `--output` não for definido, o default é:
`data/generated/<variant>.csv`

Exemplo com output explícito:

```bash
python -m src.scripts.bulk_generate --variant uct_experimental --samples 500 --iterations 300 --output data/generated/exp.csv
```

---

### 3) GUI (estado atual)

```bash
python -m src.interfaces.gui
```

Atualmente a GUI é um **placeholder** (não implementada).

---

### 4) Notebook

Existe o notebook:

- `PopOut_Solution.ipynb`

Para usar:

```bash
jupyter notebook
```

Depois abrir o ficheiro no browser.

---

## Testes

Para correr a suite de testes:

```bash
pytest -q
```

Ou, se preferires verboso:

```bash
pytest -v
```

---

## Notas úteis

- O projeto está preparado para desenvolvimento local com `pip install -e .`.
- A implementação de algoritmos e engine está em `src/`.
- Os testes estão em `tests/` e cobrem regras, bitboard, MCTS, ID3, scripts e CLI.

---

## Troubleshooting rápido

- **`ModuleNotFoundError`**: confirma que o ambiente está ativo e que foi feito `pip install -e .`.
- **`conda: command not found`**: usar a opção de `venv` + `pip`.
- **Import errors ao correr scripts**: executa com `python -m ...` a partir da raiz `popout-ai/` (evita correr ficheiros soltos diretamente).
