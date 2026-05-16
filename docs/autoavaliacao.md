# Auto-Avaliação — PopOut AI

**Inteligência Artificial 2025/2026 — Trabalho Prático**

> **Nota**: este documento é uma proposta de auto-avaliação preenchida pelo
> grupo. Caso a equipa docente forneça um template oficial em formato próprio
> (PDF/DOCX), este ficheiro deve ser substituído ou complementado por esse.

---

## 1. Identificação

**Grupo**:

| Aluno | Nº mecanográfico | Curso |
|-------|------------------|-------|
| [ALUNO 1: Nome Completo] | [Nº] | [Curso] |
| [ALUNO 2: Nome Completo] | [Nº] | [Curso] |
| [ALUNO 3: Nome Completo] | [Nº] | [Curso] |

**Data de entrega**: 17 de Maio de 2026

**Repositório**: <https://github.com/Difl4/popout-ai>

---

## 2. Distribuição de tarefas

| Componente | Responsável principal | Contribuição |
|-----------|----------------------|--------------|
| Motor do jogo (bitboard + regras) | [ALUNO ?] | [Descrever] |
| MCTS standard + UCT | [ALUNO ?] | [Descrever] |
| MCTS-Solver com proof propagation | [ALUNO ?] | [Descrever] |
| Optimização Numba (kernels JIT) | [ALUNO ?] | [Descrever] |
| Tree reuse entre jogadas | [ALUNO ?] | [Descrever] |
| ID3 (entropia, IG, build, predict) | [ALUNO ?] | [Descrever] |
| Discretização Iris | [ALUNO ?] | [Descrever] |
| Geração paralela do dataset PopOut | [ALUNO ?] | [Descrever] |
| Pipeline ID3 PopOut + análise de leakage | [ALUNO ?] | [Descrever] |
| GUI Pygame | [ALUNO ?] | [Descrever] |
| CLI + torneio | [ALUNO ?] | [Descrever] |
| Testes | [ALUNO ?] | [Descrever] |
| Notebooks de análise | [ALUNO ?] | [Descrever] |
| Slides + apresentação | [ALUNO ?] | [Descrever] |

---

## 3. Auto-avaliação por critério

Critérios do guião (§4.5):

| Critério | Peso | Auto-avaliação |
|----------|------|----------------|
| Estratégia adversarial (MCTS) | 30% | [ /10] |
| Árvores de decisão (ID3) | 30% | [ /10] |
| Qualidade técnica + rigor da avaliação | 30% | [ /10] |
| Soft skills (comunicação) | 10% | [ /10] |

### 3.1 Estratégia adversarial (30%)

**O que foi implementado**:

- MCTS com UCT canónico (`StandardUCT`).
- Variante alternativa de selecção (`ExperimentalUCT` — prioriza filhos não visitados).
- **MCTS-Solver** com proof propagation AND/OR + distância minimax (`SolverMCTS`).
- 4 variantes optimizadas em Numba JIT (`NumbaMCTS`, `FlatNumbaMCTS`, `NumbaSolverMCTS`, `FlatNumbaSolverMCTS`).
- **Tree reuse** entre jogadas (`ReuseSolverMCTS`, `ReuseFlatNumbaSolverMCTS`) com compactação BFS.
- Análise do efeito do número de filhos seleccionados (`TopKChildMCTS` no notebook).
- Análise da constante de exploração C.

**Auto-avaliação**: [N/10]

**Justificação**: [Texto livre]

### 3.2 Árvores de decisão (30%)

**O que foi implementado**:

- ID3 from scratch (entropia de Shannon, information gain, build recursivo, predict com fallback) — **sem scikit-learn para treinar**.
- Discretização de Iris por quantis (`src/decision_tree/discretizer.py`).
- Cross-validation 5×10 fold no Iris (acc ≈ 95%).
- Geração paralela de 193k posições PopOut com oracle MCTS-Solver de 100k iterações.
- Pipeline de treino ID3 PopOut com análise de overfitting / data leakage (3 splits comparados).
- Dois agentes ID3 jogáveis (`ID3Agent` com features tácticas, `ID3AgentRaw` com cells brutas).

**Auto-avaliação**: [N/10]

**Justificação**: [Texto livre]

### 3.3 Qualidade técnica (30%)

**O que foi implementado**:

- Bitboard 7×6 com 7 bits/coluna (4-shift `has_won`).
- Arquitectura two-tier standard/optimized com validação cruzada.
- Protocol-based design (`MCTSEngine` runtime-checkable).
- Factory pattern para instanciar agentes por nome.
- ~2 200 linhas de testes pytest cobrindo regras, bitboard, MCTS, Solver, kernels Numba, ID3, discretizer, dataset generator.
- Rigor na avaliação de desempenho: 3 splits comparados, OVERSAMPLE testado empiricamente, mais que 13M iter/s no Flat Numba.

**Auto-avaliação**: [N/10]

**Justificação**: [Texto livre]

### 3.4 Soft skills — comunicação (10%)

**O que foi implementado**:

- Notebook principal estruturado em 7 secções (`PopOut_Solution.ipynb`).
- Notebooks complementares para Iris (`ID3_Decision_Tree.ipynb`) e pipeline ID3 (`PopOut_Decision_Tree_Pipeline.ipynb`).
- Markdown explicativo em cada secção, incluindo *Methodological note* sobre data leakage.
- README com estrutura completa + instruções de execução.
- Documentação técnica (`AI_CONTEXT.md`) para handoff.
- Slides em PDF.

**Auto-avaliação**: [N/10]

**Justificação**: [Texto livre]

---

## 4. Aspectos a destacar

[Texto livre — o que o grupo considera serem os pontos fortes do trabalho]

---

## 5. Limitações conhecidas

[Texto livre — o que o grupo reconhece como limitações ou trabalho futuro]

Exemplos a discutir:

- Os agentes ID3 jogáveis são **híbridos**: `_forced_move` (immediate-win /
  immediate-block) actua antes da árvore. Os win-rates reportados são do
  conjunto, não da árvore em isolamento.
- A capacidade de generalização real do ID3 PopOut (split por estado) é
  ~44 % test acc, não os ~88 % que apareceriam com split row-level enviesado.
- Tactical accuracy do ID3 puro é baixa (~27 % em `opp_wins_next=1`); a
  performance prática vem da combinação com `_forced_move`.
- Não foi implementado *transposition table* via Zobrist hashing
  (mencionado como trabalho futuro).

---

## 6. Total proposto

| Componente | Peso | Pontuação | Subtotal |
|-----------|------|-----------|----------|
| Adversarial | 30% | [ /10] | [ /3.0] |
| Decision Trees | 30% | [ /10] | [ /3.0] |
| Qualidade técnica | 30% | [ /10] | [ /3.0] |
| Soft skills | 10% | [ /10] | [ /1.0] |
| **Total** | **100%** | — | **[ /10]** |

(Equivalente em escala 0–20: **[ /20]**.)
