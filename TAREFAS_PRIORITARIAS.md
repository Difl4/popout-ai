# 🎯 TODO - TAREFAS PRIORITÁRIAS

**status do Projeto:** 80% Completo | **104 testes passando** | **Pronto para melhorias**

---

## 🔴 CRÍTICO (Fazer ESTA SEMANA)

### 1. Testes para GUI - `test_gui.py` ⏰ 2-3 horas
**Por quê:** GUI é a maior interface visual, sem testes não há guarantee de funcionamento

**O fazer:**
```bash
# Criar arquivo
touch tests/test_gui.py

# Testar funções:
- _player_color()
- _player_glow()
- _draw_vertical_gradient() - mock pygame
- _encode_move()
- _x_to_column()
- launch_gui() - mockar pygame.init()
```

**Referência:** `tests/test_mcts.py` e `tests/test_cli.py` como exemplo

---

### 2. Preencher PopOut_Solution.ipynb ⏰ 2-3 horas
**Por quê:** Notebook vazio perde toda documentação executável e resultados

**O fazer:**
```
1. Célula 1: Intro e setup
   - Título: "PopOut AI - Solution"
   - Descrição do projeto
   - Imports

2. Células 2-3: Game Engine Demo
   - Criar board
   - Aplicar moves
   - Mostrar string representation

3. Células 4-5: MCTS Demo
   - Criar StandardUCT
   - Jogar algumas iterações
   - Print stats

4. Célula 6: Rules Demo
   - Mostrar vitória, empate, repetição
   - Exemplos visuais

5. Célula 7: ID3 Training
   - Gerar dataset com bulk_generate
   - Treinar ID3Classifier
   - Score accuracy

6. Célula 8: Comparação MCTS vs ID3
   - Qual é mais rápido?
   - Qual é mais preciso?
   - Gráficos de accuracy

7. Célula 9: Conclusions
   - Resumo dos resultados
```

---

### 3. Adicionar Error Handling em GUI ⏰ 30 minutos
**Ficheiro:** `src/interfaces/gui.py`

**O fazer:**
```python
def launch_gui() -> None:
    """Inicia GUI pygame com tratamento de erros."""
    try:
        pygame.init()
        # ... resto do código
    except pygame.error as e:
        print(f"❌ Erro pygame: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        sys.exit(1)
    finally:
        pygame.quit()
```

---

### 4. Completar Type Hints ⏰ 1 hora
**Área:** Todos os `*.py` que têm `# type: ignore` ou sem tipos

**Ficheiros prioritários:**
- [ ] `src/interfaces/gui.py` - adicionar tipos em funções private
- [ ] `src/algorithms/id3/learner.py` - melhor cobertura
- [ ] `tests/*.py` - adicionar tipos em fixtures

**Comando para validar:**
```bash
python -m mypy src/ --ignore-missing-imports
```

---

## 🟠 ALTA (Fazer em 2-3 dias)

### 5. Adicionar Testes de Integração ⏰ 1-2 horas
**Ficheiro:** `tests/test_integration.py`

**Testes:**
```python
def test_full_cli_game_flow():
    """Simula um jogo CLI completo"""
    
def test_gui_full_game_pvp():
    """Simula jogo PvP na GUI"""
    
def test_gui_full_game_vs_ai():
    """Simula jogo vs IA na GUI"""
    
def test_mcts_to_id3_pipeline():
    """Gera dataset e treina ID3"""
```

---

### 6. Melhorar GUI - Features Importantes ⏰ 3-4 horas

**Adicionar:**

#### a. Menu de Pausa
```python
- ESC key para pausar
- Mostrar menu: Resume / New Game / Difficulty / Quit
```

#### b. Seletor de Dificuldade
```python
- Easy: 100 MCTS iterations
- Medium: 500 iterations
- Hard: 1000+ iterations
```

#### c. Save/Load Game
```python
def save_game_state(board, game_mode, filename="save.pkl"):
    """Salva estado do jogo"""
    
def load_game_state(filename="save.pkl"):
    """Carrega último jogo"""
```

#### d. Estatísticas
```python
- Tempo de jogo
- Número de movimentos
- Taxa de vitória vs IA
```

---

### 7. Input Validation Robusto ⏰ 30-45 minutos

**Ficheiros:**
- [ ] `src/algorithms/id3/learner.py` - validar NaN, empty DataFrames
- [ ] `src/engine/bitboard.py` - validar moves fora de range
- [ ] `src/scripts/bulk_generate.py` - validar parâmetros

**Exemplo:**
```python
def fit(self, df: pd.DataFrame, target: str) -> None:
    if df.empty:
        raise ValueError("DataFrame vazio")
    if df.isnull().any().any():
        raise ValueError("DataFrame contém NaN")
    if target not in df.columns:
        raise ValueError(f"Target '{target}' não encontrado")
    # ...
```

---

### 8. Adicionar Métodos a PopOutBoard ⏰ 20 minutos

**Ficheiro:** `src/engine/bitboard.py`

**Adicionar:**
```python
def __eq__(self, other: "PopOutBoard") -> bool:
    """Comparar dois boards"""
    if not isinstance(other, PopOutBoard):
        return False
    return (self.mask_p1 == other.mask_p1 and 
            self.mask_p2 == other.mask_p2 and
            self.current_player == other.current_player)

def __hash__(self) -> int:
    """Para usar em sets/dicts"""
    return hash((self.mask_p1, self.mask_p2, self.current_player))
```

---

## 🔵 MÉDIA (Fazer no próximo sprint)

### 9. Feature Importance em ID3 ⏰ 30 minutos
```python
def get_feature_importance(self) -> dict[str, float]:
    """Retorna importância de cada feature"""
    # Contar quantas vezes cada feature foi usada na árvore
```

---

### 10. Tree Visualization ⏰ 45 minutos
```python
def print_tree(self, node: DecisionNode = None, prefix: str = ""):
    """Printa a árvore em ASCII art"""
    # if node is None, node = self.root
    # Recursivamente print children com indentation
```

---

### 11. Performance Benchmarking ⏰ 1 hora
**Ficheiro:** `tests/test_performance.py`

```python
def test_mcts_speed():
    """Benchmark MCTS iterations/second"""
    
def test_id3_training_speed():
    """Benchmark tempo de treino ID3"""
    
def test_bitboard_operations():
    """Benchmark velocidade de operações bitboard"""
```

---

### 12. Code Coverage Report ⏰ 30 minutos
```bash
# Setup
pip install pytest-cov

# Gerar report
pytest --cov=src --cov-report=html

# Abrir
open htmlcov/index.html
```

---

### 13. Logging Melhorado ⏰ 30 minutos
**Ficheiro:** `src/scripts/bulk_generate.py`

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_dataset(...):
    logger.info(f"Gerando dataset com {n_samples} amostras...")
    for i, sample in enumerate(samples):
        if i % 50 == 0:
            logger.info(f"Progresso: {i}/{n_samples}")
```

---

## 🟢 BAIXA (Adicional/Polish)

### 14. ExperimentalUCT Documentação ⏰ 15 minutos
- Adicionar docstring explicando política experimental
- Adicionar comentários em `best_child()`

---

### 15. Sound Effects (Opcional) ⏰ 1-2 horas
```python
# Em _draw_disc ou ao aplicar move
pygame.mixer.Sound("assets/drop.wav").play()
pygame.mixer.Sound("assets/pop.wav").play()
pygame.mixer.Sound("assets/win.wav").play()
```

---

### 16. GUI Screenshots para README ⏰ 30 minutos
- Tirar 3-4 screenshots do jogo
- Adicionar a README.md com instruções

---

### 17. Contribution Guide ⏰ 30 minutos
**Ficheiro:** `CONTRIBUTING.md`
```markdown
# Como Contribuir

1. Fork repo
2. Create branch (feature/sua-feature)
3. Commit changes
4. Push para branch
5. Open Pull Request

## Setup dev:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest pytest-cov
```

## Antes de PR:
```bash
pytest -q
mypy src/
```

---

## 📝 CHECKLIST DE IMPLEMENTAÇÃO

Copiar e colar na plataforma de gestão de projetos (GitHub Issues, Notion, etc):

```markdown
## Fase 1: Robustez (CRÍTICO)
- [ ] Criar test_gui.py com 10+ testes
- [ ] Preencher PopOut_Solution.ipynb
- [ ] Adicionar try/except em launch_gui()
- [ ] Completar type hints em gui.py

## Fase 2: Features (ALTA)
- [ ] test_integration.py
- [ ] Pausa e menu em GUI
- [ ] Seletor de dificuldade
- [ ] Input validation melhorado
- [ ] __eq__ e __hash__ em PopOutBoard

## Fase 3: Polish (MÉDIA)
- [ ] Feature importance ID3
- [ ] Tree visualization
- [ ] Performance tests
- [ ] Code coverage reports
- [ ] Logging melhorado

## Fase 4: Extras (BAIXA)
- [ ] ExperimentalUCT docs
- [ ] Sound effects
- [ ] Screenshots README
- [ ] CONTRIBUTING.md
```

---

## ⏱️ ESTIMATIVA TOTAL DE TEMPO

| Fase | Tarefas | Tempo | Cumulative |
|------|---------|-------|-----------|
| **Crítico** | 4 | 6-7h | 6-7h |
| **Alta** | 5 | 5-8h | 11-15h |
| **Média** | 5 | 4-5h | 15-20h |
| **Baixa** | 4+ | 3-4h | 18-24h |

**Total para versão "polida" completa:** ~20-24 horas de desenvolvimento

---

## 🚀 RECOMENDAÇÃO DE PRIORIDADE

**Se tem 1 dia:** 
→ Faire Fase 1 (Robustez) - garante que tudo funciona

**Se tem 3 dias:**
→ Fazer Fase 1 + Parte de Fase 2 (Features essenciais)

**Se tem 1 semana:**
→ Fazer Fases 1, 2 e 3 completos (versão "production-ready")

**Se tem 2 semanas:**
→ Fazer tudo (versão "polida" com extras)

---

## 📊 MÉTRICAS DE SUCESSO

Após completar estas tarefas:
- ✅ 100% testado (incluindo GUI)
- ✅ 100% type hints
- ✅ ~95% documentação
- ✅ Pronto para distribuição
- ✅ Código profissional qualidade

