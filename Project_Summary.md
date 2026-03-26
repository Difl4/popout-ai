# Project Summary:

## Directory: Root
### File: `PopOut_Solution.ipynb`
  - Empty notebook.
## Directory: src
### File: `__init__.py`
  - Contains helper logic/imports.
## Directory: src/algorithms
### File: `__init__.py`
  - Contains helper logic/imports.
## Directory: src/algorithms/id3
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `discretizer.py`
  - Function: `fit_quantile_bins()`  | Aprende limites de discretização por quantis para colunas numéricas.
  - Function: `apply_bins()`  | Aplica limites de discretização e devolve DataFrame categórico.

### File: `learner.py`
  - Class: `DecisionNode`
    - Method: `is_leaf()`  | Indica se o nó é folha.
  - Class: `ID3Classifier`
    - Method: `__init__()`  | Inicializa classificador sem árvore treinada.
    - Method: `entropy()`  | Calcula entropia de Shannon de uma variável categórica.
    - Method: `information_gain()`  | Calcula ganho de informação de um atributo.
    - Method: `majority_class()`  | Obtém classe majoritária.
    - Method: `build_tree()`  | Constrói árvore ID3 recursivamente.
    - Method: `fit()`  | Treina o classificador ID3.
    - Method: `predict_one()`  | Prediz classe para uma observação.
    - Method: `predict()`  | Prediz classes para múltiplas observações.
    - Method: `score()`  | Calcula accuracy em dataset rotulado.
## Directory: src/algorithms/mcts
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `base.py`
  - Class: `MCTSNode`
    - Method: `__post_init__()`  | Inicializa lista de jogadas não testadas para o estado atual.
    - Method: `is_terminal()`  | Indica se o nó representa um estado terminal conhecido.
    - Method: `q()`  | Valor médio do nó.
  - Class: `BaseMCTS`
    - Method: `__init__()`  | Inicializa hiperparâmetros do MCTS.
    - Method: `best_child()`  | Seleciona filho pelo critério UCT.
    - Method: `select()`  | Fase de seleção: desce pela árvore até nó expansível/terminal.
    - Method: `expand()`  | Fase de expansão: cria um novo filho a partir de jogada não explorada.
    - Method: `simulate()`  | Fase de simulação (rollout) a partir do nó.
    - Method: `backpropagate()`  | Fase de retropropagação: atualiza estatísticas até à raiz.
    - Method: `run()`  | Executa iterações MCTS e devolve a melhor jogada por visitas.

### File: `uct_experimental.py`
  - Class: `ExperimentalUCT`
    - Method: `best_child()`  | Escolhe melhor filho com política experimental.

### File: `uct_standard.py`
  - Class: `StandardUCT`
    - Method: `__init__()`  | Inicializa o UCT padrão.
## Directory: src/engine
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `bitboard.py`
  - Class: `PopOutBoard`
    - Method: `clone()`  | Cria uma cópia profunda do estado atual.
    - Method: `legal_drop_moves()`  | Lista colunas válidas para jogada de drop.
    - Method: `legal_pop_moves()`  | Lista colunas válidas para jogada pop do jogador.
    - Method: `legal_moves()`  | Lista todas as jogadas legais (drop e pop).
    - Method: `apply_drop()`  | Executa uma jogada drop numa coluna.
    - Method: `apply_pop()`  | Executa uma jogada pop removendo a peça da base da coluna.
    - Method: `apply_move()`  | Aplica uma jogada genérica ('drop' ou 'pop').
    - Method: `is_full()`  | Verifica se o tabuleiro está cheio (sem drops possíveis).
    - Method: `to_feature_dict()`  | Converte o estado em dicionário de features para ID3.
    - Method: `__str__()`  | Representação textual do tabuleiro.

### File: `rules.py`
  - Function: `check_winner_for_player()`  | Verifica se um jogador tem 4-em-linha no estado atual.
  - Function: `evaluate_after_move()`  | Resolve resultado após uma jogada, considerando regra de conflito.
  - Function: `board_signature()`  | Gera assinatura imutável para deteção de repetição de estado.
  - Function: `is_threefold_repetition()`  | Verifica empate por repetição tripla de estado.
  - Function: `is_draw()`  | Verifica condições de empate.
## Directory: src/interfaces
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `cli.py`
  - Function: `parse_move()`  | Converte texto do utilizador para formato de jogada.
  - Function: `run_cli_game()`  | Corre um jogo humano (P1) vs MCTS (P2) em terminal.

### File: `gui.py`
  - Function: `launch_gui()`  | Lança uma GUI placeholder para demonstração.
## Directory: src/scripts
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `bulk_generate.py`
  - Function: `make_agent()`  | Cria agente MCTS conforme variante pedida.
  - Function: `randomize_state()`  | Gera estado plausível aplicando jogadas aleatórias.
  - Function: `generate_dataset()`  | Gera dataset de treino para ID3 baseado em decisões de MCTS.
  - Function: `main()`  | Entry point de linha de comandos para geração em lote.
## Directory: tests
### File: `conftest.py`
  - Contains helper logic/imports.

### File: `test_bitboard.py`
  - Function: `_fill_column()`  | Preenche *count* células de uma coluna com peças do jogador.
  - Class: `TestDrop`
    - Method: `test_drop_empty_column()` 
    - Method: `test_drop_stacks_pieces()` 
    - Method: `test_drop_full_column_returns_false()` 
    - Method: `test_drop_invalid_column_negative()` 
    - Method: `test_drop_invalid_column_too_large()` 
  - Class: `TestPop`
    - Method: `test_pop_own_piece()` 
    - Method: `test_pop_opponent_piece_returns_false()` 
    - Method: `test_pop_empty_column_returns_false()` 
    - Method: `test_pop_invalid_column()` 
    - Method: `test_gravity_after_pop()`  | Após pop, todas as peças acima descem uma posição.
  - Class: `TestLegalMoves`
    - Method: `test_legal_drop_moves_initial()` 
    - Method: `test_legal_drop_moves_full_column_excluded()` 
    - Method: `test_legal_pop_moves_empty_board()` 
    - Method: `test_legal_pop_moves_with_own_pieces()` 
    - Method: `test_legal_moves_combines_drop_and_pop()` 
    - Method: `test_legal_moves_uses_current_player_default()` 
  - Class: `TestApplyMove`
    - Method: `test_apply_move_drop_switches_player()` 
    - Method: `test_apply_move_pop_switches_player()` 
    - Method: `test_apply_move_no_switch()` 
    - Method: `test_apply_move_invalid_returns_false()` 
  - Class: `TestClone`
    - Method: `test_clone_is_independent()` 
    - Method: `test_clone_preserves_state()` 
  - Class: `TestIsFull`
    - Method: `test_empty_board_not_full()` 
    - Method: `test_full_board()` 
  - Class: `TestFeatureDict`
    - Method: `test_feature_dict_keys()` 
    - Method: `test_feature_dict_values_empty()` 
  - Class: `TestStr`
    - Method: `test_str_contains_column_numbers()` 
    - Method: `test_str_shows_pieces()` 

### File: `test_bulk_generate.py`
  - Class: `TestMakeAgent`
    - Method: `test_standard_variant()` 
    - Method: `test_experimental_variant()` 
    - Method: `test_unknown_variant_raises()` 
  - Class: `TestRandomizeState`
    - Method: `test_returns_popout_board()` 
    - Method: `test_zero_steps_returns_empty()` 
    - Method: `test_many_steps_does_not_crash()` 
    - Method: `test_state_has_pieces_after_steps()` 
  - Class: `TestGenerateDataset`
    - Method: `test_returns_dataframe()` 
    - Method: `test_has_best_move_column()` 
    - Method: `test_has_current_player_column()` 
    - Method: `test_has_cell_columns()` 
    - Method: `test_best_move_format()`  | best_move deve ter formato 'tipo_coluna', ex: 'drop_3'.
    - Method: `test_row_count()`  | Número de linhas deve ser <= n_samples (pode ser menor se estados sem jogadas).
    - Method: `test_experimental_variant_works()` 

### File: `test_cli.py`
  - Class: `TestParseMove`
    - Method: `test_drop_move()` 
    - Method: `test_drop_move_zero()` 
    - Method: `test_drop_move_six()` 
    - Method: `test_pop_move()` 
    - Method: `test_pop_move_five()` 
    - Method: `test_uppercase_accepted()`  | Input é convertido para lowercase internamente.
    - Method: `test_whitespace_stripped()` 
    - Method: `test_invalid_type_raises()` 
    - Method: `test_too_short_raises()` 
    - Method: `test_empty_string_raises()` 
    - Method: `test_invalid_column_high_raises()` 
    - Method: `test_invalid_column_negative_raises()` 
    - Method: `test_non_numeric_column_raises()` 

### File: `test_discretizer.py`
  - Class: `TestFitQuantileBins`
    - Method: `test_basic_bins()` 
    - Method: `test_bins_are_sorted()` 
    - Method: `test_multiple_columns()` 
    - Method: `test_constant_column_no_bins()`  | Coluna constante → quantis iguais → set vazio ou mínimo.
  - Class: `TestApplyBins`
    - Method: `test_applies_labels()` 
    - Method: `test_does_not_modify_original()` 
    - Method: `test_single_bin_edge()`  | Um único limite → 2 classes.
    - Method: `test_no_bins_for_column()`  | Coluna sem bins definidos não é alterada.
  - Class: `TestRoundTrip`
    - Method: `test_discretize_then_id3()`  | Fluxo completo: dados numéricos → discretizar → treinar ID3 → prever.
    - Method: `test_predict_after_discretize()`  | Prever uma nova observação após discretização.

### File: `test_id3.py`
  - Function: `_weather_dataset()`  | Dataset clássico 'play tennis' para testes categóricos.
  - Function: `_simple_pure_dataset()`  | Dataset onde todas as linhas têm a mesma classe.
  - Function: `_simple_binary_dataset()`  | Dataset binário equilibrado.
  - Class: `TestEntropy`
    - Method: `test_pure_set_entropy_zero()` 
    - Method: `test_uniform_binary_entropy_one()` 
    - Method: `test_single_element()` 
    - Method: `test_three_classes_uniform()` 
  - Class: `TestInformationGain`
    - Method: `test_gain_positive_for_useful_feature()` 
    - Method: `test_gain_zero_for_useless_feature()`  | Feature com valor único não separa nada → ganho = 0.
  - Class: `TestMajorityClass`
    - Method: `test_majority()` 
  - Class: `TestFitPredict`
    - Method: `test_fit_creates_tree()` 
    - Method: `test_predict_training_data_high_accuracy()`  | ID3 deve ter accuracy perfeita ou quase perfeita no treino.
    - Method: `test_predict_pure_dataset()` 
    - Method: `test_predict_one()` 
    - Method: `test_predict_unseen_category_falls_back()`  | Valor de feature não visto no treino → usa majority_label.
    - Method: `test_predict_before_fit_raises()` 
  - Class: `TestDecisionNode`
    - Method: `test_leaf_node()` 
    - Method: `test_internal_node()` 
    - Method: `test_children_dict()` 
  - Class: `TestScore`
    - Method: `test_score_returns_float()` 

### File: `test_mcts.py`
  - Class: `TestMCTSNode`
    - Method: `test_untried_moves_populated_on_init()` 
    - Method: `test_is_terminal_false_initially()` 
    - Method: `test_is_terminal_true_when_winner_set()` 
    - Method: `test_q_zero_when_no_visits()` 
    - Method: `test_q_after_updates()` 
  - Class: `TestBaseMCTSRun`
    - Method: `test_returns_legal_move_initial_state()` 
    - Method: `test_returns_legal_move_mid_game()` 
    - Method: `test_deterministic_with_same_seed()` 
    - Method: `test_single_legal_move()`  | Quando só há uma jogada legal, deve devolvê-la.
    - Method: `test_raises_on_no_legal_moves()`  | Estado sem jogadas legais deve lançar ValueError.
  - Class: `TestMCTSSteps`
    - Method: `test_expand_creates_child()` 
    - Method: `test_expand_terminal_returns_self()` 
    - Method: `test_simulate_returns_valid_reward()` 
    - Method: `test_backpropagate_updates_visits()` 
    - Method: `test_select_returns_node()` 
  - Class: `TestStandardUCT`
    - Method: `test_returns_legal_move()` 
    - Method: `test_inherits_base()` 
  - Class: `TestExperimentalUCT`
    - Method: `test_returns_legal_move()` 
    - Method: `test_inherits_base()` 
    - Method: `test_best_child_prefers_unvisited()`  | ExperimentalUCT deve escolher filhos não visitados primeiro.

### File: `test_rules.py`
  - Function: `_set_cell()`  | Define diretamente uma célula (atalho para testes).
  - Function: `_make_board_with()`  | Cria tabuleiro com células específicas preenchidas: [(row, col, player), ...].
  - Class: `TestHorizontalWin`
    - Method: `test_horizontal_bottom_row()` 
    - Method: `test_horizontal_middle_row()` 
    - Method: `test_horizontal_right_edge()` 
    - Method: `test_three_in_a_row_not_win()` 
  - Class: `TestVerticalWin`
    - Method: `test_vertical_bottom()` 
    - Method: `test_vertical_top()` 
    - Method: `test_three_vertical_not_win()` 
  - Class: `TestDiagonalWin`
    - Method: `test_diagonal_down_right()`  | Diagonal ↘ (r+k, c+k).
    - Method: `test_diagonal_down_left()`  | Diagonal ↙ (r+k, c-k).
    - Method: `test_diagonal_at_corner()`  | Diagonal começando no canto superior esquerdo.
    - Method: `test_three_diagonal_not_win()` 
  - Class: `TestConflictRule`
    - Method: `test_both_win_mover_wins()`  | Se ambos têm 4-em-linha, quem jogou (mover) vence.
    - Method: `test_only_player1_wins()` 
    - Method: `test_no_winner()` 
  - Class: `TestBoardSignature`
    - Method: `test_signature_is_hashable()` 
    - Method: `test_different_states_different_signatures()` 
    - Method: `test_same_state_same_signature()` 
  - Class: `TestThreefoldRepetition`
    - Method: `test_no_repetition()` 
    - Method: `test_exactly_three_repetitions()` 
    - Method: `test_two_repetitions_not_enough()` 
  - Class: `TestIsDraw`
    - Method: `test_draw_full_board()` 
    - Method: `test_draw_threefold()` 
    - Method: `test_no_draw_normal_state()` 

---
## 📊 Internal Research Logic Frequency
Functions you wrote, sorted by how often they are used in this project:
* `PopOutBoard`: 60 calls
* `apply_drop`: 30 calls
* `check_winner_for_player`: 16 calls
* `parse_move`: 15 calls
* `MCTSNode`: 13 calls
* `apply_move`: 12 calls
* `_make_board_with`: 12 calls
* `ID3Classifier`: 11 calls
* `BaseMCTS`: 11 calls
* `legal_moves`: 10 calls
* `run`: 10 calls
* `evaluate_after_move`: 9 calls
* `board_signature`: 8 calls
* `generate_dataset`: 8 calls
* `fit`: 8 calls
* `expand`: 7 calls
* `apply_pop`: 7 calls
* `apply_bins`: 7 calls
* `entropy`: 6 calls
* `DecisionNode`: 6 calls
* `fit_quantile_bins`: 6 calls
* `_weather_dataset`: 6 calls
* `clone`: 5 calls
* `randomize_state`: 5 calls
* `is_leaf`: 4 calls
* `is_draw`: 4 calls
* `legal_pop_moves`: 4 calls
* `is_threefold_repetition`: 4 calls
* `make_agent`: 4 calls
* `predict_one`: 4 calls
* `information_gain`: 3 calls
* `best_child`: 3 calls
* `q`: 3 calls
* `legal_drop_moves`: 3 calls
* `is_full`: 3 calls
* `StandardUCT`: 3 calls
* `to_feature_dict`: 3 calls
* `score`: 3 calls
* `ExperimentalUCT`: 3 calls
* `_set_cell`: 3 calls
* `majority_class`: 2 calls
* `build_tree`: 2 calls
* `predict`: 2 calls
* `select`: 2 calls
* `simulate`: 2 calls
* `backpropagate`: 2 calls
* `__init__`: 1 calls
* `run_cli_game`: 1 calls
* `launch_gui`: 1 calls
* `main`: 1 calls
* `_fill_column`: 1 calls
* `_simple_pure_dataset`: 1 calls