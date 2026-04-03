# Project Summary:

## Directory: notebooks
### File: `PopOut_Solution.ipynb`
  - Cell 1:
    - Markdown Hint: Setup e Imports...
  - Cell 2:
    - Markdown Hint: 1. Demonstração do Game Engine (Bitboard)...
    - Code Hint (Internal): PopOutBoard, legal_moves
  - Cell 3:
    - Code Hint (Internal): PopOutBoard, apply_move, evaluate_after_move
  - Cell 4:
    - Markdown Hint: 2. MCTS — Monte Carlo Tree Search...
    - Code Hint (Internal): PopOutBoard, StandardUCT, run
  - Cell 5:
    - Markdown Hint: 2.1 Análise do Parâmetro de Exploração C (UCT)...
    - Code Hint (Internal): PopOutBoard, StandardUCT, run
  - Cell 6:
    - Markdown Hint: 2.2 Standard UCT vs Experimental UCT...
    - Code Hint (Internal): ExperimentalUCT, StandardUCT, randomize_state, run
  - Cell 7:
    - Markdown Hint: 2.3 Efeito do Número de Iterações...
    - Code Hint (Internal): PopOutBoard, StandardUCT, run
  - Cell 8:
    - Markdown Hint: 3. Árvore de Decisão ID3 — Dataset Iris (Warm-Up)...
    - Code Hint (Internal): apply_bins, fit_quantile_bins
  - Cell 9:
    - Code Hint (Internal): ID3Classifier, fit, predict, score
  - Cell 10:
    - Code Hint (Internal): get_feature_importance, print_tree
  - Cell 11:
    - Markdown Hint: 4. Árvore de Decisão ID3 — Dataset PopOut...
    - Code Hint (Internal): ID3Classifier, fit, generate_dataset, score
  - Cell 12:
    - Markdown Hint: 5. Análise Comparativa & Computer vs Computer...
    - Code Hint (Internal): PopOutBoard, StandardUCT, predict, run
  - Cell 13:
    - Markdown Hint: 5.2 Computer vs Computer — Torneio StandardUCT vs ExperimentalUCT...
    - Code Hint (Internal): ExperimentalUCT, PopOutBoard, StandardUCT, apply_move, board_signature, evaluate_after_move, is_full, is_threefold_repetition, run
## Directory: src
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `__main__.py`
  - Function: `main()` 

### File: `config.py`
  - Contains helper logic/imports.

### File: `game_state.py`
  - Class: `GameState`
  - Class: `GameSaveManager`
    - Method: `__init__()`  | Inicializa manager de save.
    - Method: `save_path()`  | Retorna caminho completo do arquivo de save.
    - Method: `has_save()`  | Verifica se existe um save guardado.
    - Method: `save_game()`  | Guarda estado de jogo em arquivo.
    - Method: `load_game()`  | Carrega estado de jogo de arquivo.
    - Method: `delete_save()`  | Apaga arquivo de save.
    - Method: `list_saves()`  | Lista todos os arquivos de save disponíveis.
    - Method: `get_save_info()`  | Obtém informações sobre o save (sem carregar completo).
## Directory: src/algorithms
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `factory.py`
  - Function: `get_agent()`  | Instantiate and return an MCTS engine by name.
## Directory: src/algorithms/mcts
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `protocol.py`
  - Class: `MCTSEngine`
    - Method: `run()`  | Return the best move integer (0-13) for the given board state.
## Directory: src/algorithms/mcts/optimized
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `numba_mcts.py`
  - Class: `_NumbaNode`
    - Method: `__post_init__()` 
  - Class: `NumbaMCTS`
    - Method: `__init__()` 
    - Method: `best_child()` 
    - Method: `expand()` 
    - Method: `simulate()` 
  - Class: `FlatNumbaMCTS`
    - Method: `__init__()` 
    - Method: `run()`  | Return the best move integer (0-13) for the given board state.
  - Function: `warmup()`  | Trigger JIT compilation of all @njit functions once.

### File: `numba_search.py`
  - Function: `nb_expand_step()`  | Apply a move, evaluate for a winner, and compute the resulting legal moves.
  - Function: `nb_simulate()`  | Full random rollout in pure int64 arithmetic.
  - Function: `_nb_best_child_id()`  | UCT child selection on flat arrays. Returns child node id.
  - Function: `nb_mcts_run()`  | Complete MCTS loop in Numba — select, expand, simulate, backpropagate.
## Directory: src/algorithms/mcts/standard
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `base.py`
  - Class: `MCTSNode`
    - Method: `__post_init__()` 
    - Method: `is_terminal()` 
    - Method: `q()` 
  - Class: `BaseMCTS`
    - Method: `__init__()` 
    - Method: `best_child()` 
    - Method: `select()` 
    - Method: `expand()` 
    - Method: `simulate()` 
    - Method: `backpropagate()` 
    - Method: `run()` 

### File: `uct_experimental.py`
  - Class: `ExperimentalUCT`
    - Method: `best_child()`  | Escolhe melhor filho com política experimental.

### File: `uct_standard.py`
  - Class: `StandardUCT`
    - Method: `__init__()`  | Inicializa o UCT padrão.
## Directory: src/engine
### File: `__init__.py`
  - Contains helper logic/imports.
## Directory: src/engine/optimized
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `numba_bitboard.py`
  - Function: `nb_legal_moves()`  | Return (moves_array, n) with legal move ints in moves_array[:n].
  - Function: `nb_apply_move()`  | Apply move (0-13) and return (new_mask_p1, new_mask_p2, next_player).
  - Function: `nb_is_full()`  | True when every column's top row bit is occupied.

### File: `numba_rules.py`
  - Function: `nb_has_won()`  | 4-in-a-row check via bitwise shifts. Mirrors rules.has_won.
  - Function: `nb_check_winner_for_player()`  | Mirrors rules.check_winner_for_player.
  - Function: `nb_evaluate_after_move()`  | Mirrors rules.evaluate_after_move — PopOut tiebreak rule included.
  - Function: `nb_is_threefold_repetition()`  | Mirrors rules.is_threefold_repetition.
  - Function: `nb_is_draw()`  | Mirrors rules.is_draw — full board or threefold repetition.
## Directory: src/engine/standard
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `bitboard.py`
  - Class: `PopOutBoard`
    - Method: `clone()` 
    - Method: `merged_mask()` 
    - Method: `legal_moves()`  | Gera jogadas como inteiros:
    - Method: `apply_move()`  | Aplica a jogada (0-13) diretamente.
    - Method: `is_full()` 
    - Method: `to_feature_dict()`  | Converte o estado do tabuleiro em dicionário de features para ML.
    - Method: `__str__()` 
    - Method: `__eq__()`  | Comparar dois boards por estado.
    - Method: `__hash__()`  | Hash do estado do board para usar em sets/dicts.

### File: `rules.py`
  - Function: `has_won()`  | Checks for 4-in-a-row using bitwise shifts (Connect 4 logic).
  - Function: `check_winner_for_player()`  | Verifica se o jogador especificado venceu.
  - Function: `evaluate_after_move()`  | Resolve resultado após uma jogada.
  - Function: `board_signature()`  | Assinatura única imutável para o estado do tabuleiro.
  - Function: `is_threefold_repetition()`  | Verifica empate por repetição tripla de estado.
  - Function: `is_draw()`  | Verifica condições de empate (cheio ou repetição).
## Directory: src/interfaces
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `cli.py`
  - Function: `parse_move()`  | Converte 'd<col>' ou 'p<col>' para inteiro do motor (0-13).
  - Function: `decode_move()`  | Converte inteiro do motor para texto legível.
  - Function: `_print_board()`  | Imprime o tabuleiro com indicador do jogador atual.
  - Function: `_check_winner()`  | Verifica vitória ou empate por repetição.
  - Function: `run_hvh()`  | Jogo Humano vs Humano via linha de comandos.
  - Function: `run_hvc()`  | Jogo Humano vs Computador (MCTS Standard).
  - Function: `run_cvc()`  | Jogo Computador vs Computador entre dois agentes MCTS.
  - Function: `run_cvc_tournament()`  | Torneio automático: StandardUCT vs ExperimentalUCT.
  - Function: `run_cli_game()`  | Menu principal da CLI — seleciona o modo de jogo.

### File: `gui.py`
  - Contains helper logic/imports.
## Directory: src/interfaces/gui
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `assets.py`
  - Function: `create_fonts()`  | Create and return game fonts. Must be called after pygame.init().

### File: `components.py`
  - Class: `PauseMenu`
    - Method: `__init__()` 
    - Method: `toggle_pause()` 
    - Method: `navigate()` 
    - Method: `select_current()` 
    - Method: `handle_event()`  | Process a pygame event and return the selected option name if confirmed.
    - Method: `draw()`  | Draw the pause menu overlay.

### File: `core.py`
  - Function: `_column_from_mouse()`  | Converte coordenada X do rato para coluna 0..6, ou None fora do tabuleiro.
  - Function: `_encode_move()`  | Codifica jogada no formato do motor: drop(0..6) ou pop(7..13).
  - Function: `_make_ai_engine()`  | Instantiate the correct AI engine for the given difficulty level.
  - Function: `launch_gui()`  | Inicia a janela pygame com interface melhorada e executa o ciclo principal.

### File: `renderer.py`
  - Function: `_player_color()`  | Devolve a cor base associada ao jogador.
  - Function: `_player_glow()`  | Devolve a cor de glow (brilho) do jogador.
  - Function: `_draw_vertical_gradient()`  | Desenha um gradiente vertical no fundo com mais qualidade.
  - Function: `_draw_glow_circle()`  | Desenha uma aura de brilho ao redor de um círculo.
  - Function: `_draw_disc()`  | Desenha uma peça com sombra, brilho e animação de chegada.
  - Function: `_draw_board()`  | Desenha tabuleiro com animações, HUD superior/inferior, preview, hover e overlay final.
  - Function: `_draw_pause_menu()`  | Desenha menu de pausa com opcoes navegaveis.

### File: `state.py`
  - Class: `AnimationState`
    - Method: `__init__()`  | Inicializa estado de animação vazio.
    - Method: `add_piece_animation()`  | Inicia animação de entrada para uma peça.
    - Method: `update()`  | Atualiza todas as animações.
  - Class: `Difficulty`
    - Method: `iterations()` 
    - Method: `label()` 
    - Method: `engine_type()` 
  - Class: `CoordinateMapper`
    - Method: `col_from_x()`  | Convert mouse x position to board column.
## Directory: src/ml
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `dataset_generator.py`
  - Function: `make_agent()`  | Cria agente MCTS conforme variante pedida.
  - Function: `randomize_state()`  | Gera estado plausível aplicando jogadas aleatórias.
  - Function: `generate_dataset()`  | Gera dataset de treino para ID3 baseado em decisões de MCTS.
  - Function: `main()`  | Entry point de linha de comandos para geração em lote.

### File: `discretizer.py`
  - Function: `fit_quantile_bins()`  | Aprende limites de discretização por quantis para colunas numéricas.
  - Function: `apply_bins()`  | Aplica limites de discretização e devolve DataFrame categórico.
## Directory: src/ml/id3
### File: `__init__.py`
  - Contains helper logic/imports.

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
    - Method: `_count_feature_usage()`  | Conta recursivamente o uso de cada feature na árvore.
    - Method: `get_feature_importance()`  | Calcula importância de cada feature na árvore treinada.
    - Method: `print_tree()`  | Imprime a árvore de decisão com ASCII art.
    - Method: `tree_to_string()`  | Converte a árvore para string (alternativa a print_tree).
    - Method: `_tree_to_string_recursive()`  | Helper recursivo para tree_to_string.
## Directory: src/utils
### File: `__init__.py`
  - Contains helper logic/imports.

### File: `numba_tools.py`
  - Function: `clear_numba_cache()`  | Delete all .nbi and .nbc files under *root*. Returns count of deleted files.
  - Function: `recompile()`  | Clear Numba cache and trigger warmup recompilation.
## Directory: tests
### File: `conftest.py`
  - Contains helper logic/imports.

### File: `test_bitboard.py`
  - Function: `_get_cell()`  | Lê célula: row 0 é o fundo, row 5 é o topo.
  - Class: `TestDrop`
    - Method: `test_drop_empty_column()` 
    - Method: `test_drop_stacks_pieces()` 
    - Method: `test_is_full_detection()` 
  - Class: `TestPop`
    - Method: `test_pop_removes_bottom()` 
    - Method: `test_gravity_after_pop()` 
  - Class: `TestLegalMoves`
    - Method: `test_legal_moves_initial()` 
    - Method: `test_legal_moves_pop_inclusion()` 

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
    - Method: `test_constant_column_no_bins()`  | Coluna constante → quantis iguais → set pode ter um valor.
  - Class: `TestApplyBins`
    - Method: `test_applies_labels()` 
    - Method: `test_does_not_modify_original()` 
    - Method: `test_single_bin_edge()`  | Um único limite → 2 classes.
    - Method: `test_no_bins_for_column()`  | Coluna sem bins definidos não é alterada.
  - Class: `TestRoundTrip`
    - Method: `test_discretize_then_id3()`  | Fluxo completo: dados numéricos → discretizar → treinar ID3 → prever.
    - Method: `test_predict_after_discretize()`  | Prever uma nova observação após discretização.

### File: `test_game_state.py`
  - Class: `TestGameState`
    - Method: `test_game_state_creation()`  | Deve criar estado de jogo válido.
    - Method: `test_game_state_defaults()`  | Deve usar valores padrão para campos opcionais.
  - Class: `TestGameSaveManager`
    - Method: `temp_save_dir()`  | Cria diretório temporário para saves.
    - Method: `test_manager_creation()`  | Deve criar manager com diretório de save.
    - Method: `test_save_path_property()`  | Deve retornar caminho completo de save.
    - Method: `test_has_save_initially_false()`  | Não deve ter save inicialmente.
    - Method: `test_save_and_load_game()`  | Deve guardar e carregar jogo com sucesso.
    - Method: `test_load_game_none_when_no_save()`  | Deve retornar None se não houver save.
    - Method: `test_delete_save()`  | Deve apagar arquivo de save.
    - Method: `test_list_saves()`  | Deve listar todos os saves disponíveis.
    - Method: `test_get_save_info()`  | Deve retornar informações sobre save sem carregar completo.
    - Method: `test_get_save_info_none_when_no_save()`  | Deve retornar None quando não há save.
    - Method: `test_board_serialization()`  | PopOutBoard deve ser serializável com pickle.

### File: `test_gui.py`
  - Class: `TestPlayerColor`
    - Method: `test_player1_color()`  | Player 1 deve retornar cor vermelha.
    - Method: `test_player2_color()`  | Player 2 deve retornar cor amarela.
    - Method: `test_invalid_player_returns_p2()`  | Jogador inválido retorna cor default (P2).
  - Class: `TestPlayerGlow`
    - Method: `test_player1_glow()`  | Player 1 glow deve ser vermelho brilhante.
    - Method: `test_player2_glow()`  | Player 2 glow deve ser amarelo brilhante.
  - Class: `TestColumnFromMouse`
    - Method: `test_column0_at_left_edge()`  | X=0 deve retornar coluna 0.
    - Method: `test_column3_at_middle()`  | Meio do board deve retornar coluna 3.
    - Method: `test_column6_at_right_edge()`  | Coluna 6 no limite direito.
    - Method: `test_out_of_bounds_left()`  | X negativo retorna None.
    - Method: `test_out_of_bounds_right()`  | X fora do board retorna None.
    - Method: `test_exact_boundary_cases()`  | Testa limites exatos entre colunas.
  - Class: `TestEncodeMove`
    - Method: `test_drop_mode_column0()`  | Mode DROP, coluna 0 = move 0.
    - Method: `test_drop_mode_column6()`  | Mode DROP, coluna 6 = move 6.
    - Method: `test_pop_mode_column0()`  | Mode POP, coluna 0 = move 7 (7 + 0).
    - Method: `test_pop_mode_column6()`  | Mode POP, coluna 6 = move 13 (7 + 6).
    - Method: `test_symmetric_encoding()`  | Testa simetria DROP/POP para todas as colunas.
  - Class: `TestAnimationState`
    - Method: `test_animation_state_creation()`  | AnimationState deve ser criado vazio.
    - Method: `test_add_piece_animation()`  | Adicionar animação de peça.
    - Method: `test_animation_update_progress()`  | Animação deve progredir com tempo.
    - Method: `test_animation_completion()`  | Animação deve ser removida quando 100% completa.
    - Method: `test_multiple_animations()`  | Deve gerenciar múltiplas animações.
  - Class: `TestDrawFunctions`
    - Method: `test_draw_vertical_gradient()`  | Gradiente deve desenhar linhas do topo ao fundo.
    - Method: `test_draw_glow_circle()`  | Glow circle deve desenhar múltiplos círculos.
  - Function: `test_launch_gui_initialization()`  | Teste que launch_gui inicializa pygame corretamente.
  - Class: `TestGuiConstants`
    - Method: `test_cell_size_is_positive()`  | CELL_SIZE deve ser positivo.
    - Method: `test_board_dimensions()`  | Dimensões do board devem ser válidas.
    - Method: `test_window_dimensions()`  | Dimensões da janela devem conter o board.
    - Method: `test_colors_are_tuples()`  | Cores devem ser tuplas RGB ou RGBA.
    - Method: `test_colors_valid_rgb_values()`  | Valores de cor devem estar entre 0-255.
  - Class: `TestGuiIntegration`
    - Method: `test_encode_decode_roundtrip()`  | Encoding DROP então POP deve voltar ao original.
    - Method: `test_mouse_to_column_to_move()`  | Pipeline: mouse X -> coluna -> move.
  - Class: `TestDifficulty`
    - Method: `test_difficulty_easy()`  | Dificuldade Easy = 100 iterações.
    - Method: `test_difficulty_medium()`  | Dificuldade Medium = 500 iterações.
    - Method: `test_difficulty_hard()`  | Dificuldade Hard = 1000 iterações.
    - Method: `test_difficulty_extreme()`  | Dificuldade Extreme = 10000 iterações.
  - Class: `TestPauseMenu`
    - Method: `test_pause_menu_creation()`  | PauseMenu deve inicializar desativado.
    - Method: `test_pause_menu_toggle()`  | Deve alternar estado de pausa.
    - Method: `test_pause_menu_navigate_down()`  | Deve navegar para baixo nas opções.
    - Method: `test_pause_menu_navigate_up()`  | Deve navegar para cima nas opções.
    - Method: `test_pause_menu_navigate_wrap_around()`  | Deve fazer wrap-around ao navegar além dos limites.
    - Method: `test_pause_menu_select_current()`  | Deve retornar opção selecionada correta.
    - Method: `test_pause_menu_reset_on_toggle()`  | Deve resetar opção para "Retomar" ao pausar.
  - Class: `TestDrawPauseMenu`
    - Method: `test_draw_pause_menu_renders()`  | Menu de pausa deve renderizar sem erros.

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
  - Class: `TestFeatureImportance`
    - Method: `test_importance_before_fit_raises()`  | get_feature_importance() sem fit() deve lançar ValueError.
    - Method: `test_importance_pure_dataset_empty()`  | Dataset puro (só um label) → árvore é apenas leaf → vazio.
    - Method: `test_importance_returns_dict()`  | get_feature_importance() retorna dict[str, float].
    - Method: `test_importance_all_positive()`  | Todos os scores de importância são positivos.
    - Method: `test_importance_normalizes_to_one()`  | Soma dos scores de importância ≈ 1.0.
    - Method: `test_importance_sorted_by_count()`  | Features em ordem decrescente de importância.
    - Method: `test_importance_with_single_split_feature()`  | Feature com um único split contribui menos.
    - Method: `test_importance_consistent_across_runs()`  | Importância é determinística (mesmos dados = mesma árvore).
    - Method: `test_importance_binary_dataset()`  | Importância em dataset binário simples.
    - Method: `test_importance_multifeature()`  | Importância com múltiplas features em jogo.
  - Class: `TestTreeVisualization`
    - Method: `test_print_tree_before_fit_raises()`  | print_tree() sem fit should handle gracefully.
    - Method: `test_tree_to_string_before_fit()`  | tree_to_string() sem fit retorna mensagem.
    - Method: `test_print_tree_pure_dataset()`  | print_tree com dataset puro (sem splits).
    - Method: `test_tree_to_string_pure_dataset()`  | tree_to_string com dataset puro.
    - Method: `test_print_tree_with_splits()`  | print_tree com árvore que tem splits.
    - Method: `test_tree_to_string_with_splits()`  | tree_to_string com múltiplos splits.
    - Method: `test_print_tree_contains_ascii_chars()`  | print_tree usa caracteres ASCII para desenhar árvore.
    - Method: `test_tree_to_string_indentation()`  | tree_to_string mantém indentação apropriada.
    - Method: `test_tree_visualization_with_binary_feature()`  | Visualização com dataset binário.
    - Method: `test_tree_visualization_output_format()`  | Formato da visualização é consistente.

### File: `test_integration.py`
  - Class: `TestFullMCTSGameFlow`
    - Method: `test_mcts_plays_complete_game()`  | MCTS deve conseguir jogar um jogo completo sem erros.
    - Method: `test_mcts_deterministic_with_seed()`  | MCTS com mesmo seed deve produzir mesmo resultado.
  - Class: `TestID3TrainingPipeline`
    - Method: `test_dataset_generation_and_training()`  | Deve gerar dataset e treinar ID3 com sucesso.
    - Method: `test_id3_predictions_are_valid()`  | Predições ID3 devem ser labels válidas do dataset.
  - Class: `TestBoardStateEquality`
    - Method: `test_board_equality()`  | Boards com mesmo estado devem ser iguais.
    - Method: `test_board_inequality_after_move()`  | Boards divergentes devem ser diferentes.
    - Method: `test_board_hashable()`  | Board deve ser hashable e usável em sets/dicts.
    - Method: `test_board_hash_consistency()`  | Hash do mesmo board deve ser sempre igual.
  - Class: `TestGameFlowWithDifferentModes`
    - Method: `test_pvp_game_flow()`  | Simula um jogo Player vs Player.
    - Method: `test_pvai_game_flow()`  | Simula Player vs AI com clima de gameplay real.
  - Class: `TestBulkDatasetGeneration`
    - Method: `test_generate_multiple_samples_different_seeds()`  | Datasets com seeds diferentes devem conter amostras diferentes.
    - Method: `test_dataset_structure_consistency()`  | Dataset deve ter estrutura consistente.
  - Class: `TestEndToEndPipeline`
    - Method: `test_complete_pipeline()`  | Executa pipeline completo com sucesso.
    - Method: `test_pipeline_with_different_mcts_variants()`  | Validar que pipeline funciona com diferentes variantes de MCTS.

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

### File: `test_numba_mcts.py`
  - Function: `_warm()`  | Pre-compile all JIT functions once for the entire test session.
  - Class: `TestMCTSEngineProtocol`
    - Method: `test_base_mcts_satisfies_protocol()` 
    - Method: `test_numba_mcts_satisfies_protocol()` 
    - Method: `test_flat_numba_mcts_satisfies_protocol()` 
    - Method: `test_object_without_run_does_not_satisfy()` 
  - Class: `TestNumbaMCTS`
    - Method: `test_returns_legal_move_initial_state()` 
    - Method: `test_returns_legal_move_mid_game()` 
    - Method: `test_inherits_base_mcts()` 
    - Method: `test_simulate_returns_valid_reward()` 
    - Method: `test_expand_builds_valid_child()` 
    - Method: `test_expand_child_has_valid_untried_moves()` 
    - Method: `test_best_child_returns_child_of_node()` 
    - Method: `test_is_faster_than_base_mcts()`  | NumbaMCTS must be measurably faster for 5k iterations.
  - Class: `TestFlatNumbaMCTS`
    - Method: `test_returns_legal_move_initial_state()` 
    - Method: `test_returns_legal_move_mid_game()` 
    - Method: `test_return_type_is_int()` 
    - Method: `test_reusable_across_multiple_calls()`  | The pre-allocated arrays must be safe to reuse without state leak.
    - Method: `test_hits_100k_iterations_under_5_seconds()`  | FlatNumbaMCTS must complete 100k iterations in < 5 s.
    - Method: `test_at_least_5x_faster_than_base_mcts()`  | FlatNumbaMCTS must be at least 5× faster than BaseMCTS.
    - Method: `test_agrees_with_numba_mcts_on_move_legality()`  | Both optimised engines must return a legal move from the same position.
  - Class: `TestWarmup`
    - Method: `test_warmup_runs_without_error()` 
    - Method: `test_cached_warmup_is_fast()`  | Second warmup() call must finish well under 2 seconds (cache hit).
    - Method: `test_engines_work_immediately_after_warmup()` 

### File: `test_numba_rules.py`
  - Function: `_warm_kernels()`  | Trigger JIT compilation once for the entire test session.
  - Function: `_h_mask()`  | Build a bitmask from (col, row) pairs using the 7-bits-per-col layout.
  - Function: `_board_ints()` 
  - Class: `TestNbHasWon`
    - Method: `test_empty_board_no_win()` 
    - Method: `test_horizontal_4_in_a_row()` 
    - Method: `test_horizontal_3_not_win()` 
    - Method: `test_vertical_4_in_a_row()` 
    - Method: `test_vertical_3_not_win()` 
    - Method: `test_diagonal_down_right()` 
    - Method: `test_diagonal_down_left()` 
    - Method: `test_agrees_with_python_reference()`  | nb_has_won must match rules.has_won for arbitrary masks.
  - Class: `TestNbLegalMoves`
    - Method: `test_empty_board_only_7_drops()` 
    - Method: `test_no_pops_for_player_without_pieces()` 
    - Method: `test_pop_available_when_player_has_bottom_piece()` 
    - Method: `test_full_column_not_in_drops()` 
    - Method: `test_matches_python_legal_moves()` 
  - Class: `TestNbApplyMove`
    - Method: `test_drop_sets_bit_at_bottom()` 
    - Method: `test_drop_stacks_second_piece_above()` 
    - Method: `test_pop_clears_bottom_bit()` 
    - Method: `test_player_alternates()` 
    - Method: `test_matches_python_apply_move()`  | Kernel output must equal Python reference for every step.
  - Class: `TestNbEvaluateAfterMove`
    - Method: `test_no_winner_empty_board()` 
    - Method: `test_player1_horizontal_win()` 
    - Method: `test_player2_vertical_win()` 
    - Method: `test_both_win_mover_takes_precedence()` 
    - Method: `test_agrees_with_python_reference()` 
  - Class: `TestNbIsFull`
    - Method: `test_empty_board_not_full()` 
    - Method: `test_agrees_with_python_is_full()` 

### File: `test_numba_search.py`
  - Function: `_warm_kernels()`  | Trigger JIT compilation once for the entire test session.
  - Class: `TestNbSimulate`
    - Method: `test_returns_valid_reward_on_fresh_board()` 
    - Method: `test_terminal_win_returns_one()` 
    - Method: `test_terminal_loss_returns_zero()` 
    - Method: `test_reward_in_range_across_many_rollouts()` 
  - Class: `TestNbExpandStep`
    - Method: `test_returns_correct_board_state()` 
    - Method: `test_legal_moves_match_python()` 

### File: `test_performance.py`
  - Class: `TestBitboardPerformance`
    - Method: `test_apply_move_completes_quickly()`  | Test apply_move() completes in reasonable time.
    - Method: `test_legal_moves_speed()`  | Test legal_moves() computation speed.
    - Method: `test_board_state_operations_batch()`  | Test batch board operations complete quickly.
    - Method: `test_legal_moves_iterations_per_second()`  | Measure legal_moves() iterations per second.
  - Class: `TestID3Performance`
    - Method: `_create_dataset()`  | Create synthetic dataset for benchmarking.
    - Method: `test_id3_fit_speed_small()`  | Test ID3 training completes quickly on small dataset.
    - Method: `test_id3_fit_speed_medium()`  | Test ID3 training on medium dataset.
    - Method: `test_id3_fit_speed_large()`  | Test ID3 training on larger dataset.
    - Method: `test_id3_predict_speed()`  | Test ID3 prediction speed.
    - Method: `test_id3_feature_importance_speed()`  | Test feature importance calculation speed.
    - Method: `test_id3_tree_visualization_speed()`  | Test tree visualization speed.
    - Method: `test_id3_scaling_with_dataset_size()`  | Test ID3 scaling as dataset grows.
  - Class: `TestMCTSPerformance`
    - Method: `test_standard_uct_completes_quickly()`  | Test StandardUCT completes in reasonable time.
    - Method: `test_experimental_uct_completes_quickly()`  | Test ExperimentalUCT completes in reasonable time.
    - Method: `test_mcts_scaling_with_iterations()`  | Test MCTS scales linearly with iterations.
    - Method: `test_mcts_determinism()`  | Test MCTS output with same seed is consistent.
    - Method: `test_mcts_different_seeds_different_moves()`  | Test different seeds produce different results.
  - Class: `TestIntegrationPerformance`
    - Method: `test_full_game_simulation_speed()`  | Benchmark full game simulation.
    - Method: `test_id3_training_with_small_dataset()`  | Test ID3 training runs quickly with synthetic data.
    - Method: `test_multiple_bitboard_operations()`  | Test combining multiple bitboard operations.

### File: `test_rules.py`
  - Function: `_set_cell()`  | Define diretamente uma célula no bitboard.
  - Function: `_make_board_with()` 
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

### File: `test_validation.py`
  - Class: `TestID3Validation`
    - Method: `test_fit_none_dataframe()`  | fit() rejeita None como DataFrame.
    - Method: `test_fit_wrong_type_dataframe()`  | fit() rejeita tipos não-DataFrame.
    - Method: `test_fit_wrong_type_target()`  | fit() rejeita target não-string.
    - Method: `test_fit_empty_dataframe()`  | fit() rejeita DataFrame vazio.
    - Method: `test_fit_missing_target_column()`  | fit() rejeita DataFrames sem coluna target.
    - Method: `test_fit_with_nan_values()`  | fit() rejeita DataFrames com NaN.
    - Method: `test_fit_valid_dataframe()`  | fit() aceita DataFrame válido.
    - Method: `test_fit_multiple_columns()`  | fit() funciona com múltiplas features.
  - Class: `TestBitboardValidation`
    - Method: `test_apply_move_wrong_type()`  | apply_move() rejeita tipos não-inteiros.
    - Method: `test_apply_move_float()`  | apply_move() rejeita floats.
    - Method: `test_apply_move_boolean()`  | apply_move() rejeita booleans (mesmo sendo int em Python).
    - Method: `test_apply_move_negative()`  | apply_move() rejeita moves negativos.
    - Method: `test_apply_move_too_large()`  | apply_move() rejeita moves > 13.
    - Method: `test_apply_move_valid_drops()`  | apply_move() aceita drops válidos (0-6).
    - Method: `test_apply_move_valid_pops()`  | apply_move() aceita pops válidos (7-13).
    - Method: `test_apply_move_boundary_0()`  | apply_move() aceita 0 (limite inferior).
    - Method: `test_apply_move_boundary_13()`  | apply_move() aceita 13 (limite superior).
  - Class: `TestBulkGenerateValidation`
    - Method: `test_make_agent_wrong_variant_type()`  | make_agent() rejeita variant não-string.
    - Method: `test_make_agent_wrong_seed_type()`  | make_agent() rejeita seed não-int/None.
    - Method: `test_make_agent_invalid_variant()`  | make_agent() rejeita variante desconhecida.
    - Method: `test_make_agent_valid_standard()`  | make_agent() cria StandardUCT corretamente.
    - Method: `test_make_agent_valid_experimental()`  | make_agent() cria ExperimentalUCT corretamente.
    - Method: `test_make_agent_none_seed()`  | make_agent() aceita seed=None.
    - Method: `test_randomize_state_wrong_steps_type()`  | randomize_state() rejeita steps não-inteiro.
    - Method: `test_randomize_state_wrong_rng_type()`  | randomize_state() rejeita rng não-Random.
    - Method: `test_randomize_state_negative_steps()`  | randomize_state() rejeita steps negativos.
    - Method: `test_randomize_state_zero_steps()`  | randomize_state() aceita steps=0.
    - Method: `test_randomize_state_valid()`  | randomize_state() cria board válido.
    - Method: `test_generate_dataset_wrong_variant_type()`  | generate_dataset() rejeita variant não-string.
    - Method: `test_generate_dataset_wrong_n_samples_type()`  | generate_dataset() rejeita n_samples não-int.
    - Method: `test_generate_dataset_wrong_iterations_type()`  | generate_dataset() rejeita iterations não-int.
    - Method: `test_generate_dataset_wrong_seed_type()`  | generate_dataset() rejeita seed não-int.
    - Method: `test_generate_dataset_negative_n_samples()`  | generate_dataset() rejeita n_samples <= 0.
    - Method: `test_generate_dataset_negative_iterations()`  | generate_dataset() rejeita iterations <= 0.
    - Method: `test_generate_dataset_negative_seed()`  | generate_dataset() rejeita seed < 0.
    - Method: `test_generate_dataset_invalid_variant()`  | generate_dataset() rejeita variante inválida.
    - Method: `test_generate_dataset_valid_small()`  | generate_dataset() cria dataset válido.
  - Class: `TestValidationIntegration`
    - Method: `test_validation_chain_id3_to_bitboard()`  | ID3 validation interage corretamente com bitboard.
    - Method: `test_validation_chain_bulk_to_id3()`  | Geração de dataset e ID3 validation integram.
    - Method: `test_multiple_validation_errors_caught()`  | Múltiplos erros de validação são detectados independentemente.

---
## 📊 Internal Research Logic Frequency
Functions you wrote, sorted by how often they are used in this project:
* `PopOutBoard`: 125 calls
* `apply_move`: 84 calls
* `ID3Classifier`: 57 calls
* `fit`: 51 calls
* `run`: 45 calls
* `legal_moves`: 44 calls
* `generate_dataset`: 26 calls
* `StandardUCT`: 21 calls
* `MCTSNode`: 17 calls
* `evaluate_after_move`: 16 calls
* `parse_move`: 16 calls
* `_weather_dataset`: 16 calls
* `FlatNumbaMCTS`: 15 calls
* `check_winner_for_player`: 14 calls
* `get_feature_importance`: 13 calls
* `BaseMCTS`: 13 calls
* `board_signature`: 12 calls
* `GameSaveManager`: 12 calls
* `NumbaMCTS`: 12 calls
* `expand`: 12 calls
* `_make_board_with`: 12 calls
* `randomize_state`: 11 calls
* `is_full`: 11 calls
* `nb_has_won`: 11 calls
* `make_agent`: 11 calls
* `score`: 10 calls
* `nb_apply_move`: 10 calls
* `_column_from_mouse`: 10 calls
* `_h_mask`: 10 calls
* `GameState`: 9 calls
* `nb_evaluate_after_move`: 9 calls
* `nb_legal_moves`: 9 calls
* `navigate`: 9 calls
* `PauseMenu`: 9 calls
* `toggle_pause`: 9 calls
* `_encode_move`: 9 calls
* `is_leaf`: 9 calls
* `ExperimentalUCT`: 8 calls
* `apply_bins`: 8 calls
* `warmup`: 8 calls
* `add_piece_animation`: 8 calls
* `_get_cell`: 8 calls
* `_create_dataset`: 8 calls
* `fit_quantile_bins`: 7 calls
* `predict`: 7 calls
* `print_tree`: 7 calls
* `save_game`: 7 calls
* `select_current`: 7 calls
* `tree_to_string`: 7 calls
* `is_threefold_repetition`: 6 calls
* `nb_simulate`: 6 calls
* `clone`: 6 calls
* `_print_board`: 6 calls
* `AnimationState`: 6 calls
* `_player_color`: 6 calls
* `entropy`: 6 calls
* `DecisionNode`: 6 calls
* `load_game`: 5 calls
* `reset_game`: 5 calls
* `predict_one`: 5 calls
* `_board_ints`: 5 calls
* `nb_is_full`: 4 calls
* `best_child`: 4 calls
* `has_won`: 4 calls
* `update`: 4 calls
* `_simple_pure_dataset`: 4 calls
* `get_save_info`: 3 calls
* `nb_expand_step`: 3 calls
* `simulate`: 3 calls
* `q`: 3 calls
* `_check_winner`: 3 calls
* `_draw_pause_menu`: 3 calls
* `get_agent`: 3 calls
* `_draw_glow_circle`: 3 calls
* `_draw_disc`: 3 calls
* `to_feature_dict`: 3 calls
* `information_gain`: 3 calls
* `_set_cell`: 3 calls
* `is_draw`: 3 calls
* `main`: 2 calls
* `run_cli_game`: 2 calls
* `__init__`: 2 calls
* `nb_is_threefold_repetition`: 2 calls
* `select`: 2 calls
* `backpropagate`: 2 calls
* `run_hvc`: 2 calls
* `decode_move`: 2 calls
* `_make_ai_engine`: 2 calls
* `_draw_board`: 2 calls
* `apply_and_animate`: 2 calls
* `check_winner`: 2 calls
* `_draw_vertical_gradient`: 2 calls
* `majority_class`: 2 calls
* `build_tree`: 2 calls
* `_count_feature_usage`: 2 calls
* `_tree_to_string_recursive`: 2 calls
* `_player_glow`: 2 calls
* `_simple_binary_dataset`: 2 calls
* `launch_gui`: 1 calls
* `_NumbaNode`: 1 calls
* `nb_mcts_run`: 1 calls
* `_nb_best_child_id`: 1 calls
* `run_cvc`: 1 calls
* `run_hvh`: 1 calls
* `run_cvc_tournament`: 1 calls
* `create_fonts`: 1 calls
* `clear_numba_cache`: 1 calls
* `recompile`: 1 calls
* `delete_save`: 1 calls
* `list_saves`: 1 calls
* `NoRun`: 1 calls