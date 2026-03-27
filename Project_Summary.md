# Project Summary:

## Directory: Root
### File: `PopOut_Solution.ipynb`
  - Cell 1:
    - Markdown Hint: Setup e Imports...
  - Cell 2:
    - Markdown Hint: 1. Demonstração do Game Engine (Bitboard)...
    - Code Hint (Internal): PopOutBoard, legal_moves
  - Cell 3:
    - Code Hint (Internal): PopOutBoard, apply_move, evaluate_after_move
  - Cell 4:
    - Markdown Hint: 2. MCTS - Monte Carlo Tree Search...
    - Code Hint (Internal): PopOutBoard, StandardUCT, run
  - Cell 5:
    - Markdown Hint: 3. ID3 Classifier - Árvore de Decisão...
    - Code Hint (Internal): ID3Classifier, fit, generate_dataset, predict, score
  - Cell 6:
    - Markdown Hint: 4. Comparação de Performance: MCTS vs ID3...
    - Code Hint (Internal): PopOutBoard, StandardUCT, predict, run

### File: `play.py`
  - Contains helper logic/imports.
## Directory: src
### File: `__init__.py`
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

### File: `numba_mcts.py`
  - Function: `nb_has_won()`  | 4-in-a-row check via bitwise shifts. Identical logic to rules.has_won.
  - Function: `nb_legal_moves()`  | Returns (moves_array, n) where moves_array[:n] are the legal move ints.
  - Function: `nb_apply_move()`  | Applies move (0-13) to (mask_p1, mask_p2, current_player).
  - Function: `nb_evaluate_after_move()`  | Mirrors rules.evaluate_after_move — PopOut tiebreak rule included.
  - Function: `nb_is_full()` 
  - Function: `nb_expand_step()`  | Apply a move, evaluate for a winner, and compute the new legal moves —
  - Function: `nb_simulate()`  | Full random rollout in pure int64 arithmetic.
  - Class: `_NumbaNode`
    - Method: `__post_init__()` 
  - Class: `NumbaMCTS`
    - Method: `__init__()` 
    - Method: `best_child()` 
    - Method: `expand()` 
    - Method: `simulate()` 
  - Function: `_nb_best_child_id()`  | UCT child selection on flat arrays. Returns child node id.
  - Function: `nb_mcts_run()`  | Complete MCTS loop in Numba — select, expand, simulate, backpropagate.
  - Class: `FlatNumbaMCTS`
    - Method: `__init__()` 
    - Method: `run()`  | Return the best move integer (0-13) for the given board state.
  - Function: `warmup()`  | Run a dummy search to force Numba to compile all @njit functions.

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
  - Function: `parse_move()`  | Converte d<col> ou p<col> no formato inteiro do motor (0-13).
  - Function: `decode_move()`  | Converte o inteiro do motor de volta para texto legível.
  - Function: `check_and_print_winner()`  | Verifica se há vencedor após uma jogada e imprime o resultado.
  - Function: `run_cli_game()` 

### File: `gui.py`
  - Class: `AnimationState`
    - Method: `__init__()`  | Inicializa estado de animação vazio.
    - Method: `add_piece_animation()`  | Inicia animação de entrada para uma peça.
    - Method: `update()`  | Atualiza todas as animações.
  - Class: `Difficulty`
  - Class: `PauseMenu`
    - Method: `__init__()`  | Inicializa menu de pausa.
    - Method: `toggle_pause()`  | Alterna estado de pausa.
    - Method: `navigate()`  | Navega menu com setas (up/down).
    - Method: `select_current()`  | Retorna a opção selecionada (para processamento).
  - Function: `_draw_pause_menu()`  | Desenha menu de pausa com opções navegáveis.
  - Function: `_player_color()`  | Devolve a cor base associada ao jogador.
  - Function: `_player_glow()`  | Devolve a cor de glow (brilho) do jogador.
  - Function: `_draw_vertical_gradient()`  | Desenha um gradiente vertical no fundo com mais qualidade.
  - Function: `_draw_glow_circle()`  | Desenha uma aura de brilho ao redor de um círculo.
  - Function: `_draw_disc()`  | Desenha uma peça com sombra, brilho e animação de chegada.
  - Function: `_draw_board()`  | Desenha tabuleiro com animações, HUD superior/inferior, preview, hover e overlay final.
  - Function: `_column_from_mouse()`  | Converte coordenada X do rato para coluna 0..6, ou None fora do tabuleiro.
  - Function: `_encode_move()`  | Codifica jogada no formato do motor: drop(0..6) ou pop(7..13).
  - Function: `launch_gui()`  | Inicia a janela pygame com interface melhorada e executa o ciclo principal.
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
    - Method: `test_difficulty_extreme()`  | Dificuldade Extreme = 2000 iterações.
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

---
## 📊 Internal Research Logic Frequency
Functions you wrote, sorted by how often they are used in this project:
* `PopOutBoard`: 68 calls
* `apply_move`: 39 calls
* `legal_moves`: 20 calls
* `run`: 18 calls
* `generate_dataset`: 16 calls
* `ID3Classifier`: 16 calls
* `parse_move`: 15 calls
* `evaluate_after_move`: 14 calls
* `check_winner_for_player`: 14 calls
* `fit`: 13 calls
* `MCTSNode`: 13 calls
* `GameSaveManager`: 12 calls
* `_make_board_with`: 12 calls
* `StandardUCT`: 10 calls
* `_column_from_mouse`: 10 calls
* `BaseMCTS`: 10 calls
* `GameState`: 9 calls
* `AnimationState`: 9 calls
* `PauseMenu`: 9 calls
* `add_piece_animation`: 9 calls
* `toggle_pause`: 9 calls
* `_encode_move`: 9 calls
* `board_signature`: 8 calls
* `_get_cell`: 8 calls
* `score`: 7 calls
* `save_game`: 7 calls
* `expand`: 7 calls
* `navigate`: 7 calls
* `apply_bins`: 7 calls
* `predict`: 6 calls
* `entropy`: 6 calls
* `DecisionNode`: 6 calls
* `_player_color`: 6 calls
* `fit_quantile_bins`: 6 calls
* `_weather_dataset`: 6 calls
* `load_game`: 5 calls
* `predict_one`: 5 calls
* `is_full`: 5 calls
* `select_current`: 5 calls
* `randomize_state`: 5 calls
* `is_leaf`: 4 calls
* `nb_legal_moves`: 4 calls
* `is_threefold_repetition`: 4 calls
* `update`: 4 calls
* `make_agent`: 4 calls
* `get_save_info`: 3 calls
* `information_gain`: 3 calls
* `clone`: 3 calls
* `best_child`: 3 calls
* `q`: 3 calls
* `nb_apply_move`: 3 calls
* `nb_evaluate_after_move`: 3 calls
* `has_won`: 3 calls
* `_draw_glow_circle`: 3 calls
* `_draw_disc`: 3 calls
* `ExperimentalUCT`: 3 calls
* `_set_cell`: 3 calls
* `is_draw`: 3 calls
* `launch_gui`: 2 calls
* `majority_class`: 2 calls
* `build_tree`: 2 calls
* `select`: 2 calls
* `simulate`: 2 calls
* `backpropagate`: 2 calls
* `nb_has_won`: 2 calls
* `__init__`: 2 calls
* `nb_simulate`: 2 calls
* `check_and_print_winner`: 2 calls
* `_draw_vertical_gradient`: 2 calls
* `_draw_pause_menu`: 2 calls
* `_player_glow`: 2 calls
* `nb_is_full`: 1 calls
* `nb_expand_step`: 1 calls
* `_NumbaNode`: 1 calls
* `_nb_best_child_id`: 1 calls
* `nb_mcts_run`: 1 calls
* `NumbaMCTS`: 1 calls
* `FlatNumbaMCTS`: 1 calls
* `run_cli_game`: 1 calls
* `decode_move`: 1 calls
* `_draw_board`: 1 calls
* `main`: 1 calls
* `to_feature_dict`: 1 calls
* `delete_save`: 1 calls
* `list_saves`: 1 calls
* `_simple_pure_dataset`: 1 calls