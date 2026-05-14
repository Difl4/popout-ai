"""Agente de jogo baseado em árvore de decisão ID3.

Treina (ou carrega) um ID3Classifier sobre o dataset gerado pelo notebook
PopOut_Decision_Tree_Pipeline.ipynb e utiliza-o para selecionar a próxima
jogada.  É compatível com o protocolo MCTSEngine.

Velocidade típica: < 1 ms por decisão (vs ~100-500 ms para MCTS).
"""

from __future__ import annotations

import gc
import pickle
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import extended_features, has_won
from src.decision_tree.id3.learner import ID3Classifier

_TARGET = "best_move"
_OVERSAMPLE_FACTOR = 4  # proven positions duplicated this many extra times (matches notebook)

# Project root derived from this file's location so paths work regardless of CWD.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ID3Agent:
    """Agente jogável treinado com ID3 usando o dataset do notebook.

    Na primeira chamada a run(), o agente treina o modelo automaticamente:
      1. Tenta carregar o CSV do notebook em *dataset_path*.
      2. Se não existir, cai de volta para o dataset mais pequeno legado.
      3. Treina o ID3Classifier com oversampling de posições provadas.

    Args:
        dataset_path: Caminho para o CSV de treino.  Se None, usa o padrão.
        max_depth: Profundidade máxima da árvore ID3.
        seed: Semente aleatória para geração do dataset de fallback.
        n_samples: Amostras a gerar se nenhum dataset existir.
        mcts_iterations: Iterações MCTS por amostra durante a geração de fallback.
    """

    DEFAULT_DATASET = str(_PROJECT_ROOT / "data/generated/v2_5000games_100k/popout_dt_dataset.csv")
    FALLBACK_DATASET = str(_PROJECT_ROOT / "data/generated/uct_standard.csv")
    MODEL_PICKLE = str(_PROJECT_ROOT / "data/generated/v2_5000games_100k/id3_model.pkl")

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        max_depth: int = 10,
        seed: int = 42,
        n_samples: int = 200,
        mcts_iterations: int = 150,
        pickle_path: Optional[str] = None,
    ) -> None:
        self._dataset_path = Path(dataset_path or self.DEFAULT_DATASET)
        self._pickle_path = Path(pickle_path or self.MODEL_PICKLE)
        self._max_depth = max_depth
        self._seed = seed
        self._n_samples = n_samples
        self._mcts_iterations = mcts_iterations
        self._classifier: Optional[ID3Classifier] = None

    # ── Inicialização lazy ────────────────────────────────────────────────────

    def _ensure_trained(self) -> None:
        if self._classifier is not None:
            return

        # Fast path: load pre-trained model from pickle
        if self._pickle_path.exists():
            print(f"  [ID3Agent] A carregar modelo de {self._pickle_path} ...")
            with open(self._pickle_path, "rb") as f:
                self._classifier = pickle.load(f)
            print("  [ID3Agent] Modelo carregado.")
            return

        # Slow path: train from CSV and save pickle for future runs
        self._classifier = self._train_and_save()

    def _train_and_save(self) -> ID3Classifier:
        """Train the ID3 classifier from CSV and persist it to disk."""
        if self._dataset_path.exists():
            df = pd.read_csv(self._dataset_path)
        else:
            fallback = Path(self.FALLBACK_DATASET)
            if fallback.exists():
                print(f"  [ID3Agent] Dataset do notebook não encontrado; a usar fallback {fallback}")
                df = pd.read_csv(fallback)
            else:
                from src.decision_tree.dataset_generator import generate_dataset
                print(f"  [ID3Agent] Gerando dataset de fallback ({self._n_samples} amostras)...")
                df = generate_dataset(
                    variant="uct_standard",
                    n_samples=self._n_samples,
                    iterations=self._mcts_iterations,
                    seed=self._seed,
                )
                fallback.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(fallback, index=False)

        feature_cols = [c for c in df.columns if c not in (_TARGET, "is_proven")]

        # Oversample proven positions (mirrors notebook).
        # Single concat to avoid creating multiple full copies in memory.
        if "is_proven" in df.columns:
            proven = df[df["is_proven"] == 1]
            if not proven.empty:
                df = pd.concat([df] + [proven] * _OVERSAMPLE_FACTOR, ignore_index=True)
                del proven
                gc.collect()

        print(f"  [ID3Agent] A treinar ID3 (depth={self._max_depth}, {len(df)} linhas)...")

        # Convert to strings in-place column by column to avoid a full copy.
        for col in feature_cols:
            df[col] = df[col].astype(str)
        df[_TARGET] = df[_TARGET].astype(str)

        clf = ID3Classifier(max_depth=self._max_depth)
        clf.fit(df[feature_cols + [_TARGET]], target=_TARGET)

        # Free the training DataFrame before pickling.
        del df
        gc.collect()

        self._pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._pickle_path, "wb") as f:
            pickle.dump(clf, f)
        print(f"  [ID3Agent] Modelo guardado em {self._pickle_path}")
        return clf

    # ── Interface MCTSEngine ──────────────────────────────────────────────────

    def run(self, board: PopOutBoard, iterations: int = 0) -> int:
        """Seleciona a melhor jogada usando a árvore de decisão ID3.

        Prioridades aplicadas antes de consultar a árvore:
          1. Ganhar imediatamente se possível.
          2. Bloquear vitória imediata do adversário.
          3. Predição da árvore ID3.

        O parâmetro *iterations* é ignorado — existe apenas para compatibilidade
        com o protocolo MCTSEngine.
        """
        self._ensure_trained()

        if self._classifier is None:
            raise RuntimeError("Classifier não inicializado após _ensure_trained()")

        forced = self._forced_move(board)
        if forced is not None:
            return forced

        features = extended_features(board)
        row = pd.Series({k: str(v) for k, v in features.items() if k != "is_proven"})
        prediction = self._classifier.predict_one(row)
        return self._parse_move(prediction, board.legal_moves())

    # ── Auxiliares ────────────────────────────────────────────────────────────

    @staticmethod
    def _forced_move(board: PopOutBoard) -> int | None:
        """Devolve jogada forçada (vitória ou bloqueio imediato), ou None.

        Cobre ameaças por drop E por pop: para bloquear uma ameaça de pop do
        adversário procura-se qualquer jogada minha que elimine todas as suas
        vitórias imediatas (lookahead de 2 meios-lances).
        """
        me = board.current_player
        opp = 3 - me
        legal = board.legal_moves()

        # Priority 1: win immediately (drop or pop)
        for move in legal:
            b = board.clone()
            b.apply_move(move)
            mask = b.mask_p1 if me == 1 else b.mask_p2
            if has_won(mask):
                return move

        # Priority 2: block opponent's immediate win.
        # Find all moves that would win for the opponent on their next turn.
        def _opp_wins_after(b: PopOutBoard) -> bool:
            """True if the opponent can win in one move from board b."""
            b_opp = b.clone()
            b_opp.current_player = opp
            for m in b_opp.legal_moves():
                c = b_opp.clone()
                c.apply_move(m)
                opp_mask = c.mask_p1 if opp == 1 else c.mask_p2
                if has_won(opp_mask):
                    return True
            return False

        if _opp_wins_after(board):
            # Find a move of mine that removes all opponent wins.
            for move in legal:
                b = board.clone()
                b.apply_move(move)
                if not _opp_wins_after(b):
                    return move

        return None


    @staticmethod
    def _parse_move(prediction: str, legal_moves: list[int]) -> int:
        """Converte predição "drop_3" / "pop_1" para inteiro do motor.

        Se a predição for inválida ou ilegal, cai de volta para um drop aleatório.
        Pop moves só acontecem quando o classificador os prediz explicitamente —
        nunca como fallback, pois remover uma peça própria por acidente é
        geralmente catastrófico.
        """
        try:
            move_type, col_str = prediction.split("_")
            col = int(col_str)
            move = col if move_type == "drop" else col + 7
            if move in legal_moves:
                return move
        except (ValueError, AttributeError):
            pass

        # Prefer a random drop — pops should only happen when explicitly predicted.
        drop_moves = [m for m in legal_moves if m < 7]
        return random.choice(drop_moves if drop_moves else legal_moves)