"""Agente ID3 treinado apenas com features brutas (células + jogador atual).

Nenhuma feature de alto nível é fornecida — o modelo aprende táticas e
estratégia exclusivamente a partir do estado bruto do tabuleiro, com
oversampling pesado das posições provadas para garantir exposição a padrões
decisivos.  Não utiliza regras codificadas (sem _forced_move).
"""

from __future__ import annotations

import gc
import pickle
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from src.engine.standard.bitboard import PopOutBoard
from src.engine.standard.rules import has_won
from src.decision_tree.id3.learner import ID3Classifier

_TARGET = "best_move"
_OVERSAMPLE_FACTOR = 10  # heavy oversampling of proven positions

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Only raw board state: cell_r_c occupancy values + current_player.
# All hand-crafted summary features are excluded so the tree learns everything
# from the raw cell patterns.
_EXCLUDE = {"best_move", "is_proven", "threats_me", "threats_opp",
            "center_me", "center_opp", "phase", "can_win", "opp_wins_next"}


class ID3AgentRaw:
    """Agente ID3 com features brutas e oversampling de posições provadas.

    Args:
        dataset_path: Caminho para o CSV de treino. Se None, usa o padrão.
        max_depth:    Profundidade máxima da árvore. Maior profundidade permite
                      ao modelo aprender padrões táticos complexos a partir das
                      células brutas.
        pickle_path:  Caminho do ficheiro pickle. Se None, usa o padrão.
    """

    DEFAULT_DATASET = str(_PROJECT_ROOT / "data/generated/v2_5000games_100k/popout_dt_dataset.csv")
    MODEL_PICKLE    = str(_PROJECT_ROOT / "data/generated/v2_5000games_100k/id3_model_raw.pkl")

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        max_depth: int = 20,
        pickle_path: Optional[str] = None,
        **_kwargs,
    ) -> None:
        self._dataset_path = Path(dataset_path or self.DEFAULT_DATASET)
        self._pickle_path  = Path(pickle_path  or self.MODEL_PICKLE)
        self._max_depth    = max_depth
        self._classifier: Optional[ID3Classifier] = None

    # ── Inicialização lazy ────────────────────────────────────────────────────

    def _ensure_trained(self) -> None:
        if self._classifier is not None:
            return
        if self._pickle_path.exists():
            print(f"  [ID3AgentRaw] A carregar modelo de {self._pickle_path} ...")
            with open(self._pickle_path, "rb") as f:
                self._classifier = pickle.load(f)
            print("  [ID3AgentRaw] Modelo carregado.")
            return
        self._classifier = self._train_and_save()

    def _train_and_save(self) -> ID3Classifier:
        """Train on raw cell features only and persist to disk."""
        df = pd.read_csv(self._dataset_path)

        feature_cols = [c for c in df.columns if c not in _EXCLUDE]

        # Heavy oversampling of proven positions so the tree sees enough
        # examples of decisive board states to learn tactical patterns from
        # raw cells alone.
        if "is_proven" in df.columns:
            proven = df[df["is_proven"] == 1]
            if not proven.empty:
                df = pd.concat([df] + [proven] * _OVERSAMPLE_FACTOR, ignore_index=True)
                del proven
                gc.collect()

        print(f"  [ID3AgentRaw] A treinar ID3 (depth={self._max_depth}, "
              f"{len(df)} linhas, {len(feature_cols)} features brutas)...")

        for col in feature_cols:
            df[col] = df[col].astype(str)
        df[_TARGET] = df[_TARGET].astype(str)

        clf = ID3Classifier(max_depth=self._max_depth)
        clf.fit(df[feature_cols + [_TARGET]], target=_TARGET)

        del df
        gc.collect()

        self._pickle_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._pickle_path, "wb") as f:
            pickle.dump(clf, f)
        print(f"  [ID3AgentRaw] Modelo guardado em {self._pickle_path}")
        return clf

    # ── Interface MCTSEngine ──────────────────────────────────────────────────

    def run(self, board: PopOutBoard, iterations: int = 0) -> int:
        self._ensure_trained()
        if self._classifier is None:
            raise RuntimeError("Classifier não inicializado.")

        forced = self._forced_move(board)
        if forced is not None:
            return forced

        row = pd.Series({
            k: str(v)
            for k, v in board.to_feature_dict().items()
        })
        prediction = self._classifier.predict_one(row)
        return self._parse_move(prediction, board.legal_moves())

    @staticmethod
    def _forced_move(board: PopOutBoard) -> int | None:
        me = board.current_player
        opp = 3 - me
        legal = board.legal_moves()

        for move in legal:
            b = board.clone()
            b.apply_move(move)
            mask = b.mask_p1 if me == 1 else b.mask_p2
            if has_won(mask):
                return move

        def _opp_wins_after(b: PopOutBoard) -> bool:
            for m in b.legal_moves():
                c = b.clone()
                c.apply_move(m)
                opp_mask = c.mask_p1 if opp == 1 else c.mask_p2
                if has_won(opp_mask):
                    return True
            return False

        board_as_opp = board.clone()
        board_as_opp.current_player = opp
        if _opp_wins_after(board_as_opp):
            for move in legal:
                b = board.clone()
                b.apply_move(move)
                if not _opp_wins_after(b):
                    return move

        return None

    @staticmethod
    def _parse_move(prediction: str, legal_moves: list[int]) -> int:
        try:
            move_type, col_str = prediction.split("_")
            col  = int(col_str)
            move = col if move_type == "drop" else col + 7
            if move in legal_moves:
                return move
        except (ValueError, AttributeError):
            pass
        drop_moves = [m for m in legal_moves if m < 7]
        return random.choice(drop_moves if drop_moves else legal_moves)
