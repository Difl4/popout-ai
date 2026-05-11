"""Agente de jogo baseado em árvore de decisão ID3.

Treina (ou carrega) um ID3Classifier sobre um dataset gerado por MCTS e
utiliza-o para selecionar a próxima jogada.  É compatível com o protocolo
MCTSEngine — qualquer chamador que aceite um MCTS aceita também este agente.

Velocidade típica: < 1 ms por decisão (vs ~100-500 ms para MCTS).
Qualidade: inferior ao MCTS, mas suficiente para jogar de forma coerente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.engine.standard.bitboard import PopOutBoard
from src.decision_tree.id3.learner import ID3Classifier
from src.decision_tree.discretizer import fit_quantile_bins, apply_bins


class ID3Agent:
    """Agente jogável treinado com ID3.

    Na primeira chamada a run(), o agente treina o modelo automaticamente:
      1. Tenta carregar o CSV em *dataset_path*.
      2. Se não existir, gera um novo dataset usando StandardUCT e guarda-o.
      3. Treina o ID3Classifier no dataset.

    Args:
        dataset_path: Caminho para o CSV de treino.  Se None, usa o padrão.
        max_depth: Profundidade máxima da árvore ID3.
        seed: Semente aleatória para geração do dataset.
        n_samples: Amostras a gerar se o dataset não existir.
        mcts_iterations: Iterações MCTS por amostra durante a geração.
    """

    DEFAULT_DATASET = "data/generated/uct_standard.csv"

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        max_depth: int = 8,
        seed: int = 42,
        n_samples: int = 200,
        mcts_iterations: int = 150,
    ) -> None:
        self._dataset_path = Path(dataset_path or self.DEFAULT_DATASET)
        self._max_depth = max_depth
        self._seed = seed
        self._n_samples = n_samples
        self._mcts_iterations = mcts_iterations
        self._classifier: Optional[ID3Classifier] = None

    # ── Inicialização lazy ────────────────────────────────────────────────────

    def _ensure_trained(self) -> None:
        if self._classifier is not None:
            return

        if self._dataset_path.exists():
            df = pd.read_csv(self._dataset_path)
        else:
            from src.decision_tree.dataset_generator import generate_dataset
            print(f"  [ID3Agent] Gerando dataset ({self._n_samples} amostras, {self._mcts_iterations} iter)...")
            df = generate_dataset(
                variant="uct_standard",
                n_samples=self._n_samples,
                iterations=self._mcts_iterations,
                seed=self._seed,
            )
            self._dataset_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(self._dataset_path, index=False)
            print(f"  [ID3Agent] Dataset guardado em {self._dataset_path}")

        # As features são inteiros categóricos (0/1/2 por célula, 1/2 para jogador).
        # O ID3 trabalha com strings — converter aqui.
        df = df.astype(str)

        clf = ID3Classifier(max_depth=self._max_depth)
        clf.fit(df, target="best_move")
        self._classifier = clf

    # ── Interface MCTSEngine ──────────────────────────────────────────────────

    def run(self, board: PopOutBoard, iterations: int = 0) -> int:
        """Seleciona a melhor jogada usando a árvore de decisão ID3.

        O parâmetro *iterations* é ignorado — existe apenas para compatibilidade
        com o protocolo MCTSEngine.

        Returns:
            int: Jogada seleccionada (0-6 drop, 7-13 pop).
        """
        self._ensure_trained()

        assert self._classifier is not None

        features = board.to_feature_dict()
        row = pd.Series({k: str(v) for k, v in features.items()})
        prediction = self._classifier.predict_one(row)

        move = self._parse_move(prediction, board.legal_moves())
        return move

    # ── Auxiliares ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_move(prediction: str, legal_moves: list[int]) -> int:
        """Converte predição "drop_3" / "pop_1" para inteiro do motor.

        Usa o primeiro movimento legal como fallback se a predição for inválida
        ou ilegal (pode acontecer em estados não vistos durante o treino).
        """
        try:
            move_type, col_str = prediction.split("_")
            col = int(col_str)
            move = col if move_type == "drop" else col + 7
            if move in legal_moves:
                return move
        except (ValueError, AttributeError):
            pass
        return legal_moves[0]
