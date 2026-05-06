"""Implementação de Decision Tree ID3 sem scikit-learn."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


@dataclass
class DecisionNode:
    """
    Nó da árvore de decisão ID3.

    Attributes:
        feature (Optional[str]): Atributo usado na divisão.
        label (Optional[str]): Classe final se nó folha.
        children (Dict[str, DecisionNode]): Ramos por valor categórico.
        majority_label (Optional[str]): Classe majoritária neste nó.
    """

    feature: Optional[str] = None
    label: Optional[str] = None
    children: Dict[str, "DecisionNode"] = field(default_factory=dict)
    majority_label: Optional[str] = None

    def is_leaf(self) -> bool:
        """
        Indica se o nó é folha.

        Returns:
            bool: True quando possui label final.
        """
        return self.label is not None


class ID3Classifier:
    """
    Classificador ID3 para atributos categóricos.
    """

    def __init__(self, max_depth: Optional[int] = None) -> None:
        """
        Inicializa classificador sem árvore treinada.

        Args:
            max_depth (Optional[int]): Profundidade máxima da árvore.
                Se None, cresce até esgotarem os atributos ou amostras.
                Limitar a profundidade reduz overfitting.
        """
        if max_depth is not None:
            if not isinstance(max_depth, int) or isinstance(max_depth, bool):
                raise TypeError(f"max_depth deve ser int ou None, recebeu {type(max_depth).__name__}")
            if max_depth < 1:
                raise ValueError(f"max_depth deve ser >= 1, recebeu {max_depth}")
        self.max_depth: Optional[int] = max_depth
        self.root: Optional[DecisionNode] = None
        self.target_name: Optional[str] = None

    @staticmethod
    def entropy(labels: pd.Series) -> float:
        """
        Calcula entropia de Shannon de uma variável categórica.

        Args:
            labels (pd.Series): Série de classes.

        Returns:
            float: Entropia em bits.

        OTIMIZAÇÃO: Vetorizado com NumPy para evitar lambda.
        """
        probs = labels.value_counts(normalize=True).values
        # Evitar log(0) com np.maximum
        return float(-np.sum(probs * np.log2(np.maximum(probs, 1e-10))))

    def information_gain(self, df: pd.DataFrame, feature: str, target: str) -> float:
        """
        Calcula ganho de informação de um atributo.

        Args:
            df (pd.DataFrame): Dataset atual.
            feature (str): Atributo candidato.
            target (str): Nome da variável alvo.

        Returns:
            float: Ganho de informação.
        """
        base_entropy = self.entropy(df[target])
        weighted_entropy = 0.0

        for value, subset in df.groupby(feature):
            weight = len(subset) / len(df)
            weighted_entropy += weight * self.entropy(subset[target])

        return base_entropy - weighted_entropy

    @staticmethod
    def majority_class(labels: pd.Series) -> str:
        """
        Obtém classe majoritária.

        Args:
            labels (pd.Series): Série alvo.

        Returns:
            str: Classe mais frequente.
        """
        return str(labels.value_counts().idxmax())

    def build_tree(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        depth: int = 0,
    ) -> DecisionNode:
        """
        Constrói árvore ID3 recursivamente.

        Args:
            df (pd.DataFrame): Subconjunto de treino.
            features (list[str]): Atributos disponíveis.
            target (str): Nome da classe.
            depth (int): Profundidade atual (uso interno).

        Returns:
            DecisionNode: Nó raiz do subproblema.

        OTIMIZAÇÃO: Early stopping se encontrar feature com ganho muito elevado.
        """
        node = DecisionNode()
        node.majority_label = self.majority_class(df[target])

        unique_labels = df[target].unique()
        if len(unique_labels) == 1:
            node.label = str(unique_labels[0])
            return node

        if not features:
            node.label = node.majority_label
            return node

        # Parar se atingiu profundidade máxima
        if self.max_depth is not None and depth >= self.max_depth:
            node.label = node.majority_label
            return node

        gains = {}
        best_gain = -1.0
        best_feature = None

        # OTIMIZAÇÃO: Early stopping se encontrar feature com ganho muito bom
        for f in features:
            gain = self.information_gain(df, f, target)
            gains[f] = gain

            if gain > best_gain:
                best_gain = gain
                best_feature = f

            # Early stopping: se ganho é próximo do máximo teórico, para
            if best_gain > 0.95:
                break

        if best_feature is None:
            best_feature = features[0]

        node.feature = best_feature
        remaining = [f for f in features if f != best_feature]

        for val, subset in df.groupby(best_feature):
            if subset.empty:
                leaf = DecisionNode(label=node.majority_label, majority_label=node.majority_label)
                node.children[str(val)] = leaf
            else:
                node.children[str(val)] = self.build_tree(subset, remaining, target, depth + 1)

        return node

    def fit(self, df: pd.DataFrame, target: str) -> None:
        """
        Treina o classificador ID3.

        Args:
            df (pd.DataFrame): Dataset com features e alvo.
            target (str): Nome da coluna target.

        Raises:
            ValueError: Se df é None, vazio, contém NaN ou target não existe.
            TypeError: Se df não é DataFrame ou target não é string.
        """
        # Type validation
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        if not isinstance(target, str):
            raise TypeError(f"Expected target name as string, got {type(target).__name__}")

        # Empty validation
        if df.empty:
            raise ValueError("DataFrame is empty")

        # Target exists validation
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame. Available columns: {list(df.columns)}")

        # NaN validation
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            raise ValueError(f"DataFrame contains NaN values in columns: {nan_cols}")

        self.target_name = target
        features = [c for c in df.columns if c != target]
        self.root = self.build_tree(df, features, target, depth=0)

    def predict_one(self, row: pd.Series) -> str:
        """
        Prediz classe para uma observação.

        Args:
            row (pd.Series): Linha de entrada.

        Returns:
            str: Classe prevista.
        """
        if self.root is None:
            raise ValueError("Modelo ainda não treinado.")

        node = self.root
        while not node.is_leaf():
            assert node.feature is not None
            value = str(row.get(node.feature))
            if value in node.children:
                node = node.children[value]
            else:
                return str(node.majority_label)
        return str(node.label)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Prediz classes para múltiplas observações.

        Args:
            df (pd.DataFrame): DataFrame de entrada.

        Returns:
            pd.Series: Previsões.

        OTIMIZAÇÃO: Usa list comprehension em vez de apply().
        """
        if self.root is None:
            raise ValueError("Modelo ainda não treinado.")

        predictions = [self.predict_one(row) for _, row in df.iterrows()]
        return pd.Series(predictions, index=df.index)

    def score(self, df: pd.DataFrame, target: str) -> float:
        """
        Calcula accuracy em dataset rotulado.

        Args:
            df (pd.DataFrame): Dados de avaliação.
            target (str): Coluna alvo real.

        Returns:
            float: Accuracy entre 0 e 1.
        """
        preds = self.predict(df.drop(columns=[target]))
        return float((preds == df[target].astype(str)).mean())

    def _count_feature_usage(self, node: Optional[DecisionNode], counts: Dict[str, int]) -> None:
        """
        Conta recursivamente o uso de cada feature na árvore.

        Args:
            node (Optional[DecisionNode]): Nó atual.
            counts (Dict[str, int]): Dicionário acumulador de contagens.
        """
        if node is None or node.is_leaf():
            return

        # Contar uso da feature neste nó de decisão
        if node.feature is not None:
            counts[node.feature] = counts.get(node.feature, 0) + 1

        # Recursivamente contar em todos os filhos
        for child in node.children.values():
            self._count_feature_usage(child, counts)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calcula importância de cada feature na árvore treinada.

        Returns:
            Dict[str, float]: Mapa de feature -> importance score (normalizado).
                            Features não usadas não aparecem no dicionário.
                            Scores somam a aproximadamente 1.0.

        Raises:
            ValueError: Se modelo ainda não foi treinado (root é None).
        """
        if self.root is None:
            raise ValueError("Modelo ainda não treinado. Chame fit() primeiro.")

        # Contabilizar uso de cada feature
        feature_counts: Dict[str, int] = {}
        self._count_feature_usage(self.root, feature_counts)

        # Se nenhuma feature foi usada (árvore é apenas um leaf)
        if not feature_counts:
            return {}

        # Normalizar: dividir cada contagem pela soma total
        total_splits = sum(feature_counts.values())
        importance: Dict[str, float] = {
            feature: count / total_splits
            for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        }

        return importance

    def print_tree(self, node: Optional[DecisionNode] = None, prefix: str = "",
                   value_label: str = "") -> None:
        """
        Imprime a árvore de decisão com ASCII art.

        Args:
            node (Optional[DecisionNode]): Nó atual. Se None, usa root.
            prefix (str): Prefixo de indentação (uso interno).
            value_label (str): Rótulo do valor anterior (uso interno).
        """
        if node is None:
            if self.root is None:
                print("Modelo ainda não treinado.")
                return
            print(f"[{self.target_name}]")
            self.print_tree(self.root, "", "")
            return

        # Leaf node
        if node.is_leaf():
            print(f"{prefix}{value_label} → {node.label}")
            return

        # Decision node - show feature name at root
        if value_label:
            print(f"{prefix}{value_label}")

        # Get children sorted by key
        children = sorted(node.children.items())
        for idx, (value, child) in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("   " if is_last else "│  ")

            if child.is_leaf():
                label_str = f"{value} → {child.label}"
                print(f"{prefix}{connector}{label_str}")
            else:
                # Show feature name for decision nodes
                feature_str = f"[{child.feature}]" if child.feature else "?"
                print(f"{prefix}{connector}{value}")
                self.print_tree(child, child_prefix, "")

    def tree_to_string(self) -> str:
        """
        Converte a árvore para string (alternativa a print_tree).

        Returns:
            str: Representação em string da árvore.
        """
        if self.root is None:
            return "Modelo ainda não treinado."

        lines = [f"[{self.target_name}]"]
        self._tree_to_string_recursive(self.root, "", "", lines)
        return "\n".join(lines)

    def _tree_to_string_recursive(
        self,
        node: Optional[DecisionNode],
        prefix: str,
        value_label: str,
        lines: List[str],
    ) -> None:
        """Helper recursivo para tree_to_string."""
        if node is None:
            return

        if node.is_leaf():
            line = f"{prefix}{value_label} → {node.label}"
            lines.append(line)
            return

        if value_label:
            lines.append(f"{prefix}{value_label}")

        children = sorted(node.children.items())
        for idx, (value, child) in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("   " if is_last else "│  ")

            if child.is_leaf():
                line = f"{prefix}{connector}{value} → {child.label}"
                lines.append(line)
            else:
                line = f"{prefix}{connector}{value}"
                lines.append(line)
                self._tree_to_string_recursive(child, child_prefix, "", lines)
