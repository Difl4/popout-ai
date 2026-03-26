"""Implementação de Decision Tree ID3 sem scikit-learn."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


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

    def __init__(self) -> None:
        """
        Inicializa classificador sem árvore treinada.
        """
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
        """
        probs = labels.value_counts(normalize=True)
        return float(-(probs * probs.apply(lambda p: math.log2(p) if p > 0 else 0.0)).sum())

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

    def build_tree(self, df: pd.DataFrame, features: list[str], target: str) -> DecisionNode:
        """
        Constrói árvore ID3 recursivamente.

        Args:
            df (pd.DataFrame): Subconjunto de treino.
            features (list[str]): Atributos disponíveis.
            target (str): Nome da classe.

        Returns:
            DecisionNode: Nó raiz do subproblema.
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

        gains = {f: self.information_gain(df, f, target) for f in features}
        best_feature = max(gains, key=gains.get)

        node.feature = best_feature
        remaining = [f for f in features if f != best_feature]

        for val, subset in df.groupby(best_feature):
            if subset.empty:
                leaf = DecisionNode(label=node.majority_label, majority_label=node.majority_label)
                node.children[str(val)] = leaf
            else:
                node.children[str(val)] = self.build_tree(subset, remaining, target)

        return node

    def fit(self, df: pd.DataFrame, target: str) -> None:
        """
        Treina o classificador ID3.

        Args:
            df (pd.DataFrame): Dataset com features e alvo.
            target (str): Nome da coluna target.
        """
        self.target_name = target
        features = [c for c in df.columns if c != target]
        self.root = self.build_tree(df, features, target)

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
        """
        return df.apply(self.predict_one, axis=1)

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
