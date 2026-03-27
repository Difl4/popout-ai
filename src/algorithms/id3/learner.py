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

    def build_tree(self, df: pd.DataFrame, features: List[str], target: str) -> DecisionNode:
        """
        Constrói árvore ID3 recursivamente.

        Args:
            df (pd.DataFrame): Subconjunto de treino.
            features (list[str]): Atributos disponíveis.
            target (str): Nome da classe.

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
