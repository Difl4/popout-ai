"""Implementação de Decision Tree ID3 sem scikit-learn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
import numpy as np


@dataclass
class DecisionNode:
    feature: Optional[str] = None
    label: Optional[str] = None
    children: Dict[str, "DecisionNode"] = field(default_factory=dict)
    majority_label: Optional[str] = None

    def is_leaf(self) -> bool:
        return self.label is not None


class ID3Classifier:
    """Classificador ID3 para atributos categóricos."""

    def __init__(self, max_depth: Optional[int] = None) -> None:
        if max_depth is not None:
            if not isinstance(max_depth, int) or isinstance(max_depth, bool):
                raise TypeError(f"max_depth deve ser int ou None, recebeu {type(max_depth).__name__}")
            if max_depth < 1:
                raise ValueError(f"max_depth deve ser >= 1, recebeu {max_depth}")
        self.max_depth: Optional[int] = max_depth
        self.root: Optional[DecisionNode] = None
        self.target_name: Optional[str] = None

    @staticmethod
    def entropy(labels) -> float:
        arr = labels.values if hasattr(labels, 'values') else np.asarray(labels)
        _, counts = np.unique(arr, return_counts=True)
        probs = counts / len(arr)
        return float(-np.sum(probs * np.log2(np.maximum(probs, 1e-10))))

    def information_gain(self, df: pd.DataFrame, feature: str, target: str) -> float:
        feat = df[feature].values
        tgt  = df[target].values
        base_h = self.entropy(tgt)
        n = len(feat)
        weighted = 0.0
        for v in np.unique(feat):
            mask = feat == v
            weighted += mask.sum() / n * self.entropy(tgt[mask])
        return base_h - weighted

    @staticmethod
    def majority_class(labels) -> str:
        arr = labels.values if hasattr(labels, 'values') else np.asarray(labels)
        vals, counts = np.unique(arr, return_counts=True)
        return str(vals[counts.argmax()])

    def build_tree(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        depth: int = 0,
    ) -> DecisionNode:
        node = DecisionNode()
        node.majority_label = self.majority_class(df[target])

        unique_labels = df[target].unique()
        if len(unique_labels) == 1:
            node.label = str(unique_labels[0])
            return node

        if not features:
            node.label = node.majority_label
            return node

        if self.max_depth is not None and depth >= self.max_depth:
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
                node.children[str(val)] = self.build_tree(subset, remaining, target, depth + 1)

        return node

    def fit(self, df: pd.DataFrame, target: str) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        if not isinstance(target, str):
            raise TypeError(f"Expected target name as string, got {type(target).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in DataFrame. Available columns: {list(df.columns)}")
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            raise ValueError(f"DataFrame contains NaN values in columns: {nan_cols}")

        self.target_name = target
        features = [c for c in df.columns if c != target]
        self.root = self.build_tree(df, features, target, depth=0)

    def predict_one(self, row: pd.Series) -> str:
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
        if self.root is None:
            raise ValueError("Modelo ainda não treinado.")
        predictions = [self.predict_one(row) for _, row in df.iterrows()]
        return pd.Series(predictions, index=df.index)

    def score(self, df: pd.DataFrame, target: str) -> float:
        preds = self.predict(df.drop(columns=[target]))
        return float((preds == df[target].astype(str)).mean())

    def _count_feature_usage(self, node: Optional[DecisionNode], counts: Dict[str, int]) -> None:
        if node is None or node.is_leaf():
            return
        if node.feature is not None:
            counts[node.feature] = counts.get(node.feature, 0) + 1
        for child in node.children.values():
            self._count_feature_usage(child, counts)

    def get_feature_importance(self) -> Dict[str, float]:
        if self.root is None:
            raise ValueError("Modelo ainda não treinado. Chame fit() primeiro.")
        feature_counts: Dict[str, int] = {}
        self._count_feature_usage(self.root, feature_counts)
        if not feature_counts:
            return {}
        total_splits = sum(feature_counts.values())
        return {
            feature: count / total_splits
            for feature, count in sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        }

    def print_tree(self, node: Optional[DecisionNode] = None, prefix: str = "",
                   value_label: str = "") -> None:
        if node is None:
            if self.root is None:
                print("Modelo ainda não treinado.")
                return
            print(f"[{self.target_name}]")
            self.print_tree(self.root, "", "")
            return
        if node.is_leaf():
            print(f"{prefix}{value_label} → {node.label}")
            return
        if value_label:
            print(f"{prefix}{value_label}")
        children = sorted(node.children.items())
        for idx, (value, child) in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("   " if is_last else "│  ")
            if child.is_leaf():
                print(f"{prefix}{connector}{value} → {child.label}")
            else:
                print(f"{prefix}{connector}{value}")
                self.print_tree(child, child_prefix, "")

    def tree_to_string(self) -> str:
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
        if node is None:
            return
        if node.is_leaf():
            lines.append(f"{prefix}{value_label} → {node.label}")
            return
        if value_label:
            lines.append(f"{prefix}{value_label}")
        children = sorted(node.children.items())
        for idx, (value, child) in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("   " if is_last else "│  ")
            if child.is_leaf():
                lines.append(f"{prefix}{connector}{value} → {child.label}")
            else:
                lines.append(f"{prefix}{connector}{value}")
                self._tree_to_string_recursive(child, child_prefix, "", lines)
