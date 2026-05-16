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
        n = len(feat)

        feat_codes, feat_cats = pd.factorize(feat, sort=True)
        tgt_codes,  tgt_cats  = pd.factorize(tgt,  sort=True)
        n_feat, n_tgt = len(feat_cats), len(tgt_cats)

        tgt_counts = np.bincount(tgt_codes, minlength=n_tgt).astype(np.float64)
        probs  = tgt_counts / n
        base_h = float(-np.sum(probs * np.log2(np.maximum(probs, 1e-10))))

        joint = np.bincount(feat_codes * n_tgt + tgt_codes,
                            minlength=n_feat * n_tgt).reshape(n_feat, n_tgt).astype(np.float64)
        fc = joint.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            cp = np.where(fc > 0, joint / fc, 0.0)
        log_cp = np.zeros_like(cp)
        pos = cp > 0
        log_cp[pos] = np.log2(cp[pos])
        cond_h = -np.sum(joint / n * log_cp)

        return base_h - cond_h

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

    def _build_fast(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: np.ndarray,
        feat_indices: List[int],
        n_tgt: int,
        depth: int,
    ) -> DecisionNode:
        """Recursively build the tree on pre-encoded integer arrays.

        Works on a subset of rows identified by `indices` rather than copying
        DataFrame slices, which avoids pandas overhead at every node.
        Information gain is computed via a joint-frequency bincount — fully
        vectorised, no Python loop over unique values.
        """
        node = DecisionNode()
        tgt_here = y[indices]
        counts = np.bincount(tgt_here, minlength=n_tgt)
        best_cls = int(counts.argmax())
        node.majority_label = str(self._tgt_classes_[best_cls])

        if counts.max() == len(indices):
            node.label = str(self._tgt_classes_[best_cls])
            return node

        if not feat_indices or (self.max_depth is not None and depth >= self.max_depth):
            node.label = node.majority_label
            return node

        n = len(indices)
        base_probs = counts.astype(np.float64) / n
        base_h = float(-np.sum(base_probs * np.log2(np.maximum(base_probs, 1e-10))))

        best_ig, best_fi = -1.0, feat_indices[0]
        for fi in feat_indices:
            n_fv = self._feat_n_vals_[fi]
            feat_here = X[indices, fi]
            joint = np.bincount(feat_here * n_tgt + tgt_here,
                                minlength=n_fv * n_tgt).reshape(n_fv, n_tgt).astype(np.float64)
            fc = joint.sum(axis=1, keepdims=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                cp = np.where(fc > 0, joint / fc, 0.0)
            log_cp = np.zeros_like(cp)
            pos = cp > 0
            log_cp[pos] = np.log2(cp[pos])
            cond_h = -np.sum(joint / n * log_cp)
            ig = base_h - cond_h
            if ig > best_ig:
                best_ig, best_fi = ig, fi

        node.feature = self._feat_cols_[best_fi]
        remaining = [fi for fi in feat_indices if fi != best_fi]

        feat_vals = X[indices, best_fi]
        for v in range(self._feat_n_vals_[best_fi]):
            child_indices = indices[feat_vals == v]
            val_str = str(self._feat_maps_[best_fi][v])
            if len(child_indices) == 0:
                leaf = DecisionNode(label=node.majority_label, majority_label=node.majority_label)
            else:
                leaf = self._build_fast(X, y, child_indices, remaining, n_tgt, depth + 1)
            node.children[val_str] = leaf

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
        self.target_name = target
        feature_cols = [c for c in df.columns if c != target]

        # Encode target to integer codes once
        tgt_cat = pd.Categorical(df[target])
        self._tgt_classes_: list = tgt_cat.categories.tolist()
        y = np.asarray(tgt_cat.codes, dtype=np.int32)

        # Encode every feature column to integer codes once
        n_feats = len(feature_cols)
        X = np.empty((len(df), n_feats), dtype=np.int32)
        self._feat_cols_: list = feature_cols
        self._feat_maps_: list = []
        self._feat_n_vals_: list = []
        for j, col in enumerate(feature_cols):
            cat = pd.Categorical(df[col])
            X[:, j] = np.asarray(cat.codes, dtype=np.int32)
            self._feat_maps_.append(cat.categories.tolist())
            self._feat_n_vals_.append(len(cat.categories))

        n_tgt = len(self._tgt_classes_)
        indices = np.arange(len(df), dtype=np.int32)
        self.root = self._build_fast(X, y, indices, list(range(n_feats)), n_tgt, depth=0)

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
        results = np.empty(len(df), dtype=object)
        self._predict_batch(self.root, df, np.arange(len(df), dtype=np.int32), results)
        return pd.Series(results, index=df.index)

    def _predict_batch(
        self,
        node: DecisionNode,
        df: pd.DataFrame,
        indices: np.ndarray,
        results: np.ndarray,
    ) -> None:
        """Predict for a batch of rows simultaneously, avoiding per-row Python overhead."""
        if node.is_leaf():
            results[indices] = node.label
            return
        feat_vals = df[node.feature].values
        for val, child in node.children.items():
            mask = feat_vals[indices] == val
            child_indices = indices[mask]
            if len(child_indices) > 0:
                self._predict_batch(child, df, child_indices, results)
        # Rows with unseen feature values fall back to the majority label
        seen_mask = np.isin(feat_vals[indices], list(node.children.keys()))
        unseen = indices[~seen_mask]
        if len(unseen) > 0:
            results[unseen] = node.majority_label

    def score(self, df: pd.DataFrame, target: str) -> float:
        feat_cols = [c for c in df.columns if c != target]
        preds = self.predict(df[feat_cols])
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
        """Print the tree to stdout in ASCII form.

        For every non-leaf node, the feature being tested is shown in square
        brackets next to the branch value, e.g. ``low [petal_length]``, so
        the reader can always tell which attribute is being split on.
        """
        if node is None:
            if self.root is None:
                print("Modelo ainda não treinado.")
                return
            if self.root.is_leaf():
                print(f"[{self.target_name}] → {self.root.label}")
                return
            print(f"[{self.target_name}] — split on [{self.root.feature}]")
            self.print_tree(self.root, "", "")
            return
        if node.is_leaf():
            print(f"{prefix}{value_label} → {node.label}")
            return
        if value_label:
            feat_marker = f" [{node.feature}]" if node.feature else ""
            print(f"{prefix}{value_label}{feat_marker}")
        children = sorted(node.children.items())
        for idx, (value, child) in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("   " if is_last else "│  ")
            if child.is_leaf():
                print(f"{prefix}{connector}{value} → {child.label}")
            else:
                feat_marker = f" [{child.feature}]" if child.feature else ""
                print(f"{prefix}{connector}{value}{feat_marker}")
                self.print_tree(child, child_prefix, "")

    def tree_to_string(self) -> str:
        """Return the tree as a single multiline string.

        Output format matches :meth:`print_tree` — non-leaf branches show the
        feature being tested in square brackets.
        """
        if self.root is None:
            return "Modelo ainda não treinado."
        if self.root.is_leaf():
            return f"[{self.target_name}] → {self.root.label}"
        lines = [f"[{self.target_name}] — split on [{self.root.feature}]"]
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
            feat_marker = f" [{node.feature}]" if node.feature else ""
            lines.append(f"{prefix}{value_label}{feat_marker}")
        children = sorted(node.children.items())
        for idx, (value, child) in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "└─ " if is_last else "├─ "
            child_prefix = prefix + ("   " if is_last else "│  ")
            if child.is_leaf():
                lines.append(f"{prefix}{connector}{value} → {child.label}")
            else:
                feat_marker = f" [{child.feature}]" if child.feature else ""
                lines.append(f"{prefix}{connector}{value}{feat_marker}")
                self._tree_to_string_recursive(child, child_prefix, "", lines)
