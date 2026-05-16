"""Discretização de atributos numéricos para uso no ID3 categórico."""

from __future__ import annotations

from typing import Dict

import pandas as pd


def fit_quantile_bins(df: pd.DataFrame, columns: list[str], q: int = 3) -> Dict[str, list[float]]:
    """
    Aprende limites de discretização por quantis para colunas numéricas.

    Args:
        df (pd.DataFrame): Dados de treino.
        columns (list[str]): Colunas numéricas a discretizar.
        q (int): Número de grupos (quantis).

    Returns:
        Dict[str, list[float]]: Mapa coluna -> limites ordenados.
    """
    bins: Dict[str, list[float]] = {}
    for col in columns:
        quantiles = sorted(set(df[col].quantile([i / q for i in range(1, q)]).tolist()))
        bins[col] = quantiles
    return bins


def apply_bins(df: pd.DataFrame, bins: Dict[str, list[float]]) -> pd.DataFrame:
    """
    Aplica limites de discretização e devolve DataFrame categórico.

    Labels gerados: low, medium, high, ... (dependendo do nº de bins).

    Args:
        df (pd.DataFrame): Dados a transformar.
        bins (Dict[str, list[float]]): Limites por coluna.

    Returns:
        pd.DataFrame: Cópia transformada com colunas discretizadas.
    """
    out = df.copy()
    base_labels = ["very_low", "low", "medium", "high", "very_high", "extreme"]

    for col, edges in bins.items():
        n_classes = len(edges) + 1
        labels = base_labels[:n_classes]
        cut_edges = [-float("inf")] + edges + [float("inf")]
        out[col] = pd.cut(out[col], bins=cut_edges, labels=labels, include_lowest=True).astype(object)

    return out
