"""Testes para src/algorithms/id3/discretizer.py — discretização por quantis."""

import pytest
import pandas as pd

from src.algorithms.id3.discretizer import apply_bins, fit_quantile_bins
from src.algorithms.id3.learner import ID3Classifier


# ── fit_quantile_bins ────────────────────────────────────────────────────────

class TestFitQuantileBins:
    def test_basic_bins(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        bins = fit_quantile_bins(df, columns=["x"], q=3)
        assert "x" in bins
        assert len(bins["x"]) > 0  # pelo menos 1 limite
        # Com q=3, esperamos ~2 limites (tercis)
        assert len(bins["x"]) <= 2

    def test_bins_are_sorted(self):
        df = pd.DataFrame({"a": [10, 20, 30, 40, 50, 60]})
        bins = fit_quantile_bins(df, columns=["a"], q=4)
        assert bins["a"] == sorted(bins["a"])

    def test_multiple_columns(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6],
            "y": [10, 20, 30, 40, 50, 60],
        })
        bins = fit_quantile_bins(df, columns=["x", "y"], q=3)
        assert "x" in bins
        assert "y" in bins

    def test_constant_column_no_bins(self):
        """Coluna constante → quantis iguais → set pode ter um valor."""
        df = pd.DataFrame({"c": [5, 5, 5, 5, 5]})
        bins = fit_quantile_bins(df, columns=["c"], q=3)
        # Coluna constante: quantis são todos iguais, set remove duplicados
        # Pode resultar em 0 ou 1 item dependendo da implementação
        assert len(bins["c"]) <= 1


# ── apply_bins ───────────────────────────────────────────────────────────────

class TestApplyBins:
    def test_applies_labels(self):
        df = pd.DataFrame({"x": [1, 5, 9]})
        bins = {"x": [3.0, 7.0]}  # 3 classes
        result = apply_bins(df, bins)
        assert result["x"].dtype == object  # string/categorical
        # A função usa "very_low", "low", "medium", "high", "very_high", "extreme"
        # Com 2 bins (3.0, 7.0) temos 3 classes: very_low, low, medium
        assert set(result["x"].unique()).issubset({"very_low", "low", "medium", "high", "very_high", "extreme"})

    def test_does_not_modify_original(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        bins = {"x": [2.0]}
        _ = apply_bins(df, bins)
        assert df["x"].dtype == float  # original intacto

    def test_single_bin_edge(self):
        """Um único limite → 2 classes."""
        df = pd.DataFrame({"v": [0, 1, 2, 3, 4]})
        bins = {"v": [2.0]}
        result = apply_bins(df, bins)
        unique = set(result["v"].unique())
        assert len(unique) == 2

    def test_no_bins_for_column(self):
        """Coluna sem bins definidos não é alterada."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        bins = {"x": [2.0]}
        result = apply_bins(df, bins)
        assert list(result["y"]) == ["a", "b", "c"]


# ── Round-trip: discretize + ID3 ────────────────────────────────────────────

class TestRoundTrip:
    def test_discretize_then_id3(self):
        """Fluxo completo: dados numéricos → discretizar → treinar ID3 → prever."""
        df = pd.DataFrame({
            "sepal_length": [5.1, 4.9, 7.0, 6.5, 5.0, 6.3, 4.7, 7.2, 6.0, 5.5],
            "sepal_width":  [3.5, 3.0, 3.2, 2.8, 3.6, 3.3, 3.2, 3.6, 2.2, 2.4],
            "species": ["setosa", "setosa", "versicolor", "versicolor",
                        "setosa", "versicolor", "setosa", "virginica",
                        "virginica", "virginica"],
        })
        num_cols = ["sepal_length", "sepal_width"]
        bins = fit_quantile_bins(df, columns=num_cols, q=3)
        df_cat = apply_bins(df, bins)

        clf = ID3Classifier()
        clf.fit(df_cat, target="species")
        acc = clf.score(df_cat, target="species")
        assert acc >= 0.5  # Deve ser razoável no treino

    def test_predict_after_discretize(self):
        """Prever uma nova observação após discretização."""
        df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "label": ["a", "a", "a", "a", "b", "b", "b", "b"],
        })
        bins = fit_quantile_bins(df, columns=["x"], q=2)
        df_cat = apply_bins(df, bins)

        clf = ID3Classifier()
        clf.fit(df_cat, target="label")

        # Nova observação
        new_row = pd.DataFrame({"x": [2]})
        new_cat = apply_bins(new_row, bins)
        pred = clf.predict_one(new_cat.iloc[0])
        assert pred in ("a", "b")
