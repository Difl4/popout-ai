"""Testes para src/algorithms/id3/learner.py — ID3Classifier."""

import pytest
import pandas as pd

from src.algorithms.id3.learner import DecisionNode, ID3Classifier


# ── Helpers ──────────────────────────────────────────────────────────────────

def _weather_dataset() -> pd.DataFrame:
    """Dataset clássico 'play tennis' para testes categóricos."""
    data = {
        "outlook": ["sunny", "sunny", "overcast", "rain", "rain", "rain",
                     "overcast", "sunny", "sunny", "rain", "sunny",
                     "overcast", "overcast", "rain"],
        "temperature": ["hot", "hot", "hot", "mild", "cool", "cool",
                        "cool", "mild", "cool", "mild", "mild",
                        "mild", "hot", "mild"],
        "humidity": ["high", "high", "high", "high", "normal", "normal",
                     "normal", "high", "normal", "normal", "normal",
                     "high", "normal", "high"],
        "wind": ["weak", "strong", "weak", "weak", "weak", "strong",
                 "strong", "weak", "weak", "weak", "strong",
                 "strong", "weak", "strong"],
        "play": ["no", "no", "yes", "yes", "yes", "no",
                 "yes", "no", "yes", "yes", "yes",
                 "yes", "yes", "no"],
    }
    return pd.DataFrame(data)


def _simple_pure_dataset() -> pd.DataFrame:
    """Dataset onde todas as linhas têm a mesma classe."""
    return pd.DataFrame({
        "feat": ["a", "b", "c"],
        "label": ["yes", "yes", "yes"],
    })


def _simple_binary_dataset() -> pd.DataFrame:
    """Dataset binário equilibrado."""
    return pd.DataFrame({
        "feat": ["a", "a", "b", "b"],
        "label": ["yes", "no", "yes", "no"],
    })


# ── Entropy ──────────────────────────────────────────────────────────────────

class TestEntropy:
    def test_pure_set_entropy_zero(self):
        labels = pd.Series(["yes", "yes", "yes"])
        assert ID3Classifier.entropy(labels) == pytest.approx(0.0)

    def test_uniform_binary_entropy_one(self):
        labels = pd.Series(["yes", "no", "yes", "no"])
        assert ID3Classifier.entropy(labels) == pytest.approx(1.0)

    def test_single_element(self):
        labels = pd.Series(["a"])
        assert ID3Classifier.entropy(labels) == pytest.approx(0.0)

    def test_three_classes_uniform(self):
        labels = pd.Series(["a", "b", "c", "a", "b", "c"])
        import math
        expected = -3 * (1 / 3) * math.log2(1 / 3)
        assert ID3Classifier.entropy(labels) == pytest.approx(expected, abs=1e-6)


# ── Information Gain ─────────────────────────────────────────────────────────

class TestInformationGain:
    def test_gain_positive_for_useful_feature(self):
        df = _weather_dataset()
        clf = ID3Classifier()
        gain = clf.information_gain(df, "outlook", "play")
        assert gain > 0.0

    def test_gain_zero_for_useless_feature(self):
        """Feature com valor único não separa nada → ganho = 0."""
        df = pd.DataFrame({
            "feat": ["x", "x", "x", "x"],
            "label": ["a", "b", "a", "b"],
        })
        clf = ID3Classifier()
        gain = clf.information_gain(df, "feat", "label")
        assert gain == pytest.approx(0.0)


# ── Majority Class ───────────────────────────────────────────────────────────

class TestMajorityClass:
    def test_majority(self):
        labels = pd.Series(["a", "b", "a", "a", "b"])
        assert ID3Classifier.majority_class(labels) == "a"


# ── Fit & Predict ────────────────────────────────────────────────────────────

class TestFitPredict:
    def test_fit_creates_tree(self):
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        assert clf.root is not None
        assert clf.target_name == "play"

    def test_predict_training_data_high_accuracy(self):
        """ID3 deve ter accuracy perfeita ou quase perfeita no treino."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        acc = clf.score(df, target="play")
        assert acc >= 0.9  # ID3 tipicamente memoriza o treino

    def test_predict_pure_dataset(self):
        df = _simple_pure_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="label")
        preds = clf.predict(df.drop(columns=["label"]))
        assert all(p == "yes" for p in preds)

    def test_predict_one(self):
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        row = df.drop(columns=["play"]).iloc[0]
        pred = clf.predict_one(row)
        assert pred in ("yes", "no")

    def test_predict_unseen_category_falls_back(self):
        """Valor de feature não visto no treino → usa majority_label."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        # Criar linha com valor nunca visto
        row = pd.Series({
            "outlook": "NEVER_SEEN",
            "temperature": "hot",
            "humidity": "high",
            "wind": "weak",
        })
        pred = clf.predict_one(row)
        assert pred in ("yes", "no")  # Não deve crashar

    def test_predict_before_fit_raises(self):
        clf = ID3Classifier()
        with pytest.raises(ValueError, match="não treinado"):
            clf.predict_one(pd.Series({"a": 1}))


# ── DecisionNode ─────────────────────────────────────────────────────────────

class TestDecisionNode:
    def test_leaf_node(self):
        node = DecisionNode(label="yes")
        assert node.is_leaf() is True

    def test_internal_node(self):
        node = DecisionNode(feature="outlook")
        assert node.is_leaf() is False

    def test_children_dict(self):
        child = DecisionNode(label="no")
        parent = DecisionNode(feature="wind", children={"strong": child})
        assert "strong" in parent.children
        assert parent.children["strong"].is_leaf()


# ── Score ────────────────────────────────────────────────────────────────────

class TestScore:
    def test_score_returns_float(self):
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        s = clf.score(df, target="play")
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0
