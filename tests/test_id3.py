"""Testes para src/decision_tree/id3/learner.py — ID3Classifier."""

import pytest
import pandas as pd

from src.decision_tree.id3.learner import DecisionNode, ID3Classifier


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


# ── Feature Importance ───────────────────────────────────────────────────────

class TestFeatureImportance:
    def test_importance_before_fit_raises(self):
        """get_feature_importance() sem fit() deve lançar ValueError."""
        clf = ID3Classifier()
        with pytest.raises(ValueError, match="não treinado"):
            clf.get_feature_importance()

    def test_importance_pure_dataset_empty(self):
        """Dataset puro (só um label) → árvore é apenas leaf → vazio."""
        df = _simple_pure_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="label")
        importance = clf.get_feature_importance()
        assert importance == {}  # Sem splits → sem features

    def test_importance_returns_dict(self):
        """get_feature_importance() retorna dict[str, float]."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        importance = clf.get_feature_importance()
        assert isinstance(importance, dict)
        assert all(isinstance(k, str) for k in importance.keys())
        assert all(isinstance(v, float) for v in importance.values())

    def test_importance_all_positive(self):
        """Todos os scores de importância são positivos."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        importance = clf.get_feature_importance()
        assert all(v > 0.0 for v in importance.values())

    def test_importance_normalizes_to_one(self):
        """Soma dos scores de importância ≈ 1.0."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        importance = clf.get_feature_importance()
        total = sum(importance.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_importance_sorted_by_count(self):
        """Features em ordem decrescente de importância."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        importance = clf.get_feature_importance()
        values = list(importance.values())
        # Verificar que está em ordem decrescente
        assert values == sorted(values, reverse=True)

    def test_importance_with_single_split_feature(self):
        """Feature com um único split contribui menos."""
        # Dataset simples onde apenas uma feature importa
        df = pd.DataFrame({
            "critical": ["a", "a", "b", "b"],
            "ignored": ["x", "y", "x", "y"],
            "label": ["yes", "yes", "no", "no"],
        })
        clf = ID3Classifier()
        clf.fit(df, target="label")
        importance = clf.get_feature_importance()
        
        # Verificar que temos importância
        assert len(importance) > 0
        
        # A feature "critical" deve ter importância alta
        if "critical" in importance:
            assert importance["critical"] > 0.0

    def test_importance_consistent_across_runs(self):
        """Importância é determinística (mesmos dados = mesma árvore)."""
        df = _weather_dataset()
        
        clf1 = ID3Classifier()
        clf1.fit(df, target="play")
        imp1 = clf1.get_feature_importance()
        
        clf2 = ID3Classifier()
        clf2.fit(df, target="play")
        imp2 = clf2.get_feature_importance()
        
        # Mesmas features
        assert imp1.keys() == imp2.keys()
        
        # Mesmos scores
        for key in imp1.keys():
            assert imp1[key] == pytest.approx(imp2[key], abs=1e-6)

    def test_importance_binary_dataset(self):
        """Importância em dataset binário simples."""
        df = _simple_binary_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="label")
        importance = clf.get_feature_importance()
        
        # Deve ter importância se a árvore não é trivial
        if importance:  # Se há splits
            assert sum(importance.values()) == pytest.approx(1.0, abs=1e-6)

    def test_importance_multifeature(self):
        """Importância com múltiplas features em jogo."""
        df = pd.DataFrame({
            "feat1": ["a", "a", "b", "b", "a", "b"],
            "feat2": ["x", "y", "x", "y", "y", "x"],
            "feat3": [1, 2, 1, 2, 1, 2],
            "label": ["yes", "no", "yes", "no", "yes", "no"],
        })
        clf = ID3Classifier()
        clf.fit(df, target="label")
        importance = clf.get_feature_importance()
        
        # No mínimo uma feature foi usada
        assert len(importance) >= 1
        
        # Normalização
        assert sum(importance.values()) == pytest.approx(1.0, abs=1e-6)


# ── Tree Visualization ───────────────────────────────────────────────────────

class TestTreeVisualization:
    def test_print_tree_before_fit_raises(self):
        """print_tree() sem fit should handle gracefully."""
        clf = ID3Classifier()
        # print_tree should not raise but print message
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        clf.print_tree()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        assert "não treinado" in output

    def test_tree_to_string_before_fit(self):
        """tree_to_string() sem fit retorna mensagem."""
        clf = ID3Classifier()
        result = clf.tree_to_string()
        assert "não treinado" in result

    def test_print_tree_pure_dataset(self):
        """print_tree com dataset puro (sem splits)."""
        df = _simple_pure_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="label")
        
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        clf.print_tree()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Deve conter o target name
        assert "label" in output

    def test_tree_to_string_pure_dataset(self):
        """tree_to_string com dataset puro."""
        df = _simple_pure_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="label")
        
        result = clf.tree_to_string()
        assert "label" in result
        assert isinstance(result, str)

    def test_print_tree_with_splits(self):
        """print_tree com árvore que tem splits."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        clf.print_tree()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Deve conter o target e features
        assert "play" in output
        assert "→" in output  # Arrow towards leaf labels
        
    def test_tree_to_string_with_splits(self):
        """tree_to_string com múltiplos splits."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        
        result = clf.tree_to_string()
        assert isinstance(result, str)
        assert "play" in result
        assert "→" in result  # Arrow towards labels
        assert len(result) > 0

    def test_print_tree_contains_ascii_chars(self):
        """print_tree usa caracteres ASCII para desenhar árvore."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        clf.print_tree()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Verificar connectors ASCII
        assert any(c in output for c in ["├", "└", "─"])

    def test_tree_to_string_indentation(self):
        """tree_to_string mantém indentação apropriada."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        
        result = clf.tree_to_string()
        lines = result.split("\n")
        
        # Deve ter múltiplas linhas
        assert len(lines) > 1
        
        # Algumas linhas com indentação
        indented_lines = [l for l in lines if l.startswith((" ", "│", "├", "└"))]
        assert len(indented_lines) > 0

    def test_tree_visualization_with_binary_feature(self):
        """Visualização com dataset binário."""
        df = _simple_binary_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="label")
        
        result = clf.tree_to_string()
        assert "label" in result

    def test_tree_visualization_output_format(self):
        """Formato da visualização é consistente."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")

        # Ambos métodos devem produzir outputs relacionados
        tree_str = clf.tree_to_string()

        # Verificar que tem labels e estrutura
        assert "yes" in tree_str or "no" in tree_str
        assert "play" in tree_str  # Target name deve constar
        assert "→" in tree_str  # Arrows para leaf labels

    def test_tree_to_string_includes_root_feature(self):
        """O root deve indicar a feature em que é feito o primeiro split."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        tree_str = clf.tree_to_string()

        # Pelo menos uma das features deve aparecer perto do topo
        assert clf.root is not None
        assert clf.root.feature in tree_str
        # E deve haver um marcador explícito "split on"
        assert "split on" in tree_str

    def test_tree_to_string_includes_inner_feature_names(self):
        """Em árvores não-triviais, features de splits interiores também aparecem."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")
        tree_str = clf.tree_to_string()

        # Recolhe nomes de features de todos os nós internos da árvore
        inner_features: set = set()

        def _walk(node):
            if node is None or node.is_leaf():
                return
            if node.feature is not None:
                inner_features.add(node.feature)
            for child in node.children.values():
                _walk(child)

        _walk(clf.root)
        # Cada feature usada num split interno deve aparecer em algum lugar no output
        for feat in inner_features:
            assert feat in tree_str, f"Feature '{feat}' missing from tree output"

    def test_print_tree_includes_feature_names(self):
        """O output de print_tree deve mostrar nomes de features (não só valores)."""
        df = _weather_dataset()
        clf = ID3Classifier()
        clf.fit(df, target="play")

        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        clf.print_tree()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        # O root deve indicar a feature do primeiro split
        assert clf.root is not None and clf.root.feature is not None
        assert clf.root.feature in output


