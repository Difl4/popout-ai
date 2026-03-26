"""Testes para src/interfaces/cli.py — parsing de jogadas."""

import pytest

from src.interfaces.cli import parse_move


# ── parse_move ───────────────────────────────────────────────────────────────

class TestParseMove:
    def test_drop_move(self):
        assert parse_move("d3") == ("drop", 3)

    def test_drop_move_zero(self):
        assert parse_move("d0") == ("drop", 0)

    def test_drop_move_six(self):
        assert parse_move("d6") == ("drop", 6)

    def test_pop_move(self):
        assert parse_move("p0") == ("pop", 0)

    def test_pop_move_five(self):
        assert parse_move("p5") == ("pop", 5)

    def test_uppercase_accepted(self):
        """Input é convertido para lowercase internamente."""
        assert parse_move("D4") == ("drop", 4)
        assert parse_move("P2") == ("pop", 2)

    def test_whitespace_stripped(self):
        assert parse_move("  d3  ") == ("drop", 3)

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="[Tt]ipo inválido"):
            parse_move("x3")

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="[Ff]ormato inválido"):
            parse_move("d")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="[Ff]ormato inválido"):
            parse_move("")

    def test_invalid_column_high_raises(self):
        with pytest.raises(ValueError, match="[Cc]oluna inválida"):
            parse_move("d7")

    def test_invalid_column_negative_raises(self):
        with pytest.raises(ValueError):
            parse_move("d-1")

    def test_non_numeric_column_raises(self):
        with pytest.raises(ValueError):
            parse_move("da")
