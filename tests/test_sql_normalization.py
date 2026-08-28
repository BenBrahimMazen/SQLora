"""Offline unit tests for src/sql_normalization.py.

No network, no model, no dataset — pure string logic. Runs in CI.
"""

from src.sql_normalization import normalize_sql, sql_exact_match


class TestNormalizeBasics:
    def test_lowercase_keywords_and_identifiers(self):
        assert normalize_sql("SELECT Name FROM Singer") == "select name from singer"

    def test_trailing_semicolon_and_whitespace(self):
        assert normalize_sql("SELECT a FROM t ;\n") == "select a from t"

    def test_newlines_collapse_to_single_spaces(self):
        assert normalize_sql("SELECT a\nFROM t\nWHERE a = 1") == "select a from t where a = 1"

    def test_values_are_lowercased(self):
        # Documented behavior: whole-string lowercase, including literals.
        assert normalize_sql("SELECT * FROM t WHERE name = 'Hello'") == \
            "select * from t where name = 'hello'"


class TestQuotesAndOperators:
    def test_double_quoted_literal_matches_single_quoted(self):
        assert sql_exact_match("SELECT * FROM t WHERE name = \"bob\"",
                               "SELECT * FROM t WHERE name = 'bob'")

    def test_operator_spacing_ignored(self):
        assert sql_exact_match("SELECT * FROM t WHERE a>= b", "select * from t where a >=b")

    def test_angle_bracket_not_equals_unified(self):
        assert sql_exact_match("SELECT * FROM t WHERE a <> 1", "SELECT * FROM t WHERE a != 1")

    def test_comma_spacing_ignored(self):
        assert normalize_sql("SELECT a , b FROM t") == "select a, b from t"

    def test_paren_spacing_ignored(self):
        assert sql_exact_match("SELECT count (*) FROM t", "SELECT count(*) FROM t")


class TestSelectListSorting:
    def test_column_order_sorted(self):
        # Result-set semantics don't depend on projection order.
        assert sql_exact_match("SELECT b, a FROM t", "SELECT a, b FROM t")

    def test_distinct_kept_as_prefix_and_sorted(self):
        assert normalize_sql("SELECT DISTINCT b, a FROM t") == "select distinct a, b from t"

    def test_nested_parens_not_split(self):
        assert sql_exact_match("SELECT name, count(*) FROM t GROUP BY name",
                               "SELECT count(*), name FROM t GROUP BY name")

    def test_comma_inside_string_literal_not_split(self):
        assert sql_exact_match("SELECT c, 'a,b' FROM t", "SELECT 'a,b', c FROM t")

    def test_bare_star_skips_sorting_but_does_not_crash(self):
        norm = normalize_sql("SELECT *, name FROM t")
        assert norm.startswith("select") and "*" in norm

    def test_subquery_in_from_kept_intact(self):
        assert sql_exact_match(
            "SELECT a FROM (SELECT x FROM t) ORDER BY a",
            "select a from (select x from t) order by a",
        )


class TestExactMatch:
    def test_empty_prediction_never_matches(self):
        assert not sql_exact_match("", "SELECT 1")
        assert not sql_exact_match(None or "", "SELECT 1")

    def test_genuinely_different_queries_do_not_match(self):
        assert not sql_exact_match("SELECT a FROM t", "SELECT b FROM t")

    def test_fenced_prediction_matches(self):
        # clean_prediction-style fences are tolerated end-to-end via normalize
        # only when the caller strips them; normalization itself treats the
        # fence as content — so exact match uses cleaned input at call sites.
        assert not sql_exact_match("```sql\nSELECT a FROM t\n```", "SELECT a FROM t")
