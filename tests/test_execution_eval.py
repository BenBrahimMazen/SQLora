"""Offline tests for src/execution_eval.py using tiny in-memory-built SQLite
fixtures (written to tmp_path). No network, no model, no Spider download.

These tests are the safety net for the defensive-execution requirements:
read-only in-memory copies, statement denial, timeouts, error classification,
and order-(in)sensitive result comparison.
"""

import sqlite3

import pytest

from src.execution_eval import (
    canonicalize_rows,
    clean_prediction,
    evaluate_predictions,
    results_match,
    run_query,
)

RECURSIVE_COUNT_SQL = (
    "WITH RECURSIVE c(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM c) "
    "SELECT count(*) FROM c"
)


@pytest.fixture()
def demo_db(tmp_path):
    """A miniature Spider-style database: database/<db_id>/<db_id>.sqlite."""
    db_dir = tmp_path / "database" / "demo_db"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "demo_db.sqlite"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE singer (id INTEGER PRIMARY KEY, name TEXT, age INTEGER);
        CREATE TABLE concert (id INTEGER PRIMARY KEY, singer_id INTEGER);
        INSERT INTO singer VALUES (1, 'A', 20), (2, 'B', 30), (3, 'C', 40);
        INSERT INTO concert VALUES (10, 1), (11, 1), (12, 2);
        """
    )
    conn.commit()
    conn.close()
    return db_path


class TestRunQueryHappyPath:
    def test_select_returns_rows_and_columns(self, demo_db):
        res = run_query(demo_db, "SELECT name FROM singer ORDER BY name")
        assert res["status"] == "ok"
        assert res["rows"] == [("A",), ("B",), ("C",)]
        assert res["columns"] == ["name"]

    def test_source_file_untouched_after_queries(self, demo_db):
        before = demo_db.read_bytes()
        run_query(demo_db, "DELETE FROM singer")  # denied, see below
        run_query(demo_db, "SELECT * FROM singer")
        assert demo_db.read_bytes() == before


class TestRunQueryDefensive:
    def test_syntax_error_is_a_failure_not_a_crash(self, demo_db):
        res = run_query(demo_db, "SELEC nope FROM")
        assert res["status"] == "error"
        assert res["error"]

    def test_write_statement_denied(self, demo_db):
        res = run_query(demo_db, "DELETE FROM singer")
        assert res["status"] == "error"
        assert "author" in res["error"].lower()

    def test_ddl_denied(self, demo_db):
        res = run_query(demo_db, "CREATE TABLE evil (x INT)")
        assert res["status"] == "error"

    def test_attach_blocked_before_execution(self, demo_db):
        res = run_query(demo_db, "ATTACH DATABASE 'evil.db' AS evil")
        assert res["status"] == "error"
        assert "forbidden" in res["error"]

    def test_pragma_blocked(self, demo_db):
        res = run_query(demo_db, "PRAGMA journal_mode = DELETE")
        assert res["status"] == "error"

    def test_multi_statement_rejected(self, demo_db):
        res = run_query(demo_db, "SELECT 1; DELETE FROM singer")
        assert res["status"] == "error"

    def test_empty_query(self, demo_db):
        assert run_query(demo_db, "   ")["status"] == "error"

    def test_missing_database_file(self, tmp_path):
        res = run_query(tmp_path / "nope" / "nope.sqlite", "SELECT 1")
        assert res["status"] == "error"
        assert "not found" in res["error"]

    def test_timeout_aborts_instead_of_hanging(self, demo_db):
        res = run_query(demo_db, RECURSIVE_COUNT_SQL, timeout_s=0.5)
        assert res["status"] == "error"
        assert "timeout" in res["error"].lower()

    def test_row_cap(self, demo_db):
        sql = ("WITH RECURSIVE c(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM c) "
               "SELECT x FROM c LIMIT 100000")
        res = run_query(demo_db, sql, max_rows=500)
        assert res["status"] == "error"
        assert "too large" in res["error"]

    def test_markdown_fences_stripped(self, demo_db):
        res = run_query(demo_db, "```sql\nSELECT count(*) FROM singer\n```;")
        assert res["status"] == "ok"
        assert res["rows"] == [(3,)]


class TestComparison:
    def test_identical_results_match(self, demo_db):
        gold = run_query(demo_db, "SELECT name FROM singer ORDER BY name")
        pred = run_query(demo_db, "SELECT name FROM singer")
        assert results_match(pred, gold)

    def test_different_results_do_not_match(self, demo_db):
        gold = run_query(demo_db, "SELECT name FROM singer")
        pred = run_query(demo_db, "SELECT name FROM singer WHERE id = 1")
        assert not results_match(pred, gold)

    def test_row_order_ignored_by_default(self, demo_db):
        gold = run_query(demo_db, "SELECT name FROM singer ORDER BY name ASC")
        pred = run_query(demo_db, "SELECT name FROM singer ORDER BY name DESC")
        assert results_match(pred, gold)
        assert not results_match(pred, gold, order_sensitive=True)

    def test_int_and_float_canonicalized_equal(self):
        assert canonicalize_rows([(3,)]) == canonicalize_rows([(3.0,)])
        assert canonicalize_rows([(2.5,)]) == canonicalize_rows([(2.50004,)])
        assert canonicalize_rows([(None,)]) != canonicalize_rows([("",)])

    def test_failed_prediction_never_matches(self, demo_db):
        gold = run_query(demo_db, "SELECT 1")
        pred = run_query(demo_db, "SELECT nope")
        assert not results_match(pred, gold)


class TestEvaluatePredictions:
    def _gold(self):
        return [
            {"db_id": "demo_db", "difficulty": "easy", "output": "SELECT count(*) FROM singer"},
            {"db_id": "demo_db", "difficulty": "hard", "output": "SELECT max(age) FROM singer"},
        ]

    def _db_dir(self, demo_db):
        # demo_db = <tmp>/database/demo_db/demo_db.sqlite -> database dir is two parents up.
        return demo_db.parent.parent

    def test_summary_accuracy_and_tiers(self, demo_db):
        preds = [
            {"predicted": "SELECT count(*) FROM singer"},          # correct
            {"predicted": "SELECT min(age) FROM singer"},          # wrong
        ]
        per, summary = evaluate_predictions(preds, self._gold(), self._db_dir(demo_db))
        assert summary["n"] == 2
        assert summary["execution_accuracy"] == 0.5
        assert summary["by_difficulty"]["easy"]["accuracy"] == 1.0
        assert summary["by_difficulty"]["hard"]["accuracy"] == 0.0
        assert per[0]["execution_match"] is True
        assert per[1]["execution_match"] is False

    def test_error_prediction_recorded_not_raised(self, demo_db):
        preds = [{"predicted": "DROP TABLE singer"}, {"predicted": "SELECT max(age) FROM singer"}]
        per, summary = evaluate_predictions(preds, self._gold(), self._db_dir(demo_db))
        assert per[0]["pred_status"] == "error"
        assert summary["pred_errors"] == 1

    def test_length_mismatch_rejected(self, demo_db):
        with pytest.raises(ValueError):
            evaluate_predictions([{"predicted": "SELECT 1"}], self._gold(),
                                 self._db_dir(demo_db))


class TestCleanPrediction:
    def test_strips_fences_and_semicolon(self):
        assert clean_prediction("```sql\nSELECT 1\n```") == "SELECT 1"
        assert clean_prediction("  SELECT 1; ") == "SELECT 1"

    def test_empty(self):
        assert clean_prediction("") == ""
        assert clean_prediction(None) == ""
