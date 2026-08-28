"""Offline tests for src/preprocessing.py — schema serialization, difficulty
estimation, and split conversion. Uses small hand-built tables.json-style
fixtures (same structure as the real file). No network, no model.
"""

import json

from src.preprocessing import (
    build_user_message,
    convert_split,
    estimate_difficulty,
    load_tables_json,
    read_jsonl,
    serialize_schema,
    write_jsonl,
)

# Hand-built tables.json entry with the same structure as the real Spider file.
DEMO_SCHEMA = {
    "db_id": "demo_db",
    "table_names": ["singer", "concert"],
    "table_names_original": ["singer", "concert"],
    "column_names": [[-1, "*"], [0, "id"], [0, "name"], [0, "age"], [1, "id"], [1, "singer id"]],
    "column_names_original": [[-1, "*"], [0, "id"], [0, "name"], [0, "age"],
                              [1, "id"], [1, "singer_id"]],
    "column_types": ["text", "integer", "text", "integer", "integer", "integer"],
    "primary_keys": [1, 4],
    "foreign_keys": [[5, 1]],  # concert.singer_id -> singer.id
}


class TestSerializeSchema:
    def test_tables_and_columns_with_types(self):
        s = serialize_schema(DEMO_SCHEMA)
        assert "database: demo_db" in s
        assert "singer(id INTEGER [PK], name TEXT, age INTEGER)" in s
        assert "concert(id INTEGER [PK], singer_id INTEGER)" in s

    def test_star_column_omitted(self):
        assert "*" not in serialize_schema(DEMO_SCHEMA)

    def test_foreign_key_rendered(self):
        s = serialize_schema(DEMO_SCHEMA)
        assert "concert.singer_id -> singer.id" in s

    def test_no_foreign_keys_section_when_empty(self):
        schema = dict(DEMO_SCHEMA, foreign_keys=[])
        assert "foreign keys" not in serialize_schema(schema)

    def test_malformed_fk_skipped_not_crashing(self):
        schema = dict(DEMO_SCHEMA, foreign_keys=[[99, 100]])
        assert "foreign keys" in serialize_schema(schema)  # header still there
        assert "->" not in serialize_schema(schema)


class TestBuildUserMessage:
    def test_contains_schema_and_question(self):
        msg = build_user_message("database: demo_db\ntables:\n  singer(id)", "Who is 30?")
        assert msg.startswith("Database schema:")
        assert "Who is 30?" in msg
        assert "SQLite" in msg


class TestEstimateDifficulty:
    def test_plain_select_is_easy(self):
        assert estimate_difficulty("SELECT name FROM singer") == "easy"

    def test_aggregate_only_is_easy(self):
        assert estimate_difficulty("SELECT count(*) FROM singer") == "easy"

    def test_single_where_is_easy(self):
        assert estimate_difficulty("SELECT name FROM singer WHERE age > 20") == "easy"

    def test_where_plus_group_by_is_medium(self):
        assert estimate_difficulty(
            "SELECT name FROM singer WHERE age > 20 GROUP BY name"
        ) == "medium"

    def test_three_components_is_hard(self):
        sql = ("SELECT name FROM singer WHERE age > 20 GROUP BY name "
               "ORDER BY name LIMIT 3")
        assert estimate_difficulty(sql) == "hard"

    def test_lone_subquery_is_hard(self):
        assert estimate_difficulty("SELECT name FROM (SELECT name FROM singer)") == "hard"

    def test_subquery_with_where_is_extra_hard(self):
        sql = ("SELECT name FROM singer WHERE id IN "
               "(SELECT singer_id FROM concert)")
        assert estimate_difficulty(sql) == "extra hard"

    def test_set_operation_is_extra_hard(self):
        sql = "SELECT name FROM singer INTERSECT SELECT name FROM singer"
        assert estimate_difficulty(sql) == "extra hard"


class TestConvertSplit:
    def test_record_shape(self):
        schemas = {"demo_db": DEMO_SCHEMA}
        examples = [{"db_id": "demo_db", "question": "How many singers?",
                     "query": "SELECT count(*) FROM singer"}]
        records, missing = convert_split(examples, schemas)
        assert missing == {}
        rec = records[0]
        assert set(rec) == {"db_id", "question", "instruction", "output", "difficulty"}
        assert rec["db_id"] == "demo_db"
        assert rec["question"] == "How many singers?"
        assert rec["output"] == "SELECT count(*) FROM singer"
        assert rec["difficulty"] == "easy"
        assert "singer(" in rec["instruction"]  # schema embedded in instruction

    def test_unknown_db_uses_fallback_and_is_counted(self):
        examples = [{"db_id": "ghost_db", "question": "?", "query": "SELECT 1"}]
        records, missing = convert_split(examples, {})
        assert missing == {"ghost_db": 1}
        assert "schema metadata missing" in records[0]["instruction"]
        assert records[0]["output"] == "SELECT 1"


class TestJsonlRoundtrip:
    def test_write_then_read(self, tmp_path):
        records = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        path = tmp_path / "round.jsonl"
        write_jsonl(records, path)
        assert read_jsonl(path) == records

    def test_load_tables_json(self, tmp_path):
        path = tmp_path / "tables.json"
        path.write_text(json.dumps([DEMO_SCHEMA]), encoding="utf-8")
        assert load_tables_json(path) == {"demo_db": DEMO_SCHEMA}
