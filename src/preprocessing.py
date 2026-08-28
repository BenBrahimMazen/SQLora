"""Convert Spider examples into instruction-tuning format.

Pipeline position: runs AFTER scripts/download_spider.py has populated
``data/spider/`` and BEFORE training (src/train_qlora.py) and inference
(src/baseline_prompting.py).

Reads:
    data/spider/train.jsonl, data/spider/dev.jsonl   (db_id, question, query)
    data/spider/tables.json                          (schema metadata)

Writes:
    data/processed/train.jsonl, data/processed/dev.jsonl — one JSON object per
    line:
        {"db_id": str,
         "question": str,
         "instruction": "<serialized schema> + <natural-language question>",
         "output": "<gold SQL>",
         "difficulty": "easy" | "medium" | "hard" | "extra hard"}

The ``instruction`` text is model-agnostic: chat templates (Qwen, Llama, ...)
are applied later by the training/inference scripts, not baked in here, so the
same processed data works for every model and ablation.

This module is stdlib-only: it is imported by scripts/download_spider.py and by
the offline test suite (tests/test_preprocessing.py), neither of which should
pull in torch/datasets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# Chosen to match the order Spider papers report difficulty tiers in.
DIFFICULTY_TIERS = ("easy", "medium", "hard", "extra hard")

SYSTEM_PROMPT = (
    "You are an expert text-to-SQL assistant. Given a database schema and a "
    "question, write a single SQLite query that answers the question. Reply "
    "with the SQL query only — no explanation, no markdown fences."
)


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a .jsonl file into a list of dicts (raises on malformed lines)."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:  # pragma: no cover - trivial guard
                raise ValueError(f"{path}:{i}: invalid JSON: {e}") from e
    return records


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write a list of dicts as .jsonl (one compact JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_tables_json(path: str | Path) -> dict[str, dict]:
    """Load Spider's tables.json into {db_id: schema_dict}.

    Each schema dict has the original Spider fields: ``table_names``,
    ``table_names_original``, ``column_names`` / ``column_names_original``
    (lists of [table_index, column_name]; index 0 is [-1, "*"]),
    ``column_types``, ``primary_keys`` (column indexes) and ``foreign_keys``
    (pairs of column indexes).
    """
    with open(path, "r", encoding="utf-8") as f:
        tables = json.load(f)
    return {t["db_id"]: t for t in tables}


def serialize_schema(schema: dict) -> str:
    """Serialize one tables.json schema into a compact prompt representation.

    Format (original identifiers, since those are what SQL must use):

        database: concert_singer
        tables:
          stadium(Stadium_ID INTEGER [PK], Name TEXT, Capacity INTEGER)
          singer(Singer_ID INTEGER [PK], Name TEXT, Country TEXT)
        foreign keys:
          singer.Concert_ID -> concert.Concert_ID

    ``*`` (the implicit all-columns entry) is skipped; primary keys are marked
    inline with ``[PK]``.
    """
    table_names = schema["table_names_original"]
    col_names = schema["column_names_original"]
    col_types = schema["column_types"]
    primary = set(schema.get("primary_keys", []))
    foreign_keys = schema.get("foreign_keys", [])

    # Column index -> owning table index (needed to render foreign keys).
    col_table = {i: t for i, (t, _name) in enumerate(col_names)}

    lines = [f"database: {schema['db_id']}", "tables:"]
    for t_idx, t_name in enumerate(table_names):
        cols = []
        for c_idx, (tbl, col) in enumerate(col_names):
            if tbl != t_idx:
                continue
            marker = " [PK]" if c_idx in primary else ""
            # tables.json stores lowercase types ("text", "integer"...) — render
            # them uppercase, SQL-style, for a familiar-looking prompt.
            ctype = str(col_types[c_idx]).upper() if c_idx < len(col_types) else "TEXT"
            cols.append(f"{col} {ctype}{marker}")
        lines.append(f"  {t_name}({', '.join(cols)})")

    if foreign_keys:
        lines.append("foreign keys:")
        for src, dst in foreign_keys:
            try:
                src_ref = f"{table_names[col_table[src]]}.{col_names[src][1]}"
                dst_ref = f"{table_names[col_table[dst]]}.{col_names[dst][1]}"
                lines.append(f"  {src_ref} -> {dst_ref}")
            except (KeyError, IndexError):
                # Malformed FK entry in tables.json — skip rather than crash.
                continue
    return "\n".join(lines)


def build_user_message(schema_str: str, question: str) -> str:
    """Build the user turn: serialized schema + natural-language question.

    Kept separate from SYSTEM_PROMPT so chat-template callers can compose
    [system, user, assistant] turns themselves.
    """
    return (
        f"Database schema:\n{schema_str}\n\n"
        f"Question: {question}\n\n"
        "Write one SQLite query that answers the question."
    )


# --- Difficulty estimation ---------------------------------------------------

_SUBQUERY_RE = re.compile(r"\bselect\b", re.IGNORECASE)
_SETOP_RE = re.compile(r"\b(intersect|union|except)\b", re.IGNORECASE)
_JOIN_RE = re.compile(r"\bjoin\b", re.IGNORECASE)
_GROUP_RE = re.compile(r"\bgroup\s+by\b", re.IGNORECASE)
_ORDER_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)
_HAVING_RE = re.compile(r"\bhaving\b", re.IGNORECASE)
_WHERE_RE = re.compile(r"\bwhere\b", re.IGNORECASE)
_AGG_RE = re.compile(r"\b(count|sum|avg|min|max)\s*\(", re.IGNORECASE)


def estimate_difficulty(sql: str) -> str:
    """Estimate the Spider difficulty tier for a query from its SQL text.

    The official labels are produced by the Spider evaluation script from a
    fully parsed SQL structure, which the HuggingFace parquet release does not
    ship. This function approximates the same tiers from keyword features and
    is used consistently for training data, predictions and evaluation
    breakdowns — so results by tier are internally comparable (counts may
    differ slightly from the official evaluation script's labels).

    Rules (score = number of present features among WHERE / GROUP BY /
    ORDER BY / HAVING / JOIN / aggregate):

    - easy        score <= 1, no subquery, no set operation
    - medium      score == 2, no subquery, no set operation
    - hard        score >= 3 with no subquery, or a lone subquery with score 0
    - extra hard  a set operation (INTERSECT/UNION/EXCEPT), or a subquery
                  combined with score >= 1
    """
    s = sql.strip()
    has_subquery = len(_SUBQUERY_RE.findall(s)) > 1
    has_setop = bool(_SETOP_RE.search(s))
    score = sum(
        bool(rx.search(s))
        for rx in (_WHERE_RE, _GROUP_RE, _ORDER_RE, _HAVING_RE, _JOIN_RE, _AGG_RE)
    )
    if has_setop:
        return "extra hard"
    if has_subquery:
        return "extra hard" if score >= 1 else "hard"
    if score <= 1:
        return "easy"
    if score == 2:
        return "medium"
    return "hard" if score == 3 else "extra hard"


# --- Split conversion --------------------------------------------------------

_FALLBACK_SCHEMA_NOTE = "(schema metadata missing for this database)"


def convert_split(
    examples: list[dict],
    schemas: dict[str, dict],
    sql_key: str = "query",
    question_key: str = "question",
    db_key: str = "db_id",
) -> tuple[list[dict], Counter]:
    """Convert raw Spider examples to instruction records.

    Returns (records, missing_schema_count). Records keep ``db_id`` and
    ``difficulty`` alongside ``instruction``/``output`` — evaluation needs them
    to locate the right SQLite file and to group accuracy by tier.
    """
    records: list[dict] = []
    missing: Counter = Counter()
    for ex in examples:
        db_id = ex[db_key]
        question = ex[question_key]
        sql = ex[sql_key].strip()
        schema = schemas.get(db_id)
        if schema is None:
            missing[db_id] += 1
            schema_str = f"database: {db_id} {_FALLBACK_SCHEMA_NOTE}"
        else:
            schema_str = serialize_schema(schema)
        records.append(
            {
                "db_id": db_id,
                "question": question,
                "instruction": build_user_message(schema_str, question),
                "output": sql,
                "difficulty": estimate_difficulty(sql),
            }
        )
    return records, missing


def print_split_stats(name: str, records: list[dict]) -> None:
    """Print example count, distinct DB count and difficulty distribution."""
    dist = Counter(r["difficulty"] for r in records)
    n_dbs = len({r["db_id"] for r in records})
    print(f"\n[{name}] {len(records)} examples, {n_dbs} distinct databases")
    for tier in DIFFICULTY_TIERS:
        n = dist.get(tier, 0)
        pct = 100.0 * n / len(records) if records else 0.0
        print(f"  {tier:<11} {n:>5}  ({pct:5.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Spider jsonl + tables.json into instruction-tuning jsonl."
    )
    parser.add_argument("--raw-dir", default="data/spider",
                        help="Directory written by scripts/download_spider.py")
    parser.add_argument("--out-dir", default="data/processed",
                        help="Where to write train.jsonl / dev.jsonl")
    parser.add_argument("--tables", default=None,
                        help="Path to tables.json (default: <raw-dir>/tables.json)")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    tables_path = Path(args.tables) if args.tables else raw_dir / "tables.json"
    if not tables_path.exists():
        raise SystemExit(
            f"{tables_path} not found — run scripts/download_spider.py first."
        )
    schemas = load_tables_json(tables_path)
    print(f"Loaded schemas for {len(schemas)} databases from {tables_path}")

    out_dir = Path(args.out_dir)
    for split, out_name in (("train", "train.jsonl"), ("dev", "dev.jsonl")):
        in_path = raw_dir / f"{split}.jsonl"
        if not in_path.exists():
            print(f"[{split}] {in_path} missing — skipping (download first?)")
            continue
        examples = read_jsonl(in_path)
        records, missing = convert_split(examples, schemas)
        out_path = out_dir / out_name
        write_jsonl(records, out_path)
        print_split_stats(split, records)
        if missing:
            print(f"  WARNING: {sum(missing.values())} examples had no schema for: "
                  f"{sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
        print(f"  wrote {len(records)} records -> {out_path}")

    sample = records[0]
    print("\nSample record (dev[0]):")
    print("  db_id     :", sample["db_id"])
    print("  difficulty:", sample["difficulty"])
    print("  instruction (first 500 chars):")
    print("   ", sample["instruction"][:500].replace("\n", "\n    "))
    print("  output    :", sample["output"])


if __name__ == "__main__":
    main()
