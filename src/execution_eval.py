"""Execution-accuracy evaluation against the real Spider SQLite databases.

Pipeline position: the core evaluation. Consumes a predictions jsonl (from
src/baseline_prompting.py, one record per dev example) plus the gold dev
records (data/processed/dev.jsonl), executes predicted AND gold SQL against
the corresponding Spider database, and compares result sets. Used standalone
or orchestrated from src/evaluate.py (which also adds exact match + charts).

Safety model — a generated query can neither corrupt data nor hang the run:

  1. Each SQLite database is opened READ-ONLY (URI mode=ro) and backed up into
     a private in-memory copy; queries execute against the copy only. An
     in-flight copy per database is cached (small LRU) because dev asks ~6
     questions per database.
  2. An sqlite3 authorizer denies everything that is not a read — INSERT,
     UPDATE, DELETE, CREATE/DROP, ATTACH/DETACH, PRAGMA, transactions — so a
     model that hallucinates "DROP TABLE" just scores a failure.
  3. ATTACH/DETACH/PRAGMA are additionally rejected by keyword scan before
     execution (defense in depth: ATTACH could otherwise touch the filesystem).
  4. A progress handler enforces a per-query wall-clock timeout: an expensive
     query (e.g. runaway cartesian join) aborts with an error, not a hang.
  5. Row fetches are capped (max_rows) so a pathological result cannot exhaust
     memory.
  6. Every failure (syntax error, denied statement, timeout, missing database)
     is recorded as a normal per-example result — the evaluator never crashes
     on bad SQL.

Comparison is order-insensitive multiset equality by default (official-Spider
style), configurable to order-sensitive. Floats compare after rounding to 4
decimals; 3 and 3.0 compare equal (SQLite semantics).

Stdlib only — covered by tests/test_execution_eval.py (in-memory fixtures,
no network, no model).
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

# Allow running as a CLI (python src/execution_eval.py) from the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing import DIFFICULTY_TIERS, read_jsonl, write_jsonl  # noqa: E402

# SQLite authorizer action codes (stable ABI values; the sqlite3 module only
# exposes named constants on Python >= 3.11, so we hardcode the small set we
# allow). Everything not listed here is denied.
_ACT_READ = 20        # SQLITE_READ
_ACT_SELECT = 21      # SQLITE_SELECT
_ACT_FUNCTION = 31    # SQLITE_FUNCTION (max, replace, strftime, ...)
_ACT_RECURSIVE = 33   # SQLITE_RECURSIVE (recursive CTEs)
_ALLOWED_ACTIONS = {_ACT_READ, _ACT_SELECT, _ACT_FUNCTION, _ACT_RECURSIVE}
_SQLITE_OK, _SQLITE_DENY = 0, 1

# Statements that should never run, even on the in-memory copy: ATTACH could
# write to arbitrary files, PRAGMA is out of scope for text-to-SQL.
_FORBIDDEN_KEYWORDS_RE = re.compile(r"\b(attach|detach|pragma)\b", re.IGNORECASE)

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*|\s*```$", re.MULTILINE)

# In-memory database copies cached across queries (dev reuses each DB ~6x).
_MEM_CACHE_MAX = 8
_MEM_CACHE: "OrderedDict[str, sqlite3.Connection]" = OrderedDict()


def clean_prediction(text: str) -> str:
    """Best-effort cleanup of raw model output into a bare SQL string.

    Strips surrounding markdown fences (```sql ... ```), whitespace and a
    trailing semicolon. Applied defensively here so a model that wraps its
    answer in fences isn't penalized twice (inference scripts also extract).
    """
    if not text:
        return ""
    s = text.strip()
    if s.startswith("```"):
        s = _FENCE_RE.sub("", s).strip()
    return s.rstrip(";").strip()


def _authorizer(action: int, *_args) -> int:
    """Allow reads only; deny writes/DDL/ATTACH/transactions (see module docstring)."""
    return _SQLITE_OK if action in _ALLOWED_ACTIONS else _SQLITE_DENY


def _memory_copy(db_path: Path) -> sqlite3.Connection:
    """Open a fresh in-memory copy of db_path (read-only source), LRU-cached."""
    key = str(db_path)
    conn = _MEM_CACHE.get(key)
    if conn is not None:
        _MEM_CACHE.move_to_end(key)
        return conn
    while len(_MEM_CACHE) >= _MEM_CACHE_MAX:
        _old_key, _old = _MEM_CACHE.popitem(last=False)
        _old.close()
    src = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    dst = sqlite3.connect(":memory:")
    src.backup(dst)
    src.close()
    _MEM_CACHE[key] = dst
    return dst


def _evict(db_path: Path) -> None:
    conn = _MEM_CACHE.pop(str(db_path), None)
    if conn is not None:
        conn.close()


def _canonical_value(v) -> str:
    """Canonical string form of one SQLite value for result comparison."""
    if v is None:
        return "<null>"
    if isinstance(v, bool):  # sqlite returns ints, but be safe
        return f"n:{int(v)}"
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v.is_integer():
            v = int(v)  # 3.0 == 3 under SQLite semantics
        return f"n:{round(v, 4) if isinstance(v, float) else v}"
    if isinstance(v, bytes):
        return f"b:{v.hex()}"
    return f"s:{v}"


def canonicalize_rows(rows: list[tuple], order_sensitive: bool = False) -> list[tuple]:
    """Canonicalize a result set: per-value strings; sorted unless order matters."""
    canon = [tuple(_canonical_value(v) for v in row) for row in rows]
    return canon if order_sensitive else sorted(canon)


def run_query(
    db_path: str | Path,
    sql: str,
    timeout_s: float = 10.0,
    max_rows: int = 100_000,
) -> dict:
    """Execute one SQL statement safely; never raises for query-level failures.

    Returns {"status": "ok", "rows": [...]} or {"status": "error", "error": msg}.
    Errors include: empty query, forbidden statement, syntax error, denied
    (non-read) statement, timeout, result too large, missing database file.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return {"status": "error", "error": f"database file not found: {db_path}"}
    query = clean_prediction(sql)
    if not query:
        return {"status": "error", "error": "empty query"}
    if _FORBIDDEN_KEYWORDS_RE.search(query):
        return {"status": "error", "error": "forbidden statement (attach/detach/pragma)"}

    deadline = time.perf_counter() + timeout_s

    def _on_progress() -> int:
        # Returning nonzero aborts the running statement (SQLITE_INTERRUPT).
        return 1 if time.perf_counter() > deadline else 0

    try:
        conn = _memory_copy(db_path)
        conn.set_progress_handler(_on_progress, 2_000)  # check every 2k VM ops
        conn.set_authorizer(_authorizer)
        cur = conn.execute(query)  # single statement; multi-statement raises
        rows: list[tuple] = []
        while True:
            chunk = cur.fetchmany(1_000)
            if not chunk:
                break
            rows.extend(chunk)
            if len(rows) > max_rows:
                return {"status": "error",
                        "error": f"result too large (> {max_rows} rows)"}
        columns = [d[0] for d in cur.description] if cur.description else []
        return {"status": "ok", "rows": rows, "columns": columns}
    except (sqlite3.Error, sqlite3.Warning) as e:
        if time.perf_counter() >= deadline:
            return {"status": "error", "error": f"timeout after {timeout_s:g}s"}
        # The cached copy might be in a weird state after an abort — drop it.
        _evict(db_path)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}


def results_match(pred_result: dict, gold_result: dict, order_sensitive: bool = False) -> bool:
    """True iff both queries executed and their result sets are equivalent."""
    if pred_result.get("status") != "ok" or gold_result.get("status") != "ok":
        return False
    return canonicalize_rows(pred_result["rows"], order_sensitive) == canonicalize_rows(
        gold_result["rows"], order_sensitive
    )


def db_file_for(db_dir: str | Path, db_id: str) -> Path:
    """Spider layout: database/<db_id>/<db_id>.sqlite."""
    return Path(db_dir) / db_id / f"{db_id}.sqlite"


def evaluate_predictions(
    predictions: list[dict],
    gold_records: list[dict],
    db_dir: str | Path,
    timeout_s: float = 10.0,
    order_sensitive: bool = False,
    max_rows: int = 100_000,
) -> tuple[list[dict], dict]:
    """Execute pred vs gold for every example; return per-example rows + summary.

    Predictions and gold are aligned by index; each prediction record supplies
    the predicted SQL ("predicted" or "prediction" key), the gold record
    supplies db_id / difficulty / gold SQL ("output" or "query" key).
    """
    if len(predictions) != len(gold_records):
        raise ValueError(
            f"prediction/gold length mismatch: {len(predictions)} vs {len(gold_records)} "
            "(run inference on the full dev set, or pass matching --limit to both)"
        )

    per_example: list[dict] = []
    by_tier: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "correct": 0})
    n_pred_err = n_gold_err = 0
    n_correct = 0

    for i, (pred, gold) in enumerate(zip(predictions, gold_records)):
        db_id = gold.get("db_id", pred.get("db_id"))
        difficulty = gold.get("difficulty", pred.get("difficulty", "unknown"))
        gold_sql = gold.get("output", gold.get("query", ""))
        predicted_sql = clean_prediction(pred.get("predicted", pred.get("prediction", "")))
        db_path = db_file_for(db_dir, db_id)

        pred_res = run_query(db_path, predicted_sql, timeout_s, max_rows)
        gold_res = run_query(db_path, gold_sql, timeout_s, max_rows)
        match = results_match(pred_res, gold_res, order_sensitive)

        if pred_res["status"] == "error":
            n_pred_err += 1
        if gold_res["status"] == "error":
            n_gold_err += 1
        if match:
            n_correct += 1
        tier = by_tier[difficulty]
        tier["n"] += 1
        tier["correct"] += match

        per_example.append(
            {
                "index": i,
                "db_id": db_id,
                "difficulty": difficulty,
                "question": gold.get("question", pred.get("question")),
                "gold": gold_sql,
                "predicted": predicted_sql,
                "pred_status": pred_res["status"],
                "pred_error": pred_res.get("error"),
                "gold_status": gold_res["status"],
                "gold_error": gold_res.get("error"),
                "execution_match": match,
            }
        )

    n = len(per_example)
    summary = {
        "n": n,
        "execution_accuracy": round(n_correct / n, 4) if n else 0.0,
        "order_sensitive": order_sensitive,
        "timeout_s": timeout_s,
        "pred_errors": n_pred_err,
        "gold_errors": n_gold_err,
        "by_difficulty": {
            tier: {
                "n": stats["n"],
                "accuracy": round(stats["correct"] / stats["n"], 4) if stats["n"] else None,
            }
            for tier, stats in by_tier.items()
        },
    }
    return per_example, summary


def print_summary(summary: dict, title: str = "Execution accuracy") -> None:
    """Human-readable summary: overall accuracy and per-tier breakdown."""
    print(f"\n=== {title} ===")
    print(f"examples          : {summary['n']}")
    print(f"execution accuracy: {100 * summary['execution_accuracy']:.2f}%")
    print(f"pred errors       : {summary['pred_errors']}"
          f"   gold errors: {summary['gold_errors']}"
          f"   (order_sensitive={summary['order_sensitive']}, timeout={summary['timeout_s']}s)")
    by = summary["by_difficulty"]
    for tier in list(DIFFICULTY_TIERS) + sorted(set(by) - set(DIFFICULTY_TIERS)):
        if tier not in by:
            continue
        acc = by[tier]["accuracy"]
        acc_str = f"{100 * acc:6.2f}%" if acc is not None else "  n/a "
        print(f"  {tier:<11} {by[tier]['n']:>5} examples   {acc_str}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execution-accuracy eval of a predictions jsonl against Spider dev."
    )
    parser.add_argument("--predictions", required=True,
                        help="jsonl with predicted SQL per dev example (baseline_prompting.py output)")
    parser.add_argument("--gold", default="data/processed/dev.jsonl",
                        help="Gold jsonl (processed dev.jsonl, or raw dev.jsonl with 'query')")
    parser.add_argument("--db-dir", default="data/spider/database")
    parser.add_argument("--out", default=None,
                        help="Per-example results jsonl (default: <predictions>.exec_results.jsonl)")
    parser.add_argument("--summary-out", default=None, help="Optional summary json path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples (must match predictions)")
    parser.add_argument("--timeout", type=float, default=10.0, help="Per-query timeout (s)")
    parser.add_argument("--order-sensitive", action="store_true",
                        help="Compare result rows in order (default: multiset)")
    parser.add_argument("--max-rows", type=int, default=100_000)
    args = parser.parse_args()

    predictions = read_jsonl(args.predictions)
    gold_records = read_jsonl(args.gold)
    if args.limit is not None:
        predictions = predictions[: args.limit]
        gold_records = gold_records[: args.limit]

    per_example, summary = evaluate_predictions(
        predictions, gold_records, args.db_dir,
        timeout_s=args.timeout, order_sensitive=args.order_sensitive,
        max_rows=args.max_rows,
    )

    out = Path(args.out) if args.out else Path(str(args.predictions) + ".exec_results.jsonl")
    write_jsonl(per_example, out)
    print(f"wrote per-example results -> {out}")
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote summary -> {args.summary_out}")
    print_summary(summary, title=Path(args.predictions).stem)


if __name__ == "__main__":
    main()
