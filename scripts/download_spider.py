"""Download the Spider dataset: questions, gold SQL, and SQLite databases.

Pipeline position: the very first step. Everything downstream (preprocessing,
training, evaluation, demo) expects ``data/spider/`` to contain:

    train.jsonl, dev.jsonl   — db_id, question, query, difficulty
    tables.json              — schema metadata (tables, columns, types, PKs, FKs)
    database/<db_id>/<db_id>.sqlite — the 166 Spider databases

Sources (public, no authentication; verified 2026-08):

  1. Questions + gold SQL — HuggingFace ``datasets``:
     ``load_dataset("xlangai/spider")`` (parquet copy of the official release
     by the Spider authors; 7,000 train / 1,034 validation). Older releases
     also accepted the bare id ``"spider"``, which is tried as a fallback.
  2. SQLite databases + tables.json — the HuggingFace repo
     ``HAL-9001/spider-databases``, whose ``spider_data.zip`` packages the
     original Spider release's ``database/`` folder and ``tables.json``
     (Spider is CC BY-SA 4.0).

Usage:
    python scripts/download_spider.py                 # everything
    python scripts/download_spider.py --skip-databases
    python scripts/download_spider.py --databases-only
    python scripts/download_spider.py --force         # re-download even if present

Prints basic stats per split: number of examples, number of distinct
databases, and the difficulty distribution.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Local import kept after the sys.path shim so the script works when run as
# `python scripts/download_spider.py` from anywhere.
from src.preprocessing import DIFFICULTY_TIERS, estimate_difficulty, write_jsonl  # noqa: E402

# Questions/SQL. First entry is tried first; the rest are fallbacks.
# Note: "xlangai/spider" is the canonical repo; the bare id "spider" worked with
# older `datasets` releases but current huggingface_hub rejects single-segment
# repo ids — it is kept as a fallback, tried second.
DATASET_ID_CANDIDATES = ["xlangai/spider", "spider"]

# SQLite databases + tables.json (a repackaging of the original spider.zip
# contents; see module docstring).
DEFAULT_DB_ZIP_URL = (
    "https://huggingface.co/datasets/HAL-9001/spider-databases/resolve/main/spider_data.zip"
)

# HF split name -> local file name.
SPLIT_FILES = {"train": "train.jsonl", "validation": "dev.jsonl"}


def download_questions(data_dir: Path, dataset_id: str | None, force: bool) -> None:
    """Download question/SQL pairs via HF datasets and write train/dev jsonl."""
    from datasets import load_dataset  # deferred: heavy import, not needed for --databases-only

    existing = [data_dir / f for f in SPLIT_FILES.values()]
    if not force and all(p.exists() for p in existing):
        print(f"Questions already present ({', '.join(p.name for p in existing)}) — use --force to re-download.")
        return

    candidates = [dataset_id] if dataset_id else DATASET_ID_CANDIDATES
    seen: set[str] = set()
    ds = None
    used_id = None
    for cid in candidates:
        if cid in seen:
            continue
        seen.add(cid)
        try:
            print(f"Loading HuggingFace dataset '{cid}' ...")
            ds = load_dataset(cid)
            used_id = cid
            break
        except Exception as e:  # noqa: BLE001 — try the next mirror
            print(f"  failed: {type(e).__name__}: {e}")
    if ds is None:
        raise SystemExit(
            "Could not load Spider from HuggingFace. Tried: "
            f"{sorted(seen)}. Check your connection, or pass --dataset-id."
        )
    print(f"Using dataset '{used_id}'. Splits: {sorted(ds.keys())}")

    for hf_split, fname in SPLIT_FILES.items():
        # Some mirrors name the dev split "validation", others "dev".
        split = hf_split if hf_split in ds else ("dev" if "dev" in ds else None)
        if split is None:
            print(f"  WARNING: split '{hf_split}' not found in dataset — skipping {fname}")
            continue
        rows = [
            {
                "db_id": r["db_id"],
                "question": r["question"],
                "query": r["query"].strip(),
                "difficulty": estimate_difficulty(r["query"]),
            }
            for r in ds[split]
        ]
        out = data_dir / fname
        write_jsonl(rows, out)
        dist = Counter(r["difficulty"] for r in rows)
        n_dbs = len({r["db_id"] for r in rows})
        print(f"\n[{fname}] {len(rows)} examples, {n_dbs} distinct databases")
        for tier in DIFFICULTY_TIERS:
            n = dist.get(tier, 0)
            pct = 100.0 * n / len(rows) if rows else 0.0
            print(f"  {tier:<11} {n:>5}  ({pct:5.1f}%)")
        print(f"  wrote {out}")


def _download_file(url: str, dest: Path, chunk_mb_log: int = 8) -> None:
    """Stream a file to disk with basic progress logging (requests)."""
    import requests

    with requests.get(url, stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        next_log = chunk_mb_log * 1024 * 1024
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                done += len(chunk)
                if done >= next_log:
                    total_str = f"/{total / 1e6:.0f} MB" if total else ""
                    print(f"  downloaded {done / 1e6:.0f} MB{total_str}")
                    next_log += chunk_mb_log * 1024 * 1024
    print(f"  downloaded {done / 1e6:.0f} MB total -> {dest.name}")


def _relocate_extracted(data_dir: Path, extracted_root: Path) -> None:
    """Move database/ and tables.json from wherever the zip put them to data_dir.

    The known zip extracts to spider_data/database + spider_data/tables.json;
    this also tolerates other nesting so a mirror change doesn't break the run.
    """
    def find_one(pattern: str) -> Path | None:
        matches = sorted(extracted_root.rglob(pattern))
        # Prefer the shallowest match.
        return min(matches, key=lambda p: len(p.parts)) if matches else None

    db_dir = find_one("database")
    tables = find_one("tables.json")
    if db_dir is None or tables is None:
        raise SystemExit(
            "Extracted zip did not contain a 'database' directory and 'tables.json' "
            f"(looked under {extracted_root}). Pass --db-zip-url pointing at a "
            "spider.zip-style archive."
        )
    target_db = data_dir / "database"
    target_tables = data_dir / "tables.json"
    if target_db.exists():
        shutil.rmtree(target_db)
    shutil.move(str(db_dir), str(target_db))
    if target_tables.exists():
        target_tables.unlink()
    shutil.move(str(tables), str(target_tables))
    # Clean up the now-mostly-empty extraction root.
    if extracted_root != data_dir and extracted_root.exists():
        shutil.rmtree(extracted_root, ignore_errors=True)


def download_databases(data_dir: Path, db_zip_url: str, force: bool) -> None:
    """Download spider_data.zip and lay out database/ + tables.json under data_dir."""
    have_db = (data_dir / "database").is_dir()
    have_tables = (data_dir / "tables.json").is_file()
    if not force and have_db and have_tables:
        n = sum(1 for _ in (data_dir / "database").glob("*/*.sqlite"))
        print(f"Databases already present ({n} .sqlite files) — use --force to re-download.")
        return

    print(f"Downloading databases from {db_zip_url} ...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "spider_data.zip"
        _download_file(db_zip_url, zip_path)
        print("Extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        _relocate_extracted(data_dir, Path(tmp))

    n = sum(1 for _ in (data_dir / "database").glob("*/*.sqlite"))
    print(f"Extracted {n} .sqlite databases -> {data_dir / 'database'}")
    print(f"Schema metadata -> {data_dir / 'tables.json'}")


def verify(data_dir: Path) -> None:
    """Cross-check that every db_id referenced by the jsonl files has a database."""
    from src.preprocessing import read_jsonl

    db_dir = data_dir / "database"
    if not db_dir.is_dir():
        print("NOTE: database/ not present — run without --skip-databases for execution eval.")
        return
    available = {p.parent.name for p in db_dir.glob("*/*.sqlite")}
    for fname in ("train.jsonl", "dev.jsonl"):
        path = data_dir / fname
        if not path.exists():
            continue
        db_ids = {r["db_id"] for r in read_jsonl(path)}
        missing = db_ids - available
        status = "OK" if not missing else f"MISSING {len(missing)}: {sorted(missing)[:3]}..."
        print(f"  {fname}: {len(db_ids)} db_ids referenced -> {status}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Spider dataset (questions/SQL + SQLite databases)."
    )
    parser.add_argument("--data-dir", default="data/spider",
                        help="Where to place train/dev jsonl, tables.json, database/")
    parser.add_argument("--dataset-id", default=None,
                        help="HF dataset id (default: try 'spider' then 'xlangai/spider')")
    parser.add_argument("--db-zip-url", default=DEFAULT_DB_ZIP_URL,
                        help="Zip containing database/ + tables.json")
    parser.add_argument("--skip-databases", action="store_true",
                        help="Only download question/SQL jsonl files")
    parser.add_argument("--databases-only", action="store_true",
                        help="Only download the SQLite databases + tables.json")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files already exist")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if not args.databases_only:
        download_questions(data_dir, args.dataset_id, args.force)
    if not args.skip_databases:
        download_databases(data_dir, args.db_zip_url, args.force)

    print("\nVerification:")
    verify(data_dir)
    print("\nDone. Next step: python src/preprocessing.py")


if __name__ == "__main__":
    main()
