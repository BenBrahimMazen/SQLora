# Data directory

Nothing in this directory is committed to git. All data is fetched at runtime
from public sources by `scripts/download_spider.py`.

## How to populate

From the project root:

```bash
python scripts/download_spider.py
```

This downloads two things:

1. **Questions + gold SQL** — the official Spider dataset from HuggingFace
   (`datasets.load_dataset("xlangai/spider")`, parquet copy of the official
   release: 7,000 train / 1,034 validation examples; the bare id `"spider"`
   is tried as a fallback for older `datasets` versions).
2. **SQLite databases + schemas** — `spider_data.zip` from
   `https://huggingface.co/datasets/HAL-9001/spider-databases`, which packages
   the original Spider release's `database/` folder (166 `.sqlite` files) and
   `tables.json` (schema metadata used to build prompts).

After the script finishes you should see:

```
data/
└── spider/
    ├── train.jsonl        # db_id, question, query, difficulty  (from HF)
    ├── dev.jsonl          # same fields, dev split
    ├── tables.json        # schema metadata (tables, columns, types, PKs, FKs)
    └── database/          # <db_id>/<db_id>.sqlite — used by execution_eval
```

`src/preprocessing.py` then converts `train.jsonl`/`dev.jsonl` + `tables.json`
into instruction-tuning format under `data/processed/`.

Useful flags: `--skip-databases`, `--databases-only`, `--force`.

## Size

Expect roughly ~210 MB downloaded (206 MB zip + parquet questions), ~700 MB
extracted on disk. The dataset is CC BY-SA 4.0 (original Spider release, Yale
LILY) — keep the attribution if you republish results.
