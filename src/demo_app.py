"""Streamlit demo: natural-language question -> SQL -> live result table.

Pipeline position: the showpiece at the end of the pipeline. Selects one of
the sample Spider databases, serializes its schema with the same code used in
training (src/preprocessing.serialize_schema), generates SQL with the
fine-tuned model (transformers, or llama.cpp with a GGUF from
merge_and_quantize.py), executes it SAFELY against the real SQLite database
(src/execution_eval.run_query: in-memory copy, reads-only, timeout) and shows
the result table — a generated query can neither corrupt the DB nor hang the app.

Run:
    streamlit run src/demo_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config_utils import get, load_yaml
from src.execution_eval import run_query
from src.preprocessing import (
    SYSTEM_PROMPT,
    build_user_message,
    load_tables_json,
    read_jsonl,
    serialize_schema,
)

# Small, readable schemas first — nicest for a live demo. Anything present on
# disk is offered; missing ones are skipped silently.
PREFERRED_DBS = [
    "concert_singer", "pets_1", "car_1", "student_1", "world_1",
    "department_management", "wta_1", "soccer_1",
]

st.set_page_config(page_title="text2sql-lora demo", page_icon="🐬", layout="centered")


@st.cache_data(show_spinner=False)
def cached_schemas(tables_json: str) -> dict[str, dict]:
    return load_tables_json(tables_json)


@st.cache_data(show_spinner=False)
def cached_examples(dev_file: str, train_file: str) -> list[dict]:
    """Gold examples (dev first, then train) to seed suggested questions."""
    records: list[dict] = []
    for path in (dev_file, train_file):
        p = Path(path)
        if p.exists():
            records.extend(read_jsonl(p))
    return records


@st.cache_resource(show_spinner="Loading model (first run downloads weights)...")
def load_transformers_model(model_name: str, adapter_path: str | None):
    from src.baseline_prompting import load_model_and_tokenizer

    return load_model_and_tokenizer(model_name, adapter_path or None)


@st.cache_resource(show_spinner="Loading GGUF model...")
def load_llama_cpp_model(gguf_path: str):
    from llama_cpp import Llama

    return Llama(model_path=gguf_path, n_ctx=4096, verbose=False)


def generate_sql_transformers(model_name: str, adapter_path: str | None,
                              schema_str: str, question: str,
                              max_new_tokens: int, temperature: float | None) -> str:
    from src.baseline_prompting import build_messages, generate_sql

    model, tokenizer = load_transformers_model(model_name, adapter_path)
    # Same prompt construction as training/inference — consistency matters.
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_message(schema_str, question)}]
    return generate_sql(model, tokenizer, messages,
                        max_new_tokens=max_new_tokens, temperature=temperature)


def generate_sql_llama_cpp(gguf_path: str, schema_str: str, question: str,
                           max_new_tokens: int, temperature: float) -> str:
    llm = load_llama_cpp_model(gguf_path)
    out = llm.create_chat_completion(
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": build_user_message(schema_str, question)}],
        max_tokens=max_new_tokens,
        temperature=temperature,  # 0 = greedy, matching the transformers path
    )
    return out["choices"][0]["message"]["content"]


def main() -> None:
    cfg_path = ROOT / "configs" / "default.yaml"
    cfg = load_yaml(cfg_path) if cfg_path.exists() else {}

    st.title("🐬 text2sql-lora")
    st.caption(
        "Ask a question in English about a real Spider database — the fine-tuned "
        "model writes the SQL, and it runs live against the SQLite file."
    )

    with st.sidebar:
        st.header("Model")
        backend = st.selectbox("Backend", ["transformers", "llama_cpp (GGUF)"],
                               help="transformers: HF model (+optional adapter). llama_cpp: GGUF from merge_and_quantize.py — CPU-friendly.")
        model_name = st.text_input(
            "Model (HF id or local path)",
            value=get(cfg, "model.base_model", "Qwen/Qwen2.5-Coder-3B-Instruct"),
        )
        adapter_path = st.text_input(
            "LoRA adapter (optional)", value=get(cfg, "model.adapter_path") or "",
            help="e.g. outputs/qlora_run/final_adapter — leave empty for the base model",
        )
        max_new_tokens = st.number_input("Max new tokens", 64, 1024, 256, 32)
        temperature = st.slider("Temperature (0 = greedy)", 0.0, 1.5, 0.0, 0.05)
        st.header("Data")
        tables_json = st.text_input("tables.json", value=get(cfg, "data.tables_json", "data/spider/tables.json"))
        db_dir = st.text_input("Spider database dir", value=get(cfg, "data.db_dir", "data/spider/database"))
        timeout_s = st.number_input("Query timeout (s)", 1.0, 60.0, 10.0, 1.0)

    db_dir_path = Path(db_dir)
    if not db_dir_path.is_dir():
        st.info("Spider databases not found. Run `python scripts/download_spider.py` first.")
        return
    available = sorted(p.parent.name for p in db_dir_path.glob("*/*.sqlite"))
    if not available:
        st.info(f"No `<db_id>/<db_id>.sqlite` files under {db_dir}.")
        return
    ordered = [d for d in PREFERRED_DBS if d in available] + \
              [d for d in available if d not in PREFERRED_DBS]

    db_id = st.selectbox("Database", ordered)
    db_path = db_dir_path / db_id / f"{db_id}.sqlite"

    schemas = cached_schemas(tables_json)
    schema = schemas.get(db_id)
    if schema is None:
        st.error(f"No schema for '{db_id}' in {tables_json}")
        return

    st.subheader("Schema")
    schema_str = st.text_area("schema", value=serialize_schema(schema), height=180,
                              label_visibility="collapsed")

    # Suggested real questions for this DB, from the actual dataset.
    examples = cached_examples(
        get(cfg, "data.dev_file", "data/processed/dev.jsonl"),
        get(cfg, "data.train_file", "data/processed/train.jsonl"),
    )
    db_examples = [e for e in examples if e.get("db_id") == db_id][:3]
    if db_examples:
        with st.expander(f"Example questions for {db_id} (from the real dataset)"):
            for e in db_examples:
                st.markdown(f"• **{e['question']}**")
                st.code(e["output"], language="sql")

    question = st.text_input(
        "Your question",
        placeholder="e.g. " + (db_examples[0]["question"] if db_examples else "How many ... ?"),
    )

    if st.button("Generate SQL", type="primary", disabled=not question.strip()):
        temp = temperature if temperature > 0 else None
        with st.spinner("Generating SQL ..."):
            try:
                if backend.startswith("llama_cpp"):
                    sql = generate_sql_llama_cpp(model_name, schema_str, question,
                                                 int(max_new_tokens), float(temperature))
                else:
                    sql = generate_sql_transformers(model_name, adapter_path.strip(),
                                                    schema_str, question,
                                                    int(max_new_tokens), temp)
            except Exception as e:  # noqa: BLE001 — surface any load/generation failure
                st.error(f"Generation failed: {e}")
                return
        st.session_state["generated_sql"] = sql.strip()

    sql = st.text_area(
        "SQL (edit before running)",
        value=st.session_state.get("generated_sql", ""),
        height=120,
    )

    if st.button("Execute", disabled=not sql.strip()):
        result = run_query(db_path, sql, timeout_s=timeout_s)
        if result["status"] != "ok":
            st.error(f"Query failed: {result['error']}")
        else:
            rows, columns = result["rows"], result.get("columns") or []
            if not rows:
                st.warning("Query ran successfully but returned no rows.")
            else:
                st.success(f"{len(rows)} rows")
                st.dataframe(pd.DataFrame(rows, columns=columns or None))


main()
