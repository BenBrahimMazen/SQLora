"""text2sql-lora source package.

Modules (see each file's docstring for details):
    preprocessing      Spider examples -> instruction-tuning format (stdlib-only)
    sql_normalization  SQL string normalization for exact-match eval (stdlib-only)
    execution_eval     safe execution-accuracy eval against Spider SQLite DBs (stdlib-only)
    baseline_prompting zero/few-shot inference with any HF model (also used for fine-tuned preds)
    train_qlora        QLoRA fine-tuning (transformers + peft + bitsandbytes + trl)
    merge_and_quantize merge LoRA adapter into base model, export GGUF
    evaluate           orchestrate EM + EX evaluation, comparison table + chart
    demo_app           Streamlit demo app
    config_utils       small YAML config helpers shared by entry-point scripts

The first three modules are intentionally dependency-free (stdlib only) so the
offline test suite (and CI) runs with just pytest — no torch, no network.
"""
