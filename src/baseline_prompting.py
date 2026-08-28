"""Run a HF causal-LM over the Spider dev set (zero-shot or k-shot) and save predictions.

Pipeline position: after src/preprocessing.py (consumes data/processed/*.jsonl),
before src/execution_eval.py / src/evaluate.py.

Despite the name this is the project's general inference entry point:

    - baseline:      --model Qwen/Qwen2.5-Coder-3B-Instruct --few-shot-k 0
    - few-shot:      ... --few-shot-k 3
    - fine-tuned:    --adapter outputs/qlora_run/final_adapter
                     (or --model outputs/merged after merge_and_quantize.py)

Few-shot exemplars are drawn ONCE from train.jsonl with a fixed seed and
reused for every dev question, so runs are deterministic and comparable.
Generation is greedy by default (deterministic benchmarking).

Output jsonl (one record per dev example, aligned by index with dev.jsonl):
    {"db_id", "difficulty", "question", "gold", "predicted", "model", "few_shot_k"}

Example:
    python src/baseline_prompting.py --config configs/default.yaml --out preds/base.jsonl
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.config_utils import deep_update, get, load_yaml  # noqa: E402
from src.execution_eval import clean_prediction  # noqa: E402
from src.preprocessing import SYSTEM_PROMPT, read_jsonl, write_jsonl  # noqa: E402


def load_model_and_tokenizer(
    model_name: str,
    adapter_path: str | None = None,
    device: str | None = None,
):
    """Load a causal LM (+ optional LoRA adapter) for inference.

    Returns (model, tokenizer). Shared with src/demo_app.py.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"Loaded LoRA adapter: {adapter_path}")
    model.eval()
    return model, tokenizer


def build_messages(instruction: str, exemplars: list[dict]) -> list[dict]:
    """Chat messages for one example: system + k (user, assistant) pairs + user.

    `instruction` is the processed record's instruction (schema + question);
    exemplars are processed train records with instruction/output.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in exemplars:
        messages.append({"role": "user", "content": ex["instruction"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": instruction})
    return messages


def sample_few_shot_pool(train_records: list[dict], k: int, seed: int) -> list[dict]:
    """Deterministically sample k exemplars from train (fixed across questions)."""
    if k <= 0:
        return []
    rng = random.Random(seed)
    pool = rng.sample(train_records, min(k, len(train_records)))
    print(f"Few-shot: drew {len(pool)} fixed exemplars from train (seed={seed})")
    return pool


def generate_sql(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 256,
    temperature: float | None = None,
    top_p: float = 0.95,
) -> str:
    """Generate one completion for a chat prompt and return the raw decoded text."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature is not None and temperature > 0
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def extract_sql(raw: str) -> str:
    """Extract a bare SQL string from raw model output (strips fences etc.)."""
    return clean_prediction(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML config (e.g. configs/default.yaml)")
    parser.add_argument("--model", default=None, help="Base model id/path (default from config)")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--train-file", default=None, help="Processed train jsonl (for few-shot)")
    parser.add_argument("--dev-file", default=None, help="Processed dev jsonl to predict on")
    parser.add_argument("--out", default="preds/dev_predictions.jsonl")
    parser.add_argument("--few-shot-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only predict the first N dev examples (smoke runs)")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature; default greedy")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    """configs/default.yaml < --config YAML < CLI flags (None = not given)."""
    cfg: dict = {}
    default_cfg_path = ROOT / "configs" / "default.yaml"
    if default_cfg_path.exists():
        cfg = load_yaml(default_cfg_path)
    if args.config:
        cfg = deep_update(cfg, load_yaml(args.config))
    return deep_update(
        cfg,
        {
            "model": {"base_model": args.model, "adapter_path": args.adapter},
            "data": {"train_file": args.train_file, "dev_file": args.dev_file},
            "few_shot": {"k": args.few_shot_k, "seed": args.seed},
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
        },
    )


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)

    dev_file = get(cfg, "data.dev_file", "data/processed/dev.jsonl")
    train_file = get(cfg, "data.train_file", "data/processed/train.jsonl")
    model_name = get(cfg, "model.base_model")
    adapter = get(cfg, "model.adapter_path")
    k = get(cfg, "few_shot.k", 0)
    seed = get(cfg, "few_shot.seed", 42)
    max_new_tokens = get(cfg, "generation.max_new_tokens", 256)
    temperature = get(cfg, "generation.temperature")
    top_p = get(cfg, "generation.top_p", 0.95)

    dev_records = read_jsonl(dev_file)
    if args.limit is not None:
        dev_records = dev_records[: args.limit]
    print(f"Loaded {len(dev_records)} dev examples from {dev_file}")

    train_records = []
    if k > 0:
        train_records = read_jsonl(train_file)

    model, tokenizer = load_model_and_tokenizer(model_name, adapter)
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device — running on CPU, this will be slow. "
              "Use --limit for smoke tests.")
    exemplars = sample_few_shot_pool(train_records, k, seed)

    model_label = adapter or model_name
    out_path = Path(args.out)
    predictions = []
    for rec in tqdm(dev_records, desc=f"generating ({model_label})"):
        raw = generate_sql(
            model, tokenizer,
            build_messages(rec["instruction"], exemplars),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        predictions.append(
            {
                "db_id": rec["db_id"],
                "difficulty": rec.get("difficulty"),
                "question": rec.get("question"),
                "gold": rec["output"],
                "predicted": extract_sql(raw),
                "model": model_label,
                "few_shot_k": k,
            }
        )

    write_jsonl(predictions, out_path)
    print(f"wrote {len(predictions)} predictions -> {out_path}")
    print("\nSample predictions:")
    for p in predictions[:3]:
        print(f"  [{p['db_id']}] {p['question']}")
        print(f"    gold: {p['gold']}")
        print(f"    pred: {p['predicted'][:200]}")
    print("\nNext: python src/evaluate.py --pred "
          f"{Path(out_path).stem}={out_path}")


if __name__ == "__main__":
    main()
