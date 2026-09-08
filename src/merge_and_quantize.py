"""Merge a trained LoRA adapter into the base model and export to GGUF.

Pipeline position: after src/train_qlora.py. Produces
    outputs/<name>/merged/      full merged HF model (transformers format)
    outputs/<name>/gguf/        model-f16.gguf (+ optional quantized variant)

The merged model is what src/baseline_prompting.py and src/demo_app.py load
for fine-tuned inference on GPU; the GGUF is for CPU inference via llama.cpp
(e.g. running the Streamlit demo on a laptop without a GPU).

GGUF export shells out to llama.cpp, which is NOT a pip package — point
--llama-cpp-dir at a checkout (or set LLAMA_CPP_DIR). Everything the script
needs is printed if it is missing:

    git clone https://github.com/ggml-org/llama.cpp
    cd llama.cpp && pip install -r requirements.txt && cmake -B build && cmake --build build

Only the merge step runs by default; add --gguf for the llama.cpp export and
--gguf-type q4_k_m (say) for a quantized variant via llama-quantize. The
quantize binary can come from a source build (llama.cpp/build/bin) or a
prebuilt release zip (--llama-quantize / LLAMA_QUANTIZE_BIN). Conversion and
quantization are skipped when their output file already exists, so re-running
with a different --gguf-type costs only the new variant.

Examples:
    python src/merge_and_quantize.py --adapter outputs/completion_only_run/final_adapter --gguf
    python src/merge_and_quantize.py --adapter outputs/completion_only_run/final_adapter \
        --skip-merge --gguf --gguf-type q4_k_m --out-dir outputs/completion_only_run
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.config_utils import deep_update, get, load_yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML config (e.g. configs/default.yaml)")
    parser.add_argument("--base-model", default=None,
                        help="Base model id/path (must match training; default from config)")
    parser.add_argument("--adapter", required=True, help="Trained adapter dir (or a checkpoint)")
    parser.add_argument("--out-dir", default=None,
                        help="Output dir (default: outputs/merged_<adapter-stem>)")
    parser.add_argument("--gguf", action="store_true", help="Also export GGUF via llama.cpp")
    parser.add_argument("--gguf-type", default="f16",
                        help="GGUF variant: f16 (default), q8_0, q4_k_m, ...")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Reuse the existing merged/ dir instead of re-merging")
    parser.add_argument("--llama-cpp-dir", default=None,
                        help="Path to a llama.cpp checkout (or set LLAMA_CPP_DIR)")
    parser.add_argument("--llama-quantize", default=None,
                        help="Path to a llama-quantize binary, e.g. from a prebuilt "
                             "llama.cpp release zip (or set LLAMA_QUANTIZE_BIN)")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    cfg: dict = {}
    default_cfg_path = ROOT / "configs" / "default.yaml"
    if default_cfg_path.exists():
        cfg = load_yaml(default_cfg_path)
    if args.config:
        cfg = deep_update(cfg, load_yaml(args.config))
    return deep_update(cfg, {"model": {"base_model": args.base_model}})


def merge_adapter(base_model: str, adapter: Path, out_dir: Path) -> None:
    """Load base + adapter in fp16 on CPU, merge, and save a standalone model."""
    print(f"Loading base model '{base_model}' (fp16, CPU) ...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter))
    print("Merging adapter weights ...")
    model = model.merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(out_dir))
    print(f"Merged model -> {out_dir}")


def _find_llama_cpp(hint: str | None) -> Path | None:
    """Locate a llama.cpp checkout: explicit hint, LLAMA_CPP_DIR, ./llama.cpp."""
    candidates = []
    if hint:
        candidates.append(Path(hint))
    env = os.environ.get("LLAMA_CPP_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(ROOT / "llama.cpp")
    for cand in candidates:
        if cand.is_dir() and any(cand.glob("convert_hf_to_gguf.py")):
            return cand
    return None


def _find_quantize(hint: str | None, llama_cpp: Path | None) -> Path | None:
    """Locate a llama-quantize binary: explicit hint, env, or a llama.cpp build."""
    candidates: list[Path] = []
    if hint:
        candidates.append(Path(hint))
    env = os.environ.get("LLAMA_QUANTIZE_BIN")
    if env:
        candidates.append(Path(env))
    if llama_cpp:
        candidates.extend(llama_cpp.glob("build/bin/*quantize*"))   # source build
        candidates.extend(llama_cpp.glob("*quantize*"))             # prebuilt release zip
        candidates.extend(llama_cpp.glob("bin/*quantize*"))
    for cand in candidates:
        if cand.is_file() and cand.suffix in ("", ".exe"):
            return cand
    return None


def export_gguf(merged_dir: Path, out_dir: Path, gguf_type: str,
                llama_cpp: Path | None, quantize_hint: str | None) -> None:
    """Convert the merged model to GGUF; optionally quantize with llama-quantize."""
    if llama_cpp is None:
        print(
            "\nGGUF export skipped — llama.cpp not found. To enable:\n"
            "  git clone https://github.com/ggml-org/llama.cpp\n"
            "  cd llama.cpp && pip install -r requirements.txt\n"
            "  cmake -B build && cmake --build build -j\n"
            "then re-run with --llama-cpp-dir /path/to/llama.cpp"
        )
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)
    f16_path = out_dir / "model-f16.gguf"
    if f16_path.exists():
        print(f"{f16_path} already exists — skipping conversion")
    else:
        print(f"Converting to GGUF (f16) -> {f16_path}")
        subprocess.run(
            [sys.executable, str(llama_cpp / "convert_hf_to_gguf.py"),
             str(merged_dir), "--outfile", str(f16_path), "--outtype", "f16"],
            check=True,
        )

    if gguf_type and gguf_type.lower() not in ("f16", "f32"):
        quant_bin = _find_quantize(quantize_hint, llama_cpp)
        if quant_bin is None:
            print("WARNING: llama-quantize binary not found (build it, or pass "
                  f"--llama-quantize) — skipping {gguf_type} quantization "
                  "(f16 GGUF is ready).")
            return
        quant_path = out_dir / f"model-{gguf_type}.gguf"
        if quant_path.exists():
            print(f"{quant_path} already exists — skipping quantization")
            return
        print(f"Quantizing f16 -> {gguf_type} -> {quant_path}")
        subprocess.run([str(quant_bin), str(f16_path), str(quant_path), gguf_type],
                       check=True)


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)

    base_model = get(cfg, "model.base_model", "Qwen/Qwen2.5-Coder-3B-Instruct")
    adapter = Path(args.adapter)
    if not adapter.exists():
        raise SystemExit(f"Adapter not found: {adapter}")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "outputs" / f"merged_{adapter.stem}"
    merged_dir = out_dir / "merged"

    if args.skip_merge:
        if not (merged_dir / "config.json").exists():
            raise SystemExit(f"--skip-merge set but no merged model found at {merged_dir}")
        print(f"Reusing merged model at {merged_dir}")
    else:
        merge_adapter(base_model, adapter, merged_dir)

    if args.gguf:
        export_gguf(merged_dir, out_dir / "gguf", args.gguf_type,
                    _find_llama_cpp(args.llama_cpp_dir), args.llama_quantize)
        print("\nGGUF ready — run the demo on CPU with llama.cpp, or:")
        print(f"  llama-server -m {out_dir / 'gguf' / f'model-{args.gguf_type}.gguf'}")


if __name__ == "__main__":
    main()
