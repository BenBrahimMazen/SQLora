"""Modal runner: the whole GPU pipeline on modal.com, from the local CLI.

Pipeline position: cloud alternative to notebooks/finetune_colab.ipynb for
when Colab/Kaggle GPU quotas are exhausted. Same scripts, same commands as the
notebook — executed in Modal containers on a T4, with every stage's output
(data, model-weight cache, predictions, adapter, results) persisted on a
Modal Volume, so stages are individually re-runnable and a failed stage costs
only itself.

Setup (one-time, local):
    pip install modal
    modal token new            # opens the browser; log in with GitHub

Run — logs stream to the local terminal:
    modal run scripts/modal_pipeline.py --stage all
    modal run scripts/modal_pipeline.py --stage smoke     # 20-question + 10-step checks
    modal run scripts/modal_pipeline.py --stage train     # any single stage

Stages (in order): data, smoke, baseline_zeroshot, baseline_fewshot, train,
predict, evaluate, collect. All state lives on the Volume "text2sql-lora-state"
(data/, hf/ model cache, preds/, outputs/, results/) and is committed
automatically when each stage's container exits.

After `collect`, bring the artifacts back to the repo root:
    modal volume get text2sql-lora-state qlora_artifacts.zip .

Cost note: ~5-6 T4-hours for the full pipeline — well inside Modal's free
monthly compute credits.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = "/root/SQLora"  # repo location inside each container
VOL = "/root/vol"  # Modal Volume mount — persistent state across stages
REPO_URL = "https://github.com/BenBrahimMazen/SQLora.git"
VOLUME_NAME = "text2sql-lora-state"

# The repo's pinned stack, parsed client-side so image builds never depend on
# the container having a copy of requirements.txt first. Inline comments are
# stripped (pip would accept them, but be explicit).
_reqs = []
for line in (HERE.parent / "requirements.txt").read_text(encoding="utf-8").splitlines():
    line = line.split("#", 1)[0].strip()
    if line:
        _reqs.append(line)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(*_reqs)
    # Model weights land on the Volume too, so the ~6 GB download happens once.
    .env({"HF_HOME": f"{VOL}/hf"})
)

app = modal.App("text2sql-lora")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

GPU_KWARGS = {"image": image, "gpu": "T4", "timeout": 2 * 60 * 60, "volumes": {VOL: volume}}
CPU_KWARGS = {"image": image, "timeout": 60 * 60, "volumes": {VOL: volume}}


def _shell(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def _prepare_repo() -> None:
    """Clone the repo (once per container) and symlink state dirs to the Volume.

    The pipeline scripts use relative paths (data/, preds/, outputs/, results/),
    so symlinking those names at the repo root onto the Volume makes every
    stage's output persistent with zero changes to the scripts themselves.
    """
    if not Path(ROOT, ".git").exists():
        subprocess.run(["git", "clone", REPO_URL, ROOT], check=True)
    for name in ("data", "preds", "outputs", "results"):
        target = Path(VOL, name)
        target.mkdir(parents=True, exist_ok=True)
        link = Path(ROOT, name)
        if link.is_symlink():
            continue
        if link.exists():  # placeholder content from the clone (data/README.md)
            link.unlink() if link.is_file() else shutil.rmtree(link)
        link.symlink_to(target, target_is_directory=True)


@app.function(**CPU_KWARGS)
def stage_data() -> None:
    """Download real Spider (~210 MB) and build instruction-format files."""
    _prepare_repo()
    _shell(["python", "scripts/download_spider.py"])
    _shell(["python", "src/preprocessing.py"])


@app.function(**GPU_KWARGS)
def stage_smoke() -> None:
    """20-question inference + 10 optimizer steps, before committing hours."""
    _gpu_banner()
    _prepare_repo()
    _shell(["python", "src/baseline_prompting.py", "--limit", "20",
            "--out", "preds/smoke_baseline.jsonl"])
    _shell(["python", "src/train_qlora.py", "--max-steps", "10",
            "--output-dir", "outputs/smoke_run"])


@app.function(**GPU_KWARGS)
def stage_baseline_zeroshot() -> None:
    """All 1,034 dev questions, zero-shot, greedy decoding."""
    _prepare_repo()
    _shell(["python", "src/baseline_prompting.py", "--out", "preds/baseline_zeroshot.jsonl"])


@app.function(**{**GPU_KWARGS, "timeout": 4 * 60 * 60})
def stage_baseline_fewshot() -> None:
    """Few-shot baseline: 3 fixed train exemplars per question."""
    _prepare_repo()
    _shell(["python", "src/baseline_prompting.py", "--few-shot-k", "3",
            "--out", "preds/baseline_fewshot.jsonl"])


@app.function(**{**GPU_KWARGS, "timeout": 6 * 60 * 60})
def stage_train() -> None:
    """QLoRA fine-tuning, 3 epochs (~2 h on a T4)."""
    _prepare_repo()
    _shell(["python", "src/train_qlora.py", "--config", "configs/default.yaml"])


@app.function(**{**GPU_KWARGS, "timeout": 3 * 60 * 60})
def stage_predict() -> None:
    """Fine-tuned predictions over the dev set, from the saved adapter."""
    _prepare_repo()
    _shell(["python", "src/baseline_prompting.py",
            "--adapter", "outputs/qlora_run/final_adapter",
            "--out", "preds/qlora.jsonl"])


@app.function(**CPU_KWARGS)
def stage_evaluate() -> None:
    """Exact match + execution accuracy; writes results/summary.csv + chart."""
    _prepare_repo()
    _shell(["python", "src/evaluate.py",
            "--pred", "baseline-zeroshot=preds/baseline_zeroshot.jsonl",
            "--pred", "baseline-fewshot=preds/baseline_fewshot.jsonl",
            "--pred", "qlora=preds/qlora.jsonl"])


@app.function(**CPU_KWARGS)
def stage_collect() -> None:
    """Zip adapter, config, loss history, predictions, results onto the Volume."""
    include = [
        "outputs/qlora_run/final_adapter",
        "outputs/qlora_run/log_history.json",
        "outputs/qlora_run/resolved_config.yaml",
        "preds",
        "results",
    ]
    out = Path(VOL, "qlora_artifacts.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in include:
            path = Path(VOL, rel)
            if not path.exists():
                print(f"skipping missing {rel}")
                continue
            files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
            for p in files:
                zf.write(p, p.relative_to(VOL))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print(f"retrieve with: modal volume get {VOLUME_NAME} qlora_artifacts.zip .")


def _gpu_banner() -> None:
    import torch

    assert torch.cuda.is_available(), "no GPU assigned to this container"
    print("GPU:", torch.cuda.get_device_name(0))


STAGES = {
    "data": stage_data,
    "smoke": stage_smoke,
    "baseline_zeroshot": stage_baseline_zeroshot,
    "baseline_fewshot": stage_baseline_fewshot,
    "train": stage_train,
    "predict": stage_predict,
    "evaluate": stage_evaluate,
    "collect": stage_collect,
}
ORDER = list(STAGES)


@app.local_entrypoint()
def main(stage: str = "all") -> None:
    """modal run scripts/modal_pipeline.py --stage <all|data|smoke|...>"""
    key = stage.replace("-", "_")
    todo = ORDER if key == "all" else [key]
    if key != "all" and key not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from: all, {', '.join(ORDER)}")
    for name in todo:
        print(f"\n=== stage: {name} ===", flush=True)
        STAGES[name].remote()
