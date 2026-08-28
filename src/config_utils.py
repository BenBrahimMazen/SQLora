"""Tiny shared config helpers: YAML loading + dot-path access + deep merge.

Pipeline position: support module for the entry-point scripts
(train_qlora.py, baseline_prompting.py, merge_and_quantize.py, evaluate.py,
demo_app.py). Those scripts all follow the same rule:

    defaults in configs/default.yaml  <  values in --config YAML  <  CLI flags

so ablations can be a one-line CLI change or a copied YAML file — never an
edit to the source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file; empty file -> {} (so callers can apply defaults)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def deep_update(base: dict, patch: dict) -> dict:
    """Recursively merge `patch` into `base` (in place) and return `base`.

    None values in `patch` do NOT overwrite — that is how "CLI flag not given"
    is represented by the argparse layers in the entry-point scripts.
    """
    for key, value in patch.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Read a nested value with a dot-separated path: get(cfg, "lora.r")."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
