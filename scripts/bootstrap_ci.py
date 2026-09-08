"""Bootstrapped confidence intervals for the committed benchmark results.

Pipeline position: after src/evaluate.py. The per-query execution outcomes in
results/*.exec_results.jsonl are the committed evidence behind the Results
table; this script resamples them to show how much of a row-to-row difference
could be plain sampling noise. Single-run intervals resample one model's
outcomes; model-vs-model comparisons resample the SAME questions for both
models (the paired test, honest when every variant answered the identical
dev set — correlated errors cancel, so paired intervals are much tighter
than the single-run ones).

Runs on the standard library only, with a fixed seed, so the printed numbers
are reproducible from the committed evidence:
    python scripts/bootstrap_ci.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Results-table order (file stems in results/; "baseline-fewshot" is the k=3 run).
MODELS = [
    "baseline-zeroshot", "few-shot-k1", "baseline-fewshot",
    "few-shot-k5", "qlora", "qlora-completion-only",
]

# Comparisons that answer "is this difference real?" (numerator - denominator).
DELTAS = [
    ("qlora-completion-only", "qlora"),
    ("qlora", "baseline-zeroshot"),
    ("few-shot-k5", "baseline-zeroshot"),
    ("few-shot-k1", "baseline-zeroshot"),
]


def load_outcomes() -> dict[str, list[int]]:
    """Return {model: [0/1 execution match per question]}, verified aligned."""
    outcomes: dict[str, list[int]] = {}
    ref_questions: list[str] | None = None
    for stem in MODELS:
        path = RESULTS / f"{stem}.exec_results.jsonl"
        records = [json.loads(line) for line in
                   path.read_text(encoding="utf-8").splitlines() if line.strip()]
        outcomes[stem] = [int(bool(r["execution_match"])) for r in records]
        questions = [r["question"] for r in records]
        if ref_questions is None:
            ref_questions = questions
        elif questions != ref_questions:
            raise SystemExit(f"{path.name} is not question-aligned with {MODELS[0]}")
    return outcomes


def boot_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean -> (mean, lo, hi), all in [0, 1] units.

    Works for single-model 0/1 outcomes and, unchanged, for paired
    per-question differences (-1/0/+1) — resampling the difference list IS
    the paired bootstrap.
    """
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return sum(values) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outcomes = load_outcomes()
    n = len(next(iter(outcomes.values())))
    print(f"{n} dev questions per model, {args.n_boot} bootstrap resamples "
          f"(seed {args.seed})\n")

    print("single-run execution accuracy (unpaired):")
    for stem in MODELS:
        mean, lo, hi = boot_ci([float(v) for v in outcomes[stem]], args.n_boot, args.seed)
        print(f"  {stem:<22} {mean * 100:5.1f}   [{lo * 100:5.1f}, {hi * 100:5.1f}]")

    print("\npaired differences (same questions resampled for both models):")
    for num, den in DELTAS:
        diff = [float(a - b) for a, b in zip(outcomes[num], outcomes[den])]
        mean, lo, hi = boot_ci(diff, args.n_boot, args.seed)
        verdict = "excludes 0" if (lo > 0 or hi < 0) else "includes 0"
        print(f"  {num} - {den:<18} {mean * 100:+5.1f}   "
              f"[{lo * 100:+5.1f}, {hi * 100:+5.1f}]   {verdict}")


if __name__ == "__main__":
    main()
