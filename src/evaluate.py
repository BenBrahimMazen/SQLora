"""Orchestrate final evaluation: exact match + execution accuracy, compared.

Pipeline position: last step of the modeling pipeline. For each prediction
file produced by src/baseline_prompting.py (baseline AND fine-tuned models):

    1. exact match       — src/sql_normalization.sql_exact_match
    2. execution accuracy — src/execution_eval.evaluate_predictions (safe
       execution against the real Spider SQLite databases)

then writes:

    results/summary.csv                       one row per model variant
    results/<name>.exec_results.jsonl         per-example execution results
    results/accuracy_by_difficulty.png        grouped bar chart by tier

The CSV columns match the README's Results table exactly, so filling the README
is a copy-paste after real runs. Nothing here invents numbers — models that
were never run simply have no row.

Usage (repeat --pred for each model variant):
    python src/evaluate.py --pred baseline=preds/base.jsonl --pred qlora=preds/qlora.jsonl
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config_utils import deep_update, get, load_yaml  # noqa: E402
from src.execution_eval import (  # noqa: E402
    clean_prediction,
    evaluate_predictions,
    print_summary,
)
from src.preprocessing import DIFFICULTY_TIERS, read_jsonl, write_jsonl  # noqa: E402
from src.sql_normalization import sql_exact_match  # noqa: E402

# --- chart palette: colorblind-safe categorical colors, assigned in fixed
# --- slot order per series (never cycled or re-derived)
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

TIERS = list(DIFFICULTY_TIERS) + ["overall"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default=None, help="YAML config (execution section)")
    parser.add_argument("--gold", default=None, help="Gold jsonl (default from config)")
    parser.add_argument("--db-dir", default=None)
    parser.add_argument("--pred", action="append", required=True, metavar="NAME=PATH",
                        help="Predictions jsonl + label, repeatable")
    parser.add_argument("--out-dir", default=None, help="Default: results/")
    parser.add_argument("--limit", type=int, default=None,
                        help="Evaluate only the first N examples (must match every pred file)")
    parser.add_argument("--order-sensitive", action="store_true",
                        help="Order-sensitive execution comparison (default: multiset)")
    parser.add_argument("--timeout", type=float, default=None, help="Per-query timeout (s)")
    parser.add_argument("--no-chart", action="store_true")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    cfg: dict = {}
    default_cfg_path = ROOT / "configs" / "default.yaml"
    if default_cfg_path.exists():
        cfg = load_yaml(default_cfg_path)
    if args.config:
        cfg = deep_update(cfg, load_yaml(args.config))
    return cfg


def evaluate_one(name: str, pred_path: Path, gold_records: list[dict],
                 db_dir: str, timeout_s: float, order_sensitive: bool,
                 out_dir: Path, max_rows: int) -> dict:
    """EM + EX for one prediction file; writes per-example jsonl; returns metrics."""
    predictions = read_jsonl(pred_path)
    if len(predictions) != len(gold_records):
        raise SystemExit(
            f"[{name}] {pred_path} has {len(predictions)} predictions but gold has "
            f"{len(gold_records)} — re-run inference (or pass matching --limit)."
        )

    n_em = sum(
        1
        for p, g in zip(predictions, gold_records)
        if sql_exact_match(clean_prediction(p.get("predicted", "")), g.get("output", g.get("query", "")))
    )

    per_example, summary = evaluate_predictions(
        predictions, gold_records, db_dir,
        timeout_s=timeout_s, order_sensitive=order_sensitive, max_rows=max_rows,
    )
    detail_path = out_dir / f"{name}.exec_results.jsonl"
    write_jsonl(per_example, detail_path)

    metrics = {
        "name": name,
        "n": summary["n"],
        "exact_match": round(n_em / len(gold_records), 4) if gold_records else 0.0,
        "execution_accuracy": summary["execution_accuracy"],
        "pred_errors": summary["pred_errors"],
        "gold_errors": summary["gold_errors"],
        "by_difficulty": summary["by_difficulty"],
        "detail_path": str(detail_path),
    }
    print_summary(summary, title=f"{name} — execution accuracy")
    print(f"  exact match: {100 * metrics['exact_match']:.2f}%")
    return metrics


def write_summary_csv(metrics: list[dict], path: Path) -> None:
    """CSV with the same columns as the README Results table (percentages)."""
    header = ["Model", "Exact Match", "Execution Accuracy (overall)",
              "Execution Accuracy (easy)", "Execution Accuracy (medium)",
              "Execution Accuracy (hard)", "Execution Accuracy (extra hard)"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for m in metrics:
            by = m["by_difficulty"]

            def pct(tier: str) -> str:
                if tier == "overall":
                    return f"{100 * m['execution_accuracy']:.1f}"
                acc = by.get(tier, {}).get("accuracy")
                return f"{100 * acc:.1f}" if acc is not None else ""

            writer.writerow(
                [m["name"], f"{100 * m['exact_match']:.1f}", pct("overall"),
                 pct("easy"), pct("medium"), pct("hard"), pct("extra hard")]
            )


def make_chart(metrics: list[dict], gold_records: list[dict], path: Path,
               order_sensitive: bool) -> None:
    """Grouped bar chart: accuracy by difficulty tier, one series per model.

    Static PNG following the project chart spec: categorical colors in fixed
    slot order, thin bars with a small surface gap, recessive hairline grid,
    legend for the series, direct value labels in ink (never series color).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tier_counts = {tier: 0 for tier in DIFFICULTY_TIERS}
    for g in gold_records:
        d = g.get("difficulty")
        if d in tier_counts:
            tier_counts[d] += 1

    n_models = len(metrics)
    colors = SERIES_COLORS[:n_models]
    if n_models > len(SERIES_COLORS):  # fold — never invent a 9th hue
        raise SystemExit("Chart supports at most 8 model variants.")

    x = range(len(TIERS))
    slot = 0.8 / n_models
    bar_w = slot * 0.9  # thin marks with a small surface gap between fills

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for i, m in enumerate(metrics):
        by = m["by_difficulty"]
        values = []
        for tier in TIERS:
            if tier == "overall":
                values.append(m["execution_accuracy"])
            else:
                acc = by.get(tier, {}).get("accuracy")
                values.append(acc if acc is not None else 0.0)
        xs = [xi - 0.4 + slot * (i + 0.5) for xi in x]
        bars = ax.bar(xs, [100 * v for v in values], width=bar_w,
                      color=colors[i], label=m["name"], edgecolor="none")
        for rect, v, tier in zip(bars, values, TIERS):
            if v is None or (tier != "overall" and by.get(tier, {}).get("accuracy") is None):
                continue  # tier absent in this (subset) run
            ax.annotate(f"{100 * v:.1f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.5, color=INK_SECONDARY)

    labels = [
        t if t == "overall" else f"{t}\nn={tier_counts.get(t, 0)}" for t in TIERS
    ]
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)", fontsize=9.5, color=INK_SECONDARY)
    ax.set_title("Text-to-SQL accuracy by Spider difficulty tier",
                 fontsize=12, color=INK, pad=12, loc="left")
    n_note = metrics[0]["n"] if metrics else 0
    match_note = "ordered" if order_sensitive else "order-insensitive"
    ax.annotate(f"Spider dev · n={n_note} per model · execution match ({match_note}) · exact match reported in summary.csv",
                xy=(0, 1.02), xycoords="axes fraction", fontsize=8, color=MUTED)

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"chart -> {path}")


def print_console_table(metrics: list[dict]) -> None:
    """Console mirror of the CSV (so terminal runs see the comparison)."""
    print("\n=== Comparison (accuracy %) ===")
    header = f"{'model':<20} {'EM':>6} {'EX':>6} {'easy':>6} {'medium':>6} {'hard':>6} {'extra':>6}"
    print(header)
    print("-" * len(header))
    for m in metrics:
        by = m["by_difficulty"]

        def cell(t: str) -> str:
            if t == "overall":
                return f"{100 * m['execution_accuracy']:6.1f}"
            acc = by.get(t, {}).get("accuracy")
            return f"{100 * acc:6.1f}" if acc is not None else "   n/a"

        print(f"{m['name']:<20} {100 * m['exact_match']:6.1f} {cell('overall')} "
              f"{cell('easy')} {cell('medium')} {cell('hard')} {cell('extra hard')}")


def main() -> None:
    args = parse_args()
    cfg = resolve_config(args)

    gold_file = args.gold or get(cfg, "data.dev_file", "data/processed/dev.jsonl")
    db_dir = args.db_dir or get(cfg, "data.db_dir", "data/spider/database")
    out_dir = Path(args.out_dir or get(cfg, "paths.results_dir", "results"))
    timeout_s = args.timeout if args.timeout is not None else float(get(cfg, "execution.timeout_s", 10.0))
    order_sensitive = args.order_sensitive or bool(get(cfg, "execution.order_sensitive", False))
    max_rows = int(get(cfg, "execution.max_rows", 100_000))
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_records = read_jsonl(gold_file)
    if args.limit is not None:
        gold_records = gold_records[: args.limit]
    print(f"Gold: {len(gold_records)} examples from {gold_file}")

    preds: list[tuple[str, Path]] = []
    for spec in args.pred:
        name, _, path = spec.partition("=")
        if not name or not path:
            raise SystemExit(f"--pred expects NAME=PATH, got: {spec!r}")
        preds.append((name, Path(path)))

    metrics = [
        evaluate_one(name, path, gold_records, db_dir, timeout_s,
                     order_sensitive, out_dir, max_rows)
        for name, path in preds
    ]

    csv_path = out_dir / "summary.csv"
    write_summary_csv(metrics, csv_path)
    print(f"\nsummary -> {csv_path}")

    if not args.no_chart:
        make_chart(metrics, gold_records,
                   out_dir / "accuracy_by_difficulty.png", order_sensitive)

    print_console_table(metrics)
    print("\nFill the README Results table from results/summary.csv "
          "(only after real runs — no placeholder numbers).")


if __name__ == "__main__":
    main()
