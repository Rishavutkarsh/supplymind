from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


SERIES_ORDER = [
    ("center_sft", "center SFT", "#2563eb"),
    ("center_grpo", "center GRPO", "#7c3aed"),
    ("warehouse_sft", "warehouse SFT", "#0f766e"),
    ("warehouse_grpo", "warehouse GRPO", "#047857"),
    ("center", "center", "#2563eb"),
    ("warehouse", "warehouse", "#0f766e"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create static SupplyMind training/eval plots.")
    parser.add_argument("--input", type=Path, default=Path("results/training_dashboard.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def value(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = row.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def active_series(data: dict[str, Any]) -> list[tuple[str, str, str]]:
    series = data.get("training_series", {})
    active = [item for item in SERIES_ORDER if item[0] in series]
    if "center_sft" in series or "center_grpo" in series or "warehouse_sft" in series or "warehouse_grpo" in series:
        return [item for item in active if item[0].endswith(("_sft", "_grpo"))]
    return active


def line_plot(data: dict[str, Any], y_keys: tuple[str, ...], title: str, ylabel: str, output: Path) -> None:
    series = data.get("training_series", {})
    plt.figure(figsize=(10, 5.2))
    plotted = False
    for key, label, color in active_series(data):
        rows = series.get(key, {}).get("steps", [])
        xs: list[float] = []
        ys: list[float] = []
        for idx, row in enumerate(rows, start=1):
            y = value(row, *y_keys)
            if y is None:
                continue
            xs.append(value(row, "step", "global_step") or idx)
            ys.append(y)
        if xs:
            plt.plot(xs, ys, label=label, color=color, linewidth=2)
            plotted = True
    plt.title(title)
    plt.xlabel("training step")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    if plotted:
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No series available", ha="center", va="center", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def invalid_plot(data: dict[str, Any], output: Path) -> None:
    series = data.get("training_series", {})
    labels: list[str] = []
    payloads: list[float] = []
    actions: list[float] = []
    for key, label, _color in active_series(data):
        batches = series.get(key, {}).get("reward_batches", [])
        if not batches:
            continue
        labels.append(label)
        payloads.append(sum(value(row, "invalid_payloads") or 0 for row in batches))
        actions.append(sum(value(row, "invalid_actions") or 0 for row in batches))
    plt.figure(figsize=(10, 5.2))
    if labels:
        xs = range(len(labels))
        plt.bar([x - 0.18 for x in xs], payloads, width=0.36, label="invalid payloads", color="#c2410c")
        plt.bar([x + 0.18 for x in xs], actions, width=0.36, label="invalid env actions", color="#b7791f")
        plt.xticks(list(xs), labels, rotation=20, ha="right")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No invalid-action diagnostics available", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title("Invalid Payloads / Actions")
    plt.ylabel("count across logged reward batches")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def heldout_plot(data: dict[str, Any], output: Path) -> None:
    comparisons = data.get("comparisons", [])
    roles = ["center", "warehouse"]
    variants = [("base", "#64748b"), ("sft", "#2563eb"), ("grpo", "#7c3aed")]
    groups: list[tuple[str, str, str]] = []
    for role in roles:
        groups.append((role, "global", "mean_global_score"))
        groups.append((role, "role", "mean_center_role_score" if role == "center" else "mean_warehouse_role_score"))

    plt.figure(figsize=(11, 5.5))
    group_xs = list(range(len(groups)))
    width = 0.22
    any_rows = False
    for offset, (variant, color) in zip([-width, 0, width], variants, strict=True):
        ys = []
        for role, _metric, key in groups:
            row = next((item for item in comparisons if item.get("role") == role and item.get("label") == variant), None)
            ys.append(value(row or {}, key) if row else None)
        xs = [x + offset for x, y in zip(group_xs, ys, strict=True) if y is not None]
        vals = [y for y in ys if y is not None]
        if vals:
            any_rows = True
            plt.bar(xs, vals, width=width, label=variant.upper(), color=color)

    if any_rows:
        plt.xticks(group_xs, [f"{role}\n{metric}" for role, metric, _key in groups])
        plt.ylim(0, 1)
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No held-out comparisons available", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title("Held-out Scores: Base vs SFT vs GRPO")
    plt.ylabel("normalized score")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    data = load_json(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    line_plot(data, ("loss",), "Loss Over Step", "loss", args.output_dir / "loss.png")
    line_plot(data, ("reward", "rewards/reward_completions/mean"), "Reward Over Step", "reward", args.output_dir / "reward.png")
    line_plot(data, ("completions/clipped_ratio", "clipped_ratio"), "Clipped Ratio Over Step", "clipped ratio", args.output_dir / "clipped_ratio.png")
    line_plot(data, ("completions/mean_length", "completion_length", "mean_completion_length"), "Completion Length Over Step", "tokens", args.output_dir / "completion_length.png")
    invalid_plot(data, args.output_dir / "invalids.png")
    heldout_plot(data, args.output_dir / "heldout_comparison.png")
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
