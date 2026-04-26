from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib.pyplot as plt


RESULTS = Path("results")
PLOTS = RESULTS / "plots"
OUT_JSON = RESULTS / "submission_evidence.json"
OUT_MD = RESULTS / "submission_evidence.md"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


TRAINING_LOGS = {
    "warehouse_sft_v1": RESULTS / "warehouse_sft_local_action_l40s.log",
    "warehouse_sft_v2": RESULTS / "warehouse_sft_v2.log",
    "warehouse_grpo_v2": RESULTS / "warehouse_grpo_from_sft_v2.log",
    "center_sft_v1": RESULTS / "center_sft_pressure.log",
}

EVAL_LOGS = {
    "warehouse_sft_v1": RESULTS / "warehouse_eval_local_sft.log",
    "warehouse_grpo_v2": RESULTS / "warehouse_eval_grpo_v2.log",
    "warehouse_sft_v2": RESULTS / "warehouse_eval_sft_v1_v2.log",
    "center_sft_v1": RESULTS / "center_eval_pressure_sft_v2.log",
    "center_sft_v2_grpo": RESULTS / "center_eval_sft_v2_grpo.log",
}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def strip(line: str) -> str:
    return ANSI_RE.sub("", line).strip()


def extract_json_objects(line: str) -> list[dict[str, Any]]:
    """Extract JSON dicts from noisy HF progress lines."""
    cleaned = strip(line)
    objects: list[dict[str, Any]] = []
    starts = [idx for idx, ch in enumerate(cleaned) if ch == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(cleaned)):
            ch = cleaned[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        objects.append(parsed)
                    break
    return objects


def parse_training_log(path: Path) -> dict[str, Any]:
    steps: list[dict[str, float]] = []
    reward_batches: list[dict[str, Any]] = []
    json_events: list[dict[str, Any]] = []
    for raw in read_text(path).splitlines():
        line = strip(raw)
        for obj in extract_json_objects(line):
            json_events.append(obj)
            if obj.get("message") == "reward_batch":
                reward_batches.append(obj)
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            parsed = ast.literal_eval(line)
        except (SyntaxError, ValueError):
            continue
        if not isinstance(parsed, dict):
            continue
        if "loss" in parsed and "train_runtime" not in parsed:
            row: dict[str, float] = {}
            for key in ("loss", "reward", "rewards/reward_completions/mean", "completions/clipped_ratio", "completions/mean_length"):
                if key in parsed:
                    try:
                        row[key] = float(parsed[key])
                    except (TypeError, ValueError):
                        pass
            row["step"] = float(len(steps) + 1)
            steps.append(row)
        elif "train_loss" in parsed:
            json_events.append({"message": "trainer_summary", **parsed})

    final_loss = None
    for event in json_events:
        if "train_loss" in event:
            try:
                final_loss = float(event["train_loss"])
            except (TypeError, ValueError):
                pass

    return {
        "path": str(path),
        "exists": path.exists(),
        "steps": steps,
        "reward_batches": reward_batches,
        "events": json_events,
        "step_count": len(steps),
        "first_loss": steps[0].get("loss") if steps else None,
        "last_loss": steps[-1].get("loss") if steps else final_loss,
        "min_loss": min((row["loss"] for row in steps if "loss" in row), default=None),
        "final_train_loss": final_loss,
        "invalid_payloads": sum(int(row.get("invalid_payloads", 0) or 0) for row in reward_batches),
        "invalid_actions": sum(int(row.get("invalid_actions", 0) or 0) for row in reward_batches),
    }


def parse_eval_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in read_text(path).splitlines():
        for obj in extract_json_objects(raw):
            if obj.get("message") == "eval_result":
                rows.append({key: value for key, value in obj.items() if key != "episodes"})
    return rows


def dedupe_eval_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in reversed(rows):
        key = (str(row.get("role")), str(row.get("label")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return list(reversed(out))


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "-"
    return f"{number:.4f}" if abs(number) < 10 else f"{number:.2f}"


def plot_training(training: dict[str, dict[str, Any]]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plotted = False
    for key, label, color in [
        ("warehouse_sft_v1", "warehouse SFT v1 loss", "#0f766e"),
        ("warehouse_sft_v2", "warehouse SFT v2 loss", "#14b8a6"),
        ("warehouse_grpo_v2", "warehouse GRPO loss", "#7c3aed"),
        ("center_sft_v1", "center SFT loss", "#2563eb"),
    ]:
        rows = training.get(key, {}).get("steps", [])
        xs = [row["step"] for row in rows if "loss" in row]
        ys = [row["loss"] for row in rows if "loss" in row]
        if xs:
            plt.plot(xs, ys, label=label, color=color, linewidth=2)
            plotted = True
    plt.title("Training Loss")
    plt.xlabel("logged step")
    plt.ylabel("loss")
    plt.grid(alpha=0.25)
    if plotted:
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No loss rows available", ha="center", va="center", transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig(PLOTS / "submission_training_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    rows = training.get("warehouse_grpo_v2", {}).get("steps", [])
    xs = [row["step"] for row in rows if "reward" in row or "rewards/reward_completions/mean" in row]
    ys = [row.get("reward", row.get("rewards/reward_completions/mean")) for row in rows if "reward" in row or "rewards/reward_completions/mean" in row]
    if xs:
        plt.plot(xs, ys, color="#7c3aed", linewidth=2, marker="o")
    else:
        plt.text(0.5, 0.5, "Warehouse GRPO reward not available yet", ha="center", va="center", transform=plt.gca().transAxes)
    plt.title("Warehouse GRPO Reward")
    plt.xlabel("logged step")
    plt.ylabel("role reward")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOTS / "submission_warehouse_grpo_reward.png", dpi=180)
    plt.close()


def plot_eval(rows: list[dict[str, Any]]) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    metric_by_role = {"center": "mean_center_role_score", "warehouse": "mean_warehouse_role_score"}
    for role in ("warehouse", "center"):
        role_rows = [row for row in rows if row.get("role") == role]
        if not role_rows:
            continue
        labels = [str(row.get("label")) for row in role_rows]
        global_scores = [as_float(row.get("mean_global_score")) or 0.0 for row in role_rows]
        role_scores = [as_float(row.get(metric_by_role[role])) or 0.0 for row in role_rows]
        invalid_payloads = [as_float(row.get("invalid_payloads")) or 0.0 for row in role_rows]
        invalid_actions = [as_float(row.get("invalid_actions")) or 0.0 for row in role_rows]

        plt.figure(figsize=(9, 5))
        xs = range(len(labels))
        width = 0.35
        plt.bar([x - width / 2 for x in xs], global_scores, width=width, label="global score", color="#2563eb")
        plt.bar([x + width / 2 for x in xs], role_scores, width=width, label=f"{role} role score", color="#0f766e")
        plt.xticks(list(xs), labels)
        plt.ylim(0, 1)
        plt.title(f"{role.title()} Held-out Scores")
        plt.ylabel("normalized score")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS / f"submission_{role}_scores.png", dpi=180)
        plt.close()

        plt.figure(figsize=(9, 5))
        plt.bar([x - width / 2 for x in xs], invalid_payloads, width=width, label="invalid payloads", color="#dc2626")
        plt.bar([x + width / 2 for x in xs], invalid_actions, width=width, label="invalid env actions", color="#d97706")
        plt.xticks(list(xs), labels)
        plt.title(f"{role.title()} Invalid Outputs")
        plt.ylabel("count across held-out eval")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS / f"submission_{role}_invalids.png", dpi=180)
        plt.close()


def make_markdown(training: dict[str, dict[str, Any]], eval_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SupplyMind Training Evidence",
        "",
        "This file is generated from local HF job logs. It is intentionally compact for README/blog reuse.",
        "",
        "## Training Runs",
        "",
        "| run | steps | first loss | last loss | min loss | invalid payloads | invalid actions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in training.items():
        if not row.get("exists"):
            continue
        lines.append(
            f"| {key} | {row.get('step_count', 0)} | {fmt(row.get('first_loss'))} | {fmt(row.get('last_loss'))} | "
            f"{fmt(row.get('min_loss'))} | {int(row.get('invalid_payloads') or 0)} | {int(row.get('invalid_actions') or 0)} |"
        )
    lines.extend(["", "## Held-out Evaluation", "", "| role | label | global score | role score | raw reward | invalid payloads | invalid actions | action totals |", "|---|---|---:|---:|---:|---:|---:|---|"])
    for row in eval_rows:
        role = str(row.get("role", "-"))
        role_metric = "mean_center_role_score" if role == "center" else "mean_warehouse_role_score"
        action_totals = row.get("action_totals") or {}
        compact_totals = ", ".join(f"{key}:{value}" for key, value in action_totals.items() if value) or "-"
        lines.append(
            f"| {role} | {row.get('label', '-')} | {fmt(row.get('mean_global_score'))} | {fmt(row.get(role_metric))} | "
            f"{fmt(row.get('mean_raw_reward'))} | {int(row.get('invalid_payloads') or 0)} | {int(row.get('invalid_actions') or 0)} | {compact_totals} |"
        )

    warehouse_rows = [row for row in eval_rows if row.get("role") == "warehouse"]
    center_rows = [row for row in eval_rows if row.get("role") == "center"]
    lines.extend(["", "## Current Read"])
    if warehouse_rows:
        best = max(warehouse_rows, key=lambda row: as_float(row.get("mean_warehouse_role_score")) or -1)
        lines.append(f"- Best warehouse role score so far: `{best.get('label')}` at `{fmt(best.get('mean_warehouse_role_score'))}`.")
    if center_rows:
        best = max(center_rows, key=lambda row: as_float(row.get("mean_center_role_score")) or -1)
        lines.append(f"- Best center role score so far: `{best.get('label')}` at `{fmt(best.get('mean_center_role_score'))}`.")
    lines.extend(
        [
            "- Promotion rule: prefer the checkpoint that improves held-out role score without increasing invalid payloads/actions.",
            "",
            "## Plots",
            "",
            "- `results/plots/submission_training_loss.png`",
            "- `results/plots/submission_warehouse_grpo_reward.png`",
            "- `results/plots/submission_warehouse_scores.png`",
            "- `results/plots/submission_warehouse_invalids.png`",
            "- `results/plots/submission_center_scores.png`",
            "- `results/plots/submission_center_invalids.png`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    training = {key: parse_training_log(path) for key, path in TRAINING_LOGS.items()}
    eval_rows: list[dict[str, Any]] = []
    for path in EVAL_LOGS.values():
        eval_rows.extend(parse_eval_log(path))
    eval_rows = dedupe_eval_rows(eval_rows)

    plot_training(training)
    plot_eval(eval_rows)

    payload = {"training": training, "eval_rows": eval_rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(make_markdown(training, eval_rows), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote plots to {PLOTS}")


if __name__ == "__main__":
    main()
