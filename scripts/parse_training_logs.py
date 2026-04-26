"""Parse HF training job logs into the local training dashboard JSON."""

from __future__ import annotations

import argparse
import ast
import json
import re
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_OUTPUT = Path("results/training_dashboard.json")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ROLE_RE = re.compile(r"\b(center|warehouse)\b", re.IGNORECASE)
PHASE_RE = re.compile(r"\b(sft|grpo)\b", re.IGNORECASE)


def read_text(path: Path) -> str:
    """Read UTF-8 and PowerShell UTF-16 redirected logs."""
    data = path.read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if data[:200].count(b"\x00") > 20:
        return data.decode("utf-16-le")

    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    text = read_text(path).strip()
    if not text:
        return {}

    loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return loaded


def to_number(value: Any) -> Any:
    if isinstance(value, (int, float)) or value is None:
        return value
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if not stripped:
        return value
    try:
        return float(stripped)
    except ValueError:
        return value


def normalize_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: to_number(value) for key, value in values.items()}


def parse_json_line(line: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_python_dict_line(line: str) -> dict[str, Any] | None:
    if not (line.startswith("{") and line.endswith("}")):
        return None
    try:
        parsed = ast.literal_eval(line)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def looks_like_step(record: dict[str, Any]) -> bool:
    return any(
        key in record
        for key in (
            "loss",
            "reward",
            "rewards/reward_completions/mean",
            "completions/mean_length",
            "completions/clipped_ratio",
        )
    ) and "train_runtime" not in record


def infer_role(path: Path, records: list[dict[str, Any]]) -> str:
    for record in records:
        role = record.get("role")
        if isinstance(role, str) and role.lower() in {"center", "warehouse"}:
            return role.lower()

    match = ROLE_RE.search(path.stem)
    if match:
        return match.group(1).lower()

    raise ValueError(f"Could not infer role for {path}; use --center-log or --warehouse-log")


def infer_phase(path: Path, records: list[dict[str, Any]]) -> str | None:
    for record in records:
        phase = record.get("phase") or record.get("training_phase") or record.get("stage")
        if isinstance(phase, str) and phase.lower() in {"sft", "grpo"}:
            return phase.lower()

    match = PHASE_RE.search(path.stem)
    return match.group(1).lower() if match else None


def series_key(role: str, phase: str | None) -> str:
    return f"{role}_{phase}" if phase else role


def parse_log(path: Path, role_hint: str | None = None, phase_hint: str | None = None) -> tuple[str, dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    reward_batches: list[dict[str, Any]] = []
    role_records: list[dict[str, Any]] = []

    for raw_line in read_text(path).splitlines():
        line = ANSI_RE.sub("", raw_line).strip()
        if not line:
            continue

        json_record = parse_json_line(line)
        if json_record:
            normalized = normalize_mapping(json_record)
            role_records.append(normalized)
            if normalized.get("message") == "reward_batch":
                reward_batches.append(normalized)
            continue

        dict_record = parse_python_dict_line(line)
        if dict_record and looks_like_step(dict_record):
            step = normalize_mapping(dict_record)
            step.setdefault("reward", step.get("rewards/reward_completions/mean"))
            step.setdefault("completion_length", step.get("completions/mean_length"))
            step.setdefault("clipped_ratio", step.get("completions/clipped_ratio"))
            step["step"] = len(steps) + 1
            steps.append(step)

    role = role_hint or infer_role(path, role_records)
    phase = phase_hint or infer_phase(path, role_records)
    return series_key(role, phase), {
        "source_log": str(path),
        "role": role,
        "phase": phase,
        "steps": steps,
        "reward_batches": reward_batches,
    }


def parse_eval_log(path: Path) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for raw_line in read_text(path).splitlines():
        line = ANSI_RE.sub("", raw_line).strip()
        if not line:
            continue
        record = parse_json_line(line)
        if not record or record.get("message") != "eval_result":
            continue
        comparisons.append(normalize_mapping({key: value for key, value in record.items() if key != "episodes"}))
    return comparisons


def average(values: list[Any]) -> float | None:
    numbers = [value for value in values if isinstance(value, (int, float))]
    return mean(numbers) if numbers else None


def summarize(series: dict[str, Any]) -> dict[str, Any]:
    steps = series.get("steps", [])
    reward_batches = series.get("reward_batches", [])
    rewards = [step.get("reward") for step in steps]
    losses = [step.get("loss") for step in steps]
    clipped = [step.get("completions/clipped_ratio") for step in steps]
    lengths = [step.get("completions/mean_length") for step in steps]

    summary: dict[str, Any] = {
        "steps": len(steps),
        "reward_batches": len(reward_batches),
        "invalid_payloads": sum(int(batch.get("invalid_payloads") or 0) for batch in reward_batches),
        "invalid_actions": sum(int(batch.get("invalid_actions") or 0) for batch in reward_batches),
    }

    numeric_rewards = [value for value in rewards if isinstance(value, (int, float))]
    if numeric_rewards:
        summary.update(
            {
                "first_reward": rewards[0],
                "last_reward": rewards[-1],
                "best_reward": max(numeric_rewards),
                "mean_reward": average(rewards),
            }
        )
    if losses:
        summary.update(
            {
                "first_loss": losses[0],
                "last_loss": losses[-1],
                "mean_loss": average(losses),
            }
        )

    mean_clipped = average(clipped)
    mean_length = average(lengths)
    if mean_clipped is not None:
        summary["mean_clipped_ratio"] = mean_clipped
    if mean_length is not None:
        summary["mean_completion_length"] = mean_length

    return summary


def discover_logs() -> list[tuple[str | None, str | None, Path]]:
    return [(None, None, path) for path in sorted(Path("results").glob("*.log"))]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center-log", type=Path, help="HF training log for the center role")
    parser.add_argument("--warehouse-log", type=Path, help="HF training log for the warehouse role")
    parser.add_argument("--center-sft-log", type=Path, help="SFT training log for the center role")
    parser.add_argument("--warehouse-sft-log", type=Path, help="SFT training log for the warehouse role")
    parser.add_argument("--center-grpo-log", type=Path, help="GRPO training log for the center role")
    parser.add_argument("--warehouse-grpo-log", type=Path, help="GRPO training log for the warehouse role")
    parser.add_argument("--eval-log", action="append", type=Path, default=[], help="HF eval log containing eval_result JSON rows")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Dashboard JSON path")
    parser.add_argument("--status", help="Optional top-level dashboard status")
    parser.add_argument("--updated-at", help="Optional top-level dashboard updated_at value")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    requested_logs: list[tuple[str | None, str | None, Path]] = []
    if args.center_log:
        requested_logs.append(("center", None, args.center_log))
    if args.warehouse_log:
        requested_logs.append(("warehouse", None, args.warehouse_log))
    if args.center_sft_log:
        requested_logs.append(("center", "sft", args.center_sft_log))
    if args.warehouse_sft_log:
        requested_logs.append(("warehouse", "sft", args.warehouse_sft_log))
    if args.center_grpo_log:
        requested_logs.append(("center", "grpo", args.center_grpo_log))
    if args.warehouse_grpo_log:
        requested_logs.append(("warehouse", "grpo", args.warehouse_grpo_log))
    if not requested_logs:
        requested_logs = discover_logs()

    dashboard = load_json(args.output)
    dashboard.setdefault("training_series", {})
    dashboard.setdefault("training_summary", {})

    parsed_keys: list[str] = []
    valid_keys = {"center", "warehouse", "center_sft", "center_grpo", "warehouse_sft", "warehouse_grpo"}
    for role_hint, phase_hint, log_path in requested_logs:
        if not log_path.exists():
            raise FileNotFoundError(log_path)
        try:
            key, series = parse_log(log_path, role_hint, phase_hint)
        except ValueError:
            if role_hint is None:
                continue
            raise
        if key not in valid_keys:
            continue
        if not series.get("steps") and not series.get("reward_batches"):
            continue
        dashboard["training_series"][key] = series
        dashboard["training_summary"][key] = summarize(series)
        role = series.get("role")
        if key != role and role in {"center", "warehouse"}:
            dashboard["training_series"].setdefault(role, series)
            dashboard["training_summary"].setdefault(role, summarize(series))
        parsed_keys.append(key)

    comparisons: list[dict[str, Any]] = []
    for eval_log in args.eval_log:
        if not eval_log.exists():
            raise FileNotFoundError(eval_log)
        comparisons.extend(parse_eval_log(eval_log))
    if comparisons:
        seen: set[tuple[str, str]] = set()
        unique = []
        for row in reversed(comparisons):
            key = (str(row.get("role")), str(row.get("label")))
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        dashboard["comparisons"] = list(reversed(unique))

    if args.status:
        dashboard["status"] = args.status
    if args.updated_at:
        dashboard["updated_at"] = args.updated_at
    elif parsed_keys and "updated_at" not in dashboard:
        dashboard["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} with series: {', '.join(parsed_keys) or 'none'}")


if __name__ == "__main__":
    main()
