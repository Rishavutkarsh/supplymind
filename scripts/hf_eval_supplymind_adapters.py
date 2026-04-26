# /// script
# dependencies = [
#   "torch",
#   "transformers>=4.45.0",
#   "peft>=0.13.0",
#   "accelerate",
#   "huggingface_hub",
#   "pydantic",
#   "pyyaml",
# ]
# ///
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ID = "rishavutk/supplymind"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def log(message: str, **fields: Any) -> None:
    print(json.dumps({"message": message, **fields}, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["center", "warehouse"], required=True)
    parser.add_argument("--adapter-id", default="")
    parser.add_argument("--sft-adapter-id", default="")
    parser.add_argument("--grpo-adapter-id", default="")
    parser.add_argument("--task-id", default="v2_train_easy")
    parser.add_argument("--seeds", default="101,113,127")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def prepare_repo() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    local_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="space",
            allow_patterns=["src/**", "configs/**"],
            local_dir=Path("supplymind_snapshot").resolve(),
        )
    )
    os.environ["SUPPLYMIND_REWARD_CONFIG"] = str(local_dir / "configs" / "supplymind_v2_rewards.yaml")
    sys.path.insert(0, str(local_dir / "src"))
    log("repo_ready", path=str(local_dir), config_exists=Path(os.environ["SUPPLYMIND_REWARD_CONFIG"]).exists())


def extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def compact_observation(observation: Any, role: str) -> dict[str, Any]:
    data = observation.model_dump(mode="json")
    data["scenario_info"].pop("public_rules", None)
    if role == "center":
        return {
            "role": "center",
            "round_index": data["round_index"],
            "remaining_rounds": data["remaining_rounds"],
            "task_id": data["task_id"],
            "scenario_info": data["scenario_info"],
            "center": data["center"],
            "warehouse_summaries": data["center"].get("warehouse_summaries", []),
            "market_signals": data["center"].get("market_signals", []),
            "feedback": data.get("feedback", {}),
        }
    return {
        "role": "warehouse",
        "round_index": data["round_index"],
        "remaining_rounds": data["remaining_rounds"],
        "task_id": data["task_id"],
        "scenario_info": data["scenario_info"],
        "warehouses": data["warehouses"],
        "pending_transfer_proposals": data["center"].get("pending_transfer_proposals", []),
        "feedback": data.get("feedback", {}),
    }


def system_prompt(role: str) -> str:
    if role == "center":
        return (
            "You are the center policy in SupplyMind. Return only strict JSON matching CenterAction. "
            "Use central_procurements, central_liquidations, central_replenishments, inventory_transfer_proposals, offer_matches."
        )
    return (
        "You are the shared warehouse policy in SupplyMind. Return only strict JSON with key warehouse_actions mapping warehouse ids to actions. "
        "Use order_decisions, inventory_offers, inventory_requests, transfer_responses, and local_priority."
    )


def generate_action(model: Any, tokenizer: Any, role: str, observation: Any, max_new_tokens: int) -> tuple[dict[str, Any] | None, str]:
    prompt = [
        {"role": "system", "content": system_prompt(role)},
        {"role": "user", "content": json.dumps(compact_observation(observation, role), separators=(",", ":"))},
    ]
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    return extract_json(generated), generated


def _len_list(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key, [])
    return len(value) if isinstance(value, list) else 0


def center_action_stats(payload: dict[str, Any] | None) -> dict[str, int]:
    payload = payload or {}
    return {
        "central_procurements": _len_list(payload, "central_procurements"),
        "central_liquidations": _len_list(payload, "central_liquidations"),
        "central_replenishments": _len_list(payload, "central_replenishments"),
        "inventory_transfer_proposals": _len_list(payload, "inventory_transfer_proposals"),
        "offer_matches": _len_list(payload, "offer_matches"),
    }


def warehouse_action_stats(payload: dict[str, Any] | None) -> dict[str, int]:
    payload = payload or {}
    warehouses = payload.get("warehouse_actions", {})
    if not isinstance(warehouses, dict):
        warehouses = {}
    stats = {
        "warehouses_controlled": len(warehouses),
        "order_decisions": 0,
        "inventory_offers": 0,
        "inventory_requests": 0,
        "transfer_responses": 0,
        "local_priority": 0,
    }
    for action in warehouses.values():
        if isinstance(action, dict):
            for key in ("order_decisions", "inventory_offers", "inventory_requests", "transfer_responses", "local_priority"):
                stats[key] += _len_list(action, key)
    return stats


def action_stats(role: str, payload: dict[str, Any] | None) -> dict[str, int]:
    return center_action_stats(payload) if role == "center" else warehouse_action_stats(payload)


def load_model(adapter_id: str | None = None) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")
    if adapter_id:
        model = PeftModel.from_pretrained(model, adapter_id)
    model.eval()
    return model, tokenizer


def rollout(role: str, model: Any, tokenizer: Any, task_id: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
    from supplymind_env_v2.environment import V2SupplyMindEnv
    from supplymind_env_v2.models import CenterAction, V2JointAction, V2WarehouseRoleAction
    from supplymind_env_v2.policies import fixed_center_action, fixed_warehouse_actions

    env = V2SupplyMindEnv(default_task_id=task_id)
    observation = env.reset_internal(task_id, seed)
    invalid_payloads = 0
    invalid_actions = 0
    steps = 0
    action_totals: Counter[str] = Counter()
    parsed_payloads = 0
    fallback_steps = 0
    samples: list[dict[str, Any]] = []
    while not env.done:
        payload, generated = generate_action(model, tokenizer, role, observation, max_new_tokens)
        parsed_before_validation = payload is not None
        if parsed_before_validation:
            parsed_payloads += 1
            action_totals.update(action_stats(role, payload))
        try:
            if payload is None:
                raise ValueError("missing_json")
            if role == "center":
                action = V2JointAction(
                    warehouse_actions=fixed_warehouse_actions(observation),
                    central_action=CenterAction.model_validate(payload),
                )
            else:
                warehouse_role_action = V2WarehouseRoleAction.model_validate(payload)
                action = V2JointAction(
                    warehouse_actions=warehouse_role_action.warehouse_actions,
                    central_action=fixed_center_action(observation, warehouse_role_action.warehouse_actions),
                )
        except Exception:
            invalid_payloads += 1
            fallback_steps += 1
            if role == "center":
                action = V2JointAction(warehouse_actions=fixed_warehouse_actions(observation), central_action={})
            else:
                action = V2JointAction(warehouse_actions={}, central_action=fixed_center_action(observation, {}))
        result = env.step(action)
        if len(samples) < 3:
            samples.append(
                {
                    "round_index": observation.round_index,
                    "parsed": parsed_before_validation,
                    "fallback": payload is None,
                    "action_stats": action_stats(role, payload),
                    "generated_preview": generated[:600],
                    "parsed_payload": payload,
                    "invalid_action_details": result.observation.feedback.get("invalid_action_details", [])[:5],
                }
            )
        invalid_actions += len(result.observation.feedback.get("invalid_action_details", []))
        observation = result.observation
        steps += 1
    summary = dict(env.last_episode_summary or {})
    summary.update(
        {
            "seed": seed,
            "steps": steps,
            "invalid_payloads": invalid_payloads,
            "invalid_actions": invalid_actions,
            "parsed_payloads": parsed_payloads,
            "fallback_steps": fallback_steps,
            "action_totals": dict(action_totals),
            "samples": samples,
        }
    )
    return summary


def evaluate(role: str, label: str, model: Any, tokenizer: Any, task_id: str, seeds: list[int], max_new_tokens: int) -> dict[str, Any]:
    episodes = [rollout(role, model, tokenizer, task_id, seed, max_new_tokens) for seed in seeds]
    aggregate = {
        "label": label,
        "role": role,
        "episodes": episodes,
        "mean_global_score": mean(float(row.get("graded_score", 0.0)) for row in episodes),
        "mean_center_role_score": mean(float(row.get("center_role_score", 0.0)) for row in episodes),
        "mean_warehouse_role_score": mean(float(row.get("warehouse_role_score", 0.0)) for row in episodes),
        "mean_raw_reward": mean(float(row.get("raw_reward", 0.0)) for row in episodes),
        "invalid_payloads": sum(int(row["invalid_payloads"]) for row in episodes),
        "invalid_actions": sum(int(row["invalid_actions"]) for row in episodes),
        "parsed_payloads": sum(int(row.get("parsed_payloads", 0)) for row in episodes),
        "fallback_steps": sum(int(row.get("fallback_steps", 0)) for row in episodes),
        "action_totals": dict(sum((Counter(row.get("action_totals", {})) for row in episodes), Counter())),
        "sample_generations": [
            {"seed": row.get("seed"), **sample}
            for row in episodes
            for sample in row.get("samples", [])[:1]
        ][:5],
    }
    log("eval_result", **aggregate)
    return aggregate


def main() -> None:
    started = time.time()
    args = parse_args()
    if not (args.adapter_id or args.sft_adapter_id or args.grpo_adapter_id):
        raise SystemExit("Provide --adapter-id or --sft-adapter-id/--grpo-adapter-id.")
    prepare_repo()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]

    log("loading_base_model")
    base_model, tokenizer = load_model()
    base = evaluate(args.role, "base", base_model, tokenizer, args.task_id, seeds, args.max_new_tokens)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    evaluations: dict[str, Any] = {"base": base}
    adapter_specs = []
    if args.adapter_id:
        adapter_specs.append(("adapter", args.adapter_id))
    if args.sft_adapter_id:
        adapter_specs.append(("sft", args.sft_adapter_id))
    if args.grpo_adapter_id:
        adapter_specs.append(("grpo", args.grpo_adapter_id))
    for label, adapter_id in adapter_specs:
        log("loading_adapter_model", label=label, adapter_id=adapter_id)
        adapter_model, tokenizer = load_model(adapter_id)
        evaluations[label] = evaluate(args.role, label, adapter_model, tokenizer, args.task_id, seeds, args.max_new_tokens)
        del adapter_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {"message": "eval_done", "elapsed_seconds": round(time.time() - started, 2), **evaluations}
    log(**result)


if __name__ == "__main__":
    main()
