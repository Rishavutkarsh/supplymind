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
    parser.add_argument("--adapter-id", required=True)
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
    while not env.done:
        payload, _generated = generate_action(model, tokenizer, role, observation, max_new_tokens)
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
            if role == "center":
                action = V2JointAction(warehouse_actions=fixed_warehouse_actions(observation), central_action={})
            else:
                action = V2JointAction(warehouse_actions={}, central_action=fixed_center_action(observation, {}))
        result = env.step(action)
        invalid_actions += len(result.observation.feedback.get("invalid_action_details", []))
        observation = result.observation
        steps += 1
    summary = dict(env.last_episode_summary or {})
    summary.update({"seed": seed, "steps": steps, "invalid_payloads": invalid_payloads, "invalid_actions": invalid_actions})
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
    }
    log("eval_result", **aggregate)
    return aggregate


def main() -> None:
    started = time.time()
    args = parse_args()
    prepare_repo()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]

    log("loading_base_model")
    base_model, tokenizer = load_model()
    base = evaluate(args.role, "base", base_model, tokenizer, args.task_id, seeds, args.max_new_tokens)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    log("loading_adapter_model", adapter_id=args.adapter_id)
    adapter_model, tokenizer = load_model(args.adapter_id)
    trained = evaluate(args.role, "adapter", adapter_model, tokenizer, args.task_id, seeds, args.max_new_tokens)

    result = {"message": "eval_done", "elapsed_seconds": round(time.time() - started, 2), "base": base, "adapter": trained}
    log(**result)


if __name__ == "__main__":
    main()
